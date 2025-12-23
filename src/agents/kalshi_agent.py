"""
🌙 Moon Dev's Kalshi Prediction Market Agent
============================================
AI-powered analysis of Kalshi prediction markets using 6-model swarm consensus.

Features:
- Fetches active markets from Kalshi API (no auth required for public data)
- Runs 6 AI models in parallel for consensus analysis
- Filters high-value picks based on profit margins
- Outputs to CSV for review or auto-trading

Author: Moon Dev
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from termcolor import cprint
from dotenv import load_dotenv
import threading

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

load_dotenv()

# ==============================================================================
# CONFIGURATION - Customize these settings
# ==============================================================================

# Kalshi API settings
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"  # Public API
USE_DEMO_API = False  # Set True to use demo.kalshi.com

# Market filtering
MIN_VOLUME = 100  # Minimum volume to consider market (lowered - Kalshi has less volume than Polymarket)
MAX_MARKETS_TO_ANALYZE = 20  # How many markets to send to AI swarm
MARKET_STATUS_FILTER = "open"  # Only open markets

# 🌙 Moon Dev - Time Sensitivity Filters (avoid Greenland 2029 type bets!)
MAX_DAYS_TO_RESOLUTION = 90   # Skip markets resolving > 90 days out (capital locked too long)
MIN_DAYS_TO_RESOLUTION = 1    # Skip markets resolving in < 24 hours (too volatile, less edge)
PRIORITIZE_IMMINENT = True    # Boost markets resolving in 7-30 days (sweet spot)
IMMINENT_DAYS_MIN = 7         # Start of "imminent" window
IMMINENT_DAYS_MAX = 30        # End of "imminent" window

# 🌙 Moon Dev - Spike Detection (catch mispricing!)
SPIKE_THRESHOLD_PCT = 0.10    # Alert on 10%+ price moves (potential mispricing)
NEW_MARKET_HOURS = 48         # Flag markets created in last 48 hours (early mover edge)

# 🌙 Moon Dev - Liquidity Filters
MIN_OPEN_INTEREST = 100       # Minimum open interest (lowered - Kalshi has less liquidity than Polymarket)

# Profit Margin Filters for High-Value Trades
MIN_PROFIT_MARGIN = 0.05  # Minimum 5% profit margin
MAX_PRICE_FOR_YES = 0.80  # Don't buy YES above this price
MIN_PRICE_FOR_YES = 0.10  # Don't buy YES below this price
MIN_CONSENSUS_FOR_HIGH_VALUE = 5  # Minimum X out of 6 models must agree

# Position Sizing
BANKROLL_USD = 5000  # Your total trading bankroll in USD
RISK_PER_TRADE_PCT = 0.02  # Risk 2% of bankroll per trade = $100 per trade

# Paper Trading / Auto-Run Configuration
PAPER_TRADING_MODE = True  # Set True for paper trading (no real money)
AUTO_RUN_LOOP = True  # Set True to run continuously
HOURS_BETWEEN_RUNS = 1  # Hours between analysis runs (1 = 24x per day for more signals)
PAPER_STARTING_BALANCE = 5000  # Starting paper balance in USD

# Filter out sports betting terminology
IGNORE_BETTING_KEYWORDS = [
    'moneyline', 'spread', 'spreads', 'total', 'totals', 'over/under',
    'point spread', 'handicap', 'nfl', 'nba', 'mlb', 'nhl',
]

# AI Swarm Configuration
USE_SWARM_MODE = True
SWARM_MODELS = [
    {"type": "deepseek", "name": "deepseek-chat"},
    {"type": "claude", "name": "claude-sonnet-4-5"},
    {"type": "claude", "name": "claude-opus-4-5-20251101"},
    {"type": "openai", "name": "gpt-4o"},
    {"type": "gemini", "name": "gemini-2.5-flash"},
    {"type": "groq", "name": "llama-3.3-70b-versatile"},
]

# Analysis prompts
ANALYSIS_PROMPT = """You are an expert prediction market analyst. Analyze these Kalshi markets and predict the most likely outcomes.

For each market, consider:
1. Current pricing (YES price indicates market's implied probability)
2. Historical patterns and base rates
3. Recent news and developments
4. TIME SENSITIVITY - Pay special attention to:
   - 🔥 IMMINENT markets (7-30 days) = best capital efficiency, faster resolution
   - 🆕 NEW markets (< 48h old) = potential mispricing before liquidity stabilizes
   - Days to resolution affects risk/reward (shorter = less uncertainty)

Markets to analyze:
{markets_text}

For EACH market, respond with:
Market [number]: [title]
Side: YES or NO (your prediction)
Confidence: [1-10]
Reasoning: [1-2 sentences, mention time sensitivity if relevant]

PRIORITIZE markets that are:
1. Imminent (7-30 days) - faster capital turnover
2. New (< 48h) - potential mispricing opportunity
3. High conviction contrarian plays - where market seems wrong

Be contrarian when the odds seem mispriced. Look for value where the market probability differs from your estimated true probability.
"""

CONSENSUS_PROMPT = """You are aggregating predictions from 6 AI models on Kalshi prediction markets.

Here are all the model predictions:
{all_predictions}

Original markets:
{markets_text}

Create a RANKED list of the TOP 10 markets with STRONGEST CONSENSUS (most models agree).

For each pick, format EXACTLY like this:
1. Market [number]: [title]
Side: [YES/NO]
Consensus: [X] out of 6 models agree
Reasoning: [Combined reasoning from agreeing models]
Link: https://kalshi.com/markets/[ticker]

Rank by consensus strength (6/6 first, then 5/6, etc). Only include markets where at least 4 models agree.
"""

# Data paths
DATA_FOLDER = os.path.join(project_root, "src/data/kalshi")
MARKETS_CSV = os.path.join(DATA_FOLDER, "markets.csv")
PREDICTIONS_CSV = os.path.join(DATA_FOLDER, "predictions.csv")
CONSENSUS_PICKS_CSV = os.path.join(DATA_FOLDER, "consensus_picks.csv")
HIGH_VALUE_PICKS_CSV = os.path.join(DATA_FOLDER, "high_value_picks.csv")
PAPER_TRADES_CSV = os.path.join(DATA_FOLDER, "paper_trades.csv")
PAPER_PORTFOLIO_CSV = os.path.join(DATA_FOLDER, "paper_portfolio.csv")

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# ==============================================================================
# Kalshi API Client
# ==============================================================================

class KalshiClient:
    """Simple Kalshi API client for public market data"""

    def __init__(self):
        self.base_url = KALSHI_API_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def get_events(self, limit=100, status="open"):
        """Get list of events (collections of markets)"""
        try:
            url = f"{self.base_url}/events"
            params = {
                "limit": limit,
                "status": status,
            }
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get("events", [])
        except Exception as e:
            cprint(f"Error fetching events: {e}", "red")
            return []

    def get_markets(self, limit=200, status="open", cursor=None):
        """Get list of markets"""
        try:
            url = f"{self.base_url}/markets"
            params = {
                "limit": limit,
                "status": status,
            }
            if cursor:
                params["cursor"] = cursor

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("markets", []), data.get("cursor")
        except Exception as e:
            cprint(f"Error fetching markets: {e}", "red")
            return [], None

    def get_market(self, ticker):
        """Get single market by ticker"""
        try:
            url = f"{self.base_url}/markets/{ticker}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json().get("market", {})
        except Exception as e:
            cprint(f"Error fetching market {ticker}: {e}", "red")
            return {}

    def get_orderbook(self, ticker):
        """Get orderbook for a market"""
        try:
            url = f"{self.base_url}/markets/{ticker}/orderbook"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json().get("orderbook", {})
        except Exception as e:
            cprint(f"Error fetching orderbook {ticker}: {e}", "red")
            return {}


# ==============================================================================
# Kalshi Agent
# ==============================================================================

class KalshiAgent:
    """AI-powered Kalshi prediction market analyzer"""

    def __init__(self):
        self.client = KalshiClient()
        self.markets_df = pd.DataFrame()
        self.csv_lock = threading.Lock()

        # Import model factory singleton (already initialized)
        from src.models.model_factory import model_factory
        self.model_factory = model_factory

    def fetch_markets(self):
        """Fetch active markets from Kalshi with time sensitivity and spike detection"""
        cprint("\n📊 Fetching Kalshi markets...", "cyan", attrs=['bold'])

        all_markets = []
        cursor = None

        # Fetch markets with pagination (with rate limit protection)
        for _ in range(5):  # Max 5 pages
            markets, cursor = self.client.get_markets(
                limit=200,
                status=MARKET_STATUS_FILTER,
                cursor=cursor
            )
            all_markets.extend(markets)

            if not cursor:
                break

            # 🌙 Moon Dev - Rate limit protection (Kalshi returns 429 if too fast)
            time.sleep(0.5)  # Wait 500ms between pages

        cprint(f"📈 Found {len(all_markets)} total markets", "green")

        # Filter and sort markets
        filtered_markets = []
        now = datetime.now()
        skipped_time = 0
        skipped_volume = 0
        skipped_sports = 0
        skipped_oi = 0
        imminent_count = 0
        new_market_count = 0

        for m in all_markets:
            # Skip low volume
            volume = m.get("volume", 0) or 0
            if volume < MIN_VOLUME:
                skipped_volume += 1
                continue

            # Skip low open interest (liquidity filter)
            open_interest = m.get("open_interest", 0) or 0
            if open_interest < MIN_OPEN_INTEREST:
                skipped_oi += 1
                continue

            # Skip sports betting
            title = m.get("title", "").lower()
            is_sports = any(kw in title for kw in IGNORE_BETTING_KEYWORDS)
            if is_sports:
                skipped_sports += 1
                continue

            # 🌙 Moon Dev - Time Sensitivity Filter
            close_time_str = m.get("close_time", "")
            days_to_resolution = None
            is_imminent = False
            is_new_market = False

            if close_time_str:
                try:
                    # Parse ISO format close time
                    close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                    close_time = close_time.replace(tzinfo=None)  # Remove timezone for comparison
                    days_to_resolution = (close_time - now).days

                    # Skip if too far out (like Greenland 2029!)
                    if days_to_resolution > MAX_DAYS_TO_RESOLUTION:
                        skipped_time += 1
                        continue

                    # Skip if too close (less than 24 hours - too volatile)
                    if days_to_resolution < MIN_DAYS_TO_RESOLUTION:
                        skipped_time += 1
                        continue

                    # Flag imminent markets (7-30 days - sweet spot!)
                    if IMMINENT_DAYS_MIN <= days_to_resolution <= IMMINENT_DAYS_MAX:
                        is_imminent = True
                        imminent_count += 1

                except Exception:
                    pass  # If we can't parse time, include it anyway

            # 🌙 Moon Dev - New Market Detection (first 48 hours = mispricing opportunity)
            created_time_str = m.get("created_time", "") or m.get("open_time", "")
            if created_time_str:
                try:
                    created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                    created_time = created_time.replace(tzinfo=None)
                    hours_since_created = (now - created_time).total_seconds() / 3600
                    if hours_since_created <= NEW_MARKET_HOURS:
                        is_new_market = True
                        new_market_count += 1
                except Exception:
                    pass

            # Get pricing
            yes_price = m.get("yes_bid", 0) or m.get("last_price", 50) / 100
            if isinstance(yes_price, (int, float)) and yes_price > 1:
                yes_price = yes_price / 100  # Convert cents to dollars

            filtered_markets.append({
                "ticker": m.get("ticker", ""),
                "title": m.get("title", ""),
                "subtitle": m.get("subtitle", ""),
                "yes_price": yes_price,
                "no_price": 1 - yes_price,
                "volume": volume,
                "open_interest": open_interest,
                "close_time": close_time_str,
                "days_to_resolution": days_to_resolution,
                "is_imminent": is_imminent,
                "is_new_market": is_new_market,
                "event_ticker": m.get("event_ticker", ""),
                "category": m.get("category", ""),
            })

        # 🌙 Moon Dev - Smart sorting: prioritize imminent + new markets
        def sort_key(x):
            score = x["volume"]  # Base score is volume
            if PRIORITIZE_IMMINENT and x.get("is_imminent"):
                score *= 1.5  # 50% boost for imminent markets
            if x.get("is_new_market"):
                score *= 1.3  # 30% boost for new markets (mispricing opportunity)
            return score

        filtered_markets.sort(key=sort_key, reverse=True)

        # Take top N
        top_markets = filtered_markets[:MAX_MARKETS_TO_ANALYZE]

        # 🌙 Moon Dev - Show filtering stats
        cprint(f"\n📊 Filtering Stats:", "yellow", attrs=['bold'])
        cprint(f"   ⏰ Skipped {skipped_time} markets (outside {MIN_DAYS_TO_RESOLUTION}-{MAX_DAYS_TO_RESOLUTION} day window)", "yellow")
        cprint(f"   📉 Skipped {skipped_volume} markets (volume < ${MIN_VOLUME})", "yellow")
        cprint(f"   💧 Skipped {skipped_oi} markets (open interest < ${MIN_OPEN_INTEREST})", "yellow")
        cprint(f"   ⚽ Skipped {skipped_sports} sports betting markets", "yellow")
        cprint(f"   🔥 Found {imminent_count} imminent markets ({IMMINENT_DAYS_MIN}-{IMMINENT_DAYS_MAX} days)", "green")
        cprint(f"   🆕 Found {new_market_count} new markets (< {NEW_MARKET_HOURS}h old - mispricing edge!)", "green")

        cprint(f"\n✅ Selected {len(top_markets)} high-value markets for analysis", "green")

        # Convert to DataFrame and save
        self.markets_df = pd.DataFrame(top_markets)
        with self.csv_lock:
            self.markets_df.to_csv(MARKETS_CSV, index=False)

        return self.markets_df

    def run_swarm_analysis(self, markets_df):
        """Run AI swarm analysis on markets"""
        if markets_df.empty:
            cprint("⚠️ No markets to analyze", "yellow")
            return

        cprint("\n🤖 Starting AI Swarm Analysis...", "magenta", attrs=['bold'])
        cprint(f"📊 Analyzing {len(markets_df)} markets with {len(SWARM_MODELS)} AI models", "cyan")

        # Prepare markets text
        markets_text = self._format_markets_for_ai(markets_df)

        # Run all models in parallel
        all_predictions = {}
        threads = []

        for model_config in SWARM_MODELS:
            thread = threading.Thread(
                target=self._run_single_model,
                args=(model_config, markets_text, all_predictions)
            )
            threads.append(thread)
            thread.start()

        # Wait for all models
        for thread in threads:
            thread.join(timeout=120)

        cprint(f"\n✅ Got predictions from {len(all_predictions)} models", "green")

        # Get consensus
        self._get_consensus(all_predictions, markets_text, markets_df)

    def _format_markets_for_ai(self, markets_df):
        """Format markets for AI analysis with time sensitivity info"""
        lines = []
        for i, (_, row) in enumerate(markets_df.iterrows()):
            yes_price = row.get('yes_price', 0.5)
            days = row.get('days_to_resolution', 'Unknown')
            is_imminent = row.get('is_imminent', False)
            is_new = row.get('is_new_market', False)

            # Build time info string
            time_info = f"Days to Resolution: {days}"
            if is_imminent:
                time_info += " 🔥 IMMINENT (7-30 day sweet spot)"
            if is_new:
                time_info += " 🆕 NEW MARKET (< 48h old - potential mispricing!)"

            lines.append(
                f"Market {i+1}:\n"
                f"Title: {row['title']}\n"
                f"Subtitle: {row.get('subtitle', '')}\n"
                f"Current Price: YES=${yes_price:.2f} ({yes_price*100:.0f}% implied probability)\n"
                f"Volume: ${row.get('volume', 0):,}\n"
                f"Open Interest: ${row.get('open_interest', 0):,}\n"
                f"{time_info}\n"
                f"Ticker: {row['ticker']}\n"
            )
        return "\n".join(lines)

    def _run_single_model(self, model_config, markets_text, results_dict):
        """Run analysis with a single model"""
        model_name = f"{model_config['type']}:{model_config['name']}"
        try:
            cprint(f"   🔄 Starting {model_name}...", "cyan")

            model = self.model_factory.get_model(model_config['type'], model_config['name'])
            prompt = ANALYSIS_PROMPT.format(markets_text=markets_text)

            response = model.generate_response(
                system_prompt="You are an expert prediction market analyst.",
                user_content=prompt,
                temperature=0.3,
                max_tokens=4000
            )

            if response and response.content:
                results_dict[model_name] = response.content
                cprint(f"   ✅ {model_name} complete", "green")
            else:
                cprint(f"   ⚠️ {model_name} returned empty response", "yellow")

        except Exception as e:
            cprint(f"   ❌ {model_name} failed: {e}", "red")

    def _get_consensus(self, all_predictions, markets_text, markets_df):
        """Get consensus from all model predictions"""
        if not all_predictions:
            cprint("⚠️ No predictions to aggregate", "yellow")
            return

        cprint("\n🎯 Getting Consensus...", "magenta", attrs=['bold'])

        # Format all predictions
        predictions_text = ""
        for model_name, prediction in all_predictions.items():
            predictions_text += f"\n=== {model_name} ===\n{prediction}\n"

        # Save predictions
        with self.csv_lock:
            with open(PREDICTIONS_CSV, 'w', encoding='utf-8') as f:
                f.write(predictions_text)

        # Use Claude for consensus
        try:
            model = self.model_factory.get_model('claude', 'claude-sonnet-4-5')
            prompt = CONSENSUS_PROMPT.format(
                all_predictions=predictions_text,
                markets_text=markets_text
            )

            response = model.generate_response(
                system_prompt="You are aggregating AI predictions for Kalshi markets.",
                user_content=prompt,
                temperature=0.2,
                max_tokens=4000
            )

            if response and response.content:
                cprint("\n" + "="*80, "white", "on_blue", attrs=['bold'])
                cprint("🎯 KALSHI CONSENSUS PICKS", "white", "on_blue", attrs=['bold'])
                cprint("="*80, "white", "on_blue", attrs=['bold'])
                cprint(response.content, "cyan", attrs=['bold'])
                cprint("="*80 + "\n", "white", "on_blue", attrs=['bold'])

                # Save consensus and high-value picks
                self._save_consensus_picks_to_csv(response.content, markets_df)
                self._save_high_value_picks_to_csv(response.content, markets_df)

        except Exception as e:
            cprint(f"❌ Error getting consensus: {e}", "red")
            import traceback
            traceback.print_exc()

    def _save_consensus_picks_to_csv(self, consensus_response, markets_df):
        """Save consensus picks to CSV"""
        try:
            import re

            picks = []
            lines = consensus_response.split('\n')
            current_pick = {}

            for line in lines:
                line = line.strip()

                # Match "1. Market X: Title"
                market_match = re.match(r'(\d+)\.\s+Market\s+(\d+):\s+(.+)', line)
                if market_match:
                    if current_pick:
                        picks.append(current_pick)

                    current_pick = {
                        'rank': market_match.group(1),
                        'market_number': int(market_match.group(2)),
                        'market_title': market_match.group(3)
                    }

                elif line.startswith('Side:'):
                    current_pick['side'] = line.replace('Side:', '').strip()

                elif line.startswith('Consensus:'):
                    consensus_text = line.replace('Consensus:', '').strip()
                    current_pick['consensus'] = consensus_text
                    match = re.search(r'(\d+)\s+out\s+of\s+(\d+)', consensus_text)
                    if match:
                        current_pick['consensus_count'] = int(match.group(1))
                        current_pick['total_models'] = int(match.group(2))

                elif line.startswith('Link:'):
                    current_pick['link'] = line.replace('Link:', '').strip()

                elif line.startswith('Reasoning:'):
                    current_pick['reasoning'] = line.replace('Reasoning:', '').strip()

            if current_pick:
                picks.append(current_pick)

            if not picks:
                cprint("⚠️ Could not parse consensus picks", "yellow")
                return

            # Convert to records
            timestamp = datetime.now().isoformat()
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            records = []
            markets_list = list(markets_df.iterrows())

            for pick in picks:
                market_num = pick.get('market_number', 0)
                ticker = ""
                if 1 <= market_num <= len(markets_list):
                    _, row = markets_list[market_num - 1]
                    ticker = row.get('ticker', '')

                records.append({
                    'timestamp': timestamp,
                    'run_id': run_id,
                    'rank': pick.get('rank', ''),
                    'market_title': pick.get('market_title', ''),
                    'ticker': ticker,
                    'side': pick.get('side', ''),
                    'consensus_count': pick.get('consensus_count', ''),
                    'total_models': pick.get('total_models', ''),
                    'reasoning': pick.get('reasoning', ''),
                    'link': pick.get('link', '')
                })

            # Save to CSV
            if os.path.exists(CONSENSUS_PICKS_CSV):
                df = pd.read_csv(CONSENSUS_PICKS_CSV)
            else:
                df = pd.DataFrame()

            df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)

            with self.csv_lock:
                df.to_csv(CONSENSUS_PICKS_CSV, index=False)

            cprint(f"✅ Saved {len(records)} consensus picks to CSV", "green")

        except Exception as e:
            cprint(f"❌ Error saving consensus picks: {e}", "red")

    def _calculate_profit_margin(self, yes_price, side):
        """Calculate profit margin for a trade"""
        try:
            yes_price = float(yes_price)
            no_price = 1.0 - yes_price
            side = side.upper()

            if side == 'YES':
                buy_price = yes_price
                profit_margin = 1.0 - yes_price
                is_valid = MIN_PRICE_FOR_YES <= yes_price <= MAX_PRICE_FOR_YES
            elif side == 'NO':
                buy_price = no_price
                profit_margin = 1.0 - no_price
                is_valid = MIN_PRICE_FOR_YES <= no_price <= MAX_PRICE_FOR_YES
            else:
                return (0.0, 0.0, False)

            is_high_value = profit_margin >= MIN_PROFIT_MARGIN and is_valid
            return (profit_margin, buy_price, is_high_value)

        except Exception as e:
            return (0.0, 0.0, False)

    def _save_high_value_picks_to_csv(self, consensus_response, markets_df):
        """Save high-value picks with profit calculations"""
        try:
            import re

            cprint("\n💎 Analyzing picks for HIGH VALUE opportunities...", "yellow", attrs=['bold'])

            suggested_bet_usd = BANKROLL_USD * RISK_PER_TRADE_PCT
            cprint(f"💰 Suggested bet size: ${suggested_bet_usd:.2f}", "cyan")

            # Parse picks
            picks = []
            lines = consensus_response.split('\n')
            current_pick = {}

            for line in lines:
                line = line.strip()

                market_match = re.match(r'(\d+)\.\s+Market\s+(\d+):\s+(.+)', line)
                if market_match:
                    if current_pick:
                        picks.append(current_pick)
                    current_pick = {
                        'rank': market_match.group(1),
                        'market_number': int(market_match.group(2)),
                        'market_title': market_match.group(3)
                    }
                elif line.startswith('Side:'):
                    current_pick['side'] = line.replace('Side:', '').strip()
                elif line.startswith('Consensus:'):
                    match = re.search(r'(\d+)\s+out\s+of\s+(\d+)', line)
                    if match:
                        current_pick['consensus_count'] = int(match.group(1))
                        current_pick['total_models'] = int(match.group(2))
                elif line.startswith('Link:'):
                    current_pick['link'] = line.replace('Link:', '').strip()
                elif line.startswith('Reasoning:'):
                    current_pick['reasoning'] = line.replace('Reasoning:', '').strip()

            if current_pick:
                picks.append(current_pick)

            if not picks:
                return

            # Filter high-value picks
            high_value_picks = []
            markets_list = list(markets_df.iterrows())

            for pick in picks:
                market_num = pick.get('market_number', 0)
                side = pick.get('side', '').upper()
                consensus_count = pick.get('consensus_count', 0)

                if consensus_count < MIN_CONSENSUS_FOR_HIGH_VALUE:
                    cprint(f"   ❌ Market {market_num}: Only {consensus_count} models agreed", "red")
                    continue

                # Get price from markets_df
                yes_price = 0.5
                ticker = ""
                if 1 <= market_num <= len(markets_list):
                    _, row = markets_list[market_num - 1]
                    yes_price = float(row.get('yes_price', 0.5))
                    ticker = row.get('ticker', '')

                profit_margin, buy_price, is_high_value = self._calculate_profit_margin(yes_price, side)

                if is_high_value:
                    pick['ticker'] = ticker
                    pick['yes_price'] = yes_price
                    pick['no_price'] = 1.0 - yes_price
                    pick['buy_price'] = round(buy_price, 2)
                    pick['profit_margin_pct'] = round(profit_margin * 100, 1)
                    high_value_picks.append(pick)
                    cprint(f"   ✅ Market {market_num}: HIGH VALUE! Buy {side} @ ${buy_price:.2f} = {profit_margin*100:.1f}%", "green", attrs=['bold'])
                else:
                    cprint(f"   ❌ Market {market_num}: {side} @ ${buy_price:.2f} = {profit_margin*100:.1f}% (filtered)", "red")

            if not high_value_picks:
                cprint("\n💔 No HIGH VALUE picks found this round", "yellow")
                return

            # Save to CSV
            timestamp = datetime.now().isoformat()
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            records = []
            for pick in high_value_picks:
                buy_price = pick['buy_price']
                shares = suggested_bet_usd / buy_price if buy_price > 0 else 0
                payout = shares * 1.0
                profit = payout - suggested_bet_usd

                records.append({
                    'timestamp': timestamp,
                    'run_id': run_id,
                    'market_title': pick.get('market_title', ''),
                    'ticker': pick.get('ticker', ''),
                    'side': pick.get('side', ''),
                    'buy_price': buy_price,
                    'yes_price': pick.get('yes_price', ''),
                    'no_price': pick.get('no_price', ''),
                    'profit_margin_pct': pick.get('profit_margin_pct', ''),
                    'suggested_bet_usd': suggested_bet_usd,
                    'shares_you_get': round(shares, 1),
                    'payout_if_win_usd': round(payout, 2),
                    'potential_profit_usd': round(profit, 2),
                    'consensus_count': pick.get('consensus_count', ''),
                    'total_models': pick.get('total_models', ''),
                    'reasoning': pick.get('reasoning', ''),
                    'link': pick.get('link', '')
                })

            if os.path.exists(HIGH_VALUE_PICKS_CSV):
                df = pd.read_csv(HIGH_VALUE_PICKS_CSV)
            else:
                df = pd.DataFrame()

            df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)

            with self.csv_lock:
                df.to_csv(HIGH_VALUE_PICKS_CSV, index=False)

            # Display summary
            cprint("\n" + "="*80, "green")
            cprint("💎 KALSHI HIGH VALUE TRADES SUMMARY 💎", "white", "on_green", attrs=['bold'])
            cprint("="*80, "green")

            for pick in high_value_picks:
                buy_price = pick['buy_price']
                shares = suggested_bet_usd / buy_price if buy_price > 0 else 0
                profit = shares - suggested_bet_usd

                cprint(f"   🎯 Buy {pick['side']} @ ${buy_price:.2f}", "cyan", attrs=['bold'])
                cprint(f"      📌 {pick['market_title'][:60]}...", "white")
                cprint(f"      🏷️ Ticker: {pick.get('ticker', 'N/A')}", "white")
                cprint(f"      💲 YES: ${pick['yes_price']:.2f} | NO: ${pick['no_price']:.2f}", "magenta")
                cprint(f"      💵 Bet ${suggested_bet_usd:.2f} → Get {shares:.1f} shares → Profit ${profit:.2f} ({pick['profit_margin_pct']}% ROI)", "yellow", attrs=['bold'])
                cprint(f"      ✅ Consensus: {pick['consensus_count']}/{pick['total_models']} AI models agreed", "green")
                cprint("", "white")

            cprint("="*80 + "\n", "green")
            cprint(f"💎 Saved {len(records)} HIGH VALUE picks to CSV!", "green", attrs=['bold'])
            cprint(f"📁 High Value CSV: {HIGH_VALUE_PICKS_CSV}", "cyan")

        except Exception as e:
            cprint(f"❌ Error saving high value picks: {e}", "red")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main run loop"""
        cprint("\n" + "="*80, "cyan")
        cprint("🌙 Moon Dev's Kalshi Agent Starting...", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")
        cprint(f"💰 Bankroll: ${BANKROLL_USD}", "yellow")
        cprint(f"📊 Max Markets: {MAX_MARKETS_TO_ANALYZE}", "yellow")
        cprint(f"🤖 AI Models: {len(SWARM_MODELS)}", "yellow")
        cprint(f"💎 Min Consensus: {MIN_CONSENSUS_FOR_HIGH_VALUE}/{len(SWARM_MODELS)}", "yellow")
        cprint("", "yellow")
        cprint("⏰ Time Sensitivity Filters:", "magenta", attrs=['bold'])
        cprint(f"   📅 Resolution window: {MIN_DAYS_TO_RESOLUTION}-{MAX_DAYS_TO_RESOLUTION} days", "magenta")
        cprint(f"   🔥 Imminent boost: {IMMINENT_DAYS_MIN}-{IMMINENT_DAYS_MAX} days (+50% priority)", "magenta")
        cprint(f"   🆕 New market detection: < {NEW_MARKET_HOURS}h old (+30% priority)", "magenta")
        cprint("", "yellow")
        cprint("💧 Liquidity Filters:", "cyan", attrs=['bold'])
        cprint(f"   📈 Min Volume: ${MIN_VOLUME}", "cyan")
        cprint(f"   📊 Min Open Interest: ${MIN_OPEN_INTEREST}", "cyan")
        cprint("="*80 + "\n", "cyan")

        # Fetch markets
        markets_df = self.fetch_markets()

        if markets_df.empty:
            cprint("❌ No markets found. Exiting.", "red")
            return

        # Run AI analysis
        self.run_swarm_analysis(markets_df)

        cprint("\n✅ Analysis complete!", "green", attrs=['bold'])
        cprint(f"📁 Data saved to: {DATA_FOLDER}", "cyan")


def execute_paper_trade(pick_data):
    """Execute a paper trade and log it to CSV"""
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    trade_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load or create paper trades CSV
    if os.path.exists(PAPER_TRADES_CSV):
        trades_df = pd.read_csv(PAPER_TRADES_CSV)
    else:
        trades_df = pd.DataFrame(columns=[
            'trade_id', 'timestamp', 'ticker', 'market_title', 'side',
            'buy_price', 'shares', 'cost_usd', 'status', 'exit_price',
            'exit_timestamp', 'pnl_usd', 'consensus_count'
        ])

    # Calculate trade details
    bet_size = BANKROLL_USD * RISK_PER_TRADE_PCT
    buy_price = pick_data.get('buy_price', 0.5)
    shares = bet_size / buy_price if buy_price > 0 else 0

    new_trade = {
        'trade_id': trade_id,
        'timestamp': timestamp,
        'ticker': pick_data.get('ticker', ''),
        'market_title': pick_data.get('market_title', ''),
        'side': pick_data.get('side', ''),
        'buy_price': round(buy_price, 4),
        'shares': round(shares, 2),
        'cost_usd': round(bet_size, 2),
        'status': 'OPEN',
        'exit_price': None,
        'exit_timestamp': None,
        'pnl_usd': None,
        'consensus_count': pick_data.get('consensus_count', 0)
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([new_trade])], ignore_index=True)
    trades_df.to_csv(PAPER_TRADES_CSV, index=False)

    cprint(f"   📝 PAPER TRADE: {new_trade['side']} {new_trade['shares']:.1f} shares @ ${buy_price:.2f} = ${bet_size:.2f}", "green", attrs=['bold'])
    return new_trade


def get_paper_portfolio_summary():
    """Get summary of paper trading portfolio"""
    if not os.path.exists(PAPER_TRADES_CSV):
        return {
            'balance': PAPER_STARTING_BALANCE,
            'open_positions': 0,
            'total_trades': 0,
            'realized_pnl': 0
        }

    trades_df = pd.read_csv(PAPER_TRADES_CSV)
    open_trades = trades_df[trades_df['status'] == 'OPEN']
    closed_trades = trades_df[trades_df['status'] == 'CLOSED']

    total_invested = open_trades['cost_usd'].sum() if len(open_trades) > 0 else 0
    realized_pnl = closed_trades['pnl_usd'].sum() if len(closed_trades) > 0 else 0

    return {
        'balance': PAPER_STARTING_BALANCE + realized_pnl - total_invested,
        'open_positions': len(open_trades),
        'total_trades': len(trades_df),
        'realized_pnl': realized_pnl,
        'invested': total_invested
    }


def main():
    """Main entry point with continuous loop support"""
    run_count = 0

    cprint("\n" + "="*80, "cyan")
    cprint("🌙 Moon Dev's Kalshi Paper Trading Bot", "cyan", attrs=['bold'])
    cprint("="*80, "cyan")
    cprint(f"💰 Paper Balance: ${PAPER_STARTING_BALANCE}", "green")
    cprint(f"🔄 Auto-Loop: {'ON' if AUTO_RUN_LOOP else 'OFF'}", "yellow")
    cprint(f"⏱️  Run Interval: {HOURS_BETWEEN_RUNS} hours", "yellow")
    cprint(f"📊 Mode: {'PAPER TRADING' if PAPER_TRADING_MODE else 'ANALYSIS ONLY'}", "magenta")
    cprint("="*80 + "\n", "cyan")

    try:
        while True:
            run_count += 1
            cprint(f"\n{'='*80}", "blue")
            cprint(f"🚀 Run #{run_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "blue", attrs=['bold'])
            cprint(f"{'='*80}\n", "blue")

            # Show portfolio summary
            if PAPER_TRADING_MODE:
                summary = get_paper_portfolio_summary()
                cprint(f"💼 Portfolio: ${summary['balance']:.2f} | Open: {summary['open_positions']} | P&L: ${summary['realized_pnl']:.2f}", "cyan")

            # Run analysis
            agent = KalshiAgent()
            agent.run()

            # Execute paper trades if enabled - only for picks from THIS run (last 5 minutes)
            if PAPER_TRADING_MODE and os.path.exists(HIGH_VALUE_PICKS_CSV):
                try:
                    hv_df = pd.read_csv(HIGH_VALUE_PICKS_CSV)
                    if len(hv_df) > 0 and 'timestamp' in hv_df.columns:
                        # Only trade picks created in the last 5 minutes (this run)
                        hv_df['timestamp'] = pd.to_datetime(hv_df['timestamp'])
                        five_mins_ago = datetime.now() - timedelta(minutes=5)
                        current_run_picks = hv_df[hv_df['timestamp'] >= five_mins_ago]

                        if len(current_run_picks) > 0:
                            cprint(f"\n📊 Found {len(current_run_picks)} NEW high-value picks to paper trade", "yellow")
                            for _, pick in current_run_picks.iterrows():
                                execute_paper_trade(pick.to_dict())
                        else:
                            cprint(f"\n📊 No new picks from this run (skipping old CSV data)", "yellow")
                except Exception as e:
                    cprint(f"⚠️ Paper trade error: {e}", "yellow")

            if not AUTO_RUN_LOOP:
                cprint("\n✅ Single run complete. Set AUTO_RUN_LOOP=True for continuous mode.", "green")
                break

            # Sleep until next run
            next_run = datetime.now() + timedelta(hours=HOURS_BETWEEN_RUNS)
            cprint(f"\n😴 Sleeping until next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
            cprint(f"⏱️  ({HOURS_BETWEEN_RUNS} hours = {HOURS_BETWEEN_RUNS * 60} minutes)", "cyan")

            # Sleep in 1-minute chunks so we can catch keyboard interrupt
            for _ in range(HOURS_BETWEEN_RUNS * 60):
                time.sleep(60)

    except KeyboardInterrupt:
        cprint("\n\n" + "="*80, "yellow")
        cprint("⚠️ Kalshi Paper Trading Bot stopped by user", "yellow", attrs=['bold'])
        if PAPER_TRADING_MODE:
            summary = get_paper_portfolio_summary()
            cprint(f"💼 Final Portfolio: ${summary['balance']:.2f}", "cyan")
            cprint(f"📊 Total Trades: {summary['total_trades']} | Realized P&L: ${summary['realized_pnl']:.2f}", "cyan")
        cprint("="*80 + "\n", "yellow")
    except Exception as e:
        cprint(f"\n❌ Error: {e}", "red")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
