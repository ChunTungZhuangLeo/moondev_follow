"""
Kalshi Value Trading Agent
==========================
AI-powered value trading on Kalshi prediction markets.

Replaces bluff-heavy strategies with high-probability edge detection.
Optimized for Game of 25 environment with 50% loose / 50% tight opponents.

Core Principles:
1. VALUE > Bluffs: Only trade when edge is substantial
2. AI Consensus: 5/6 models must agree before entry
3. Kelly Sizing: Position size based on edge magnitude
4. Time-Optimized: Focus on 7-30 day resolution window

Architecture:
- KalshiAuthClient: RSA-PSS authenticated API access
- ValueTradingEngine: Edge detection and position sizing
- KalshiTradeExecutor: Safe trade execution with paper/live modes
- AI Swarm: Multi-model consensus for predictions

Usage:
    # Paper trading mode (default)
    python src/agents/kalshi_value_agent.py

    # Live trading (requires API keys)
    python src/agents/kalshi_value_agent.py --live

Author: Moon Dev
"""

import os
import sys
import time
import json
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from termcolor import cprint
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

load_dotenv()


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# API Settings
KALSHI_PUBLIC_API = "https://api.elections.kalshi.com/trade-api/v2"

# Market Filtering
MAX_MARKETS_TO_ANALYZE = 0  # 0 = analyze ALL markets (no limit)
MIN_VOLUME = 0  # No minimum - include all markets
MIN_OPEN_INTEREST = 0  # No minimum - Kalshi API may not return this
MARKET_STATUS = "open"

# Series to fetch - FOCUSED on categories where AI has strongest edge
# Strategy: Weather, Economics, Entertainment/Culture have clearest AI advantage
# Avoid: Politics (emotion-driven), Crypto (too swingy), Sports (efficient markets)
MARKET_SERIES = [
    # === WEATHER (AI EDGE: HIGH) ===
    # AI excels at: analyzing forecast models, historical patterns, seasonal trends
    # Market inefficiency: retail traders rely on intuition, not data
    "KXLOWNYC",         # NYC temperature lows
    "KXDENHIGH",        # Denver temperature highs
    "KXHURRICANE",      # Hurricane events
    "KXHIGHNYC",        # NYC temperature highs
    "KXLOWCHI",         # Chicago temperature lows
    "KXHIGHCHI",        # Chicago temperature highs
    "KXLOWLA",          # LA temperature lows
    "KXHIGHLA",         # LA temperature highs
    "KXSNOW",           # Snowfall predictions
    "KXRAIN",           # Rainfall predictions
    "KXWILDFIRE",       # Wildfire predictions

    # === ECONOMIC DATA (AI EDGE: HIGH) ===
    # AI excels at: modeling economic indicators, Fed behavior, data patterns
    # Market inefficiency: complex data, most traders don't understand nuances
    "KXJOBLESSCLAIMS",  # Weekly jobless claims
    "KXCPICOMBO",       # CPI data combinations
    "KXCPI",            # CPI individual
    "KXPROLLS",         # Jobs numbers/payrolls
    "KXGDP",            # GDP data
    "KXGDPUSMAX",       # GDP max quarters
    "KXPCE",            # PCE inflation
    "KXUNRATE",         # Unemployment rate
    "KXRETAIL",         # Retail sales
    "KXHOUSING",        # Housing data
    "KXFED",            # Fed funds rate decisions
    "KXFOMC",           # FOMC meeting outcomes
    "KXRATES",          # Interest rate levels
    "KXRECESSION",      # Recession predictions
    "KXINFLATION",      # Inflation levels

    # === ENTERTAINMENT & CULTURE (AI EDGE: MEDIUM-HIGH) ===
    # AI excels at: analyzing historical patterns, critic consensus, industry trends
    # Market inefficiency: emotional betting, limited data analysis by traders
    "KXGTAPRICE",       # GTA VI price
    "KXOSCARS",         # Oscar awards
    "KXGRAMMYS",        # Grammy awards
    "KXEMMYS",          # Emmy awards
    "KXGOLDENGLOBES",   # Golden Globes
    "KXBOXOFFICE",      # Box office records
    "KXSTREAMING",      # Streaming milestones
    "KXSPOTIFY",        # Spotify records
    "KXGAMEAWARDS",     # Video game awards
    "KXESPORTS",        # Esports outcomes

    # === TECH & AI (AI EDGE: MEDIUM) ===
    # AI excels at: understanding tech capabilities, development timelines
    # Market inefficiency: hype-driven pricing, misunderstood timelines
    "OAIAGI",           # OpenAI AGI announcement
    "KXAI",             # AI-related events
    "KXGPT",            # GPT model releases
    "KXSELF",           # Self-driving milestones
    "KXQUANTUM",        # Quantum computing

    # === COMMODITIES (AI EDGE: MEDIUM) ===
    # AI excels at: supply/demand modeling, weather impact analysis
    "KXNGASW",          # Natural gas weekly
    "KXOIL",            # Oil prices
    "KXGOLD",           # Gold prices

    # === RARE EVENTS (AI EDGE: HIGH for NO bets) ===
    # AI excels at: base rate analysis - these are often overpriced
    # Market inefficiency: fear/excitement drives YES prices too high
    "KXMETEOR",         # Meteor strikes
    "KXALIEN",          # Alien contact
    "KXWW3",            # World War 3
    "KXNUCLEAR",        # Nuclear events

    # === STOCK MARKET (AI EDGE: LOW - included for volume) ===
    # Efficient markets, but occasional mispricing on extreme moves
    "NASDAQ100U",       # Nasdaq above/below
    "INXD",             # S&P 500 daily
    "KXINXW",           # S&P 500 weekly
    "KXINX",            # S&P 500 levels
    "KXDOW",            # Dow Jones
    "KXVIX",            # Volatility index
]

# Time Sensitivity - REMOVED strict filtering
# We can hold long-term positions OR sell early when edge is captured
MAX_DAYS_TO_RESOLUTION = 365  # Cap at 1 year - avoid locking capital for years
MIN_DAYS_TO_RESOLUTION = 0  # Include same-day markets
MIN_POSITION_SIZE_USD = 10.0  # Don't execute trades smaller than this (waste of API calls)
SWEET_SPOT_MIN = 1  # Short-term sweet spot (for sorting priority)
SWEET_SPOT_MAX = 90  # Expanded sweet spot

# Value Trading Thresholds
MIN_EDGE_PCT = 0.08  # 8% minimum edge for short-term (<60 days)
MIN_CONSENSUS = 4  # 4/6 models must agree (lowered from 5/6)
MIN_PROFIT_MARGIN = 0.15  # 15% minimum profit margin

# Time-Adjusted Edge Requirements
# Longer-term markets need higher edge to justify capital lockup
# The edge must be high enough that price should move within ~60 days
TIME_ADJUSTED_EDGE = {
    60: 0.08,    # < 60 days: 8% edge (standard)
    180: 0.15,   # 60-180 days: 15% edge (need more edge for longer hold)
    365: 0.25,   # 180-365 days: 25% edge (significant edge for 6+ months)
}

# Volume/Momentum Filter for Long-Term Markets (>60 days)
# Only enter if market shows recent activity (people are trading it)
LONG_TERM_MIN_VOLUME = 100  # Minimum 24h volume for markets >60 days
LONG_TERM_DAYS_THRESHOLD = 60  # Apply volume filter to markets >60 days out


def get_required_edge(days_to_resolution: Optional[int]) -> float:
    """
    Calculate required edge based on days to resolution.

    Time-Adjusted Edge Requirements:
    - < 60 days: 8% edge (standard)
    - 60-180 days: 15% edge (need more edge for longer hold)
    - 180-365 days: 25% edge (significant edge for 6+ months)

    Args:
        days_to_resolution: Days until market resolves (None defaults to standard edge)

    Returns:
        Required edge as decimal (e.g., 0.08 for 8%)
    """
    if days_to_resolution is None:
        return MIN_EDGE_PCT  # Default to standard edge

    # Find the appropriate threshold
    for threshold_days, required_edge in sorted(TIME_ADJUSTED_EDGE.items()):
        if days_to_resolution <= threshold_days:
            return required_edge

    # If beyond all thresholds, return the highest requirement
    return max(TIME_ADJUSTED_EDGE.values())


def check_volume_filter(days_to_resolution: Optional[int], volume: int) -> Tuple[bool, str]:
    """
    Check if market passes volume filter for long-term markets.

    Long-term markets (>60 days) need minimum volume to ensure:
    1. Market has liquidity for entry/exit
    2. There's actual interest/trading activity
    3. Prices are being actively updated

    Args:
        days_to_resolution: Days until market resolves
        volume: 24h trading volume

    Returns:
        Tuple of (passes_filter, reason_if_failed)
    """
    if days_to_resolution is None:
        return True, ""

    if days_to_resolution > LONG_TERM_DAYS_THRESHOLD:
        if volume < LONG_TERM_MIN_VOLUME:
            return False, f"Low volume ({volume} < {LONG_TERM_MIN_VOLUME}) for {days_to_resolution}d market"

    return True, ""


# Position Sizing (moderate for paper trading)
BANKROLL_USD = 5000.0
RISK_PER_TRADE_PCT = 0.05  # 5% max per trade = $250
KELLY_FRACTION = 0.50  # Half Kelly (more aggressive)

# Execution Mode
PAPER_TRADING_MODE = True
AUTO_RUN_LOOP = True
HOURS_BETWEEN_RUNS = 1  # Run every 1 hour (24x per day) - more frequent for better coverage

# Position Management
MAX_POSITIONS = 10  # Max concurrent open positions
ALLOW_POSITION_INCREASE = False  # Don't add to existing positions (creates duplicates in CSV)

# Filter keywords
IGNORE_KEYWORDS = [
    'moneyline', 'spread', 'spreads', 'total', 'totals', 'over/under',
    'point spread', 'handicap', 'nfl', 'nba', 'mlb', 'nhl', 'tennis',
    'soccer', 'premier league', 'champions league', 'ufc', 'boxing'
]

# AI Swarm Configuration - Ordered by cost (cheapest first to save on expensive reasoning)
# Cost ranking: Groq (FREE) < DeepSeek ($0.14/1M) < Gemini ($0.075/1M) < OpenAI ($0.15/1M) < Haiku ($0.25/1M) < Sonnet ($3/1M)
SWARM_MODELS = [
    {"type": "groq", "name": "llama-3.3-70b-versatile"},   # FREE - run first
    {"type": "deepseek", "name": "deepseek-chat"},         # Very cheap - $0.14/1M tokens
    {"type": "gemini", "name": "gemini-2.0-flash"},        # Cheap - $0.075/1M input
    {"type": "openai", "name": "gpt-4o-mini"},             # Cheap - $0.15/1M input
    {"type": "claude", "name": "claude-3-haiku-20240307"}, # Medium - $0.25/1M input
    {"type": "claude", "name": "claude-sonnet-4-5"},       # Expensive - $3/1M input (reasoning)
]

# Value Trading AI Prompt (optimized for edge detection)
VALUE_ANALYSIS_PROMPT = """You are a QUANTITATIVE prediction market analyst focused on VALUE TRADING.

Your goal is to find MISPRICED markets where the true probability differs significantly from the market price.

KEY PRINCIPLES:
1. VALUE TRADING = Only bet when YOU have the edge, not the market
2. Avoid "bluffs" - these are low-probability bets hoping for lucky outcomes
3. Focus on HIGH-PROBABILITY situations where market is clearly wrong
4. Time sensitivity matters - prefer 7-30 day resolution windows

For each market, calculate your estimated TRUE PROBABILITY vs the market's implied probability.
Edge = Your Probability - Market Probability

Markets to analyze:
{markets_text}

For EACH market, respond with EXACTLY this format:

MARKET [number]: [ticker]
SIDE: YES or NO (or SKIP if no edge)
TRUE_PROBABILITY: [your estimate 0.00-1.00]
MARKET_PROBABILITY: [current price]
EDGE: [TRUE_PROB - MARKET_PROB as percentage]
CONFIDENCE: [1-10]
REASONING: [2-3 sentences explaining WHY the market is mispriced. Be specific about what information the market is missing or underweighting.]

IMPORTANT:
- Only recommend trades with TRUE EDGE (>10% difference from market)
- SKIP markets where you're uncertain (say SKIP not YES/NO)
- Better to miss opportunities than take bad bets (VALUE strategy)
- Consider time sensitivity - imminent markets (7-30 days) are preferred

Remember: Professional traders make money by being SELECTIVE, not by trading often.
"""

# Single market analysis prompt (for per-market early exit)
SINGLE_MARKET_PROMPT = """Analyze this prediction market for VALUE TRADING opportunity:

MARKET: {ticker}
TITLE: {title}
CURRENT PRICE: YES=${yes_price:.2f} / NO=${no_price:.2f}
DAYS TO RESOLUTION: {days_to_close}

Your task:
1. Estimate the TRUE probability this resolves YES
2. Compare to market price to find edge
3. Recommend YES, NO, or SKIP

Respond in EXACTLY this format (one line each):
SIDE: [YES/NO/SKIP]
TRUE_PROB: [0.00-1.00]
EDGE: [percentage, can be negative]
CONFIDENCE: [1-10]
REASON: [one sentence]
"""

CONSENSUS_PROMPT = """You are aggregating VALUE TRADING predictions from 6 AI models.

Your job is to find the HIGHEST CONVICTION trades where multiple models agree on significant edge.

Model Predictions:
{all_predictions}

Original Markets:
{markets_text}

Create a RANKED list of VALUE TRADES meeting these criteria:
1. At least 5 out of 6 models agree on direction (YES or NO)
2. Average edge across models is >12%
3. Average confidence is >7/10

For each trade, calculate:
- Consensus: How many models agree
- Average Edge: Mean edge estimate across agreeing models
- Combined Confidence: Average confidence score

Format EXACTLY like this for each qualified trade:

RANK [1-10]: [ticker]
Title: [market title]
CONSENSUS: [X] out of 6 models agree on [YES/NO]
AVERAGE_EDGE: [X.X]%
AVERAGE_CONFIDENCE: [X.X]/10
TRUE_PROBABILITY: [consensus estimate]
MARKET_PRICE: [current price]
REASONING: [synthesized reasoning from all models]
LINK: https://kalshi.com/markets/[ticker]

Only include trades that meet ALL criteria. Quality over quantity.
If no trades qualify, say "NO QUALIFIED VALUE TRADES THIS ROUND"
"""

# Web-based re-evaluation prompt for existing positions
REEVAL_PROMPT = """You are re-evaluating an EXISTING position based on NEW information.

POSITION DETAILS:
- Market: {ticker}
- Title: {title}
- Our Side: {our_side} (we are betting {our_side})
- Entry Price: ${entry_price:.2f}
- Current Price: ${current_price:.2f}
- Unrealized P&L: {pnl_pct:+.1f}%
- Original AI Probability: {original_prob:.0%}
- Days to Resolution: {days_to_close}

RECENT NEWS/INFORMATION:
{web_context}

TASK:
Based on the NEW information above, re-assess whether our position thesis is still valid.

Consider:
1. Has any news changed the fundamental probability?
2. Is the market now pricing this correctly (our edge is gone)?
3. Should we EXIT (thesis broken) or HOLD (thesis intact)?

Respond in EXACTLY this format:
ACTION: [HOLD/EXIT]
NEW_PROB: [your updated probability estimate 0.00-1.00]
CONFIDENCE: [1-10]
REASONING: [2-3 sentences explaining why to hold or exit based on new info]
"""

# Data Paths - V2 uses separate folder
DATA_FOLDER = os.path.join(project_root, "src/data/kalshi_value_v2")
LOGS_FOLDER = os.path.join(DATA_FOLDER, "logs")
MARKETS_CSV = os.path.join(DATA_FOLDER, "markets.csv")
PREDICTIONS_CSV = os.path.join(DATA_FOLDER, "predictions.csv")
VALUE_PICKS_CSV = os.path.join(DATA_FOLDER, "value_picks.csv")
TRADE_LOG_CSV = os.path.join(DATA_FOLDER, "trade_log.csv")
PORTFOLIO_CSV = os.path.join(DATA_FOLDER, "portfolio.csv")
PRICE_HISTORY_CSV = os.path.join(DATA_FOLDER, "price_history.csv")  # Hourly price snapshots
REEVAL_LOG_CSV = os.path.join(DATA_FOLDER, "reeval_log.csv")  # Re-evaluation history

# Re-evaluation Configuration
REEVAL_PRICE_MOVE_THRESHOLD = 0.20  # Re-evaluate if price moved >20% from entry
REEVAL_LOSS_THRESHOLD = -0.15  # Re-evaluate if position is down >15%
REEVAL_DAYS_BEFORE_RESOLUTION = 7  # Re-evaluate positions within 7 days of resolution
REEVAL_MIN_HOURS_BETWEEN = 24  # Don't re-evaluate same position more than once per day

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)


# ==============================================================================
# KALSHI VALUE AGENT
# ==============================================================================

class KalshiValueAgent:
    """
    AI-powered value trading agent for Kalshi prediction markets.

    Implements edge-based trading with AI swarm consensus.
    Optimized for Game of 25 opponent environment.
    """

    def __init__(self, live_mode: bool = False):
        """
        Initialize the Value Trading Agent.

        Args:
            live_mode: If True, use authenticated client for live trading
        """
        self.live_mode = live_mode
        self.csv_lock = threading.Lock()
        self.markets_df = pd.DataFrame()

        # Import model factory
        from src.models.model_factory import model_factory
        self.model_factory = model_factory

        # Import strategy engine
        from src.agents.value_trading_strategy import (
            ValueTradingEngine,
            StrategyConfig,
            TradingStyle
        )
        self.strategy_engine = ValueTradingEngine(
            StrategyConfig(
                style=TradingStyle.VALUE,
                bankroll_usd=BANKROLL_USD,
                min_edge_pct=MIN_EDGE_PCT,
                min_consensus_models=MIN_CONSENSUS,
                kelly_fraction=KELLY_FRACTION
            )
        )

        # Import executor
        from src.agents.kalshi_trade_executor import (
            KalshiTradeExecutor,
            ExecutorConfig,
            ExecutionMode
        )

        exec_mode = ExecutionMode.LIVE if live_mode else ExecutionMode.PAPER

        # V2 uses separate data folder - pass paths to executor
        trades_csv_path = os.path.join(DATA_FOLDER, "trades.csv")
        positions_csv_path = os.path.join(DATA_FOLDER, "positions.csv")
        portfolio_csv_path = os.path.join(DATA_FOLDER, "portfolio.csv")

        cprint(f"\n📁 V2 Data Folder: {DATA_FOLDER}", "cyan")
        cprint(f"   Trades CSV: {trades_csv_path}", "cyan")

        self.executor = KalshiTradeExecutor(
            ExecutorConfig(
                mode=exec_mode,
                paper_starting_balance=BANKROLL_USD,
                max_daily_trades=20,
                max_daily_loss_usd=500.0,
                trades_csv_path=trades_csv_path,
                positions_csv_path=positions_csv_path,
                portfolio_csv_path=portfolio_csv_path
            )
        )

        # Authenticated client for live mode
        self.auth_client = None
        if live_mode:
            try:
                from src.agents.kalshi_auth_client import KalshiAuthClient
                self.auth_client = KalshiAuthClient.from_env()
                cprint("Kalshi authenticated client initialized", "green")
            except Exception as e:
                cprint(f"Warning: Could not initialize authenticated client: {e}", "yellow")
                cprint("Falling back to paper trading mode", "yellow")
                self.live_mode = False

        self._print_banner()

    def _print_banner(self):
        """Print agent startup banner."""
        mode = "LIVE" if self.live_mode else "PAPER"
        cprint("\n" + "="*70, "cyan")
        cprint(" KALSHI VALUE TRADING AGENT V2 (Web Re-Eval) ", "white", "on_cyan", attrs=['bold'])
        cprint("="*70, "cyan")
        cprint(f"  Mode: {mode} TRADING", "yellow" if mode == "PAPER" else "red")
        cprint(f"  Bankroll: ${BANKROLL_USD:,.0f}", "white")
        cprint(f"  Time-Adjusted Edge Requirements:", "white")
        for days, edge in sorted(TIME_ADJUSTED_EDGE.items()):
            cprint(f"    <{days}d: {edge*100:.0f}% min edge", "white")
        cprint(f"  Min Consensus: {MIN_CONSENSUS}/6 models", "white")
        cprint(f"  Kelly Fraction: {KELLY_FRACTION*100:.0f}%", "white")
        cprint(f"  Resolution Window: {MIN_DAYS_TO_RESOLUTION}-{MAX_DAYS_TO_RESOLUTION} days", "white")
        cprint(f"  Long-Term Volume Filter: >{LONG_TERM_DAYS_THRESHOLD}d needs {LONG_TERM_MIN_VOLUME}+ vol", "yellow")
        cprint(f"  Web Re-Evaluation: Enabled for underwater positions", "magenta")
        cprint("="*70 + "\n", "cyan")

    def fetch_markets(self) -> pd.DataFrame:
        """
        Fetch and filter markets from Kalshi public API.

        Fetches from multiple series categories to find edge opportunities.
        Time filtering is relaxed - we can hold long-term OR sell early.

        Returns:
            DataFrame of filtered markets
        """
        cprint("\nFetching Kalshi markets...", "cyan", attrs=['bold'])

        all_markets = []
        session = requests.Session()

        # Fetch from configured market series
        cprint(f"Querying {len(MARKET_SERIES)} market series...", "white")

        for series_ticker in MARKET_SERIES:
            try:
                params = {
                    "series_ticker": series_ticker,
                    "status": MARKET_STATUS,
                    "limit": 50
                }

                response = session.get(
                    f"{KALSHI_PUBLIC_API}/markets",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                markets = data.get("markets", [])
                all_markets.extend(markets)
                if markets:
                    cprint(f"  {series_ticker}: {len(markets)} markets", "green")

                time.sleep(0.2)  # Rate limiting

            except Exception as e:
                cprint(f"  {series_ticker}: Error - {e}", "yellow")

        # Also fetch general markets and filter by close_time
        cprint("Fetching general markets...", "white")
        cursor = None
        for page in range(3):
            try:
                params = {"limit": 200, "status": MARKET_STATUS}
                if cursor:
                    params["cursor"] = cursor

                response = session.get(
                    f"{KALSHI_PUBLIC_API}/markets",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                markets = data.get("markets", [])
                all_markets.extend(markets)

                cursor = data.get("cursor")
                if not cursor:
                    break

                time.sleep(0.3)

            except Exception as e:
                cprint(f"Error fetching general markets: {e}", "red")
                break

        # Deduplicate by ticker
        seen_tickers = set()
        unique_markets = []
        for m in all_markets:
            ticker = m.get('ticker', '')
            if ticker and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                unique_markets.append(m)

        all_markets = unique_markets
        cprint(f"Found {len(all_markets)} unique markets", "white")

        # Filter markets
        filtered = []
        now = datetime.now()

        stats = {
            'time_filtered': 0,
            'volume_filtered': 0,
            'sports_filtered': 0,
            'imminent': 0,
            'new_markets': 0
        }

        for m in all_markets:
            # Volume filter
            volume = m.get("volume", 0) or 0
            if volume < MIN_VOLUME:
                stats['volume_filtered'] += 1
                continue

            # Open interest filter
            open_interest = m.get("open_interest", 0) or 0
            if open_interest < MIN_OPEN_INTEREST:
                continue

            # Sports filter
            title = m.get("title", "").lower()
            if any(kw in title for kw in IGNORE_KEYWORDS):
                stats['sports_filtered'] += 1
                continue

            # Time filter - try multiple date fields
            close_time_str = m.get("close_time", "") or m.get("expiration_time", "") or m.get("expected_expiration_time", "")
            days_to_resolution = None
            is_imminent = False
            is_new = False

            if close_time_str:
                try:
                    close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                    close_time = close_time.replace(tzinfo=None)
                    days_to_resolution = (close_time - now).days

                    # Only filter if we have valid time data AND it's outside range
                    if days_to_resolution > MAX_DAYS_TO_RESOLUTION:
                        stats['time_filtered'] += 1
                        continue
                    if days_to_resolution < MIN_DAYS_TO_RESOLUTION:
                        stats['time_filtered'] += 1
                        continue

                    if SWEET_SPOT_MIN <= days_to_resolution <= SWEET_SPOT_MAX:
                        is_imminent = True
                        stats['imminent'] += 1

                except Exception:
                    # If we can't parse the date, include the market anyway
                    days_to_resolution = 30  # Default to middle of sweet spot

            # New market detection
            created_time_str = m.get("created_time", "") or m.get("open_time", "")
            if created_time_str:
                try:
                    created = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                    created = created.replace(tzinfo=None)
                    hours_since = (now - created).total_seconds() / 3600
                    if hours_since <= 48:
                        is_new = True
                        stats['new_markets'] += 1
                except Exception:
                    pass

            # Get pricing - Kalshi API returns cents (1-99), convert to decimal (0.01-0.99)
            yes_price = m.get("yes_bid", 0) or m.get("last_price", 50)
            if isinstance(yes_price, (int, float)):
                # Always convert from cents to decimal if value >= 1
                # (API returns 1-99 for cents, we want 0.01-0.99)
                if yes_price >= 1:
                    yes_price = yes_price / 100.0

            filtered.append({
                "ticker": m.get("ticker", ""),
                "title": m.get("title", ""),
                "subtitle": m.get("subtitle", ""),
                "yes_price": yes_price,
                "no_price": 1.0 - yes_price,
                "volume": volume,
                "open_interest": open_interest,
                "close_time": close_time_str,
                "days_to_resolution": days_to_resolution,
                "is_imminent": is_imminent,
                "is_new_market": is_new,
                "category": m.get("category", "")
            })

        # Smart sorting: prioritize imminent + new + volume
        def sort_score(x):
            score = x["volume"]
            if x.get("is_imminent"):
                score *= 2.0  # 2x boost for sweet spot
            if x.get("is_new_market"):
                score *= 1.5  # 1.5x boost for new markets
            return score

        filtered.sort(key=sort_score, reverse=True)

        # Apply limit only if MAX_MARKETS_TO_ANALYZE > 0
        if MAX_MARKETS_TO_ANALYZE > 0:
            top_markets = filtered[:MAX_MARKETS_TO_ANALYZE]
        else:
            top_markets = filtered  # Analyze ALL markets

        # Print stats
        cprint(f"\nFiltering Stats:", "yellow")
        cprint(f"  Time-filtered: {stats['time_filtered']}", "white")
        cprint(f"  Volume-filtered: {stats['volume_filtered']}", "white")
        cprint(f"  Sports-filtered: {stats['sports_filtered']}", "white")
        cprint(f"  Imminent (sweet spot): {stats['imminent']}", "green")
        cprint(f"  New markets (<48h): {stats['new_markets']}", "green")
        cprint(f"\nSelected {len(top_markets)} markets for analysis", "cyan")

        self.markets_df = pd.DataFrame(top_markets)

        with self.csv_lock:
            self.markets_df.to_csv(MARKETS_CSV, index=False)

        return self.markets_df

    def _format_markets_for_ai(self, df: pd.DataFrame) -> str:
        """Format markets for AI analysis."""
        lines = []
        for i, (_, row) in enumerate(df.iterrows()):
            time_info = f"Days to Resolution: {row.get('days_to_resolution', 'Unknown')}"
            if row.get('is_imminent'):
                time_info += " [IMMINENT - Sweet Spot]"
            if row.get('is_new_market'):
                time_info += " [NEW MARKET - Potential Mispricing]"

            lines.append(
                f"MARKET {i+1}: {row['ticker']}\n"
                f"Title: {row['title']}\n"
                f"Current YES Price: ${row['yes_price']:.2f} (implied {row['yes_price']*100:.0f}% probability)\n"
                f"Volume: ${row.get('volume', 0):,}\n"
                f"Open Interest: ${row.get('open_interest', 0):,}\n"
                f"{time_info}\n"
            )

        return "\n".join(lines)

    def _analyze_single_market(self, market: Dict, model_config: Dict) -> Optional[Dict]:
        """
        Analyze a single market with one model.

        Returns dict with: side, true_prob, edge, confidence, reason
        Or None if error/skip
        """
        model_key = f"{model_config['type']}:{model_config['name']}"
        try:
            model = self.model_factory.get_model(
                model_config['type'],
                model_config['name']
            )

            if not model:
                return None

            prompt = SINGLE_MARKET_PROMPT.format(
                ticker=market['ticker'],
                title=market['title'],
                yes_price=market['yes_price'],
                no_price=market['no_price'],
                days_to_close=market.get('days_to_resolution', 'Unknown')
            )

            response = model.generate_response(
                system_prompt="You are a prediction market analyst. Be concise.",
                user_content=prompt,
                temperature=0.3,
                max_tokens=200
            )

            content = response.content if hasattr(response, 'content') else str(response)
            if not content:
                return None

            # Parse response
            result = {'model': model_key}
            for line in content.split('\n'):
                line = line.strip().upper()
                if line.startswith('SIDE:'):
                    side = line.replace('SIDE:', '').strip()
                    if side in ['YES', 'NO', 'SKIP']:
                        result['side'] = side
                elif line.startswith('TRUE_PROB:'):
                    try:
                        result['true_prob'] = float(line.replace('TRUE_PROB:', '').strip())
                    except:
                        pass
                elif line.startswith('EDGE:'):
                    try:
                        edge_str = line.replace('EDGE:', '').replace('%', '').strip()
                        result['edge'] = float(edge_str) / 100.0
                    except:
                        pass
                elif line.startswith('CONFIDENCE:'):
                    try:
                        result['confidence'] = int(line.replace('CONFIDENCE:', '').strip().split('/')[0])
                    except:
                        pass

            return result if result.get('side') else None

        except Exception as e:
            return None

    def _analyze_market_with_early_exit(self, market: Dict) -> Optional[Dict]:
        """
        Analyze a single market with smart early exit.

        Strategy:
        1. Query ALL 6 models (they're fast with simple per-market prompts)
        2. Check for consensus after all responses
        3. Apply time-adjusted edge requirements
        4. Apply volume filter for long-term markets

        The per-market prompts are small (~200 tokens) so this is fast.
        Early exit would save time but miss too many good trades.

        Returns qualified trade dict or None
        """
        ticker = market['ticker']
        days_to_resolution = market.get('days_to_resolution')
        volume = market.get('volume', 0) or 0

        # === VOLUME FILTER FOR LONG-TERM MARKETS ===
        passes_volume, volume_reason = check_volume_filter(days_to_resolution, volume)
        if not passes_volume:
            return None  # Skip low-volume long-term markets silently

        results = []

        # Query all models - the prompts are small so this is fast
        for model_config in SWARM_MODELS:
            result = self._analyze_single_market(market, model_config)
            if result and result.get('side') in ['YES', 'NO']:
                results.append(result)

        # Final count - which side has more?
        final_yes = sum(1 for r in results if r.get('side') == 'YES')
        final_no = sum(1 for r in results if r.get('side') == 'NO')
        total_count = len(results)

        # Determine consensus side and count
        if final_yes >= final_no:
            consensus_side = 'YES'
            agree_count = final_yes
        else:
            consensus_side = 'NO'
            agree_count = final_no

        # Need strong agreement: either 4+ raw votes OR 75%+ of responding models
        # This handles cases where some models SKIP
        agreement_pct = agree_count / total_count if total_count > 0 else 0
        if agree_count < 3 or total_count < 3:
            return None
        if agreement_pct < 0.70:  # Need 70%+ agreement
            return None

        # Calculate averages from agreeing models
        agreeing = [r for r in results if r.get('side') == consensus_side]
        avg_edge = sum(abs(r.get('edge', 0)) for r in agreeing) / len(agreeing)
        avg_confidence = sum(r.get('confidence', 5) for r in agreeing) / len(agreeing)
        avg_true_prob = sum(r.get('true_prob', 0.5) for r in agreeing) / len(agreeing)

        # === TIME-ADJUSTED EDGE REQUIREMENT ===
        required_edge = get_required_edge(days_to_resolution)

        # Check thresholds with time-adjusted edge
        if avg_edge < required_edge:
            return None
        if avg_confidence < 6:  # Lowered from 7 to allow more trades
            return None

        return {
            'ticker': ticker,
            'title': market['title'],
            'side': consensus_side,
            'consensus_count': agree_count,
            'total_models': total_count,
            'edge_pct': avg_edge,
            'required_edge': required_edge,
            'confidence': avg_confidence,
            'true_probability': avg_true_prob,
            'market_price': market['yes_price'] if consensus_side == 'YES' else market['no_price'],
            'days_to_resolution': days_to_resolution,
            'reasoning': f"{agree_count}/{total_count} models agree on {consensus_side}",
            'link': f"https://kalshi.com/markets/{ticker}"
        }

    def _execute_single_trade(self, trade: Dict, market: Dict) -> bool:
        """
        Execute a single trade immediately when found.

        Returns True if trade was executed, False if skipped.
        """
        ticker = trade.get('ticker', '')
        side = trade.get('side', '')
        consensus_count = trade.get('consensus_count', 0)
        total_models = trade.get('total_models', 6)
        edge_pct = trade.get('edge_pct', 0)
        confidence = trade.get('confidence', 5)

        yes_price = market.get('yes_price', 0.5)

        # Use strategy engine for analysis
        analysis = self.strategy_engine.evaluate_market(
            ticker=ticker,
            title=trade.get('title', ''),
            market_yes_price=yes_price,
            ai_probability=trade.get('true_probability', 0.5),
            ai_confidence=confidence / 10.0,
            ai_side=side,
            consensus_count=consensus_count,
            total_models=total_models,
            days_to_resolution=market.get('days_to_resolution'),
            is_imminent=market.get('is_imminent', False),
            open_interest=market.get('open_interest', 0)
        )

        if analysis.recommended:
            signal = self.strategy_engine.generate_trade_signal(analysis)

            if signal:
                # Check minimum position size BEFORE executing
                # Use signal.cost_usd (actual cost) not analysis.position_size_usd (Kelly recommendation)
                # because strategy engine forces minimum 1 contract which can be smaller than Kelly size
                if signal.cost_usd < MIN_POSITION_SIZE_USD:
                    cprint(f"    ⏭️  Skipped: Position too small (${signal.cost_usd:.2f} < ${MIN_POSITION_SIZE_USD})", "yellow")
                    return False

                # Check if we already have a position in this market
                existing_positions = self.executor.get_open_positions()
                if not existing_positions.empty and ticker in existing_positions['ticker'].values:
                    if not ALLOW_POSITION_INCREASE:
                        cprint(f"    ⏭️  Skipped: Already have position in {ticker}", "yellow")
                        return False

                cprint(f"    📈 EXECUTING: {ticker} - {side} | ${signal.cost_usd:.2f} ({signal.contracts} contracts) | {edge_pct*100:.1f}% edge", "green", attrs=['bold'])
                self.executor.execute_signal(signal, confirm=False)
                return True
        else:
            cprint(f"    ⏭️  Skipped: {analysis.reason}", "yellow")

        return False

    def run_swarm_analysis(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Run AI swarm analysis on markets with per-market early exit.

        New architecture:
        - Analyze each market individually
        - Exit early if first 2 models disagree (saves API calls)
        - Only continue to all 6 models if initial agreement

        Returns:
            Dict with qualified trades, or None
        """
        if df.empty:
            cprint("No markets to analyze", "yellow")
            return None

        total_markets = len(df)
        cprint(f"\nStarting Per-Market Analysis ({total_markets} markets, {len(SWARM_MODELS)} models)...",
               "magenta", attrs=['bold'])
        cprint("  Using early-exit strategy: skip market if first 2 models disagree", "white")

        qualified_trades = []
        executed_trades = []
        skipped_early = 0
        no_consensus = 0

        for i, (_, market) in enumerate(df.iterrows()):
            # Progress indicator every 50 markets
            if (i + 1) % 50 == 0:
                cprint(f"  Progress: {i+1}/{total_markets} markets analyzed "
                       f"(found {len(qualified_trades)} opportunities, executed {len(executed_trades)})", "cyan")

            result = self._analyze_market_with_early_exit(market.to_dict())

            if result:
                qualified_trades.append(result)
                days_info = f"{result.get('days_to_resolution', '?')}d" if result.get('days_to_resolution') else "?"
                req_edge = result.get('required_edge', MIN_EDGE_PCT) * 100
                cprint(f"  ✅ {result['ticker']}: {result['side']} "
                       f"({result['consensus_count']}/{result['total_models']} agree, "
                       f"{result['edge_pct']*100:.1f}% edge vs {req_edge:.0f}% req, {days_info})", "green")

                # EXECUTE IMMEDIATELY - don't wait until the end
                executed = self._execute_single_trade(result, market.to_dict())
                if executed:
                    executed_trades.append(result)

        cprint(f"\n📊 Analysis Complete:", "cyan", attrs=['bold'])
        cprint(f"  Markets analyzed: {total_markets}", "white")
        cprint(f"  Qualified trades: {len(qualified_trades)}", "green")
        cprint(f"  Executed trades: {len(executed_trades)}", "green")

        if not qualified_trades:
            return {
                "predictions": {},
                "consensus": "NO QUALIFIED VALUE TRADES THIS ROUND",
                "markets_df": df,
                "qualified_trades": [],
                "executed_count": 0
            }

        # Build consensus text for compatibility with existing code
        consensus_lines = []
        for i, trade in enumerate(qualified_trades, 1):
            consensus_lines.append(f"""
RANK {i}: {trade['ticker']}
Title: {trade['title']}
CONSENSUS: {trade['consensus_count']} out of {trade['total_models']} models agree on {trade['side']}
AVERAGE_EDGE: {trade['edge_pct']*100:.1f}%
AVERAGE_CONFIDENCE: {trade['confidence']:.1f}/10
TRUE_PROBABILITY: {trade['true_probability']:.2f}
MARKET_PRICE: ${trade['market_price']:.2f}
REASONING: {trade['reasoning']}
LINK: {trade['link']}
""")

        return {
            "predictions": {},
            "consensus": "\n".join(consensus_lines),
            "markets_df": df,
            "qualified_trades": qualified_trades,
            "executed_count": len(executed_trades)
        }

    def _get_consensus(
        self,
        predictions: Dict[str, str],
        markets_text: str,
        df: pd.DataFrame
    ) -> Optional[str]:
        """Generate consensus from all predictions."""
        cprint("\nGenerating Value Consensus...", "magenta")

        predictions_text = "\n".join([
            f"=== {key} ===\n{value}\n"
            for key, value in predictions.items()
        ])

        # Save predictions
        with self.csv_lock:
            with open(PREDICTIONS_CSV, 'w', encoding='utf-8') as f:
                f.write(predictions_text)

        try:
            model = self.model_factory.get_model('claude', 'claude-sonnet-4-5')
            prompt = CONSENSUS_PROMPT.format(
                all_predictions=predictions_text,
                markets_text=markets_text
            )

            response = model.generate_response(
                system_prompt="You are a quantitative analyst aggregating AI predictions for value trading.",
                user_content=prompt,
                temperature=0.2,
                max_tokens=4000
            )

            if response and hasattr(response, 'content'):
                consensus = response.content
            else:
                consensus = str(response) if response else None

            if consensus:
                self._display_consensus(consensus)
                self._save_value_picks(consensus, df)

            return consensus

        except Exception as e:
            cprint(f"Error getting consensus: {e}", "red")
            return None

    def _display_consensus(self, consensus: str):
        """Display consensus results."""
        cprint("\n" + "="*70, "green")
        cprint(" VALUE TRADING CONSENSUS ", "white", "on_green", attrs=['bold'])
        cprint("="*70, "green")
        cprint(consensus, "cyan")
        cprint("="*70 + "\n", "green")

    def _save_value_picks(self, consensus: str, df: pd.DataFrame):
        """Parse and save value picks from consensus."""
        import re

        timestamp = datetime.now().isoformat()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        picks = []
        lines = consensus.split('\n')
        current_pick = {}

        for line in lines:
            line = line.strip()
            # Remove markdown formatting
            clean_line = re.sub(r'^#+\s*', '', line)  # Remove leading ## or #
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_line)  # Remove **bold**

            # Match RANK patterns (handles "## RANK 1:", "RANK [1]:", "RANK 1:")
            rank_match = re.match(r'RANK\s*\[?(\d+)\]?:\s*(.+)', clean_line, re.IGNORECASE)
            if rank_match:
                if current_pick and current_pick.get('ticker'):
                    picks.append(current_pick)
                current_pick = {
                    'rank': rank_match.group(1),
                    'ticker': rank_match.group(2).strip()
                }
                continue

            if clean_line.startswith('Title:'):
                current_pick['title'] = clean_line.replace('Title:', '').strip()
            elif 'CONSENSUS:' in clean_line or 'out of' in clean_line.lower():
                current_pick['consensus_text'] = clean_line
                match = re.search(r'(\d+)\s*out\s*of\s*(\d+)', clean_line)
                if match:
                    current_pick['consensus_count'] = int(match.group(1))
                    current_pick['total_models'] = int(match.group(2))
                side_match = re.search(r'(?:agree\s*on|on)\s*\**\s*(YES|NO)\s*\**', clean_line, re.IGNORECASE)
                if side_match:
                    current_pick['side'] = side_match.group(1).upper()
            elif 'AVERAGE_EDGE:' in clean_line:
                edge_match = re.search(r'AVERAGE_EDGE:\s*([0-9.]+)', clean_line)
                if edge_match:
                    try:
                        current_pick['edge_pct'] = float(edge_match.group(1)) / 100.0
                    except ValueError:
                        pass
            elif 'AVERAGE_CONFIDENCE:' in clean_line:
                conf_match = re.search(r'AVERAGE_CONFIDENCE:\s*([0-9.]+)', clean_line)
                if conf_match:
                    try:
                        current_pick['confidence'] = float(conf_match.group(1))
                    except ValueError:
                        pass
            elif 'TRUE_PROBABILITY:' in clean_line:
                # Handle formats like "~0.05" or "0.15-0.20" or "0.5" or "45%" or "45"
                prob_match = re.search(r'TRUE_PROBABILITY:\s*~?\s*([0-9.]+)', clean_line)
                if prob_match:
                    try:
                        prob_value = float(prob_match.group(1))
                        # Normalize: if > 1, it's a percentage (like 45 or 45%)
                        if prob_value > 1.0:
                            prob_value = prob_value / 100.0
                        current_pick['true_probability'] = prob_value
                    except ValueError:
                        pass
            elif 'MARKET_PRICE:' in clean_line:
                price_match = re.search(r'MARKET_PRICE:\s*\$?([0-9.]+)', clean_line)
                if price_match:
                    try:
                        price_value = float(price_match.group(1))
                        # Normalize: if > 1, it's cents or percentage (like 27 cents = $0.27)
                        if price_value > 1.0:
                            price_value = price_value / 100.0
                        current_pick['market_price'] = price_value
                    except ValueError:
                        pass
            elif 'REASONING:' in clean_line:
                current_pick['reasoning'] = clean_line.split('REASONING:')[-1].strip()
            elif 'LINK:' in clean_line:
                link_match = re.search(r'(https?://[^\s]+)', clean_line)
                if link_match:
                    current_pick['link'] = link_match.group(1)

        if current_pick:
            picks.append(current_pick)

        if not picks:
            cprint("No value picks found in consensus", "yellow")
            return

        # Convert to records
        records = []
        for pick in picks:
            records.append({
                'timestamp': timestamp,
                'run_id': run_id,
                'rank': pick.get('rank', ''),
                'ticker': pick.get('ticker', ''),
                'title': pick.get('title', ''),
                'side': pick.get('side', ''),
                'consensus_count': pick.get('consensus_count', 0),
                'total_models': pick.get('total_models', 6),
                'edge_pct': pick.get('edge_pct', 0),
                'confidence': pick.get('confidence', 0),
                'true_probability': pick.get('true_probability', 0),
                'market_price': pick.get('market_price', 0),
                'reasoning': pick.get('reasoning', ''),
                'link': pick.get('link', '')
            })

        # Save to CSV
        if os.path.exists(VALUE_PICKS_CSV):
            existing = pd.read_csv(VALUE_PICKS_CSV)
        else:
            existing = pd.DataFrame()

        new_df = pd.DataFrame(records)
        combined = pd.concat([existing, new_df], ignore_index=True)

        with self.csv_lock:
            combined.to_csv(VALUE_PICKS_CSV, index=False)

        cprint(f"\nSaved {len(records)} value picks to CSV", "green")

        # Execute trades for qualified picks
        self._execute_value_trades(picks, df)

    def _execute_value_trades(self, picks: List[Dict], df: pd.DataFrame):
        """Execute trades for qualified value picks."""
        cprint("\nEvaluating picks for execution...", "yellow")

        for pick in picks:
            ticker = pick.get('ticker', '')
            side = pick.get('side', '')
            consensus_count = pick.get('consensus_count', 0)
            edge_pct = pick.get('edge_pct', 0)
            confidence = pick.get('confidence', 0)

            # Validation
            if consensus_count < MIN_CONSENSUS:
                cprint(f"  {ticker}: Skipped - consensus {consensus_count}/{MIN_CONSENSUS}", "yellow")
                continue

            if edge_pct < MIN_EDGE_PCT:
                cprint(f"  {ticker}: Skipped - edge {edge_pct*100:.1f}% < {MIN_EDGE_PCT*100}%", "yellow")
                continue

            # Get market data
            market_row = df[df['ticker'] == ticker]
            if market_row.empty:
                cprint(f"  {ticker}: Skipped - not in analyzed markets (AI may have hallucinated)", "yellow")
                continue

            market_row = market_row.iloc[0]
            yes_price = market_row.get('yes_price', 0.5)

            # Calculate position
            entry_price = yes_price if side == "YES" else (1.0 - yes_price)

            # Use strategy engine for analysis
            analysis = self.strategy_engine.evaluate_market(
                ticker=ticker,
                title=pick.get('title', ''),
                market_yes_price=yes_price,
                ai_probability=pick.get('true_probability', 0.5),
                ai_confidence=confidence / 10.0,
                ai_side=side,
                consensus_count=consensus_count,
                total_models=pick.get('total_models', 6),
                days_to_resolution=market_row.get('days_to_resolution'),
                is_imminent=market_row.get('is_imminent', False),
                open_interest=market_row.get('open_interest', 0)
            )

            if analysis.recommended:
                signal = self.strategy_engine.generate_trade_signal(analysis)

                if signal:
                    cprint(f"\n  EXECUTING: {ticker} - {side}", "green", attrs=['bold'])
                    self.executor.execute_signal(signal, confirm=False)
            else:
                cprint(f"  {ticker}: Skipped - {analysis.reason}", "yellow")

    def check_and_settle_positions(self):
        """
        Check open positions for:
        1. Settlement (resolved markets)
        2. Price changes (unrealized P&L tracking)

        Fetches market status from Kalshi API and updates position tracking.
        """
        open_positions = self.executor.get_open_positions()

        if open_positions.empty:
            return

        cprint(f"\n📊 Monitoring {len(open_positions)} open positions...", "cyan", attrs=['bold'])

        session = requests.Session()
        settled_count = 0
        total_unrealized_pnl = 0.0
        total_cost = 0.0
        position_updates = []  # Track all position updates for summary

        for _, pos in open_positions.iterrows():
            ticker = pos['ticker']
            our_side = pos['side'].lower()
            entry_price = pos['price_cents'] / 100.0
            contracts = pos['contracts']
            cost = pos['cost_usd']
            total_cost += cost

            try:
                # Fetch market status from Kalshi
                response = session.get(
                    f"{KALSHI_PUBLIC_API}/markets/{ticker}",
                    timeout=30
                )

                if response.status_code != 200:
                    cprint(f"  {ticker}: ⚠️ Could not fetch price", "yellow")
                    continue

                market = response.json().get('market', {})
                status = market.get('status', '')
                result = market.get('result', '')  # "yes", "no", or ""

                # Get current prices - IMPORTANT: Use BID for cash out (what you'd get if you sold)
                # yes_bid = price someone will pay for YES contracts
                # no_bid = price someone will pay for NO contracts
                yes_bid = market.get('yes_bid', 0) or 0
                no_bid = market.get('no_bid', 0) or 0
                yes_ask = market.get('yes_ask', 0) or 0
                last_price = market.get('last_price', 50)

                # Convert from cents if needed
                if yes_bid >= 1:
                    yes_bid = yes_bid / 100.0
                if no_bid >= 1:
                    no_bid = no_bid / 100.0
                if yes_ask >= 1:
                    yes_ask = yes_ask / 100.0
                if last_price >= 1:
                    last_price = last_price / 100.0

                # CASH OUT PRICE = what we'd get if we sold NOW
                # If we own YES, we sell YES at yes_bid
                # If we own NO, we sell NO at no_bid (or equivalently, buy YES at yes_ask, but no_bid is clearer)
                if our_side == 'yes':
                    cash_out_price = yes_bid if yes_bid > 0 else last_price
                else:
                    # For NO: if no_bid exists, use it. Otherwise derive from yes_ask
                    # no_bid = 1 - yes_ask (the bid for NO is the inverse of ask for YES)
                    if no_bid > 0:
                        cash_out_price = no_bid
                    elif yes_ask > 0:
                        cash_out_price = 1.0 - yes_ask
                    else:
                        cash_out_price = 1.0 - last_price

                # Calculate P&L based on CASH OUT price (not theoretical value)
                price_change = cash_out_price - entry_price
                unrealized_pnl = price_change * contracts
                pnl_pct = (unrealized_pnl / cost * 100) if cost > 0 else 0

                # Also track theoretical mid-price for comparison
                current_price = cash_out_price  # Use cash out as the "current" price

                total_unrealized_pnl += unrealized_pnl

                # Check if market has settled
                if status == 'settled' and result:
                    won = (result == our_side)
                    exit_price = 100 if won else 0
                    actual_pnl = (exit_price / 100.0 - entry_price) * contracts

                    cprint(f"  {ticker}: 🏁 SETTLED - {result.upper()} | We had {our_side.upper()} | "
                           f"P&L: ${actual_pnl:+.2f}", "green" if won else "red", attrs=['bold'])

                    self.executor.close_position(
                        trade_id=pos['trade_id'],
                        exit_price_cents=exit_price,
                        won=won
                    )
                    settled_count += 1

                elif status == 'closed':
                    # Market closed but not yet settled - check expiration
                    close_time_str = market.get('close_time', '')
                    if close_time_str:
                        try:
                            close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                            close_time = close_time.replace(tzinfo=None)
                            if close_time < datetime.now():
                                cprint(f"  {ticker}: ⏳ Closed, awaiting settlement", "yellow")
                        except Exception:
                            pass
                    position_updates.append({
                        'ticker': ticker,
                        'side': our_side,
                        'entry': entry_price,
                        'current': current_price,
                        'pnl': unrealized_pnl,
                        'pnl_pct': pnl_pct,
                        'status': 'closed'
                    })
                    # Update price in trades.csv
                    self.executor.update_position_price(
                        trade_id=pos['trade_id'],
                        current_price_cents=int(current_price * 100),
                        unrealized_pnl=unrealized_pnl
                    )

                elif status in ['open', 'active']:
                    # Kalshi uses "active" for open markets
                    # Track for display, then check for early exit opportunities
                    position_updates.append({
                        'ticker': ticker,
                        'side': our_side,
                        'entry': entry_price,
                        'current': current_price,
                        'pnl': unrealized_pnl,
                        'pnl_pct': pnl_pct,
                        'status': 'open'
                    })
                    # Update price in trades.csv
                    self.executor.update_position_price(
                        trade_id=pos['trade_id'],
                        current_price_cents=int(current_price * 100),
                        unrealized_pnl=unrealized_pnl
                    )
                    # Check for early exit opportunities (may trigger an exit)
                    self._check_early_exit(pos, market, session)

                time.sleep(0.2)  # Rate limiting

            except Exception as e:
                cprint(f"  {ticker}: Error checking status - {e}", "red")

        # Display position summary with P&L (using CASH OUT prices)
        if position_updates:
            cprint(f"\n  {'─'*68}", "white")
            cprint(f"  {'TICKER':<25} {'SIDE':<5} {'ENTRY':>8} {'CASHOUT':>8} {'P&L':>10} {'%':>8}", "white", attrs=['bold'])
            cprint(f"  {'─'*68}", "white")

            for p in position_updates:
                color = "green" if p['pnl'] >= 0 else "red"
                cprint(f"  {p['ticker']:<25} {p['side'].upper():<5} ${p['entry']:>6.2f} ${p['current']:>6.2f} "
                       f"${p['pnl']:>+8.2f} {p['pnl_pct']:>+7.1f}%", color)

            cprint(f"  {'─'*68}", "white")
            pnl_color = "green" if total_unrealized_pnl >= 0 else "red"
            total_pnl_pct = (total_unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
            cprint(f"  {'TOTAL UNREALIZED P&L:':<40} ${total_unrealized_pnl:>+8.2f} {total_pnl_pct:>+7.1f}%",
                   pnl_color, attrs=['bold'])
            cprint(f"  {'TOTAL INVESTED:':<40} ${total_cost:>8.2f}", "white")

            # Save hourly price history to CSV
            self._save_price_history(position_updates, total_unrealized_pnl, total_cost)

        if settled_count > 0:
            cprint(f"\n✅ Settled {settled_count} positions", "green", attrs=['bold'])

    def _save_price_history(
        self,
        position_updates: List[Dict],
        total_unrealized_pnl: float,
        total_cost: float
    ):
        """
        Save hourly price snapshots to price_history.csv.

        Creates a historical record of position prices for tracking P&L over time.
        Each row represents a position's state at a specific timestamp.

        Args:
            position_updates: List of position update dicts from check_and_settle_positions
            total_unrealized_pnl: Total unrealized P&L across all positions
            total_cost: Total cost of all open positions
        """
        if not position_updates:
            return

        timestamp = datetime.now().isoformat()

        # Build records for each position
        records = []
        for p in position_updates:
            records.append({
                'timestamp': timestamp,
                'ticker': p['ticker'],
                'side': p['side'],
                'entry_price': p['entry'],
                'current_price': p['current'],
                'unrealized_pnl': round(p['pnl'], 2),
                'pnl_pct': round(p['pnl_pct'], 2),
                'status': p['status']
            })

        # Also add a summary row for portfolio totals
        total_pnl_pct = (total_unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
        records.append({
            'timestamp': timestamp,
            'ticker': '_PORTFOLIO_TOTAL',
            'side': '',
            'entry_price': total_cost,
            'current_price': total_cost + total_unrealized_pnl,
            'unrealized_pnl': round(total_unrealized_pnl, 2),
            'pnl_pct': round(total_pnl_pct, 2),
            'status': 'summary'
        })

        # Load existing history or create new DataFrame
        if os.path.exists(PRICE_HISTORY_CSV):
            try:
                existing_df = pd.read_csv(PRICE_HISTORY_CSV)
            except Exception:
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()

        # Append new records
        new_df = pd.DataFrame(records)
        combined = pd.concat([existing_df, new_df], ignore_index=True)

        # Save to CSV
        with self.csv_lock:
            combined.to_csv(PRICE_HISTORY_CSV, index=False)

        cprint(f"\n📈 Saved {len(position_updates)} position snapshots to price_history.csv", "cyan")

    def _check_early_exit(self, pos: pd.Series, market: Dict, session: requests.Session):
        """
        Check if we should exit a position early (before settlement).

        TRADE-SPECIFIC EXIT STRATEGY:
        Uses the AI's original probability estimate to set dynamic thresholds.
        Don't wait for 100% of predicted edge - take profits at a portion.

        Exit Triggers (in priority order):
        0. SMALL POSITION: <$7 cost - not worth monitoring
        1. EDGE CAPTURED: Price moved X% toward AI's predicted probability
        2. EDGE LOST: Price moved against us, edge now negative
        3. TIME + PROFIT: Near resolution with profit, lock it in

        Args:
            pos: Position DataFrame row
            market: Market data from Kalshi API
            session: Requests session
        """
        ticker = pos['ticker']
        our_side = pos['side'].lower()
        entry_price = pos['price_cents'] / 100.0
        cost = pos['cost_usd']
        contracts = pos['contracts']
        original_edge = pos.get('edge_pct', 0.10)
        true_prob = pos.get('true_prob', 0.5)  # AI's estimated probability

        # Get CASH OUT PRICE (actual bid price, not theoretical value)
        yes_bid = market.get('yes_bid', 0) or 0
        no_bid = market.get('no_bid', 0) or 0
        yes_ask = market.get('yes_ask', 0) or 0
        last_price = market.get('last_price', 50)

        # Convert from cents
        if yes_bid >= 1:
            yes_bid = yes_bid / 100.0
        if no_bid >= 1:
            no_bid = no_bid / 100.0
        if yes_ask >= 1:
            yes_ask = yes_ask / 100.0
        if last_price >= 1:
            last_price = last_price / 100.0

        # Cash out price = what we'd actually get if we sold now
        if our_side == 'yes':
            cash_out_price = yes_bid if yes_bid > 0 else last_price
        else:
            if no_bid > 0:
                cash_out_price = no_bid
            elif yes_ask > 0:
                cash_out_price = 1.0 - yes_ask
            else:
                cash_out_price = 1.0 - last_price

        current_price = cash_out_price

        # === 0. SMALL POSITION EXIT ===
        MIN_POSITION_TO_MONITOR = 7.0
        if cost < MIN_POSITION_TO_MONITOR:
            unrealized_pnl = (current_price - entry_price) * contracts
            exit_reason = f"Position too small (${cost:.2f} < ${MIN_POSITION_TO_MONITOR})"
            self._execute_early_exit(pos, current_price, unrealized_pnl, cost, contracts, exit_reason, won=unrealized_pnl >= 0)
            return

        # Calculate P&L
        unrealized_pnl = (current_price - entry_price) * contracts
        pnl_pct = unrealized_pnl / cost if cost > 0 else 0

        # === TRADE-SPECIFIC EXIT THRESHOLDS ===
        # Based on AI's predicted probability vs market price

        # Target price = AI's true probability (for our side)
        # If we bet YES, target is true_prob
        # If we bet NO, target is (1 - true_prob) for YES side, so NO target is true_prob
        if our_side == 'yes':
            target_price = true_prob
        else:
            # For NO bets: if AI says 10% YES (true_prob=0.1), we want NO price to go to 0.90
            # But current_price is already the NO price (1 - yes_bid)
            target_price = 1.0 - true_prob  # This is what NO should be worth

        # How much edge was there at entry?
        total_edge = target_price - entry_price  # Positive if we had edge

        # === TAKE PROFIT THRESHOLDS (Research-Backed) ===
        # Based on tastytrade options research: "Managing at 50% of max profit
        # provides the greatest annualized ROC" (Return on Capital)
        # Source: https://support.tastytrade.com/support/s/solutions/articles/43000435423
        #
        # For prediction markets (binary outcomes like options):
        # - 50% edge capture = optimal balance of locking gains vs. capturing upside
        # - Waiting for 100% often means holding through reversals
        #
        # ORATS backtesting tested: +25%, +50%, +75%, +100% profit targets
        # tastytrade rule: close winners at 50%, stop losses at 2x original credit

        TAKE_PROFIT_PCT = 0.50  # Exit when 50% of predicted edge is captured
        STOP_LOSS_EDGE_PCT = -0.50  # Exit when edge reversed 50% (thesis likely wrong)

        price_move = current_price - entry_price
        edge_captured_pct = price_move / total_edge if abs(total_edge) > 0.01 else 0

        # === 1. TAKE PROFIT: Captured portion of predicted edge ===
        if total_edge > 0.05 and edge_captured_pct >= TAKE_PROFIT_PCT:
            exit_reason = f"Edge captured ({edge_captured_pct*100:.0f}% of {total_edge*100:.0f}¢ edge)"
            self._execute_early_exit(pos, current_price, unrealized_pnl, cost, contracts, exit_reason, won=True)
            return

        # === 2. STOP LOSS: Edge reversed significantly ===
        # If price moved against us by more than 50% of our original edge, cut losses
        if edge_captured_pct <= STOP_LOSS_EDGE_PCT:
            exit_reason = f"Edge lost ({edge_captured_pct*100:.0f}% - moved against prediction)"
            self._execute_early_exit(pos, current_price, unrealized_pnl, cost, contracts, exit_reason, won=False)
            return

        # === 3. ABSOLUTE STOP LOSS: 40% of position value ===
        if pnl_pct <= -0.40:
            exit_reason = f"Hard stop loss ({pnl_pct*100:.1f}% loss)"
            self._execute_early_exit(pos, current_price, unrealized_pnl, cost, contracts, exit_reason, won=False)
            return

        # === 4. TIME + PROFIT: Lock in gains near resolution ===
        close_time_str = market.get('close_time', '')
        if close_time_str:
            try:
                close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                close_time = close_time.replace(tzinfo=None)
                hours_to_close = (close_time - datetime.now()).total_seconds() / 3600

                # Near resolution + profitable = lock it in
                if hours_to_close < 24 and pnl_pct >= 0.15:
                    exit_reason = f"Lock profit before resolution ({hours_to_close:.0f}h left, {pnl_pct*100:.0f}% gain)"
                    self._execute_early_exit(pos, current_price, unrealized_pnl, cost, contracts, exit_reason, won=True)
                    return
            except Exception:
                pass

        # === 5. WEB-BASED RE-EVALUATION (V2 Feature) ===
        # Check if this position should be re-evaluated with fresh web info
        days_to_resolution = None
        close_time_str = market.get('close_time', '')
        if close_time_str:
            try:
                close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                close_time = close_time.replace(tzinfo=None)
                days_to_resolution = (close_time - datetime.now()).days
            except:
                pass

        should_reeval, reeval_reason = self._should_reevaluate(
            pos, current_price, pnl_pct, days_to_resolution
        )

        if should_reeval:
            action = self._reevaluate_position(
                pos, market, current_price, pnl_pct, days_to_resolution, reeval_reason
            )

            if action == 'EXIT':
                exit_reason = f"AI re-evaluation: thesis invalidated ({reeval_reason})"
                self._execute_early_exit(pos, current_price, unrealized_pnl, cost, contracts, exit_reason, won=pnl_pct > 0)
                return

        # Report position status (no exit triggered)
        status_color = "green" if pnl_pct >= 0 else "yellow"
        target_info = f"Target: ${target_price:.2f}" if total_edge > 0 else ""
        progress = f"{edge_captured_pct*100:+.0f}% to target" if total_edge > 0.01 else ""
        cprint(f"  {ticker}: {our_side.upper()} @ ${entry_price:.2f} → ${current_price:.2f} ({pnl_pct*100:+.1f}%) | {progress}",
               status_color)

    def _execute_early_exit(
        self,
        pos: pd.Series,
        current_price: float,
        unrealized_pnl: float,
        cost: float,
        contracts: int,
        reason: str,
        won: bool
    ):
        """
        Execute an early exit from a position.

        Args:
            pos: Position data
            current_price: Current market price
            unrealized_pnl: Unrealized P&L
            cost: Original cost
            contracts: Number of contracts
            reason: Exit reason for logging
            won: Whether this counts as a win
        """
        ticker = pos['ticker']
        status = 'won' if won else 'lost'
        color = 'green' if won else 'red'

        cprint(f"  {ticker}: EARLY EXIT - {reason}", color, attrs=['bold'])
        cprint(f"    Unrealized P&L: ${unrealized_pnl:.2f}", color)

        # In paper mode, simulate the sale
        if not self.live_mode:
            sale_proceeds = current_price * contracts
            actual_pnl = sale_proceeds - cost

            self.executor.trades_df.loc[
                self.executor.trades_df['trade_id'] == pos['trade_id'],
                ['status', 'exit_price_cents', 'exit_timestamp', 'pnl_usd', 'notes']
            ] = [status, int(current_price * 100), datetime.now().isoformat(), actual_pnl,
                 f"Early exit: {reason}"]

            self.executor.trades_df.to_csv(self.executor.trades_csv, index=False)
            self.executor._calculate_portfolio_state()

            cprint(f"    SOLD (paper): ${actual_pnl:+.2f}", color)

    def _should_reevaluate(
        self,
        pos: pd.Series,
        current_price: float,
        pnl_pct: float,
        days_to_resolution: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be re-evaluated with web search.

        Triggers:
        1. Large price move (>20% from entry)
        2. Significant loss (>15% underwater)
        3. Approaching resolution (within 7 days)

        Returns:
            Tuple of (should_reeval, reason)
        """
        entry_price = pos['price_cents'] / 100.0
        trade_id = pos['trade_id']

        # Check if we recently re-evaluated this position
        if os.path.exists(REEVAL_LOG_CSV):
            try:
                reeval_df = pd.read_csv(REEVAL_LOG_CSV)
                last_reeval = reeval_df[reeval_df['trade_id'] == trade_id]
                if not last_reeval.empty:
                    last_time = datetime.fromisoformat(last_reeval.iloc[-1]['timestamp'])
                    hours_since = (datetime.now() - last_time).total_seconds() / 3600
                    if hours_since < REEVAL_MIN_HOURS_BETWEEN:
                        return False, f"Re-evaluated {hours_since:.0f}h ago"
            except Exception:
                pass

        # Trigger 1: Large price move from entry
        price_change_pct = abs(current_price - entry_price) / entry_price if entry_price > 0 else 0
        if price_change_pct >= REEVAL_PRICE_MOVE_THRESHOLD:
            direction = "up" if current_price > entry_price else "down"
            return True, f"Price moved {price_change_pct*100:.0f}% {direction} from entry"

        # Trigger 2: Significant loss
        if pnl_pct <= REEVAL_LOSS_THRESHOLD:
            return True, f"Position is {pnl_pct*100:.1f}% underwater"

        # Trigger 3: Approaching resolution
        if days_to_resolution is not None and days_to_resolution <= REEVAL_DAYS_BEFORE_RESOLUTION:
            return True, f"Only {days_to_resolution:.0f} days to resolution"

        return False, "No trigger"

    def _fetch_web_context(self, ticker: str, title: str) -> str:
        """
        Fetch recent news/information about a market using web search.

        Returns:
            String with relevant web context, or error message
        """
        try:
            # Import web search capability
            from src.models.model_factory import model_factory

            # Create search query from title
            # Clean up title for better search
            search_query = title.replace("**", "").replace("?", "")
            search_query = f"{search_query} latest news 2025"

            cprint(f"    🔍 Searching: {search_query[:50]}...", "cyan")

            # Use Gemini for web search (it has good search integration)
            model = model_factory.get_model('gemini', 'gemini-2.0-flash')

            if not model:
                return "Web search unavailable - no model"

            # Ask model to search and summarize recent news
            search_prompt = f"""Search for the latest news and information about this topic:

"{title}"

Find and summarize:
1. Any recent news articles (last 7 days)
2. Expert opinions or analysis
3. Key facts that could affect the outcome
4. Any announcements or official statements

Keep your response under 300 words. Focus on FACTS, not speculation.
If you can't find recent news, say "No recent news found" and provide any relevant background."""

            response = model.generate_response(
                system_prompt="You are a news researcher. Find and summarize relevant recent news.",
                user_content=search_prompt,
                temperature=0.3,
                max_tokens=500
            )

            if response and hasattr(response, 'content'):
                return response.content
            else:
                return "No web context retrieved"

        except Exception as e:
            cprint(f"    ⚠️ Web search error: {e}", "yellow")
            return f"Web search error: {str(e)}"

    def _reevaluate_position(
        self,
        pos: pd.Series,
        market: Dict,
        current_price: float,
        pnl_pct: float,
        days_to_resolution: Optional[float],
        reason: str
    ) -> Optional[str]:
        """
        Re-evaluate a position using web search and AI analysis.

        Returns:
            "EXIT" if AI recommends closing, "HOLD" to keep, or None on error
        """
        ticker = pos['ticker']
        title = pos['title']
        our_side = pos['side'].lower()
        entry_price = pos['price_cents'] / 100.0
        true_prob = pos.get('true_prob', 0.5)

        cprint(f"\n  🔄 RE-EVALUATING: {ticker}", "magenta", attrs=['bold'])
        cprint(f"    Reason: {reason}", "magenta")

        # Fetch web context
        web_context = self._fetch_web_context(ticker, title)

        if "error" in web_context.lower() or "unavailable" in web_context.lower():
            cprint(f"    ⚠️ Could not fetch web context, skipping re-eval", "yellow")
            return None

        # Build re-evaluation prompt
        prompt = REEVAL_PROMPT.format(
            ticker=ticker,
            title=title,
            our_side=our_side.upper(),
            entry_price=entry_price,
            current_price=current_price,
            pnl_pct=pnl_pct * 100,
            original_prob=true_prob,
            days_to_close=days_to_resolution if days_to_resolution else "Unknown",
            web_context=web_context
        )

        # Use Claude Sonnet for re-evaluation (best reasoning)
        try:
            model = self.model_factory.get_model('claude', 'claude-sonnet-4-5')

            if not model:
                cprint(f"    ⚠️ No model for re-evaluation", "yellow")
                return None

            response = model.generate_response(
                system_prompt="You are a prediction market analyst re-evaluating an existing position.",
                user_content=prompt,
                temperature=0.2,
                max_tokens=500
            )

            content = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            action = None
            new_prob = None
            confidence = None
            reasoning = ""

            for line in content.split('\n'):
                line = line.strip().upper()
                if line.startswith('ACTION:'):
                    action_val = line.replace('ACTION:', '').strip()
                    if 'EXIT' in action_val:
                        action = 'EXIT'
                    elif 'HOLD' in action_val:
                        action = 'HOLD'
                elif line.startswith('NEW_PROB:'):
                    try:
                        new_prob = float(line.replace('NEW_PROB:', '').strip())
                    except:
                        pass
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = int(line.replace('CONFIDENCE:', '').strip().split('/')[0])
                    except:
                        pass
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()

            # Log the re-evaluation
            self._log_reevaluation(
                trade_id=pos['trade_id'],
                ticker=ticker,
                action=action or 'UNKNOWN',
                new_prob=new_prob,
                confidence=confidence,
                reasoning=reasoning,
                web_context=web_context[:500]  # Truncate for CSV
            )

            if action == 'EXIT':
                cprint(f"    🚨 AI RECOMMENDS EXIT: {reasoning}", "red", attrs=['bold'])
            elif action == 'HOLD':
                cprint(f"    ✅ AI RECOMMENDS HOLD: {reasoning}", "green")

            return action

        except Exception as e:
            cprint(f"    ❌ Re-evaluation error: {e}", "red")
            return None

    def _log_reevaluation(
        self,
        trade_id: str,
        ticker: str,
        action: str,
        new_prob: Optional[float],
        confidence: Optional[int],
        reasoning: str,
        web_context: str
    ):
        """Log re-evaluation to CSV for tracking."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'trade_id': trade_id,
            'ticker': ticker,
            'action': action,
            'new_prob': new_prob,
            'confidence': confidence,
            'reasoning': reasoning,
            'web_context': web_context.replace('\n', ' ')[:500]
        }

        if os.path.exists(REEVAL_LOG_CSV):
            try:
                df = pd.read_csv(REEVAL_LOG_CSV)
            except:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        new_df = pd.DataFrame([record])
        combined = pd.concat([df, new_df], ignore_index=True)
        combined.to_csv(REEVAL_LOG_CSV, index=False)

    def run_once(self):
        """Run a single analysis cycle."""
        run_timestamp = datetime.now()
        cprint(f"\n{'='*70}", "blue")
        cprint(f"  Analysis Run - {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}", "blue", attrs=['bold'])
        cprint(f"{'='*70}\n", "blue")

        # First, check and settle any resolved positions
        self.check_and_settle_positions()

        # Fetch markets
        df = self.fetch_markets()

        if df.empty:
            cprint("No markets found", "yellow")
            return

        # Run analysis (now executes trades immediately when found)
        result = self.run_swarm_analysis(df)

        qualified_trades = []
        executed_count = 0
        if result:
            qualified_trades = result.get('qualified_trades', [])
            executed_count = result.get('executed_count', 0)
            if qualified_trades:
                cprint(f"\n🎯 Found {len(qualified_trades)} qualified trades, executed {executed_count}!", "green", attrs=['bold'])
            else:
                cprint("\nNo qualified trades this round", "yellow")
        else:
            cprint("\nAnalysis completed with warnings", "yellow")

        # Export run log
        self._export_run_log(run_timestamp, df, qualified_trades, executed_count)

        # Show portfolio
        self.executor.display_open_positions()
        self.executor.display_performance_summary()

    def _export_run_log(self, run_timestamp: datetime, markets_df: pd.DataFrame,
                        qualified_trades: List[Dict], executed_count: int):
        """Export a log of the run for tracking."""
        log_filename = f"run_{run_timestamp.strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(LOGS_FOLDER, log_filename)

        with open(log_path, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"KALSHI VALUE AGENT - RUN LOG\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Timestamp: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {'LIVE' if self.live_mode else 'PAPER'}\n")
            f.write(f"Bankroll: ${BANKROLL_USD:,.0f}\n\n")

            f.write(f"MARKETS ANALYZED: {len(markets_df)}\n")
            f.write(f"QUALIFIED TRADES: {len(qualified_trades)}\n")
            f.write(f"EXECUTED TRADES: {executed_count}\n\n")

            if qualified_trades:
                f.write(f"{'='*70}\n")
                f.write(f"QUALIFIED TRADES DETAILS\n")
                f.write(f"{'='*70}\n\n")

                for i, trade in enumerate(qualified_trades, 1):
                    f.write(f"#{i}: {trade['ticker']}\n")
                    f.write(f"    Title: {trade.get('title', 'N/A')}\n")
                    f.write(f"    Side: {trade['side']}\n")
                    f.write(f"    Consensus: {trade['consensus_count']}/{trade['total_models']} models\n")
                    f.write(f"    Edge: {trade['edge_pct']*100:.1f}%\n")
                    f.write(f"    Confidence: {trade['confidence']:.1f}/10\n")
                    f.write(f"    True Prob: {trade['true_probability']:.2f}\n")
                    f.write(f"    Market Price: ${trade['market_price']:.2f}\n")
                    f.write(f"    Link: {trade['link']}\n\n")

            # Current portfolio summary
            f.write(f"{'='*70}\n")
            f.write(f"CURRENT PORTFOLIO\n")
            f.write(f"{'='*70}\n\n")

            open_positions = self.executor.get_open_positions()
            if not open_positions.empty:
                for _, pos in open_positions.iterrows():
                    f.write(f"  {pos['ticker']}: {pos['side'].upper()} @ ${pos['price_cents']/100:.2f} "
                            f"({pos['contracts']} contracts, ${pos['cost_usd']:.2f})\n")
            else:
                f.write("  No open positions\n")

        cprint(f"\n📝 Run log exported: {log_filename}", "cyan")

    def _execute_qualified_trades(self, qualified_trades: List[Dict], df: pd.DataFrame):
        """Execute trades for qualified picks from per-market analysis."""
        cprint("\nExecuting qualified trades...", "yellow")

        for trade in qualified_trades:
            ticker = trade.get('ticker', '')
            side = trade.get('side', '')
            consensus_count = trade.get('consensus_count', 0)
            total_models = trade.get('total_models', 6)
            edge_pct = trade.get('edge_pct', 0)
            confidence = trade.get('confidence', 5)

            # Get market data from df
            market_row = df[df['ticker'] == ticker]
            if market_row.empty:
                cprint(f"  {ticker}: Skipped - not in markets DataFrame", "yellow")
                continue

            market_row = market_row.iloc[0]
            yes_price = market_row.get('yes_price', 0.5)

            # Use strategy engine for analysis
            analysis = self.strategy_engine.evaluate_market(
                ticker=ticker,
                title=trade.get('title', ''),
                market_yes_price=yes_price,
                ai_probability=trade.get('true_probability', 0.5),
                ai_confidence=confidence / 10.0,
                ai_side=side,
                consensus_count=consensus_count,
                total_models=total_models,
                days_to_resolution=market_row.get('days_to_resolution'),
                is_imminent=market_row.get('is_imminent', False),
                open_interest=market_row.get('open_interest', 0)
            )

            if analysis.recommended:
                signal = self.strategy_engine.generate_trade_signal(analysis)

                if signal:
                    cprint(f"\n  📈 EXECUTING: {ticker} - {side} ({consensus_count}/{total_models}, {edge_pct*100:.1f}% edge)", "green", attrs=['bold'])
                    self.executor.execute_signal(signal, confirm=False)
            else:
                cprint(f"  {ticker}: Skipped - {analysis.reason}", "yellow")

    def run_loop(self):
        """Run continuous analysis loop."""
        cprint("\nStarting continuous trading loop...", "cyan", attrs=['bold'])
        cprint(f"Analysis interval: {HOURS_BETWEEN_RUNS} hours", "white")
        cprint("Press Ctrl+C to stop\n", "yellow")

        run_count = 0

        try:
            while True:
                run_count += 1
                cprint(f"\n{'#'*70}", "magenta")
                cprint(f"  RUN #{run_count}", "magenta", attrs=['bold'])
                cprint(f"{'#'*70}", "magenta")

                self.run_once()

                if not AUTO_RUN_LOOP:
                    break

                # Sleep until next run
                next_run = datetime.now() + timedelta(hours=HOURS_BETWEEN_RUNS)
                cprint(f"\nNext run at {next_run.strftime('%H:%M:%S')}", "cyan")

                # Sleep in 1-minute chunks
                for _ in range(int(HOURS_BETWEEN_RUNS * 60)):
                    time.sleep(60)

        except KeyboardInterrupt:
            cprint("\n\nStopped by user", "yellow", attrs=['bold'])
            self.executor.display_performance_summary()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Kalshi Value Trading Agent"
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading (requires API keys)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (no loop)'
    )

    args = parser.parse_args()

    agent = KalshiValueAgent(live_mode=args.live)

    if args.once:
        agent.run_once()
    else:
        agent.run_loop()


if __name__ == "__main__":
    main()
