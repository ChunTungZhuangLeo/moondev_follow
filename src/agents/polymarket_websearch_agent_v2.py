"""
🌙 Moon Dev's Polymarket Web Search Agent V2 - EDGE-BASED TRADING
Built with love by Moon Dev 🚀

V2 ENHANCEMENTS:
- Edge-Based Entry: AI predicts probability, calculates edge vs market price
- Momentum-Based Exit: Trailing stop loss that locks in profits
- Time-Based Exit: Auto-sell after max holding period
- Relaxed Time Filters: 90 days max (catches early movers on longer markets)
- Expected Value (EV) calculation for smarter bet sizing

This version can run side-by-side with V1 for A/B testing!
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import threading
import websocket
from datetime import datetime, timedelta
from pathlib import Path
from termcolor import cprint

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_factory import ModelFactory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
# V2 CONFIGURATION - Edge-Based Trading
# ==============================================================================

# Trade filtering (same as V1)
MIN_TRADE_SIZE_USD = 500
IGNORE_PRICE_THRESHOLD = 0.02
LOOKBACK_HOURS = 24

# Market category filters
IGNORE_CRYPTO_KEYWORDS = [
    'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol',
    'dogecoin', 'doge', 'shiba', 'cardano', 'ada', 'ripple', 'xrp',
]

IGNORE_SPORTS_KEYWORDS = [
    'nba', 'nfl', 'mlb', 'nhl', 'mls', 'ufc', 'boxing',
    'football', 'basketball', 'baseball', 'hockey', 'soccer',
    'super bowl', 'world series', 'playoffs', 'championship',
    'lakers', 'warriors', 'celtics', 'knicks', 'heat', 'bucks',
    'cowboys', 'patriots', 'chiefs', 'eagles', 'packers',
    'yankees', 'dodgers', 'red sox', 'mets',
    'premier league', 'la liga', 'champions league',
    'tennis', 'golf', 'nascar', 'formula 1', 'f1', 'cricket',
]

# Agent behavior
ANALYSIS_CHECK_INTERVAL_SECONDS = 300
NEW_MARKETS_FOR_ANALYSIS = 3
MARKETS_TO_ANALYZE = 3
MARKETS_TO_DISPLAY = 20
REANALYSIS_HOURS = 8

# AI Configuration
USE_SWARM_MODE = True
AI_MODEL_PROVIDER = "xai"
AI_MODEL_NAME = "grok-2-fast-reasoning"
SEND_PRICE_INFO_TO_AI = True  # V2: ALWAYS send price for edge calculation!

# Web Search Configuration
WEB_SEARCH_MODEL = "gpt-4o-mini-search-preview"
WEB_SEARCH_TIMEOUT = 60
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# ==============================================================================
# 🌙 V2 - EDGE-BASED AI PROMPTS (NEW!)
# ==============================================================================

MARKET_ANALYSIS_SYSTEM_PROMPT_V2 = """You are a prediction market expert analyzing Polymarket markets.
You have been provided with RECENT NEWS AND CONTEXT for each market from web search.

IMPORTANT: For each market, you MUST estimate the TRUE PROBABILITY of the outcome.
Compare your probability estimate to the market price to find EDGE (mispricing).

Edge = Your Probability - Market Probability
- Positive edge on YES side means market undervalues YES
- Negative edge means market overvalues YES (bet NO instead)

For each market, provide your prediction in this EXACT format:

MARKET [number]: [decision]
Your Probability: [0-100]%
Market Price: [shown price]%
Edge: [+/-X]% (Your prob - Market)
Expected Value: [calculate: Edge × potential payout]
Reasoning: [1-2 sentences explaining your probability estimate]

Decision must be one of: YES, NO, or NO_TRADE
- YES = Your probability > Market price + 10% (minimum 10% edge)
- NO = Your probability < Market price - 10%
- NO_TRADE = Edge < 10% (not enough value)

Be contrarian when you have conviction. Look for markets where news hasn't been priced in yet."""

CONSENSUS_AI_PROMPT_TEMPLATE_V2 = """You are analyzing predictions from multiple AI models on Polymarket markets.

MARKET REFERENCE:
{market_reference}

ALL AI RESPONSES:
{all_responses}

Based on ALL of these AI responses, identify the TOP {top_count} MARKETS with:
1. STRONGEST CONSENSUS (most models agree)
2. HIGHEST AVERAGE EDGE (biggest mispricing)

Rules:
- Look for markets where most AIs agree on the same side
- Prioritize markets with higher calculated edge
- Include the AVERAGE EDGE from all models
- Focus on actionable trades with real value

Format your response EXACTLY like this:

TOP {top_count} EDGE-BASED PICKS:

1. Market [number]: [market title]
   Side: [YES/NO/NO_TRADE]
   Consensus: [X out of Y models agreed]
   Average Edge: [+/-X]%
   Average AI Probability: [X]%
   Market Price: [X]%
   Link: [polymarket URL]
   Reasoning: [1 sentence on why this is mispriced]

2. Market [number]: [market title]
   Side: [YES/NO/NO_TRADE]
   Consensus: [X out of Y models agreed]
   Average Edge: [+/-X]%
   Average AI Probability: [X]%
   Market Price: [X]%
   Link: [polymarket URL]
   Reasoning: [1 sentence on why this is mispriced]

[Continue for all {top_count} markets...]
"""

# Data paths - V2 uses separate files for A/B testing!
DATA_FOLDER = os.path.join(project_root, "src/data/polymarket_websearch_v2")
MARKETS_CSV = os.path.join(DATA_FOLDER, "markets_v2.csv")
PREDICTIONS_CSV = os.path.join(DATA_FOLDER, "predictions_v2.csv")
CONSENSUS_PICKS_CSV = os.path.join(DATA_FOLDER, "consensus_picks_v2.csv")
WEB_SEARCH_LOG_CSV = os.path.join(DATA_FOLDER, "web_search_log_v2.csv")

# Polymarket API & WebSocket
POLYMARKET_API_BASE = "https://data-api.polymarket.com"
WEBSOCKET_URL = "wss://ws-live-data.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"

# ==============================================================================
# 🌙 V2 - PAPER TRADING Configuration (Edge-Based)
# ==============================================================================
PAPER_TRADING_MODE = True
PAPER_TRADING_BANKROLL = 10000.0
BET_SIZE_PERCENT = 0.05
MAX_OPEN_POSITIONS = 10

# V2: Edge-based entry requirements
MIN_CONSENSUS_FOR_TRADE = 5  # 5/6 models must agree
MIN_EDGE_FOR_TRADE = 0.10  # Minimum 10% edge required to enter

# Paper trading data paths (V2 separate)
PAPER_PORTFOLIO_CSV = os.path.join(DATA_FOLDER, "paper_portfolio_v2.csv")
PAPER_TRADES_CSV = os.path.join(DATA_FOLDER, "paper_trades_v2.csv")

# ==============================================================================
# 🌙 V2 - TIME SENSITIVITY FILTERS (RELAXED!)
# ==============================================================================
ENABLE_TIME_FILTERS = True
MAX_DAYS_TO_RESOLUTION = 90   # V2: Relaxed from 30 to 90 days (catch early movers!)
MIN_DAYS_TO_RESOLUTION = 1
PRIORITIZE_IMMINENT = True
IMMINENT_DAYS_MIN = 3
IMMINENT_DAYS_MAX = 30  # Still prioritize 3-30 day markets

# ==============================================================================
# 🌙 V2 - MOMENTUM-BASED EXIT STRATEGY (NEW!)
# ==============================================================================

# Take Profit / Stop Loss (same as V1)
TAKE_PROFIT_PERCENT = 0.15
STOP_LOSS_PERCENT = 0.20
CHECK_PRICES_ENABLED = True

# V2 NEW: Trailing Stop Loss
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ACTIVATION = 0.08  # Activate trailing stop after +8% gain
TRAILING_STOP_DISTANCE = 0.05    # Trail 5% behind highest price

# V2 NEW: Time-Based Exit (max holding period)
TIME_BASED_EXIT_ENABLED = True
MAX_HOLD_DAYS = 14  # Auto-sell after 14 days regardless of P&L

# V2 NEW: Momentum Detection
MOMENTUM_EXIT_ENABLED = True
MOMENTUM_THRESHOLD = 0.20  # Exit if price moves +20% in our favor (early profit taking)


# ==============================================================================
# 🌙 V2 - Paper Trading Functions (Edge-Enhanced)
# ==============================================================================

def init_paper_portfolio():
    """Initialize paper trading portfolio CSV"""
    if not os.path.exists(PAPER_PORTFOLIO_CSV):
        os.makedirs(DATA_FOLDER, exist_ok=True)
        df = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'balance': PAPER_TRADING_BANKROLL,
            'open_positions': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'realized_pnl': 0.0,
            'total_edge_captured': 0.0  # V2: Track edge captured
        }])
        df.to_csv(PAPER_PORTFOLIO_CSV, index=False)
        cprint(f"📝 V2 paper portfolio created with ${PAPER_TRADING_BANKROLL}", "green")
    return pd.read_csv(PAPER_PORTFOLIO_CSV)


def get_paper_portfolio_summary():
    """Get current paper trading portfolio summary"""
    if not os.path.exists(PAPER_PORTFOLIO_CSV):
        init_paper_portfolio()

    df = pd.read_csv(PAPER_PORTFOLIO_CSV)
    latest = df.iloc[-1] if len(df) > 0 else None

    if latest is None:
        return {
            'balance': PAPER_TRADING_BANKROLL,
            'open_positions': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'realized_pnl': 0.0,
            'total_edge_captured': 0.0,
            'win_rate': 0.0
        }

    total = latest.get('total_trades', 0)
    winning = latest.get('winning_trades', 0)

    return {
        'balance': latest.get('balance', PAPER_TRADING_BANKROLL),
        'open_positions': latest.get('open_positions', 0),
        'total_trades': total,
        'winning_trades': winning,
        'losing_trades': latest.get('losing_trades', 0),
        'realized_pnl': latest.get('realized_pnl', 0.0),
        'total_edge_captured': latest.get('total_edge_captured', 0.0),
        'win_rate': (winning / total * 100) if total > 0 else 0.0
    }


def execute_paper_trade_v2(pick: dict, market_price: float = 0.50):
    """
    V2: Execute paper trade with edge tracking

    Args:
        pick: The consensus pick dict
        market_price: Current market price (for edge calculation)
    """
    if not PAPER_TRADING_MODE:
        return

    portfolio = get_paper_portfolio_summary()
    balance = portfolio['balance']
    open_positions = portfolio['open_positions']

    if open_positions >= MAX_OPEN_POSITIONS:
        cprint(f"⚠️ Max positions ({MAX_OPEN_POSITIONS}) reached", "yellow")
        return

    bet_size = balance * BET_SIZE_PERCENT
    if bet_size < 1:
        cprint(f"⚠️ Insufficient balance (${balance:.2f})", "yellow")
        return

    consensus_count = pick.get('consensus_count', 0)
    if consensus_count < MIN_CONSENSUS_FOR_TRADE:
        cprint(f"⚠️ Consensus too low ({consensus_count}/6)", "yellow")
        return

    # V2: Check edge requirement
    edge = pick.get('average_edge', 0)
    if isinstance(edge, str):
        try:
            edge = float(edge.replace('%', '').replace('+', '')) / 100
        except:
            edge = 0

    if abs(edge) < MIN_EDGE_FOR_TRADE:
        cprint(f"⚠️ Edge too low ({edge*100:.1f}%) - need {MIN_EDGE_FOR_TRADE*100}%", "yellow")
        return

    # Extract condition_id
    link = pick.get('link', '')
    condition_id = ''
    if 'polymarket.com/event/' in link:
        condition_id = link.split('/event/')[-1].split('/')[0].split('?')[0]

    # V2: Record trade with edge info and trailing stop tracking
    trade = {
        'timestamp': datetime.now().isoformat(),
        'market_title': pick.get('market_title', 'Unknown'),
        'side': pick.get('side', 'Unknown'),
        'consensus': pick.get('consensus', ''),
        'consensus_count': consensus_count,
        'bet_size_usd': bet_size,
        'entry_price': market_price,
        'status': 'OPEN',
        'pnl': 0.0,
        'link': link,
        'condition_id': condition_id,
        # V2 NEW fields
        'entry_edge': edge,
        'ai_probability': pick.get('ai_probability', 0),
        'highest_price': market_price,  # For trailing stop
        'entry_timestamp': datetime.now().isoformat(),
        'exit_price': None,
        'exit_reason': '',
        'resolved_at': '',
        'outcome': ''
    }

    # Save trade
    if os.path.exists(PAPER_TRADES_CSV):
        trades_df = pd.read_csv(PAPER_TRADES_CSV)
    else:
        trades_df = pd.DataFrame()

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)
    trades_df.to_csv(PAPER_TRADES_CSV, index=False)

    # Update portfolio
    new_balance = balance - bet_size
    portfolio_update = {
        'timestamp': datetime.now().isoformat(),
        'balance': new_balance,
        'open_positions': open_positions + 1,
        'total_trades': portfolio['total_trades'] + 1,
        'winning_trades': portfolio['winning_trades'],
        'losing_trades': portfolio['losing_trades'],
        'realized_pnl': portfolio['realized_pnl'],
        'total_edge_captured': portfolio.get('total_edge_captured', 0)
    }

    portfolio_df = pd.read_csv(PAPER_PORTFOLIO_CSV)
    portfolio_df = pd.concat([portfolio_df, pd.DataFrame([portfolio_update])], ignore_index=True)
    portfolio_df.to_csv(PAPER_PORTFOLIO_CSV, index=False)

    cprint(f"\n{'='*60}", "green")
    cprint(f"📝 V2 PAPER TRADE - EDGE-BASED ENTRY", "green", attrs=['bold'])
    cprint(f"{'='*60}", "green")
    cprint(f"   Market: {pick.get('market_title', 'Unknown')[:50]}...", "white")
    cprint(f"   Side: {pick.get('side', 'Unknown')}", "cyan")
    cprint(f"   Entry Price: {market_price*100:.1f}%", "white")
    cprint(f"   AI Probability: {pick.get('ai_probability', 0)}%", "yellow")
    cprint(f"   Entry Edge: {edge*100:+.1f}%", "green" if edge > 0 else "red")
    cprint(f"   Bet Size: ${bet_size:.2f}", "green")
    cprint(f"   New Balance: ${new_balance:.2f}", "white")
    cprint(f"{'='*60}\n", "green")


def check_and_close_positions_v2():
    """
    V2: Check open positions with enhanced exit conditions:
    1. Take profit (fixed +15%)
    2. Stop loss (fixed -20%)
    3. Trailing stop (locks in profits)
    4. Time-based exit (max hold days)
    5. Momentum exit (early profit on big moves)
    """
    if not PAPER_TRADING_MODE or not CHECK_PRICES_ENABLED:
        return

    if not os.path.exists(PAPER_TRADES_CSV):
        return

    trades_df = pd.read_csv(PAPER_TRADES_CSV)
    open_trades = trades_df[trades_df['status'] == 'OPEN']

    if len(open_trades) == 0:
        cprint("📊 No open V2 positions to check", "white")
        return

    cprint(f"\n🔍 V2: Checking {len(open_trades)} positions (edge + momentum exits)...", "yellow")

    positions_closed = 0
    portfolio = get_paper_portfolio_summary()
    now = datetime.now()

    for idx, trade in open_trades.iterrows():
        market_title = trade.get('market_title', 'Unknown')
        entry_price = trade.get('entry_price', 0.50)
        bet_size = trade.get('bet_size_usd', 0)
        side = trade.get('side', 'YES')
        condition_id = trade.get('condition_id', '')
        highest_price = trade.get('highest_price', entry_price)
        entry_timestamp = trade.get('entry_timestamp', trade.get('timestamp'))
        entry_edge = trade.get('entry_edge', 0)

        # Get current price
        current_price = get_market_current_price(condition_id, side)
        if current_price is None:
            continue

        # Update highest price for trailing stop
        if side.upper() == 'YES' and current_price > highest_price:
            trades_df.loc[idx, 'highest_price'] = current_price
            highest_price = current_price
        elif side.upper() == 'NO' and current_price < highest_price:
            trades_df.loc[idx, 'highest_price'] = current_price
            highest_price = current_price

        # Calculate P&L
        if side.upper() == 'YES':
            price_change = current_price - entry_price
        else:
            price_change = entry_price - current_price

        pnl_percent = price_change / entry_price if entry_price > 0 else 0
        pnl_usd = bet_size * pnl_percent

        # Calculate days held
        try:
            entry_dt = pd.to_datetime(entry_timestamp)
            days_held = (now - entry_dt).days
        except:
            days_held = 0

        # ==============================================================
        # V2 EXIT CONDITIONS (Priority Order)
        # ==============================================================
        exit_reason = None

        # 1. Fixed Take Profit
        if pnl_percent >= TAKE_PROFIT_PERCENT:
            exit_reason = f"TAKE PROFIT (+{pnl_percent*100:.1f}%)"

        # 2. Fixed Stop Loss
        elif pnl_percent <= -STOP_LOSS_PERCENT:
            exit_reason = f"STOP LOSS ({pnl_percent*100:.1f}%)"

        # 3. V2: Trailing Stop Loss
        elif TRAILING_STOP_ENABLED and pnl_percent >= TRAILING_STOP_ACTIVATION:
            # Calculate trailing stop level
            if side.upper() == 'YES':
                trailing_stop_price = highest_price * (1 - TRAILING_STOP_DISTANCE)
                if current_price <= trailing_stop_price:
                    exit_reason = f"TRAILING STOP (peak: {highest_price*100:.1f}%, now: {current_price*100:.1f}%)"
            else:
                trailing_stop_price = highest_price * (1 + TRAILING_STOP_DISTANCE)
                if current_price >= trailing_stop_price:
                    exit_reason = f"TRAILING STOP (peak: {highest_price*100:.1f}%, now: {current_price*100:.1f}%)"

        # 4. V2: Time-Based Exit
        elif TIME_BASED_EXIT_ENABLED and days_held >= MAX_HOLD_DAYS:
            exit_reason = f"TIME EXIT ({days_held} days held, P&L: {pnl_percent*100:+.1f}%)"

        # 5. V2: Momentum Exit (big move in our favor)
        elif MOMENTUM_EXIT_ENABLED and pnl_percent >= MOMENTUM_THRESHOLD:
            exit_reason = f"MOMENTUM EXIT (+{pnl_percent*100:.1f}% - locking gains!)"

        # 6. Near Resolution
        elif current_price >= 0.95 or current_price <= 0.05:
            exit_reason = f"NEAR RESOLUTION (price={current_price*100:.1f}%)"

        # Execute exit if triggered
        if exit_reason:
            trades_df.loc[idx, 'status'] = 'CLOSED'
            trades_df.loc[idx, 'pnl'] = pnl_usd
            trades_df.loc[idx, 'exit_price'] = current_price
            trades_df.loc[idx, 'exit_reason'] = exit_reason
            trades_df.loc[idx, 'resolved_at'] = datetime.now().isoformat()

            is_win = pnl_usd > 0
            positions_closed += 1

            # Update portfolio with edge tracking
            new_balance = portfolio['balance'] + bet_size + pnl_usd
            edge_captured = entry_edge if is_win else 0

            portfolio_update = {
                'timestamp': datetime.now().isoformat(),
                'balance': new_balance,
                'open_positions': portfolio['open_positions'] - 1,
                'total_trades': portfolio['total_trades'],
                'winning_trades': portfolio['winning_trades'] + (1 if is_win else 0),
                'losing_trades': portfolio['losing_trades'] + (0 if is_win else 1),
                'realized_pnl': portfolio['realized_pnl'] + pnl_usd,
                'total_edge_captured': portfolio.get('total_edge_captured', 0) + edge_captured
            }

            portfolio_df = pd.read_csv(PAPER_PORTFOLIO_CSV)
            portfolio_df = pd.concat([portfolio_df, pd.DataFrame([portfolio_update])], ignore_index=True)
            portfolio_df.to_csv(PAPER_PORTFOLIO_CSV, index=False)

            portfolio = get_paper_portfolio_summary()

            color = "green" if is_win else "red"
            cprint(f"\n{'='*60}", color)
            cprint(f"💰 V2 POSITION CLOSED - {exit_reason}", color, attrs=['bold'])
            cprint(f"{'='*60}", color)
            cprint(f"   Market: {market_title[:40]}...", "white")
            cprint(f"   Side: {side}", "cyan")
            cprint(f"   Entry: {entry_price*100:.1f}% → Exit: {current_price*100:.1f}%", "white")
            cprint(f"   Entry Edge: {entry_edge*100:+.1f}%", "yellow")
            cprint(f"   Days Held: {days_held}", "white")
            cprint(f"   P&L: ${pnl_usd:.2f} ({pnl_percent*100:+.1f}%)", color)
            cprint(f"   New Balance: ${new_balance:.2f}", "white")
            cprint(f"{'='*60}\n", color)

    # Save updated trades
    trades_df.to_csv(PAPER_TRADES_CSV, index=False)

    if positions_closed > 0:
        cprint(f"\n✅ V2: Closed {positions_closed} positions", "green")
    else:
        cprint(f"📊 V2: No positions hit exit conditions", "white")


def get_market_current_price(condition_id: str, side: str) -> float:
    """Get current market price from Polymarket"""
    if not condition_id:
        return None

    try:
        url = f"{POLYMARKET_GAMMA_API}/markets/{condition_id}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            prices = data.get('outcomePrices', [])
            if prices and len(prices) >= 2:
                yes_price = float(prices[0]) if prices[0] else 0.5
                no_price = float(prices[1]) if prices[1] else 0.5
                return yes_price if side.upper() == 'YES' else no_price

        # Fallback to CLOB
        clob_url = f"https://clob.polymarket.com/markets/{condition_id}"
        response = requests.get(clob_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            tokens = data.get('tokens', [])
            for token in tokens:
                if token.get('outcome', '').upper() == side.upper():
                    return float(token.get('price', 0.5))
    except:
        pass

    return None


def print_paper_portfolio_status_v2():
    """Print V2 portfolio status with edge metrics"""
    if not PAPER_TRADING_MODE:
        return

    summary = get_paper_portfolio_summary()

    cprint(f"\n{'='*60}", "cyan")
    cprint(f"💼 V2 PAPER TRADING PORTFOLIO (EDGE-BASED)", "cyan", attrs=['bold'])
    cprint(f"{'='*60}", "cyan")
    cprint(f"   Balance: ${summary['balance']:.2f}", "green" if summary['balance'] >= PAPER_TRADING_BANKROLL else "red")
    cprint(f"   Open Positions: {summary['open_positions']}/{MAX_OPEN_POSITIONS}", "white")
    cprint(f"   Total Trades: {summary['total_trades']}", "white")
    cprint(f"   Win Rate: {summary['win_rate']:.1f}%", "green" if summary['win_rate'] >= 50 else "yellow")
    cprint(f"   Realized P&L: ${summary['realized_pnl']:.2f}", "green" if summary['realized_pnl'] >= 0 else "red")
    cprint(f"   Total Edge Captured: {summary.get('total_edge_captured', 0)*100:.1f}%", "green")
    cprint(f"{'='*60}\n", "cyan")


# ==============================================================================
# V2 Polymarket Web Search Agent
# ==============================================================================

class PolymarketWebSearchAgentV2:
    """V2 Agent with edge-based entry and momentum-based exits"""

    def __init__(self):
        cprint("\n" + "="*80, "cyan")
        cprint("🌙 Moon Dev's Polymarket WEB SEARCH Agent V2", "cyan", attrs=['bold'])
        cprint("🎯 EDGE-BASED TRADING + MOMENTUM EXITS", "yellow", attrs=['bold'])
        cprint("="*80, "cyan")

        os.makedirs(DATA_FOLDER, exist_ok=True)

        self.csv_lock = threading.Lock()
        self.last_analyzed_count = 0
        self.last_analysis_run_timestamp = None

        self.ws = None
        self.ws_connected = False
        self.total_trades_received = 0
        self.filtered_trades_count = 0
        self.ignored_crypto_count = 0
        self.ignored_sports_count = 0

        if not OPENAI_API_KEY:
            cprint("⚠️ WARNING: OPENAI_KEY not found!", "red", attrs=['bold'])
        else:
            cprint(f"✅ OpenAI API key configured", "green")
            cprint(f"🔍 Web search model: {WEB_SEARCH_MODEL}", "cyan")

        if USE_SWARM_MODE:
            cprint("🤖 Using SWARM MODE - Multiple AI models", "green")
            try:
                from src.agents.swarm_agent import SwarmAgent
                self.swarm = SwarmAgent()
                cprint("✅ Swarm agent loaded", "green")
            except Exception as e:
                cprint(f"❌ Swarm load failed: {e}", "red")
                self.swarm = None
                self.model = ModelFactory().get_model(AI_MODEL_PROVIDER, AI_MODEL_NAME)
        else:
            self.model = ModelFactory().get_model(AI_MODEL_PROVIDER, AI_MODEL_NAME)
            self.swarm = None

        self.markets_df = self._load_markets()
        self.predictions_df = self._load_predictions()
        self._init_web_search_log()

        cprint(f"📊 Loaded {len(self.markets_df)} markets", "cyan")
        cprint(f"🔮 Loaded {len(self.predictions_df)} predictions", "cyan")
        cprint("✨ V2 Initialization complete!\n", "green")

    def _init_web_search_log(self):
        if not os.path.exists(WEB_SEARCH_LOG_CSV):
            df = pd.DataFrame(columns=['timestamp', 'market_title', 'search_query', 'response_length', 'response_preview'])
            df.to_csv(WEB_SEARCH_LOG_CSV, index=False)

    def _load_markets(self):
        if os.path.exists(MARKETS_CSV):
            try:
                return pd.read_csv(MARKETS_CSV)
            except:
                pass
        return pd.DataFrame(columns=['timestamp', 'market_id', 'event_slug', 'title', 'outcome', 'price', 'size_usd', 'first_seen', 'last_analyzed', 'last_trade_timestamp'])

    def _load_predictions(self):
        if os.path.exists(PREDICTIONS_CSV):
            try:
                return pd.read_csv(PREDICTIONS_CSV)
            except:
                pass
        return pd.DataFrame(columns=['analysis_timestamp', 'analysis_run_id', 'market_title', 'market_slug', 'consensus_prediction', 'num_models_responded', 'web_search_used', 'market_link', 'average_edge', 'ai_probability'])

    def _save_markets(self):
        try:
            with self.csv_lock:
                self.markets_df.to_csv(MARKETS_CSV, index=False)
        except Exception as e:
            cprint(f"❌ Error saving markets: {e}", "red")

    def _save_predictions(self):
        try:
            with self.csv_lock:
                self.predictions_df.to_csv(PREDICTIONS_CSV, index=False)
            cprint(f"💾 Saved {len(self.predictions_df)} predictions", "green")
        except Exception as e:
            cprint(f"❌ Error saving predictions: {e}", "red")

    def search_market_context(self, market_title: str) -> str:
        """Search web for market context"""
        cprint(f"\n{'='*60}", "yellow")
        cprint(f"🔍 WEB SEARCH: {market_title[:50]}...", "yellow", attrs=['bold'])
        cprint(f"{'='*60}", "yellow")

        if not OPENAI_API_KEY:
            return "No web search context (API key missing)"

        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

            user_message = f"""Search for the latest news about: {market_title}

Find recent news, updates, and relevant context for this prediction market.
Focus on: Recent news, official announcements, expert opinions, relevant data.
Provide a concise summary of the most relevant information."""

            payload = {"model": WEB_SEARCH_MODEL, "messages": [{"role": "user", "content": user_message}]}

            response = requests.post(url, headers=headers, json=payload, timeout=WEB_SEARCH_TIMEOUT)

            if response.status_code != 200:
                return f"Web search failed (status {response.status_code})"

            response_json = response.json()
            content = ""
            if 'choices' in response_json and len(response_json['choices']) > 0:
                content = response_json['choices'][0].get('message', {}).get('content', '')

            if content:
                preview = content[:500] + "..." if len(content) > 500 else content
                cprint(f"\n{'─'*60}", "green")
                cprint("📰 WEB RESULTS:", "green", attrs=['bold'])
                cprint(preview, "white")
                cprint(f"{'─'*60}", "green")
                return content

            return "No web search results found"

        except requests.exceptions.Timeout:
            return "Web search timed out"
        except Exception as e:
            return f"Web search error: {str(e)}"

    def is_near_resolution(self, price):
        price_float = float(price)
        return price_float <= IGNORE_PRICE_THRESHOLD or price_float >= (1.0 - IGNORE_PRICE_THRESHOLD)

    def should_ignore_market(self, title):
        title_lower = title.lower()
        for keyword in IGNORE_CRYPTO_KEYWORDS:
            if keyword in title_lower:
                return (True, f"crypto ({keyword})")
        for keyword in IGNORE_SPORTS_KEYWORDS:
            if keyword in title_lower:
                return (True, f"sports ({keyword})")
        return (False, None)

    # WebSocket handlers
    def on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                if data.get('type') == 'subscribed':
                    cprint("✅ V2 WebSocket subscribed!", "green")
                    self.ws_connected = True
                    return
                if data.get('type') == 'pong':
                    return

                topic = data.get('topic')
                msg_type = data.get('type')
                payload = data.get('payload', {})

                if topic == 'activity' and msg_type == 'orders_matched':
                    self.total_trades_received += 1
                    if not self.ws_connected:
                        self.ws_connected = True

                    price = float(payload.get('price', 0))
                    size = float(payload.get('size', 0))
                    usd_amount = price * size
                    title = payload.get('title', 'Unknown')

                    should_ignore, _ = self.should_ignore_market(title)
                    if should_ignore:
                        return

                    if usd_amount >= MIN_TRADE_SIZE_USD and not self.is_near_resolution(price):
                        self.filtered_trades_count += 1
                        trade_data = {
                            'timestamp': payload.get('timestamp', time.time()),
                            'conditionId': payload.get('conditionId', f"ws_{time.time()}"),
                            'eventSlug': payload.get('eventSlug', ''),
                            'title': title,
                            'outcome': payload.get('outcome', 'Unknown'),
                            'price': price,
                            'size': usd_amount,
                        }
                        self.process_trades([trade_data])
        except:
            pass

    def on_ws_error(self, ws, error):
        cprint(f"❌ V2 WebSocket Error: {error}", "red")

    def on_ws_close(self, ws, close_status_code, close_msg):
        self.ws_connected = False
        cprint(f"🔌 V2 WebSocket closed, reconnecting...", "yellow")
        time.sleep(5)
        self.connect_websocket()

    def on_ws_open(self, ws):
        cprint("🔌 V2 WebSocket connected!", "green")
        subscription_msg = {"action": "subscribe", "subscriptions": [{"topic": "activity", "type": "orders_matched"}]}
        ws.send(json.dumps(subscription_msg))
        self.ws_connected = True

        def send_ping():
            while True:
                time.sleep(5)
                try:
                    ws.send(json.dumps({"type": "ping"}))
                except:
                    break
        threading.Thread(target=send_ping, daemon=True).start()

    def connect_websocket(self):
        cprint(f"🚀 V2 Connecting to {WEBSOCKET_URL}...", "cyan")
        self.ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_open=self.on_ws_open,
            on_message=self.on_ws_message,
            on_error=self.on_ws_error,
            on_close=self.on_ws_close
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def fetch_historical_trades(self, hours_back=None):
        if hours_back is None:
            hours_back = LOOKBACK_HOURS
        try:
            cprint(f"\n📡 V2 Fetching historical trades ({hours_back}h)...", "yellow")
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            cutoff_timestamp = int(cutoff_time.timestamp())
            url = f"{POLYMARKET_API_BASE}/trades"
            params = {'limit': 1000, '_min_timestamp': cutoff_timestamp}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            trades = response.json()
            cprint(f"✅ Fetched {len(trades)} trades", "green")

            filtered_trades = []
            for trade in trades:
                price = float(trade.get('price', 0))
                size = float(trade.get('size', 0))
                usd_amount = price * size
                title = trade.get('title', 'Unknown')
                should_ignore, _ = self.should_ignore_market(title)
                if should_ignore:
                    continue
                if usd_amount >= MIN_TRADE_SIZE_USD and not self.is_near_resolution(price):
                    filtered_trades.append(trade)

            cprint(f"💰 {len(filtered_trades)} trades over ${MIN_TRADE_SIZE_USD}", "cyan")
            return filtered_trades
        except Exception as e:
            cprint(f"❌ Historical fetch error: {e}", "red")
            return []

    def process_trades(self, trades):
        if not trades:
            return

        unique_markets = {}
        for trade in trades:
            market_id = trade.get('conditionId', '')
            if market_id and market_id not in unique_markets:
                unique_markets[market_id] = trade

        new_markets = 0
        for market_id, trade in unique_markets.items():
            try:
                if market_id in self.markets_df['market_id'].values:
                    mask = self.markets_df['market_id'] == market_id
                    self.markets_df.loc[mask, 'last_trade_timestamp'] = datetime.now().isoformat()
                    continue

                new_market = {
                    'timestamp': trade.get('timestamp', ''),
                    'market_id': trade.get('conditionId', ''),
                    'event_slug': trade.get('eventSlug', ''),
                    'title': trade.get('title', 'Unknown'),
                    'outcome': trade.get('outcome', ''),
                    'price': float(trade.get('price', 0)),
                    'size_usd': float(trade.get('size', 0)),
                    'first_seen': datetime.now().isoformat(),
                    'last_analyzed': None,
                    'last_trade_timestamp': datetime.now().isoformat()
                }

                self.markets_df = pd.concat([self.markets_df, pd.DataFrame([new_market])], ignore_index=True)
                new_markets += 1
                cprint(f"✨ V2 NEW: ${new_market['size_usd']:,.0f} - {new_market['title'][:60]}", "green")
            except:
                continue

        if new_markets > 0:
            self._save_markets()

    def get_ai_predictions_v2(self):
        """V2: Get AI predictions with EDGE CALCULATION"""
        if len(self.markets_df) == 0:
            cprint("⚠️ No markets to analyze", "yellow")
            return

        markets_to_analyze = self.markets_df.tail(MARKETS_TO_ANALYZE)
        analysis_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_timestamp = datetime.now().isoformat()

        cprint("\n" + "="*80, "magenta")
        cprint(f"🤖 V2 AI Analysis - EDGE-BASED", "magenta", attrs=['bold'])
        cprint(f"📊 Run ID: {analysis_run_id}", "magenta")
        cprint("="*80, "magenta")

        # Phase 1: Web Search
        cprint("\n🔍 PHASE 1: WEB SEARCH", "yellow", attrs=['bold'])
        web_contexts = {}
        for i, (_, row) in enumerate(markets_to_analyze.iterrows()):
            web_contexts[row['title']] = self.search_market_context(row['title'])
            if i < len(markets_to_analyze) - 1:
                time.sleep(2)

        # Phase 2: Build Edge-Based Prompt
        cprint("\n📝 PHASE 2: EDGE-BASED PROMPT", "green", attrs=['bold'])
        markets_text = ""
        for i, (_, row) in enumerate(markets_to_analyze.iterrows()):
            web_context = web_contexts.get(row['title'], "No context")[:1000]
            market_price_pct = row['price'] * 100

            markets_text += f"""
Market {i+1}:
Title: {row['title']}
Current Market Price: {market_price_pct:.1f}% (this is the market's implied probability)
Recent trade: ${row['size_usd']:,.2f} on {row['outcome']}
Link: https://polymarket.com/event/{row['event_slug']}

📰 WEB SEARCH CONTEXT:
{web_context}

---
"""

        # Use V2 edge-based prompt
        system_prompt = MARKET_ANALYSIS_SYSTEM_PROMPT_V2
        user_prompt = f"""Analyze these {len(markets_to_analyze)} Polymarket markets.

For EACH market:
1. Estimate the TRUE PROBABILITY based on news context
2. Calculate EDGE vs market price
3. Only recommend trades with 10%+ edge

{markets_text}

Provide predictions in the exact format specified."""

        # Phase 3: Swarm Analysis
        if USE_SWARM_MODE and self.swarm:
            cprint("\n🌊 PHASE 3: AI SWARM (Edge Calculation)", "blue", attrs=['bold'])

            swarm_result = self.swarm.query(prompt=user_prompt, system_prompt=system_prompt)

            if not swarm_result or not swarm_result.get('responses'):
                cprint("❌ No swarm responses", "red")
                return

            successful = [n for n, d in swarm_result.get('responses', {}).items() if d.get('success')]
            if not successful:
                cprint("❌ All models failed", "red")
                return

            cprint(f"\n✅ {len(successful)} models responded", "green", attrs=['bold'])

            # Display responses
            for model_name, model_data in swarm_result.get('responses', {}).items():
                if model_data.get('success'):
                    cprint(f"\n{'='*60}", "cyan")
                    cprint(f"✅ {model_name.upper()}", "cyan", attrs=['bold'])
                    cprint(f"{'='*60}", "cyan")
                    cprint(model_data.get('response', '')[:1500], "white")

            # Get top picks with edge
            self._get_top_consensus_picks_v2(swarm_result, markets_to_analyze)

            self._mark_markets_analyzed(markets_to_analyze, analysis_timestamp)

    def _mark_markets_analyzed(self, markets, timestamp):
        try:
            for market_id in markets['market_id'].tolist():
                mask = self.markets_df['market_id'] == market_id
                self.markets_df.loc[mask, 'last_analyzed'] = timestamp
            self._save_markets()
        except:
            pass

    def _get_top_consensus_picks_v2(self, swarm_result, markets_df):
        """V2: Get consensus picks with edge calculation"""
        try:
            cprint("\n" + "="*80, "yellow")
            cprint(f"🧠 V2 CONSENSUS - EDGE-BASED PICKS", "yellow", attrs=['bold'])
            cprint("="*80, "yellow")

            all_responses_text = ""
            for model_name, model_data in swarm_result.get('responses', {}).items():
                if model_data.get('success'):
                    all_responses_text += f"\n{'='*40}\n{model_name.upper()}:\n{'='*40}\n"
                    all_responses_text += model_data.get('response', '') + "\n"

            markets_list = list(markets_df.iterrows())
            market_reference = "\n".join([
                f"Market {i+1}: {row['title']}\nMarket Price: {row['price']*100:.1f}%\nLink: https://polymarket.com/event/{row['event_slug']}"
                for i, (_, row) in enumerate(markets_list)
            ])

            consensus_prompt = CONSENSUS_AI_PROMPT_TEMPLATE_V2.format(
                market_reference=market_reference,
                all_responses=all_responses_text,
                top_count=5
            )

            consensus_model = ModelFactory().get_model('claude', 'claude-sonnet-4-5')

            response = consensus_model.generate_response(
                system_prompt="You analyze predictions to find strongest consensus WITH calculated edge. Focus on mispricing opportunities.",
                user_content=consensus_prompt,
                temperature=0.3,
                max_tokens=1500
            )

            cprint("\n" + "="*80, "white", "on_blue", attrs=['bold'])
            cprint(f"🏆 V2 TOP EDGE-BASED PICKS", "white", "on_blue", attrs=['bold'])
            cprint("="*80, "white", "on_blue", attrs=['bold'])
            cprint(response.content, "cyan", attrs=['bold'])
            cprint("="*80 + "\n", "white", "on_blue", attrs=['bold'])

            self._save_consensus_picks_v2(response.content, markets_df)

        except Exception as e:
            cprint(f"❌ Consensus error: {e}", "red")

    def _save_consensus_picks_v2(self, consensus_response, markets_df):
        """V2: Save picks with edge info and execute trades"""
        try:
            import re
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
                    consensus_text = line.replace('Consensus:', '').strip()
                    current_pick['consensus'] = consensus_text
                    match = re.search(r'(\d+)\s+out\s+of\s+(\d+)', consensus_text)
                    if match:
                        current_pick['consensus_count'] = int(match.group(1))
                        current_pick['total_models'] = int(match.group(2))
                elif line.startswith('Average Edge:'):
                    edge_text = line.replace('Average Edge:', '').strip()
                    current_pick['average_edge'] = edge_text
                    # Parse numeric edge
                    try:
                        edge_num = float(re.search(r'[+-]?\d+\.?\d*', edge_text).group()) / 100
                        current_pick['edge_numeric'] = edge_num
                    except:
                        current_pick['edge_numeric'] = 0
                elif line.startswith('Average AI Probability:'):
                    prob_text = line.replace('Average AI Probability:', '').strip()
                    current_pick['ai_probability'] = prob_text
                elif line.startswith('Market Price:'):
                    price_text = line.replace('Market Price:', '').strip()
                    current_pick['market_price'] = price_text
                    try:
                        current_pick['market_price_numeric'] = float(re.search(r'\d+\.?\d*', price_text).group()) / 100
                    except:
                        current_pick['market_price_numeric'] = 0.5
                elif line.startswith('Link:'):
                    current_pick['link'] = line.replace('Link:', '').strip()
                elif line.startswith('Reasoning:'):
                    current_pick['reasoning'] = line.replace('Reasoning:', '').strip()

            if current_pick:
                picks.append(current_pick)

            if not picks:
                cprint("⚠️ Could not parse V2 picks", "yellow")
                return

            # Save to CSV
            timestamp = datetime.now().isoformat()
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            records = []
            for pick in picks:
                records.append({
                    'timestamp': timestamp,
                    'run_id': run_id,
                    'rank': pick.get('rank', ''),
                    'market_title': pick.get('market_title', ''),
                    'side': pick.get('side', ''),
                    'consensus': pick.get('consensus', ''),
                    'consensus_count': pick.get('consensus_count', 0),
                    'average_edge': pick.get('average_edge', ''),
                    'ai_probability': pick.get('ai_probability', ''),
                    'market_price': pick.get('market_price', ''),
                    'reasoning': pick.get('reasoning', ''),
                    'link': pick.get('link', '')
                })

            if os.path.exists(CONSENSUS_PICKS_CSV):
                consensus_df = pd.read_csv(CONSENSUS_PICKS_CSV)
            else:
                consensus_df = pd.DataFrame()

            consensus_df = pd.concat([consensus_df, pd.DataFrame(records)], ignore_index=True)

            with self.csv_lock:
                consensus_df.to_csv(CONSENSUS_PICKS_CSV, index=False)

            cprint(f"✅ V2: Saved {len(records)} edge-based picks", "green")

            # Execute paper trades for high-edge picks
            if PAPER_TRADING_MODE:
                cprint(f"\n📝 V2: Checking for edge-based trades...", "yellow")
                for pick in picks:
                    consensus_count = pick.get('consensus_count', 0)
                    edge = pick.get('edge_numeric', 0)
                    market_price = pick.get('market_price_numeric', 0.5)

                    if consensus_count >= MIN_CONSENSUS_FOR_TRADE and abs(edge) >= MIN_EDGE_FOR_TRADE:
                        execute_paper_trade_v2(pick, market_price)

                print_paper_portfolio_status_v2()

        except Exception as e:
            cprint(f"❌ V2 save error: {e}", "red")

    def status_display_loop(self):
        while True:
            try:
                time.sleep(30)
                cprint(f"\n{'='*60}", "cyan")
                cprint(f"📊 V2 Status @ {datetime.now().strftime('%H:%M:%S')}", "cyan", attrs=['bold'])
                cprint(f"{'='*60}", "cyan")
                cprint(f"   WebSocket: {'✅' if self.ws_connected else '❌'}", "green" if self.ws_connected else "red")
                cprint(f"   Total trades: {self.total_trades_received}", "white")
                cprint(f"   Filtered: {self.filtered_trades_count}", "yellow")
                cprint(f"   Markets: {len(self.markets_df)}", "white")
                cprint(f"   🎯 Edge-based trading: ENABLED", "green")
                cprint(f"{'='*60}\n", "cyan")
            except KeyboardInterrupt:
                break
            except:
                pass

    def analysis_loop(self):
        cprint("\n🤖 V2 ANALYSIS THREAD STARTED", "magenta", attrs=['bold'])

        while True:
            try:
                # V2: Check positions with enhanced exits
                if PAPER_TRADING_MODE and CHECK_PRICES_ENABLED:
                    check_and_close_positions_v2()

                # Run analysis
                if len(self.markets_df) > 0:
                    self.get_ai_predictions_v2()
                    self.last_analysis_run_timestamp = datetime.now().isoformat()

                next_check = datetime.now() + timedelta(seconds=ANALYSIS_CHECK_INTERVAL_SECONDS)
                cprint(f"⏰ V2 Next analysis: {next_check.strftime('%H:%M:%S')}\n", "magenta")

                time.sleep(ANALYSIS_CHECK_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"❌ V2 Analysis error: {e}", "red")
                time.sleep(ANALYSIS_CHECK_INTERVAL_SECONDS)


def main():
    """V2 Main - Edge-Based Trading"""
    cprint("\n" + "="*80, "cyan")
    cprint("🌙 Moon Dev's Polymarket WEB SEARCH Agent V2", "cyan", attrs=['bold'])
    cprint("🎯 EDGE-BASED TRADING + MOMENTUM EXITS", "yellow", attrs=['bold'])
    cprint("="*80, "cyan")
    cprint(f"💰 Tracking trades over ${MIN_TRADE_SIZE_USD}", "yellow")
    cprint("")
    cprint("🔥 V2 ENHANCEMENTS:", "green", attrs=['bold'])
    cprint(f"   📊 Edge-based entry: Min {MIN_EDGE_FOR_TRADE*100}% edge required", "cyan")
    cprint(f"   📈 Trailing stop: Activates at +{TRAILING_STOP_ACTIVATION*100}%, trails {TRAILING_STOP_DISTANCE*100}%", "cyan")
    cprint(f"   ⏰ Time exit: Auto-sell after {MAX_HOLD_DAYS} days", "cyan")
    cprint(f"   🚀 Momentum exit: Lock gains at +{MOMENTUM_THRESHOLD*100}%", "cyan")
    cprint(f"   📅 Relaxed time filter: Up to {MAX_DAYS_TO_RESOLUTION} days to resolution", "cyan")
    cprint("")

    if PAPER_TRADING_MODE:
        cprint("💵 V2 PAPER TRADING: ENABLED", "green", attrs=['bold'])
        cprint(f"   💰 Bankroll: ${PAPER_TRADING_BANKROLL}", "green")
        cprint(f"   📊 Bet Size: {BET_SIZE_PERCENT*100}%", "green")
        cprint(f"   🎯 Min Consensus: {MIN_CONSENSUS_FOR_TRADE}/6", "green")
        cprint(f"   📈 Min Edge: {MIN_EDGE_FOR_TRADE*100}%", "green")
        init_paper_portfolio()
        print_paper_portfolio_status_v2()
        check_and_close_positions_v2()

    cprint("="*80 + "\n", "cyan")

    agent = PolymarketWebSearchAgentV2()

    cprint(f"\n📜 Fetching historical data ({LOOKBACK_HOURS}h)...", "yellow")
    historical_trades = agent.fetch_historical_trades()
    if historical_trades:
        agent.process_trades(historical_trades)
        cprint(f"✅ V2 Database: {len(agent.markets_df)} markets", "green")

    agent.connect_websocket()

    status_thread = threading.Thread(target=agent.status_display_loop, daemon=True)
    analysis_thread = threading.Thread(target=agent.analysis_loop, daemon=True)

    try:
        cprint("🚀 V2 Starting threads...\n", "green", attrs=['bold'])
        status_thread.start()
        analysis_thread.start()

        cprint("✨ V2 Agent running! Press Ctrl+C to stop.\n", "green", attrs=['bold'])
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        cprint("\n\n⚠️ V2 Agent stopped by user", "yellow", attrs=['bold'])
        sys.exit(0)


if __name__ == "__main__":
    main()
