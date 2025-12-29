"""
Value Trading Strategy Engine
==============================
Optimized for Kalshi prediction markets with Game of 25 opponent modeling.

Core Philosophy:
- VALUE TRADING > Bluff-Heavy Strategies
- High-probability entries based on edge detection
- Adapts to opponent mix (50% loose / 50% tight players)
- Minimizes variance through Kelly Criterion position sizing

Strategy Components:
1. Edge Detection: AI-estimated probability vs market price
2. Opponent Modeling: Adjusts thresholds based on player types
3. Position Sizing: Fractional Kelly for bankroll management
4. Entry Filters: Multiple confirmation gates before trade

Author: Moon Dev
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import math
from termcolor import cprint

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


# ==============================================================================
# STRATEGY CONFIGURATION
# ==============================================================================

class TradingStyle(Enum):
    """
    Trading style selection for strategy tuning.

    VALUE: Conservative, high-edge entries (recommended)
    BALANCED: Mix of value and opportunistic plays
    AGGRESSIVE: Lower edge threshold, more trades (higher variance)
    """
    VALUE = "value"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class OpponentType(Enum):
    """
    Opponent classification for Game of 25 modeling.

    LOOSE: Trades frequently, accepts smaller edges (50% of market)
    TIGHT: Trades selectively, requires large edges (50% of market)
    UNKNOWN: Default state before classification
    """
    LOOSE = "loose"
    TIGHT = "tight"
    UNKNOWN = "unknown"


@dataclass
class StrategyConfig:
    """
    Master configuration for Value Trading Strategy.

    Edge Thresholds:
        min_edge_pct: Minimum edge required to enter (default 10%)
        high_edge_pct: Edge considered "high value" for larger positions
        extreme_edge_pct: Edge for max position sizing

    Position Sizing (Fractional Kelly):
        kelly_fraction: Fraction of full Kelly bet (0.25 = quarter Kelly)
        max_position_pct: Max single position as % of bankroll
        min_position_usd: Minimum position size in USD

    Consensus Requirements:
        min_consensus_models: Minimum AI models agreeing (out of 6)
        min_consensus_pct: Alternative - minimum agreement percentage

    Price Filters:
        avoid_extreme_prices: Skip prices near 0 or 1
        price_floor: Minimum acceptable price
        price_ceiling: Maximum acceptable price

    Time Filters:
        min_days_to_resolution: Skip very short-term markets
        max_days_to_resolution: Skip very long-term markets
        sweet_spot_min: Start of ideal resolution window
        sweet_spot_max: End of ideal resolution window
    """
    # Edge thresholds (as decimals, e.g., 0.10 = 10%)
    min_edge_pct: float = 0.10  # 10% minimum edge for VALUE trading
    high_edge_pct: float = 0.20  # 20% edge = "high value"
    extreme_edge_pct: float = 0.35  # 35%+ edge = rare opportunity

    # Kelly position sizing
    kelly_fraction: float = 0.50  # Half Kelly (moderate)
    max_position_pct: float = 0.10  # Max 10% of bankroll per trade ($500 on $5k)
    min_position_usd: float = 1.0  # No minimum - let Kelly size by edge/confidence

    # Bankroll
    bankroll_usd: float = 5000.0

    # Consensus requirements
    min_consensus_models: int = 5  # 5 out of 6 models must agree
    min_consensus_pct: float = 0.80  # Alternative: 80% agreement

    # Price filters (Kalshi uses 0.01-0.99)
    avoid_extreme_prices: bool = True
    price_floor: float = 0.10  # Don't buy below $0.10 (too risky)
    price_ceiling: float = 0.85  # Don't buy above $0.85 (low reward)

    # Time filters
    # NOTE: Time doesn't matter if edge is good - we can exit early when edge captured
    min_days_to_resolution: int = 0  # Allow imminent markets
    max_days_to_resolution: int = 9999  # No max - long-term markets OK with edge
    sweet_spot_min: int = 1  # Ideal window start
    sweet_spot_max: int = 90  # Ideal window end (bonus scoring, not hard filter)

    # Game of 25 opponent modeling
    opponent_loose_pct: float = 0.50  # 50% loose players
    opponent_tight_pct: float = 0.50  # 50% tight players

    # Trading style
    style: TradingStyle = TradingStyle.VALUE


# Style-specific overrides
STYLE_PRESETS = {
    TradingStyle.VALUE: {
        "min_edge_pct": 0.10,  # 10% minimum edge (lowered from 12%)
        "kelly_fraction": 0.20,  # 20% Kelly (very conservative)
        "min_consensus_models": 4,  # 4/6 models (lowered from 5)
        "price_floor": 0.08,
        "price_ceiling": 0.85,  # Allow slightly higher prices
    },
    TradingStyle.BALANCED: {
        "min_edge_pct": 0.08,  # 8% minimum edge
        "kelly_fraction": 0.25,  # 25% Kelly
        "min_consensus_models": 4,
        "price_floor": 0.10,
        "price_ceiling": 0.85,
    },
    TradingStyle.AGGRESSIVE: {
        "min_edge_pct": 0.05,  # 5% minimum edge
        "kelly_fraction": 0.35,  # 35% Kelly (higher variance)
        "min_consensus_models": 4,
        "price_floor": 0.08,
        "price_ceiling": 0.90,
    },
}


# ==============================================================================
# VALUE TRADING DECISION ENGINE
# ==============================================================================

@dataclass
class MarketAnalysis:
    """
    Complete analysis of a market opportunity.

    Contains AI predictions, edge calculations, and trade recommendation.
    """
    # Market identification
    ticker: str
    title: str

    # Pricing
    market_yes_price: float  # Current market YES price
    market_no_price: float  # Current market NO price (1 - yes_price)

    # AI Analysis
    ai_probability: float  # AI-estimated true probability (0-1)
    ai_confidence: float  # AI confidence in prediction (0-1)
    ai_side: str  # "YES" or "NO"
    consensus_count: int  # How many models agree
    total_models: int  # Total models queried

    # Edge calculation
    edge: float = 0.0  # Our edge over market
    expected_value: float = 0.0  # EV of the trade

    # Trade recommendation
    recommended: bool = False
    reason: str = ""
    position_size_usd: float = 0.0
    position_side: str = ""  # "YES" or "NO"
    entry_price: float = 0.0

    # Metadata
    days_to_resolution: Optional[int] = None
    is_imminent: bool = False  # 7-30 days
    is_new_market: bool = False  # < 48 hours old
    volume: int = 0
    open_interest: int = 0


@dataclass
class TradeSignal:
    """
    Final trade signal with execution parameters.
    """
    # Action
    action: str  # "BUY" or "SELL"
    side: str  # "YES" or "NO"
    ticker: str
    title: str

    # Sizing
    contracts: int
    price_cents: int  # Kalshi uses cents (1-99)
    cost_usd: float
    potential_profit_usd: float
    potential_payout_usd: float

    # Metrics
    edge_pct: float
    expected_value: float
    kelly_suggested_pct: float
    actual_position_pct: float

    # Confidence
    consensus_count: int
    total_models: int
    ai_confidence: float
    true_prob: float  # AI's estimated true probability (for exit strategy)

    # Risk
    risk_reward_ratio: float
    max_loss_usd: float
    break_even_prob: float

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    # Time info for trade classification
    days_to_resolution: Optional[int] = None  # Days until market resolves


class ValueTradingEngine:
    """
    Value Trading Strategy Engine optimized for Kalshi.

    Implements edge-based trading with opponent modeling for
    a Game of 25 environment (50% loose / 50% tight players).

    Key Principles:
    1. Only trade when YOU have the edge (not the market)
    2. Size positions based on edge magnitude (Kelly)
    3. Require consensus from AI swarm
    4. Avoid bluffing - let edge come to you
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize Value Trading Engine.

        Args:
            config: StrategyConfig or None for defaults
        """
        self.config = config or StrategyConfig()

        # Apply style presets
        if self.config.style in STYLE_PRESETS:
            preset = STYLE_PRESETS[self.config.style]
            for key, value in preset.items():
                setattr(self.config, key, value)

        # Tracking for opponent modeling
        self.trade_history: List[Dict] = []
        self.win_rate: float = 0.0
        self.total_pnl: float = 0.0

        cprint(f"\nValue Trading Engine initialized ({self.config.style.value} mode)", "green")
        self._print_config()

    def _print_config(self):
        """Print current strategy configuration."""
        cprint("\nStrategy Parameters:", "cyan")
        cprint(f"  Min Edge: {self.config.min_edge_pct*100:.1f}%", "white")
        cprint(f"  Kelly Fraction: {self.config.kelly_fraction*100:.0f}%", "white")
        cprint(f"  Max Position: {self.config.max_position_pct*100:.1f}% of bankroll", "white")
        cprint(f"  Min Consensus: {self.config.min_consensus_models}/{6} models", "white")
        cprint(f"  Price Range: ${self.config.price_floor:.2f} - ${self.config.price_ceiling:.2f}", "white")

    def calculate_edge(
        self,
        ai_probability: float,
        market_price: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Calculate trading edge and expected value.

        Edge = AI Probability - Market Implied Probability
        EV = Edge * Potential Payout - (1-Edge) * Risk

        For prediction markets:
        - If AI says 70% probability, market price is 50 cents
        - Edge = 0.70 - 0.50 = 0.20 (20% edge)
        - This means we estimate 70% chance to win $1 at cost of $0.50

        Args:
            ai_probability: AI-estimated true probability (0-1)
            market_price: Current market price for the side (0-1)
            side: "YES" or "NO"

        Returns:
            Tuple of (edge, expected_value)
        """
        # For YES side: edge = ai_prob - market_price
        # For NO side: edge = (1 - ai_prob) - market_price
        if side.upper() == "YES":
            true_prob = ai_probability
            entry_price = market_price
        else:  # NO
            true_prob = 1.0 - ai_probability
            entry_price = 1.0 - market_price

        # Edge calculation
        edge = true_prob - entry_price

        # Expected value calculation
        # EV = (probability of win * profit) - (probability of loss * loss)
        profit_if_win = 1.0 - entry_price  # Payout is always $1
        loss_if_lose = entry_price

        expected_value = (true_prob * profit_if_win) - ((1 - true_prob) * loss_if_lose)

        return edge, expected_value

    def calculate_kelly_position(
        self,
        edge: float,
        probability: float,
        entry_price: float
    ) -> float:
        """
        Calculate position size using fractional Kelly Criterion.

        Kelly Formula: f* = (bp - q) / b
        Where:
            b = odds received on the bet (payout / risk)
            p = probability of winning
            q = probability of losing (1 - p)

        For binary prediction markets:
            b = (1 - entry_price) / entry_price
            f* = (b * p - q) / b = p - q/b

        Args:
            edge: Calculated edge (our advantage)
            probability: AI-estimated win probability
            entry_price: Cost to enter position

        Returns:
            Position size as fraction of bankroll (0-1)
        """
        if edge <= 0:
            return 0.0

        if entry_price <= 0 or entry_price >= 1:
            return 0.0

        # Calculate odds (b = payout / risk)
        # Payout = $1, Risk = entry_price
        # Net profit if win = (1 - entry_price)
        odds = (1.0 - entry_price) / entry_price

        # Kelly formula
        q = 1.0 - probability
        kelly_fraction = (odds * probability - q) / odds

        # Apply fractional Kelly (risk reduction)
        adjusted_kelly = kelly_fraction * self.config.kelly_fraction

        # Cap at maximum position size
        return min(adjusted_kelly, self.config.max_position_pct)

    def apply_opponent_adjustment(
        self,
        base_edge_threshold: float,
        market_liquidity: int
    ) -> float:
        """
        Adjust edge threshold based on Game of 25 opponent modeling.

        In a mixed pool of 50% loose / 50% tight players:
        - Loose players accept smaller edges (we can be more selective)
        - Tight players only take high-edge spots (we compete with them)

        Strategy: Be TIGHTER than tight players to extract value.

        Args:
            base_edge_threshold: Starting edge requirement
            market_liquidity: Open interest (proxy for competition)

        Returns:
            Adjusted edge threshold
        """
        # High liquidity = more competition from tight players
        # We need higher edge to compensate
        if market_liquidity > 10000:
            competition_factor = 1.15  # 15% higher edge needed
        elif market_liquidity > 5000:
            competition_factor = 1.10  # 10% higher edge needed
        else:
            competition_factor = 1.00  # Low liquidity = less competition

        # In Game of 25 with 50/50 split:
        # - Loose players take marginal bets (edge 3-8%)
        # - Tight players take good bets (edge 8-15%)
        # - VALUE strategy takes great bets (edge 12%+)

        # NOTE: Reduced from 1.25 to 1.0 - the competition factor is enough
        # With 12% base + 1.15 competition, we still need ~14% edge in liquid markets
        # This allows reasonable trades while staying selective
        value_premium = 1.0  # Removed extra premium - competition factor is sufficient

        adjusted_threshold = base_edge_threshold * competition_factor * value_premium

        return adjusted_threshold

    def evaluate_market(
        self,
        ticker: str,
        title: str,
        market_yes_price: float,
        ai_probability: float,
        ai_confidence: float,
        ai_side: str,
        consensus_count: int,
        total_models: int,
        days_to_resolution: Optional[int] = None,
        is_imminent: bool = False,
        is_new_market: bool = False,
        volume: int = 0,
        open_interest: int = 0
    ) -> MarketAnalysis:
        """
        Evaluate a market opportunity for value trading.

        This is the core decision function that determines:
        1. Is there an edge?
        2. Is the edge large enough?
        3. Should we trade?
        4. How much?

        Args:
            ticker: Market ticker
            title: Market title
            market_yes_price: Current YES price (0-1)
            ai_probability: AI-estimated probability of YES
            ai_confidence: AI confidence score (0-1)
            ai_side: AI recommended side ("YES" or "NO")
            consensus_count: Number of models agreeing
            total_models: Total models queried
            days_to_resolution: Days until market resolves
            is_imminent: Is in 7-30 day sweet spot
            is_new_market: Is less than 48 hours old
            volume: Trading volume
            open_interest: Open interest

        Returns:
            MarketAnalysis with complete evaluation
        """
        analysis = MarketAnalysis(
            ticker=ticker,
            title=title,
            market_yes_price=market_yes_price,
            market_no_price=1.0 - market_yes_price,
            ai_probability=ai_probability,
            ai_confidence=ai_confidence,
            ai_side=ai_side.upper(),
            consensus_count=consensus_count,
            total_models=total_models,
            days_to_resolution=days_to_resolution,
            is_imminent=is_imminent,
            is_new_market=is_new_market,
            volume=volume,
            open_interest=open_interest
        )

        # Determine which side to evaluate
        if ai_side.upper() == "YES":
            entry_price = market_yes_price
            side = "YES"
        else:
            entry_price = 1.0 - market_yes_price  # NO price
            side = "NO"

        analysis.position_side = side
        analysis.entry_price = entry_price

        # Calculate edge
        edge, ev = self.calculate_edge(ai_probability, market_yes_price, side)
        analysis.edge = edge
        analysis.expected_value = ev

        # === FILTER GATES ===
        # Each gate must pass for trade recommendation

        # Gate 1: Minimum consensus
        if consensus_count < self.config.min_consensus_models:
            analysis.reason = f"Insufficient consensus: {consensus_count}/{self.config.min_consensus_models}"
            return analysis

        # Gate 2: Price range filter
        if self.config.avoid_extreme_prices:
            if entry_price < self.config.price_floor:
                analysis.reason = f"Price too low: ${entry_price:.2f} < ${self.config.price_floor:.2f}"
                return analysis
            if entry_price > self.config.price_ceiling:
                analysis.reason = f"Price too high: ${entry_price:.2f} > ${self.config.price_ceiling:.2f}"
                return analysis

        # Gate 3: Time filter
        if days_to_resolution is not None:
            if days_to_resolution < self.config.min_days_to_resolution:
                analysis.reason = f"Too soon: {days_to_resolution} days < {self.config.min_days_to_resolution}"
                return analysis
            if days_to_resolution > self.config.max_days_to_resolution:
                analysis.reason = f"Too far: {days_to_resolution} days > {self.config.max_days_to_resolution}"
                return analysis

        # Gate 4: Edge threshold (with opponent adjustment)
        adjusted_edge_threshold = self.apply_opponent_adjustment(
            self.config.min_edge_pct,
            open_interest
        )

        if edge < adjusted_edge_threshold:
            analysis.reason = f"Edge too small: {edge*100:.1f}% < {adjusted_edge_threshold*100:.1f}% required"
            return analysis

        # Gate 5: Positive expected value
        if ev <= 0:
            analysis.reason = f"Negative EV: {ev*100:.1f}%"
            return analysis

        # === PASSED ALL GATES ===
        # Calculate position size
        kelly_position = self.calculate_kelly_position(edge, ai_probability, entry_price)
        position_usd = kelly_position * self.config.bankroll_usd

        # Apply minimum position size
        if position_usd < self.config.min_position_usd:
            position_usd = self.config.min_position_usd

        analysis.position_size_usd = position_usd
        analysis.recommended = True
        analysis.reason = f"VALUE TRADE: {edge*100:.1f}% edge, EV={ev*100:.1f}%"

        return analysis

    def generate_trade_signal(self, analysis: MarketAnalysis) -> Optional[TradeSignal]:
        """
        Generate executable trade signal from market analysis.

        Only called when analysis.recommended is True.

        Args:
            analysis: Completed MarketAnalysis

        Returns:
            TradeSignal ready for execution, or None
        """
        if not analysis.recommended:
            return None

        # Calculate position details
        entry_price = analysis.entry_price
        position_usd = analysis.position_size_usd

        # Number of contracts
        contracts = int(position_usd / entry_price)
        if contracts < 1:
            contracts = 1

        # Adjust cost based on actual contracts
        cost_usd = contracts * entry_price
        payout_usd = contracts * 1.0  # Each contract pays $1
        profit_usd = payout_usd - cost_usd

        # Convert to Kalshi cents (1-99)
        price_cents = int(round(entry_price * 100))
        price_cents = max(1, min(99, price_cents))

        # Calculate Kelly suggestion
        kelly_pct = self.calculate_kelly_position(
            analysis.edge,
            analysis.ai_probability,
            entry_price
        )

        actual_pct = cost_usd / self.config.bankroll_usd

        # Risk metrics
        risk_reward = profit_usd / cost_usd if cost_usd > 0 else 0
        break_even = cost_usd / payout_usd if payout_usd > 0 else 1.0

        return TradeSignal(
            action="BUY",
            side=analysis.position_side,
            ticker=analysis.ticker,
            title=analysis.title,
            contracts=contracts,
            price_cents=price_cents,
            cost_usd=cost_usd,
            potential_profit_usd=profit_usd,
            potential_payout_usd=payout_usd,
            edge_pct=analysis.edge,
            expected_value=analysis.expected_value,
            kelly_suggested_pct=kelly_pct,
            actual_position_pct=actual_pct,
            consensus_count=analysis.consensus_count,
            total_models=analysis.total_models,
            ai_confidence=analysis.ai_confidence,
            true_prob=analysis.ai_probability,  # Store AI's probability for exit strategy
            risk_reward_ratio=risk_reward,
            max_loss_usd=cost_usd,
            break_even_prob=break_even,
            days_to_resolution=analysis.days_to_resolution  # For trade type classification
        )

    def rank_opportunities(
        self,
        analyses: List[MarketAnalysis]
    ) -> List[MarketAnalysis]:
        """
        Rank market opportunities by expected value and edge.

        Scoring factors:
        1. Expected Value (primary)
        2. Edge magnitude
        3. Consensus strength
        4. Time sensitivity bonus (imminent markets)

        Args:
            analyses: List of MarketAnalysis objects

        Returns:
            Sorted list (best opportunities first)
        """
        def score(analysis: MarketAnalysis) -> float:
            if not analysis.recommended:
                return -float('inf')

            # Base score: expected value
            score = analysis.expected_value * 100

            # Edge bonus
            score += analysis.edge * 50

            # Consensus bonus (normalized)
            consensus_pct = analysis.consensus_count / max(analysis.total_models, 1)
            score += consensus_pct * 20

            # Time sensitivity bonuses
            if analysis.is_imminent:
                score += 10  # Sweet spot bonus
            if analysis.is_new_market:
                score += 5  # New market bonus (potential mispricing)

            # Confidence bonus
            score += analysis.ai_confidence * 10

            return score

        return sorted(analyses, key=score, reverse=True)

    def display_trade_signal(self, signal: TradeSignal):
        """Pretty print a trade signal."""
        cprint("\n" + "="*60, "green")
        cprint(f" VALUE TRADE SIGNAL ", "white", "on_green", attrs=['bold'])
        cprint("="*60, "green")

        cprint(f"\n{signal.action} {signal.side} @ ${signal.price_cents/100:.2f}", "cyan", attrs=['bold'])
        cprint(f"Ticker: {signal.ticker}", "white")
        cprint(f"Title: {signal.title[:50]}...", "white")

        cprint(f"\nPosition:", "yellow")
        cprint(f"  Contracts: {signal.contracts}", "white")
        cprint(f"  Cost: ${signal.cost_usd:.2f}", "white")
        cprint(f"  Payout if Win: ${signal.potential_payout_usd:.2f}", "white")
        cprint(f"  Potential Profit: ${signal.potential_profit_usd:.2f} ({signal.potential_profit_usd/signal.cost_usd*100:.0f}% ROI)", "green")

        cprint(f"\nEdge Analysis:", "yellow")
        cprint(f"  Edge: {signal.edge_pct*100:.1f}%", "white")
        cprint(f"  Expected Value: {signal.expected_value*100:.1f}%", "white")
        cprint(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}x", "white")
        cprint(f"  Break-Even Prob: {signal.break_even_prob*100:.1f}%", "white")

        cprint(f"\nConfidence:", "yellow")
        cprint(f"  AI Consensus: {signal.consensus_count}/{signal.total_models} models", "white")
        cprint(f"  AI Confidence: {signal.ai_confidence*100:.0f}%", "white")
        cprint(f"  Kelly Suggested: {signal.kelly_suggested_pct*100:.1f}% of bankroll", "white")
        cprint(f"  Actual Position: {signal.actual_position_pct*100:.2f}% of bankroll", "white")

        cprint("="*60 + "\n", "green")


# ==============================================================================
# STANDALONE TESTING
# ==============================================================================

def test_value_engine():
    """Test the value trading engine with sample data."""
    cprint("\nTesting Value Trading Engine...", "cyan", attrs=['bold'])

    # Initialize with VALUE style
    engine = ValueTradingEngine(StrategyConfig(style=TradingStyle.VALUE))

    # Test scenarios
    test_markets = [
        {
            "ticker": "TRUMP-2024",
            "title": "Will Trump win the 2024 election?",
            "market_yes_price": 0.52,  # Market says 52% chance
            "ai_probability": 0.68,  # AI says 68% chance
            "ai_confidence": 0.85,
            "ai_side": "YES",
            "consensus_count": 5,
            "total_models": 6,
            "days_to_resolution": 15,
            "is_imminent": True,
            "open_interest": 5000
        },
        {
            "ticker": "FED-RATE-JAN",
            "title": "Will Fed cut rates in January?",
            "market_yes_price": 0.35,  # Market says 35% chance
            "ai_probability": 0.25,  # AI says 25% chance (lower)
            "ai_confidence": 0.75,
            "ai_side": "NO",  # AI recommends NO
            "consensus_count": 6,
            "total_models": 6,
            "days_to_resolution": 30,
            "is_imminent": True,
            "open_interest": 8000
        },
        {
            "ticker": "MARGINAL-EDGE",
            "title": "Test market with marginal edge",
            "market_yes_price": 0.50,
            "ai_probability": 0.55,  # Only 5% edge
            "ai_confidence": 0.60,
            "ai_side": "YES",
            "consensus_count": 4,
            "total_models": 6,
            "days_to_resolution": 20,
            "open_interest": 3000
        }
    ]

    analyses = []
    for market in test_markets:
        analysis = engine.evaluate_market(**market)
        analyses.append(analysis)

        cprint(f"\n{'='*50}", "blue")
        cprint(f"Market: {market['ticker']}", "blue", attrs=['bold'])
        cprint(f"  AI Side: {analysis.ai_side}", "white")
        cprint(f"  Edge: {analysis.edge*100:.1f}%", "white")
        cprint(f"  EV: {analysis.expected_value*100:.1f}%", "white")
        cprint(f"  Recommended: {analysis.recommended}", "green" if analysis.recommended else "red")
        cprint(f"  Reason: {analysis.reason}", "white")

        if analysis.recommended:
            signal = engine.generate_trade_signal(analysis)
            if signal:
                engine.display_trade_signal(signal)

    # Rank opportunities
    cprint("\n" + "="*50, "cyan")
    cprint("Ranked Opportunities:", "cyan", attrs=['bold'])
    ranked = engine.rank_opportunities(analyses)

    for i, analysis in enumerate(ranked[:5], 1):
        status = "+" if analysis.recommended else "-"
        edge_str = f"{analysis.edge*100:.1f}%" if analysis.edge > 0 else "N/A"
        cprint(f"  {i}. [{status}] {analysis.ticker}: Edge={edge_str}", "white")


if __name__ == "__main__":
    test_value_engine()
