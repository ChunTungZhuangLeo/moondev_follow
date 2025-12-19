"""
Kalshi Trade Execution Module
=============================
Executes trades on Kalshi using authenticated API with safety controls.

Features:
- Pre-trade validation and risk checks
- Paper trading mode for testing
- Live execution with confirmation
- Position tracking and P&L calculation
- Trade logging to CSV

Author: Moon Dev
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from termcolor import cprint

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.agents.kalshi_auth_client import (
    KalshiAuthClient,
    KalshiConfig,
    KalshiEnvironment,
    cents_to_dollars,
    dollars_to_cents,
    calculate_payout
)
from src.agents.value_trading_strategy import (
    TradeSignal,
    MarketAnalysis,
    StrategyConfig,
    TradingStyle
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class ExecutionMode(Enum):
    """
    Trade execution mode selection.

    PAPER: Simulated trades (no real money)
    LIVE: Real trades on Kalshi (requires authentication)
    DRY_RUN: Validates but doesn't execute
    """
    PAPER = "paper"
    LIVE = "live"
    DRY_RUN = "dry_run"


@dataclass
class ExecutorConfig:
    """
    Configuration for trade executor.

    Attributes:
        mode: Execution mode (PAPER, LIVE, DRY_RUN)
        confirm_trades: Require confirmation before live trades
        max_daily_trades: Maximum trades per day
        max_daily_loss_usd: Stop trading if daily loss exceeds this
        min_balance_usd: Minimum balance to maintain
        paper_starting_balance: Starting balance for paper trading
    """
    mode: ExecutionMode = ExecutionMode.PAPER
    confirm_trades: bool = True  # Require confirmation for live trades
    max_daily_trades: int = 20  # Max 20 trades per day
    max_daily_loss_usd: float = 500.0  # Stop if daily loss > $500
    min_balance_usd: float = 100.0  # Keep at least $100 in account
    paper_starting_balance: float = 5000.0  # Paper trading start

    # Data paths
    trades_csv_path: str = ""
    positions_csv_path: str = ""
    portfolio_csv_path: str = ""


@dataclass
class ExecutedTrade:
    """
    Record of an executed trade.
    """
    trade_id: str
    timestamp: datetime
    mode: str  # "paper" or "live"
    ticker: str
    title: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    contracts: int
    price_cents: int
    cost_usd: float
    status: str  # "open", "won", "lost", "cancelled"
    order_id: Optional[str] = None  # Kalshi order ID for live trades
    exit_price_cents: Optional[int] = None
    exit_timestamp: Optional[datetime] = None
    pnl_usd: Optional[float] = None
    edge_pct: float = 0.0
    true_prob: float = 0.5  # AI's estimated true probability for exit strategy
    consensus_count: int = 0
    notes: str = ""


@dataclass
class PortfolioState:
    """
    Current portfolio state.
    """
    balance_usd: float
    invested_usd: float
    open_positions: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    daily_trades: int
    win_rate: float = 0.0


# ==============================================================================
# TRADE EXECUTOR
# ==============================================================================

class KalshiTradeExecutor:
    """
    Trade execution engine for Kalshi with safety controls.

    Supports paper trading, live trading, and dry-run modes.
    Implements pre-trade validation and risk management.
    """

    def __init__(
        self,
        config: ExecutorConfig,
        auth_client: Optional[KalshiAuthClient] = None
    ):
        """
        Initialize trade executor.

        Args:
            config: ExecutorConfig with mode and risk settings
            auth_client: Optional authenticated client for live trading
        """
        self.config = config
        self.auth_client = auth_client
        self.mode = config.mode

        # Set up data paths
        data_folder = os.path.join(project_root, "src/data/kalshi_value")
        os.makedirs(data_folder, exist_ok=True)

        self.trades_csv = config.trades_csv_path or os.path.join(data_folder, "trades.csv")
        self.positions_csv = config.positions_csv_path or os.path.join(data_folder, "positions.csv")
        self.portfolio_csv = config.portfolio_csv_path or os.path.join(data_folder, "portfolio.csv")

        # Initialize state
        self._load_state()

        cprint(f"\nKalshi Trade Executor initialized ({self.mode.value} mode)", "green")
        self._print_portfolio_summary()

    def _load_state(self):
        """Load previous state from CSV files."""
        # Load trades
        if os.path.exists(self.trades_csv):
            self.trades_df = pd.read_csv(self.trades_csv)
        else:
            self.trades_df = pd.DataFrame(columns=[
                'trade_id', 'timestamp', 'mode', 'ticker', 'title',
                'side', 'action', 'contracts', 'price_cents', 'cost_usd',
                'status', 'order_id', 'exit_price_cents', 'exit_timestamp',
                'pnl_usd', 'edge_pct', 'true_prob', 'consensus_count', 'notes',
                'last_check', 'current_price_cents', 'unrealized_pnl'
            ])

        # Calculate portfolio state
        self._calculate_portfolio_state()

    def _calculate_portfolio_state(self):
        """Calculate current portfolio state from trades."""
        if self.trades_df.empty:
            self.portfolio = PortfolioState(
                balance_usd=self.config.paper_starting_balance,
                invested_usd=0.0,
                open_positions=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                daily_trades=0
            )
            return

        # Filter by mode
        mode_trades = self.trades_df[self.trades_df['mode'] == self.mode.value]

        # Open positions
        open_trades = mode_trades[mode_trades['status'] == 'open']
        closed_trades = mode_trades[mode_trades['status'].isin(['won', 'lost'])]

        # Calculate metrics
        invested = open_trades['cost_usd'].sum() if len(open_trades) > 0 else 0
        realized_pnl = closed_trades['pnl_usd'].sum() if len(closed_trades) > 0 else 0
        winning = len(closed_trades[closed_trades['pnl_usd'] > 0])
        losing = len(closed_trades[closed_trades['pnl_usd'] < 0])

        # Today's trades
        today = datetime.now().date()
        if 'timestamp' in mode_trades.columns:
            mode_trades_copy = mode_trades.copy()
            mode_trades_copy['date'] = pd.to_datetime(mode_trades_copy['timestamp']).dt.date
            today_trades = mode_trades_copy[mode_trades_copy['date'] == today]
            daily_trades = len(today_trades)
            daily_pnl = today_trades['pnl_usd'].sum() if 'pnl_usd' in today_trades else 0
        else:
            daily_trades = 0
            daily_pnl = 0

        # Balance calculation
        if self.mode == ExecutionMode.PAPER:
            balance = self.config.paper_starting_balance + realized_pnl - invested
        else:
            # For live mode, fetch from API
            if self.auth_client:
                try:
                    balance_response = self.auth_client.get_balance()
                    balance = balance_response.get('balance', 0) / 100.0  # cents to dollars
                except Exception:
                    balance = 0.0
            else:
                balance = 0.0

        win_rate = winning / (winning + losing) if (winning + losing) > 0 else 0.0

        self.portfolio = PortfolioState(
            balance_usd=balance,
            invested_usd=invested,
            open_positions=len(open_trades),
            total_trades=len(mode_trades),
            winning_trades=winning,
            losing_trades=losing,
            realized_pnl=realized_pnl,
            unrealized_pnl=0.0,  # Would need current prices
            daily_pnl=daily_pnl,
            daily_trades=daily_trades,
            win_rate=win_rate
        )

    def _print_portfolio_summary(self):
        """Print portfolio summary."""
        p = self.portfolio
        cprint(f"\nPortfolio Summary ({self.mode.value}):", "cyan")
        cprint(f"  Balance: ${p.balance_usd:.2f}", "white")
        cprint(f"  Invested: ${p.invested_usd:.2f}", "white")
        cprint(f"  Open Positions: {p.open_positions}", "white")
        cprint(f"  Realized P&L: ${p.realized_pnl:.2f}", "green" if p.realized_pnl >= 0 else "red")
        cprint(f"  Win Rate: {p.win_rate*100:.1f}% ({p.winning_trades}W/{p.losing_trades}L)", "white")
        cprint(f"  Today: {p.daily_trades} trades, ${p.daily_pnl:.2f} P&L", "white")

    def validate_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Validate trade signal before execution.

        Checks:
        1. Sufficient balance
        2. Daily trade limit not exceeded
        3. Daily loss limit not exceeded
        4. Minimum balance maintained
        5. Market still tradeable

        Args:
            signal: TradeSignal to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check 1: Daily trade limit
        if self.portfolio.daily_trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        # Check 2: Daily loss limit
        if self.portfolio.daily_pnl <= -self.config.max_daily_loss_usd:
            return False, f"Daily loss limit reached (${self.config.max_daily_loss_usd})"

        # Check 3: Sufficient balance
        required = signal.cost_usd
        available = self.portfolio.balance_usd - self.config.min_balance_usd

        if required > available:
            return False, f"Insufficient balance: need ${required:.2f}, have ${available:.2f}"

        # Check 4: Total open positions limit
        total_open = len(self.trades_df[self.trades_df['status'] == 'open'])
        if total_open >= 15:  # Max 15 total open positions
            return False, f"Max total positions reached ({total_open})"

        # Check 5: Position limit for this ticker (allow up to 2 entries per market)
        open_for_ticker = self.trades_df[
            (self.trades_df['ticker'] == signal.ticker) &
            (self.trades_df['status'] == 'open')
        ]
        if len(open_for_ticker) >= 2:
            return False, f"Max positions reached for {signal.ticker} (have {len(open_for_ticker)})"

        return True, "Validated"

    def execute_signal(
        self,
        signal: TradeSignal,
        confirm: bool = True
    ) -> Optional[ExecutedTrade]:
        """
        Execute a trade signal.

        Args:
            signal: TradeSignal to execute
            confirm: Require user confirmation (for live mode)

        Returns:
            ExecutedTrade record or None if failed
        """
        # Validate first
        is_valid, reason = self.validate_trade(signal)
        if not is_valid:
            cprint(f"Trade rejected: {reason}", "red")
            return None

        # Display signal
        cprint("\n" + "="*50, "yellow")
        cprint(f"TRADE SIGNAL: {signal.action} {signal.side}", "yellow", attrs=['bold'])
        cprint(f"Ticker: {signal.ticker}", "white")
        cprint(f"Price: ${signal.price_cents/100:.2f}", "white")
        cprint(f"Contracts: {signal.contracts}", "white")
        cprint(f"Cost: ${signal.cost_usd:.2f}", "white")
        cprint(f"Edge: {signal.edge_pct*100:.1f}%", "white")
        cprint("="*50, "yellow")

        # Confirm for live trades
        if self.mode == ExecutionMode.LIVE and confirm and self.config.confirm_trades:
            response = input("Execute live trade? (yes/no): ").strip().lower()
            if response != "yes":
                cprint("Trade cancelled by user", "yellow")
                return None

        # Execute based on mode
        if self.mode == ExecutionMode.DRY_RUN:
            cprint("DRY RUN - Trade validated but not executed", "cyan")
            return None

        elif self.mode == ExecutionMode.PAPER:
            return self._execute_paper_trade(signal)

        elif self.mode == ExecutionMode.LIVE:
            return self._execute_live_trade(signal)

        return None

    def _execute_paper_trade(self, signal: TradeSignal) -> ExecutedTrade:
        """Execute a paper trade (simulated)."""
        trade_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        trade = ExecutedTrade(
            trade_id=trade_id,
            timestamp=datetime.now(),
            mode="paper",
            ticker=signal.ticker,
            title=signal.title,
            side=signal.side.lower(),
            action=signal.action.lower(),
            contracts=signal.contracts,
            price_cents=signal.price_cents,
            cost_usd=signal.cost_usd,
            status="open",
            order_id=f"PAPER-{trade_id}",
            edge_pct=signal.edge_pct,
            true_prob=signal.true_prob,  # AI's probability estimate for exit strategy
            consensus_count=signal.consensus_count,
            notes=f"Paper trade - EV={signal.expected_value*100:.1f}%"
        )

        self._save_trade(trade)
        self._calculate_portfolio_state()

        cprint(f"\nPAPER TRADE EXECUTED", "green", attrs=['bold'])
        cprint(f"  Trade ID: {trade_id}", "white")
        cprint(f"  Cost: ${signal.cost_usd:.2f}", "white")
        cprint(f"  New Balance: ${self.portfolio.balance_usd:.2f}", "white")

        return trade

    def _execute_live_trade(self, signal: TradeSignal) -> Optional[ExecutedTrade]:
        """Execute a live trade on Kalshi."""
        if not self.auth_client:
            cprint("No authenticated client - cannot execute live trade", "red")
            return None

        trade_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        try:
            # Place order on Kalshi
            order_response = self.auth_client.place_order(
                ticker=signal.ticker,
                side=signal.side.lower(),
                action=signal.action.lower(),
                count=signal.contracts,
                type="limit",
                yes_price=signal.price_cents if signal.side.upper() == "YES" else None,
                no_price=signal.price_cents if signal.side.upper() == "NO" else None,
                client_order_id=trade_id
            )

            order_id = order_response.get("order", {}).get("order_id", "UNKNOWN")

            trade = ExecutedTrade(
                trade_id=trade_id,
                timestamp=datetime.now(),
                mode="live",
                ticker=signal.ticker,
                title=signal.title,
                side=signal.side.lower(),
                action=signal.action.lower(),
                contracts=signal.contracts,
                price_cents=signal.price_cents,
                cost_usd=signal.cost_usd,
                status="open",
                order_id=order_id,
                edge_pct=signal.edge_pct,
                true_prob=signal.true_prob,  # AI's probability estimate for exit strategy
                consensus_count=signal.consensus_count,
                notes=f"Live trade - Order ID: {order_id}"
            )

            self._save_trade(trade)
            self._calculate_portfolio_state()

            cprint(f"\nLIVE TRADE EXECUTED", "green", attrs=['bold'])
            cprint(f"  Order ID: {order_id}", "white")
            cprint(f"  Cost: ${signal.cost_usd:.2f}", "white")

            return trade

        except Exception as e:
            cprint(f"Live trade failed: {e}", "red")
            return None

    def _save_trade(self, trade: ExecutedTrade):
        """Save trade to CSV."""
        trade_dict = {
            'trade_id': trade.trade_id,
            'timestamp': trade.timestamp.isoformat(),
            'mode': trade.mode,
            'ticker': trade.ticker,
            'title': trade.title,
            'side': trade.side,
            'action': trade.action,
            'contracts': trade.contracts,
            'price_cents': trade.price_cents,
            'cost_usd': trade.cost_usd,
            'status': trade.status,
            'order_id': trade.order_id,
            'exit_price_cents': trade.exit_price_cents,
            'exit_timestamp': trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
            'pnl_usd': trade.pnl_usd,
            'edge_pct': trade.edge_pct,
            'true_prob': trade.true_prob,  # AI's probability for exit strategy
            'consensus_count': trade.consensus_count,
            'notes': trade.notes
        }

        new_row = pd.DataFrame([trade_dict])
        self.trades_df = pd.concat([self.trades_df, new_row], ignore_index=True)
        self.trades_df.to_csv(self.trades_csv, index=False)

    def close_position(
        self,
        trade_id: str,
        exit_price_cents: int,
        won: bool
    ) -> bool:
        """
        Close an open position and calculate P&L.

        Args:
            trade_id: Trade ID to close
            exit_price_cents: Exit price in cents
            won: Whether the position won (True) or lost (False)

        Returns:
            True if closed successfully
        """
        # Find the trade
        mask = self.trades_df['trade_id'] == trade_id

        if not mask.any():
            cprint(f"Trade {trade_id} not found", "red")
            return False

        trade_idx = mask.idxmax()
        trade = self.trades_df.loc[trade_idx]

        if trade['status'] != 'open':
            cprint(f"Trade {trade_id} is not open (status: {trade['status']})", "yellow")
            return False

        # Calculate P&L
        entry_price = trade['price_cents'] / 100.0
        contracts = trade['contracts']
        cost = trade['cost_usd']

        if won:
            payout = contracts * 1.0  # $1 per winning contract
            pnl = payout - cost
            status = 'won'
        else:
            pnl = -cost  # Lose entire cost
            status = 'lost'

        # Update trade record
        self.trades_df.loc[trade_idx, 'status'] = status
        self.trades_df.loc[trade_idx, 'exit_price_cents'] = exit_price_cents
        self.trades_df.loc[trade_idx, 'exit_timestamp'] = datetime.now().isoformat()
        self.trades_df.loc[trade_idx, 'pnl_usd'] = pnl

        self.trades_df.to_csv(self.trades_csv, index=False)
        self._calculate_portfolio_state()

        status_color = "green" if won else "red"
        cprint(f"\nPosition Closed: {trade['ticker']}", status_color, attrs=['bold'])
        cprint(f"  Result: {'WON' if won else 'LOST'}", status_color)
        cprint(f"  P&L: ${pnl:.2f}", status_color)

        return True

    def update_position_price(self, trade_id: str, current_price_cents: int, unrealized_pnl: float):
        """
        Update the current price and unrealized P&L for an open position.

        Called during each monitoring run to track price changes in trades.csv.

        Args:
            trade_id: The trade ID to update
            current_price_cents: Current cash-out price in cents
            unrealized_pnl: Current unrealized P&L in USD
        """
        trade_idx = self.trades_df[self.trades_df['trade_id'] == trade_id].index
        if len(trade_idx) == 0:
            return

        self.trades_df.loc[trade_idx, 'last_check'] = datetime.now().isoformat()
        self.trades_df.loc[trade_idx, 'current_price_cents'] = current_price_cents
        self.trades_df.loc[trade_idx, 'unrealized_pnl'] = round(unrealized_pnl, 2)

        self.trades_df.to_csv(self.trades_csv, index=False)

    def get_open_positions(self) -> pd.DataFrame:
        """Get all open positions."""
        return self.trades_df[
            (self.trades_df['mode'] == self.mode.value) &
            (self.trades_df['status'] == 'open')
        ]

    def get_trade_history(self, limit: int = 50) -> pd.DataFrame:
        """Get recent trade history."""
        mode_trades = self.trades_df[self.trades_df['mode'] == self.mode.value]
        return mode_trades.tail(limit)

    def display_open_positions(self):
        """Display all open positions."""
        open_pos = self.get_open_positions()

        if open_pos.empty:
            cprint("\nNo open positions", "yellow")
            return

        cprint(f"\n{'='*60}", "cyan")
        cprint(f"OPEN POSITIONS ({len(open_pos)})", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan")

        for _, pos in open_pos.iterrows():
            cprint(f"\n  {pos['ticker']}: {pos['side'].upper()} @ ${pos['price_cents']/100:.2f}", "white")
            cprint(f"    Contracts: {pos['contracts']} | Cost: ${pos['cost_usd']:.2f}", "white")
            cprint(f"    Edge: {pos['edge_pct']*100:.1f}% | Consensus: {pos['consensus_count']}/6", "white")

    def display_performance_summary(self):
        """Display performance summary."""
        closed = self.trades_df[
            (self.trades_df['mode'] == self.mode.value) &
            (self.trades_df['status'].isin(['won', 'lost']))
        ]

        if closed.empty:
            cprint("\nNo closed trades yet", "yellow")
            return

        total_pnl = closed['pnl_usd'].sum()
        wins = len(closed[closed['pnl_usd'] > 0])
        losses = len(closed[closed['pnl_usd'] <= 0])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        avg_win = closed[closed['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
        avg_loss = closed[closed['pnl_usd'] <= 0]['pnl_usd'].mean() if losses > 0 else 0

        cprint(f"\n{'='*60}", "magenta")
        cprint("PERFORMANCE SUMMARY", "magenta", attrs=['bold'])
        cprint(f"{'='*60}", "magenta")

        cprint(f"\n  Total Trades: {len(closed)}", "white")
        cprint(f"  Win Rate: {win_rate*100:.1f}% ({wins}W / {losses}L)", "white")
        cprint(f"  Total P&L: ${total_pnl:.2f}", "green" if total_pnl >= 0 else "red")
        cprint(f"  Avg Win: ${avg_win:.2f}", "green")
        cprint(f"  Avg Loss: ${avg_loss:.2f}", "red")

        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
            cprint(f"  Risk/Reward: {risk_reward:.2f}x", "white")

        cprint(f"\n  Current Balance: ${self.portfolio.balance_usd:.2f}", "cyan")


# ==============================================================================
# STANDALONE TESTING
# ==============================================================================

def test_executor():
    """Test the trade executor in paper mode."""
    cprint("\nTesting Kalshi Trade Executor...", "cyan", attrs=['bold'])

    # Initialize in paper mode
    config = ExecutorConfig(
        mode=ExecutionMode.PAPER,
        paper_starting_balance=5000.0,
        max_daily_trades=10
    )

    executor = KalshiTradeExecutor(config)

    # Create test signal
    test_signal = TradeSignal(
        action="BUY",
        side="YES",
        ticker="TEST-MARKET",
        title="Test Market for Executor",
        contracts=10,
        price_cents=45,
        cost_usd=4.50,
        potential_profit_usd=5.50,
        potential_payout_usd=10.00,
        edge_pct=0.15,
        expected_value=0.12,
        kelly_suggested_pct=0.02,
        actual_position_pct=0.001,
        consensus_count=5,
        total_models=6,
        ai_confidence=0.85,
        risk_reward_ratio=1.22,
        max_loss_usd=4.50,
        break_even_prob=0.45
    )

    # Execute paper trade
    trade = executor.execute_signal(test_signal, confirm=False)

    if trade:
        # Display open positions
        executor.display_open_positions()

        # Simulate win
        cprint("\nSimulating position win...", "yellow")
        executor.close_position(trade.trade_id, 100, won=True)

        # Display performance
        executor.display_performance_summary()


if __name__ == "__main__":
    test_executor()
