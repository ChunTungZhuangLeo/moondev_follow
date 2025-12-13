#!/usr/bin/env python3
"""
🌙 Moon Dev's Trading Dashboard Visualizer
Analyzes paper trading CSVs and generates visualizations

Usage:
    python src/scripts/visualize_trading.py
    python src/scripts/visualize_trading.py --v1  # Only V1
    python src/scripts/visualize_trading.py --v2  # Only V2
    python src/scripts/visualize_trading.py --compare  # Side by side
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from termcolor import cprint

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_DATA_DIR = os.path.join(BASE_DIR, "data", "polymarket_websearch")
V2_DATA_DIR = os.path.join(BASE_DIR, "data", "polymarket_websearch_v2")

# CSV filenames
PORTFOLIO_CSV = "paper_portfolio.csv"
TRADES_CSV = "paper_trades.csv"
CONSENSUS_CSV = "consensus_picks.csv"
PREDICTIONS_CSV = "predictions.csv"
MARKETS_CSV = "markets.csv"


def load_csv_safe(filepath):
    """Safely load CSV, return empty DataFrame if not found"""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            cprint(f"⚠️ Error loading {filepath}: {e}", "yellow")
    return pd.DataFrame()


def print_summary(version, data_dir):
    """Print text summary of trading data"""
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"📊 {version} Trading Summary", "cyan", attrs=['bold'])
    cprint(f"{'='*60}", "cyan")

    # Portfolio
    portfolio_df = load_csv_safe(os.path.join(data_dir, PORTFOLIO_CSV))
    if not portfolio_df.empty:
        cprint("\n💼 Portfolio Status:", "yellow", attrs=['bold'])
        if 'balance' in portfolio_df.columns:
            latest_balance = portfolio_df['balance'].iloc[-1] if len(portfolio_df) > 0 else 10000
            cprint(f"   Balance: ${latest_balance:,.2f}", "white")
        if 'total_value' in portfolio_df.columns:
            total_value = portfolio_df['total_value'].iloc[-1] if len(portfolio_df) > 0 else 10000
            cprint(f"   Total Value: ${total_value:,.2f}", "white")
        if 'realized_pnl' in portfolio_df.columns:
            realized_pnl = portfolio_df['realized_pnl'].iloc[-1] if len(portfolio_df) > 0 else 0
            color = "green" if realized_pnl >= 0 else "red"
            cprint(f"   Realized P&L: ${realized_pnl:,.2f}", color)
        if 'unrealized_pnl' in portfolio_df.columns:
            unrealized_pnl = portfolio_df['unrealized_pnl'].iloc[-1] if len(portfolio_df) > 0 else 0
            color = "green" if unrealized_pnl >= 0 else "red"
            cprint(f"   Unrealized P&L: ${unrealized_pnl:,.2f}", color)
    else:
        cprint("\n💼 No portfolio data found", "yellow")

    # Trades
    trades_df = load_csv_safe(os.path.join(data_dir, TRADES_CSV))
    if not trades_df.empty:
        cprint(f"\n📈 Trade History ({len(trades_df)} trades):", "yellow", attrs=['bold'])

        # Count buys and sells
        if 'action' in trades_df.columns:
            buys = len(trades_df[trades_df['action'] == 'BUY'])
            sells = len(trades_df[trades_df['action'] == 'SELL'])
            cprint(f"   Buys: {buys} | Sells: {sells}", "white")

        # Calculate win rate if we have closed trades
        if 'pnl' in trades_df.columns:
            closed_trades = trades_df[trades_df['pnl'].notna()]
            if len(closed_trades) > 0:
                wins = len(closed_trades[closed_trades['pnl'] > 0])
                losses = len(closed_trades[closed_trades['pnl'] < 0])
                win_rate = wins / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
                total_pnl = closed_trades['pnl'].sum()
                cprint(f"   Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)", "green" if win_rate >= 50 else "red")
                cprint(f"   Total P&L from closed: ${total_pnl:,.2f}", "green" if total_pnl >= 0 else "red")

        # Show recent trades
        cprint("\n   Recent Trades:", "white")
        recent = trades_df.tail(5)
        for _, trade in recent.iterrows():
            market = trade.get('market_title', trade.get('market', 'Unknown'))[:40]
            action = trade.get('action', 'N/A')
            side = trade.get('side', '')
            price = trade.get('price', trade.get('entry_price', 0))
            amount = trade.get('amount_usd', trade.get('amount', 0))
            cprint(f"   • {action} {side} @ ${price:.2f} (${amount:.0f}) - {market}...", "white")
    else:
        cprint("\n📈 No trade history found", "yellow")

    # Consensus Picks
    consensus_df = load_csv_safe(os.path.join(data_dir, CONSENSUS_CSV))
    if not consensus_df.empty:
        cprint(f"\n🤖 AI Consensus Picks ({len(consensus_df)} total):", "yellow", attrs=['bold'])

        # Count by side
        if 'side' in consensus_df.columns:
            yes_picks = len(consensus_df[consensus_df['side'].str.upper() == 'YES'])
            no_picks = len(consensus_df[consensus_df['side'].str.upper() == 'NO'])
            cprint(f"   YES picks: {yes_picks} | NO picks: {no_picks}", "white")

        # Show strongest consensus
        if 'consensus_count' in consensus_df.columns:
            strong = consensus_df[consensus_df['consensus_count'] >= 6]
            cprint(f"   Strong consensus (6+ models): {len(strong)}", "green")

        # Recent picks
        cprint("\n   Recent Consensus Picks:", "white")
        recent = consensus_df.tail(5)
        for _, pick in recent.iterrows():
            market = pick.get('market_title', 'Unknown')[:35]
            side = pick.get('side', 'N/A')
            consensus = pick.get('consensus', pick.get('consensus_count', 'N/A'))
            cprint(f"   • {side}: {market}... ({consensus})", "white")
    else:
        cprint("\n🤖 No consensus picks found", "yellow")

    # Predictions breakdown by model
    predictions_df = load_csv_safe(os.path.join(data_dir, PREDICTIONS_CSV))
    if not predictions_df.empty:
        cprint(f"\n🧠 Model Predictions ({len(predictions_df)} analyses):", "yellow", attrs=['bold'])

        model_cols = ['claude_prediction', 'opus_prediction', 'openai_prediction',
                      'groq_prediction', 'gemini_prediction', 'deepseek_prediction', 'xai_prediction']

        for col in model_cols:
            if col in predictions_df.columns:
                model_name = col.replace('_prediction', '').title()
                valid = predictions_df[predictions_df[col].notna() & (predictions_df[col] != 'N/A')]
                if len(valid) > 0:
                    yes_count = len(valid[valid[col].str.upper() == 'YES'])
                    no_count = len(valid[valid[col].str.upper() == 'NO'])
                    no_trade = len(valid[valid[col].str.upper() == 'NO_TRADE'])
                    cprint(f"   {model_name:12}: YES={yes_count:3} NO={no_count:3} SKIP={no_trade:3}", "white")

    # Markets tracked
    markets_df = load_csv_safe(os.path.join(data_dir, MARKETS_CSV))
    if not markets_df.empty:
        cprint(f"\n📊 Markets Tracked: {len(markets_df)}", "yellow", attrs=['bold'])

        if 'size_usd' in markets_df.columns:
            total_volume = markets_df['size_usd'].sum()
            cprint(f"   Total volume tracked: ${total_volume:,.0f}", "white")


def plot_portfolio_equity(v1_dir, v2_dir, show_v1=True, show_v2=True):
    """Plot equity curves for V1 and/or V2"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🌙 Moon Dev Trading Dashboard', fontsize=14, fontweight='bold')

    # Load data
    v1_portfolio = load_csv_safe(os.path.join(v1_dir, PORTFOLIO_CSV)) if show_v1 else pd.DataFrame()
    v2_portfolio = load_csv_safe(os.path.join(v2_dir, PORTFOLIO_CSV)) if show_v2 else pd.DataFrame()
    v1_trades = load_csv_safe(os.path.join(v1_dir, TRADES_CSV)) if show_v1 else pd.DataFrame()
    v2_trades = load_csv_safe(os.path.join(v2_dir, TRADES_CSV)) if show_v2 else pd.DataFrame()
    v1_consensus = load_csv_safe(os.path.join(v1_dir, CONSENSUS_CSV)) if show_v1 else pd.DataFrame()
    v2_consensus = load_csv_safe(os.path.join(v2_dir, CONSENSUS_CSV)) if show_v2 else pd.DataFrame()

    # Plot 1: Equity Curve
    ax1 = axes[0, 0]
    ax1.set_title('Portfolio Equity Curve')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value ($)')

    if not v1_portfolio.empty and 'timestamp' in v1_portfolio.columns:
        v1_portfolio['timestamp'] = pd.to_datetime(v1_portfolio['timestamp'])
        value_col = 'total_value' if 'total_value' in v1_portfolio.columns else 'balance'
        if value_col in v1_portfolio.columns:
            ax1.plot(v1_portfolio['timestamp'], v1_portfolio[value_col], 'b-', label='V1 (Consensus)', linewidth=2)

    if not v2_portfolio.empty and 'timestamp' in v2_portfolio.columns:
        v2_portfolio['timestamp'] = pd.to_datetime(v2_portfolio['timestamp'])
        value_col = 'total_value' if 'total_value' in v2_portfolio.columns else 'balance'
        if value_col in v2_portfolio.columns:
            ax1.plot(v2_portfolio['timestamp'], v2_portfolio[value_col], 'g-', label='V2 (Edge-based)', linewidth=2)

    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: P&L Distribution
    ax2 = axes[0, 1]
    ax2.set_title('Trade P&L Distribution')

    pnl_data = []
    labels = []

    if not v1_trades.empty and 'pnl' in v1_trades.columns:
        v1_pnl = v1_trades['pnl'].dropna()
        if len(v1_pnl) > 0:
            pnl_data.append(v1_pnl.values)
            labels.append('V1')

    if not v2_trades.empty and 'pnl' in v2_trades.columns:
        v2_pnl = v2_trades['pnl'].dropna()
        if len(v2_pnl) > 0:
            pnl_data.append(v2_pnl.values)
            labels.append('V2')

    if pnl_data:
        colors = ['blue', 'green'][:len(pnl_data)]
        ax2.hist(pnl_data, bins=20, label=labels, color=colors, alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('P&L ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No closed trades yet', ha='center', va='center', transform=ax2.transAxes)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Trade Activity Over Time
    ax3 = axes[1, 0]
    ax3.set_title('Trade Activity')

    if not v1_trades.empty and 'timestamp' in v1_trades.columns:
        v1_trades['timestamp'] = pd.to_datetime(v1_trades['timestamp'])
        v1_trades['date'] = v1_trades['timestamp'].dt.date
        v1_daily = v1_trades.groupby('date').size()
        ax3.bar(v1_daily.index, v1_daily.values, alpha=0.7, label='V1', color='blue', width=0.4)

    if not v2_trades.empty and 'timestamp' in v2_trades.columns:
        v2_trades['timestamp'] = pd.to_datetime(v2_trades['timestamp'])
        v2_trades['date'] = v2_trades['timestamp'].dt.date
        v2_daily = v2_trades.groupby('date').size()
        ax3.bar(v2_daily.index, v2_daily.values, alpha=0.7, label='V2', color='green', width=0.4)

    ax3.set_xlabel('Date')
    ax3.set_ylabel('Number of Trades')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Consensus Strength
    ax4 = axes[1, 1]
    ax4.set_title('AI Consensus Strength Distribution')

    consensus_data = []
    labels = []

    if not v1_consensus.empty and 'consensus_count' in v1_consensus.columns:
        consensus_data.append(v1_consensus['consensus_count'].dropna().values)
        labels.append('V1')

    if not v2_consensus.empty and 'consensus_count' in v2_consensus.columns:
        consensus_data.append(v2_consensus['consensus_count'].dropna().values)
        labels.append('V2')

    if consensus_data:
        colors = ['blue', 'green'][:len(consensus_data)]
        ax4.hist(consensus_data, bins=range(1, 9), label=labels, color=colors, alpha=0.7, align='left')
        ax4.set_xlabel('Models Agreeing')
        ax4.set_ylabel('Frequency')
        ax4.set_xticks(range(1, 8))
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No consensus data yet', ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(BASE_DIR, "data", "trading_dashboard.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    cprint(f"\n📊 Dashboard saved to: {output_path}", "green", attrs=['bold'])

    # Show interactive plot
    plt.show()


def plot_comparison_metrics(v1_dir, v2_dir):
    """Create comparison bar charts between V1 and V2"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('🌙 V1 vs V2 Strategy Comparison', fontsize=14, fontweight='bold')

    v1_trades = load_csv_safe(os.path.join(v1_dir, TRADES_CSV))
    v2_trades = load_csv_safe(os.path.join(v2_dir, TRADES_CSV))
    v1_portfolio = load_csv_safe(os.path.join(v1_dir, PORTFOLIO_CSV))
    v2_portfolio = load_csv_safe(os.path.join(v2_dir, PORTFOLIO_CSV))

    # Metrics to compare
    metrics = {
        'Total Trades': [
            len(v1_trades) if not v1_trades.empty else 0,
            len(v2_trades) if not v2_trades.empty else 0
        ],
        'Win Rate (%)': [0, 0],
        'Total P&L ($)': [0, 0]
    }

    # Calculate win rates
    if not v1_trades.empty and 'pnl' in v1_trades.columns:
        closed = v1_trades[v1_trades['pnl'].notna()]
        if len(closed) > 0:
            metrics['Win Rate (%)'][0] = len(closed[closed['pnl'] > 0]) / len(closed) * 100
            metrics['Total P&L ($)'][0] = closed['pnl'].sum()

    if not v2_trades.empty and 'pnl' in v2_trades.columns:
        closed = v2_trades[v2_trades['pnl'].notna()]
        if len(closed) > 0:
            metrics['Win Rate (%)'][1] = len(closed[closed['pnl'] > 0]) / len(closed) * 100
            metrics['Total P&L ($)'][1] = closed['pnl'].sum()

    # Plot bars
    x = ['V1 (Consensus)', 'V2 (Edge-based)']
    colors = ['blue', 'green']

    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        bars = ax.bar(x, values, color=colors, alpha=0.7)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}' if 'Rate' in metric_name else f'{val:.0f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = os.path.join(BASE_DIR, "data", "strategy_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    cprint(f"\n📊 Comparison saved to: {output_path}", "green", attrs=['bold'])

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='🌙 Moon Dev Trading Visualizer')
    parser.add_argument('--v1', action='store_true', help='Show V1 only')
    parser.add_argument('--v2', action='store_true', help='Show V2 only')
    parser.add_argument('--compare', action='store_true', help='Show comparison charts')
    parser.add_argument('--no-plot', action='store_true', help='Text summary only, no plots')
    args = parser.parse_args()

    cprint("\n" + "="*60, "cyan")
    cprint("🌙 Moon Dev Trading Dashboard", "cyan", attrs=['bold'])
    cprint("="*60, "cyan")

    show_v1 = not args.v2 or args.v1
    show_v2 = not args.v1 or args.v2

    # Print text summaries
    if show_v1:
        print_summary("V1 (Consensus-based)", V1_DATA_DIR)

    if show_v2:
        print_summary("V2 (Edge-based)", V2_DATA_DIR)

    # Generate plots
    if not args.no_plot:
        try:
            if args.compare:
                plot_comparison_metrics(V1_DATA_DIR, V2_DATA_DIR)
            else:
                plot_portfolio_equity(V1_DATA_DIR, V2_DATA_DIR, show_v1, show_v2)
        except Exception as e:
            cprint(f"\n⚠️ Could not generate plots: {e}", "yellow")
            cprint("Run with --no-plot for text-only summary", "yellow")

    cprint("\n" + "="*60, "cyan")
    cprint("✅ Dashboard complete!", "green", attrs=['bold'])
    cprint("="*60 + "\n", "cyan")


if __name__ == "__main__":
    main()
