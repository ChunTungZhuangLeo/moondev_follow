# Kalshi AI Swarm Value Trading Strategy

## Overview

This is an automated prediction market trading system for Kalshi that uses AI swarm consensus to identify mispriced markets and execute trades with edge-based position sizing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KALSHI VALUE TRADING AGENT                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Market    │───▶│  AI Swarm    │───▶│  Trade Executor  │   │
│  │   Fetcher   │    │  Analysis    │    │  (Paper/Live)    │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│         │                  │                     │              │
│         ▼                  ▼                     ▼              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Kalshi    │    │  6-Model     │    │  Position        │   │
│  │   API       │    │  Consensus   │    │  Monitor         │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Two Versions

### V1: Original Agent (`kalshi_value_agent.py`)
- Located in: `src/agents/kalshi_value_agent.py`
- Data directory: `src/data/kalshi_value/`
- Bankroll: $5,000
- Features: Time-adjusted edge, volume filters

### V2: Enhanced Agent (`kalshi_value_agent_v2.py`)
- Located in: `src/agents/kalshi_value_agent_v2.py`
- Data directory: `src/data/kalshi_value_v2/`
- Bankroll: $1,000
- Features: Everything in V1 + smarter early exit, trade classification

---

## Core Strategy Components

### 1. AI Swarm Consensus (6 Models)

The system queries 6 different AI models for each market:

| Model | Provider | Speed | Strength |
|-------|----------|-------|----------|
| Llama 3.3 70B | Groq | Very Fast | General reasoning |
| DeepSeek Chat | DeepSeek | Fast | Math/logic |
| Gemini 2.0 Flash | Google | Fast | Current events |
| GPT-4o Mini | OpenAI | Medium | Balanced |
| Claude Haiku | Anthropic | Fast | Nuanced analysis |
| Claude Sonnet | Anthropic | Medium | Deep reasoning |

**Consensus Rules:**
- Minimum 4/6 models must agree on direction (YES or NO)
- Average confidence must be ≥6/10
- Edge is calculated as: `|true_probability - market_price|`

### 2. Time-Adjusted Edge Requirements

Different edge thresholds based on time to resolution:

| Days to Resolution | Required Edge | Rationale |
|-------------------|---------------|-----------|
| < 60 days | 8% | Standard - quick resolution |
| 60-180 days | 15% | Need more edge for longer hold |
| 180-365 days | 25% | Significant edge for 6+ months |

**Code location:** `get_required_edge()` function in agent file

### 3. Volume Filter for Long-Term Markets

Markets >60 days out must have minimum 24h volume of 100 contracts.

**Rationale:**
- Ensures liquidity for entry/exit
- Confirms active price discovery
- Avoids stale/dead markets

### 4. Position Sizing (Half Kelly Criterion)

```python
kelly_fraction = edge / (odds - 1)  # Full Kelly
position_size = bankroll * kelly_fraction * 0.5  # Half Kelly (conservative)
```

**Parameters:**
- Max risk per trade: 5% of bankroll
- Kelly fraction: 50% (Half Kelly)
- Minimum position: $7 (avoid tiny positions)

---

## Trade Execution Flow

```
1. Fetch Markets from Kalshi API
   └── Filter by: open status, has prices, resolution window

2. For Each Market:
   ├── Check volume filter (if >60 days out)
   ├── Query all 6 AI models in parallel
   ├── Count consensus (how many agree on YES/NO)
   ├── Calculate average edge and confidence
   ├── Check time-adjusted edge requirement
   └── If qualified → Execute trade immediately

3. Position Monitoring (every 3 hours):
   ├── Update current prices
   ├── Calculate unrealized P&L
   ├── Check exit conditions:
   │   ├── Edge captured (>60% of expected edge)
   │   ├── Edge lost (moved against us significantly)
   │   └── AI re-evaluation (thesis invalidated)
   └── Execute exits as needed
```

---

## Exit Strategy (V2 Only)

### Trade Classification

At entry, trades are classified:

| Type | Criteria | Strategy |
|------|----------|----------|
| `hold_to_resolution` | ≤7 days to resolution | Hold until event resolves |
| `take_profit` | >7 days to resolution | Monitor for early exit opportunities |

### Exit Conditions for `take_profit` Trades

1. **Edge Captured** (WIN):
   - Price moved in our favor by >60% of expected edge
   - Example: Bought at 15¢, AI said 30% true prob (15¢ edge), now at 24¢+ → EXIT

2. **Edge Lost** (LOSS):
   - Position is >15% underwater AND
   - AI re-evaluation says thesis is now wrong
   - Example: Bought at 15¢, now at 10¢, AI says "no longer likely" → EXIT

3. **Thesis Invalidated**:
   - AI re-queries show <3/6 consensus (was 4+/6 at entry)
   - The swarm no longer believes in the trade

---

## File Structure

```
src/agents/
├── kalshi_value_agent.py      # V1 agent (main)
├── kalshi_value_agent_v2.py   # V2 agent (enhanced)
├── kalshi_trade_executor.py   # Trade execution (paper/live)
├── value_trading_strategy.py  # Position sizing, edge calculation
├── kalshi_auth_client.py      # RSA-PSS authenticated API client
└── KALSHI_TRADING_STRATEGY.md # This document

src/data/
├── kalshi_value/              # V1 data
│   ├── trades.csv             # Trade history
│   └── price_history.csv      # Position tracking over time
└── kalshi_value_v2/           # V2 data
    ├── trades.csv             # Trade history
    └── price_history.csv      # Position tracking over time

src/models/
├── gemini_model.py            # Gemini API wrapper
└── model_factory.py           # Model instantiation
```

---

## CSV Schema

### trades.csv

| Column | Description |
|--------|-------------|
| trade_id | Unique identifier (timestamp-based) |
| timestamp | Entry time (ISO format) |
| mode | "paper" or "live" |
| ticker | Kalshi market ticker |
| title | Market question |
| side | "yes" or "no" |
| action | "buy" |
| contracts | Number of contracts |
| price_cents | Entry price in cents |
| cost_usd | Total cost |
| status | "open", "won", "lost" |
| order_id | Kalshi order ID (or PAPER-xxx) |
| exit_price_cents | Exit price (if closed) |
| exit_timestamp | Exit time (if closed) |
| pnl_usd | Realized P&L |
| edge_pct | AI-predicted edge at entry |
| true_prob | AI-predicted true probability |
| consensus_count | Models agreeing (e.g., 4, 5, 6) |
| notes | Exit reason or other notes |
| trade_type | "hold_to_resolution" or "take_profit" (V2) |
| expected_edge_cents | Expected edge in cents (V2) |
| achieved_edge_cents | Current captured edge (V2) |

### price_history.csv

| Column | Description |
|--------|-------------|
| timestamp | Check time |
| ticker | Market ticker |
| side | "yes" or "no" |
| entry_price | Original entry price |
| current_price | Current market price |
| unrealized_pnl | Unrealized P&L in USD |
| pnl_pct | P&L as percentage |
| status | "open" or "summary" |

---

## Running the Agents

### V1 Agent
```bash
cd src/agents
python kalshi_value_agent.py           # Continuous loop
python kalshi_value_agent.py --once    # Single run
python kalshi_value_agent.py --monitor # Monitor positions only
```

### V2 Agent
```bash
cd src/agents
python kalshi_value_agent_v2.py           # Continuous loop
python kalshi_value_agent_v2.py --once    # Single run
python kalshi_value_agent_v2.py --monitor # Monitor positions only
```

---

## Environment Variables Required

```bash
# Kalshi API (for live trading)
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=/path/to/private_key.pem

# AI Model APIs
GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## Performance Tracking (as of 2025-12-20)

### V2 Portfolio
- **Invested:** $869.67
- **Current Value:** $1,002.73
- **Return:** +15.3%
- **Best Trade:** KXCPI-25DEC-T0.3 (+$164.64, 93% edge captured)
- **Trades:** 2 open, 1 won, 2 lost

### V1 Portfolio
- **Invested:** $3,158.04
- **Current Value:** $3,123.35
- **Return:** -1.1%
- **Best Trade:** KXTRILLIONAIRE-30-EM (+$33.39, 50% unrealized)
- **Trades:** 15 open (with duplicates), 5 lost (tiny)

---

## Key Insights

### What Works
1. **AI Swarm Consensus** - 4/6 agreement filters out noise
2. **Time-Adjusted Edge** - Higher bar for long-term markets
3. **Early Exit on Edge Capture** - Don't wait for resolution, take profits
4. **Volume Filter** - Avoid illiquid long-term markets

### What to Improve
1. Avoid duplicate positions (V1 has this issue)
2. Better handling of weather/short-term event markets
3. Consider market-specific edge requirements (politics vs economics)

---

## Comparison to Other Approaches

### vs. Single AI Bot (Octagon AI / Deep Trading Bot)
- They use 1 AI model + hedging
- We use 6 models for consensus (more robust signal)
- Our consensus already filters for confidence, so hedging would reduce EV

### vs. Pure Technical Analysis
- We don't use price charts or momentum
- We rely on AI's fundamental analysis of the underlying question
- This works better for prediction markets (news-driven, not chart-driven)

---

## Future Enhancements

1. **Market-Specific Models** - Use specialized prompts for politics, economics, sports
2. **Sentiment Analysis** - Incorporate social media/news sentiment
3. **Correlation Tracking** - Avoid too many correlated bets
4. **Dynamic Kelly** - Adjust Kelly fraction based on recent performance
5. **Live Trading Activation** - Switch from paper to live when confident

---

## Contact & Repository

- Repository: This codebase (`moondev_follow`)
- Main files: `src/agents/kalshi_value_agent*.py`
- Data: `src/data/kalshi_value*/`
