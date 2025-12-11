# Moon Dev Prediction Market Bots

Lightweight version containing Kalshi and Polymarket trading agents with AI swarm analysis.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file from `.env_example`:
```bash
cp .env_example .env
# Edit .env with your API keys
```

3. Run agents:
```bash
# Kalshi Agent (paper trading)
python src/agents/kalshi_agent.py

# Polymarket Agent (WebSocket + AI analysis)
python src/agents/polymarket_agent.py
```

## Configuration

Edit settings at the top of each agent file:
- `BANKROLL_USD` - Your trading bankroll
- `PAPER_TRADING_MODE` - True for paper trading
- `AUTO_RUN_LOOP` - True for continuous operation
- `HOURS_BETWEEN_RUNS` - Analysis frequency

## Required API Keys

Add to `.env`:
- `ANTHROPIC_KEY` - Claude API
- `OPENAI_KEY` - GPT-4 API
- `DEEPSEEK_KEY` - DeepSeek API
- `GROQ_API_KEY` - Groq API
- `GEMINI_KEY` - Google Gemini API
