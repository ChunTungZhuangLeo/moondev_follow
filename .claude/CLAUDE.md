🧠Smart Dev Task:

1️⃣Fix program🔧, provide bug-free🐞, well-commented code📝.

2️⃣Write detailed📏 code, implement architecture🏛️. Start with core classes🔠, functions🔢, methods🔣, brief comments🖊️.

3️⃣Output each file📂 content. Follow markdown code block format📑:
FILENAME
---LANG
CODE
--
4️⃣No placeholders❌, start with "entrypoint" file📚. Check code compatibility🧩, file naming🔤. Include module/package dependencies🔗.

5️⃣For Python🐍, NodeJS🌐, create appropriate dependency files📜. Comment on function definitions📖 and complex logic🧮.

6️⃣Use pytest, dataclasses for Python🔧.

🔍Review Task:

1️⃣Summarize unclear areas in instructions📄, ask clarification questions❓.

2️⃣As a Google engineer👷‍♂️, review a feature specification📝. Check for potential flaws💥, missing elements🔍, simplifications🧹. Make educated assumptions🎓.

📚Spec Creation Task:

1️⃣Create a detailed program specification📘. Include features, classes, functions, methods🔡, brief comments🖊️.

2️⃣Output file📂 content, follow markdown code block📑, ensure full functionality🔨.

# Kalshi Trading Bot - Project Instructions

> **Claude Code**: Read this file at the start of every session. For detailed specs, read `docs/BACKTEST_SPEC.md`.

## Quick Context

This is a Kalshi prediction market trading bot. Current goal: **build a backtesting framework** to optimize the live trading agent.

**Key Questions to Answer**:
1. Which market categories perform best?
2. Which AI model (Claude/GPT-4/DeepSeek/Groq/Gemini) performs best?
3. What information is most decisive for predictions?
4. What market patterns can be exploited?

## Project Structure

```
src/
├── agents/              # Live trading agents
│   ├── kalshi_agent.py  # Main Kalshi bot
│   └── polymarket_agent.py
├── backtest/            # Backtesting framework (building this)
│   ├── data_scraper.py  # Historical data collection
│   ├── simulator.py     # Trading simulation
│   ├── model_tester.py  # Multi-model comparison
│   ├── analyzer.py      # Performance analysis
│   └── run_backtest.py  # Main orchestrator
├── models/              # AI model interfaces
└── utils/               # Shared utilities

data/
├── raw/                 # Raw scraped data
│   ├── markets/         # Market metadata
│   ├── candlesticks/    # OHLCV per ticker
│   └── orderbooks/      # Orderbook snapshots
├── enriched/            # Processed data with features
├── backtest_results/    # Simulation outputs
└── analysis/            # Reports and charts

docs/
├── BACKTEST_SPEC.md     # Detailed implementation specs
└── API_REFERENCE.md     # Kalshi API docs

configs/
└── backtest_config.yaml # Backtest parameters
```

## Current Progress

- [x] Live agents working (kalshi_agent.py, polymarket_agent.py)
- [ ] Phase 1: Data scraper
- [ ] Phase 2: Backtesting simulator  
- [ ] Phase 3: Analysis & insights
- [ ] Phase 4: Strategy integration

## Code Style

- Python 3.11+
- Use `pathlib.Path` for file paths
- Use `dataclasses` for data structures
- Use `pandas` + `parquet` for data storage
- Type hints on all functions
- Docstrings on all public methods
- Log with `logging` module, not print()

## API Keys (in .env)

```
ANTHROPIC_KEY, OPENAI_KEY, DEEPSEEK_KEY, GROQ_API_KEY, GEMINI_KEY
```

## Common Commands

```bash
# Run data scraper
python -m src.backtest.data_scraper --start 2025-01-01 --end 2025-10-31

# Run backtest
python -m src.backtest.run_backtest --config configs/backtest_config.yaml

# Generate analysis
python -m src.backtest.analyzer --generate-report
```

## Important Rules

1. **No lookahead bias** - Never use future data in backtesting decisions
2. **Rate limit APIs** - 0.1s delay between Kalshi calls, cache LLM responses
3. **Checkpoint progress** - Save state so interrupted runs can resume
4. **Log everything** - Verbose logging for debugging

## When Implementing

1. Read `docs/BACKTEST_SPEC.md` for detailed specs
2. Look at existing code in `src/agents/` for patterns
3. Implement one file at a time, test before moving on
4. Ask if unsure about requirements