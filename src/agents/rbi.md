# 🌙 Moon Dev's RBI Execution Container System Plan

## Overview
This plan outlines the implementation of an automated execution system for the RBI (Research-Backtest-Implement) agent, enabling it to run backtests, capture results/errors, and create a continuous improvement loop.

## Current State Analysis
- ✅ Research Agent: Generates strategy ideas from various sources
- ✅ Backtest Agent: Creates backtest code using backtesting.py
- ✅ Package Agent: Fixes import/dependency issues
- ✅ Debug Agent: Fixes syntax and technical errors
- ❌ **Missing**: Actual execution of backtest code
- ❌ **Missing**: Results/error capture and feedback loop
- ❌ **Missing**: Automated strategy improvement based on results

## Proposed Architecture

### 1. Execution Container System

#### Option A: Docker-based Solution (Recommended)
```
rbi_executor/
├── Dockerfile              # Python environment with all dependencies
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # backtesting.py, talib, pandas, etc.
└── executor/
    ├── runner.py          # Safe code execution wrapper
    ├── sandbox.py         # Resource limits and timeouts
    └── result_parser.py   # Extract stats and errors
```

**Benefits:**
- Complete isolation from host system
- Reproducible environment
- Easy dependency management
- Resource limits (CPU, memory, time)
- No risk to main system

#### Option B: Python Virtual Environment + subprocess
- Lighter weight but less isolation
- Faster execution but more risk
- Suitable for trusted code only

### 2. New Agent: Execution Agent

**File:** `src/agents/execution_agent.py`

**Responsibilities:**
1. Receive backtest code from RBI agent
2. Create isolated execution environment
3. Run backtest with timeout protection
4. Capture stdout, stderr, and results
5. Parse backtest statistics
6. Return structured results

**Key Functions:**
```python
def execute_backtest(code_path: str) -> ExecutionResult:
    """
    Runs backtest in isolated container
    Returns: stats, errors, warnings, execution_time
    """

def parse_results(output: str) -> BacktestStats:
    """
    Extracts key metrics:
    - Total Return
    - Sharpe Ratio
    - Max Drawdown
    - Win Rate
    - Number of Trades
    """

def validate_code(code: str) -> ValidationResult:
    """
    Pre-execution safety checks:
    - No file system access outside data
    - No network calls
    - No dangerous imports
    """
```

### 3. New Agent: Optimization Agent

**File:** `src/agents/optimization_agent.py`

**Responsibilities:**
1. Analyze backtest results
2. Identify performance issues
3. Generate improvement suggestions
4. Create modified strategy versions
5. Track performance evolution

**Optimization Strategies:**
- Parameter tuning (stop loss, take profit, indicators)
- Entry/exit logic refinement
- Risk management improvements
- Market condition filters
- Position sizing optimization

### 4. Feedback Loop Implementation

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Research   │────▶│   Backtest   │────▶│    Package    │
│   Agent     │     │    Agent     │     │     Agent     │
└─────────────┘     └──────────────┘     └───────────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│Optimization │◀────│  Execution   │◀────│     Debug     │
│   Agent     │     │    Agent     │     │     Agent     │
└─────────────┘     └──────────────┘     └───────────────┘
      │                                            ▲
      └────────────────────────────────────────────┘
                    (If errors or poor performance)
```

### 5. Database Schema for Results

**File:** `src/data/rbi/results.db` (SQLite)

```sql
CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT,
    version INTEGER,
    code_hash TEXT,
    execution_time REAL,
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    num_trades INTEGER,
    error_message TEXT,
    created_at TIMESTAMP
);

CREATE TABLE optimization_history (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT,
    parent_version INTEGER,
    child_version INTEGER,
    optimization_type TEXT,
    changes_made TEXT,
    performance_delta REAL
);
```

### 6. Implementation Steps

#### Phase 1: Basic Execution (Week 1)
1. Create Docker container with backtesting environment
2. Implement basic execution_agent.py
3. Add timeout and resource limits
4. Test with existing backtest files

#### Phase 2: Result Parsing (Week 2)
1. Build comprehensive result parser
2. Extract all backtesting.py statistics
3. Handle error cases gracefully
4. Create structured result objects

#### Phase 3: Optimization Loop (Week 3)
1. Implement optimization_agent.py
2. Create parameter tuning logic
3. Build performance comparison system
4. Add strategy versioning

#### Phase 4: Integration (Week 4)
1. Modify rbi_agent.py to use execution system
2. Implement continuous loop logic
3. Add database persistence
4. Create monitoring dashboard

### 7. Safety Considerations

1. **Code Validation**
   - Whitelist allowed imports
   - Block file system access outside data directory
   - Prevent network calls
   - Limit execution time (5 minutes max)

2. **Resource Limits**
   - Max memory: 2GB
   - Max CPU: 1 core
   - Max disk: 100MB
   - Max execution time: 300 seconds

3. **Error Handling**
   - Graceful timeout handling
   - Memory limit exceptions
   - Syntax error capture
   - Import error resolution

### 8. Configuration Updates

**Add to `config.py`:**
```python
# Execution Configuration
EXECUTION_TIMEOUT = 300  # seconds
EXECUTION_MEMORY_LIMIT = "2g"
EXECUTION_CPU_LIMIT = "1.0"
DOCKER_IMAGE = "moondev/backtest-executor:latest"

# Optimization Parameters
MAX_OPTIMIZATION_ITERATIONS = 10
MIN_PERFORMANCE_IMPROVEMENT = 0.05  # 5%
OPTIMIZATION_ALGORITHMS = ["grid_search", "random_search", "bayesian"]
```

### 9. Monitoring and Logging

1. **Execution Metrics**
   - Success/failure rate
   - Average execution time
   - Resource usage statistics
   - Error frequency by type

2. **Performance Tracking**
   - Strategy performance over iterations
   - Best performing strategies
   - Optimization effectiveness
   - A/B testing results

### 10. Future Enhancements

1. **Parallel Execution**
   - Run multiple backtests simultaneously
   - GPU acceleration for ML strategies
   - Distributed execution across machines

2. **Advanced Optimization**
   - Genetic algorithms
   - Reinforcement learning
   - Multi-objective optimization
   - Walk-forward analysis

3. **Production Pipeline**
   - Automatic deployment of profitable strategies
   - Live paper trading validation
   - Risk management integration
   - Performance monitoring

## Success Metrics

1. **Automation**: 100% hands-free operation
2. **Throughput**: 100+ strategies tested per day
3. **Improvement**: 20%+ average performance gain through optimization
4. **Reliability**: <1% execution failure rate
5. **Safety**: Zero system compromises or crashes

## Next Steps

1. Review and approve this plan
2. Set up Docker development environment
3. Create execution_agent.py prototype
4. Test with sample backtest files
5. Iterate based on results

---

*"To the moon with automated strategy optimization!" 🚀*

*Built with love by Moon Dev* 🌙