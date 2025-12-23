"""
🌙 Moon Dev's RBI AI (Research-Backtest-Implement)
Built with love by Moon Dev 🚀

Required Setup:
1. Create folder structure:
   src/
   ├── data/
   │   └── rbi/
   │       ├── MM_DD_YYYY/         # Date-based folder (created automatically)
   │       │   ├── research/       # Strategy research outputs
   │       │   ├── backtests/      # Initial backtest code
   │       │   ├── backtests_package/ # Package-fixed code
   │       │   ├── backtests_final/ # Debugged backtest code
   │       │   └── charts/         # Charts output directory
   │       └── ideas.txt          # Trading ideas to process

2. Environment Variables:
   - No API keys needed! We're using local Ollama models 🎉

3. Create ideas.txt:
   - One trading idea per line
   - Can be YouTube URLs, PDF links, or text descriptions
   - Lines starting with # are ignored

This AI automates the RBI process:
1. Research: Analyzes trading strategies from various sources
2. Backtest: Creates backtests for promising strategies
3. Debug: Fixes technical issues in generated backtests

✨ NEW FEATURE: All outputs are now organized in date-based folders (MM_DD_YYYY)
This helps keep your strategy research organized by day!

Remember: Past performance doesn't guarantee future results!
"""


## Previous presets (kept for easy switching) 👇
# RESEARCH_CONFIG = {
#     "type": "deepseek",
#     "name": "deepseek-chat"  # Using DeepSeek Chat for research
# }
# 
# BACKTEST_CONFIG = {
#     "type": "deepseek", 
#     "name": "deepseek-reasoner"  # Using DeepSeek Reasoner for backtesting
# }
# 
# DEBUG_CONFIG = {
#     "type": "deepseek",
#     "name": "deepseek-chat"  # Using DeepSeek Chat for debugging
# }
# 
# # DEBUG_CONFIG = {
# #     "type": "ollama",
# #     "name": "deepseek-r1"  # Using Ollama's DeepSeek-R1 for debugging
# # }
# 
# PACKAGE_CONFIG = {
#     "type": "deepseek",
#     "name": "deepseek-chat"  # Using DeepSeek Chat for package optimization
# }

# Using DeepSeek for research/debug (cheap) and GPT-4o for backtesting (smart)
RESEARCH_CONFIG = {
    "type": "deepseek",
    "name": "deepseek-chat"
}

BACKTEST_CONFIG = {
    "type": "openai",
    "name": "gpt-4o"
}

DEBUG_CONFIG = {
    "type": "deepseek",
    "name": "deepseek-chat"
}

PACKAGE_CONFIG = {
    "type": "deepseek",
    "name": "deepseek-chat"
}



################

# Model Configuration
# Using a mix of Ollama models and DeepSeek API
# RESEARCH_CONFIG = {
#     "type": "ollama",
#     "name": "llama3.2"  # Using Llama 3.2 for research
# }

# RESEARCH_CONFIG = {
#     "type": "deepseek",
#     "name": "deepseek-chat"  # Using Llama 3.2 for research
# }

# BACKTEST_CONFIG = {
#     "type": "openai", 
#     "name": "o3"  # Using O3-mini for backtesting
# }

# DEBUG_CONFIG = {
#     "type": "openai",
#     "name": "o3"  # Using GPT-4.1 for debugging
# }

# # DEBUG_CONFIG = {
# #     "type": "ollama",
# #     "name": "deepseek-r1"  # Using Ollama's DeepSeek-R1 for debugging
# # }

# # PACKAGE_CONFIG = {
# #     "type": "deepseek",
# #     "name": "deepseek-chat"  # Using Llama 3.2 for package optimization
# # }

# PACKAGE_CONFIG = {
#     "type": "openai",
#     "name": "o3"  # Using Llama 3.2 for package optimization
# }


# PACKAGE_CONFIG = {
#     "type": "ollama",
#     "name": "llama3.2"  # Using Llama 3.2 for package optimization
# }


# DeepSeek Model Selection per AI
# "gemma:2b",     # Google's Gemma 2B model
#         "llama3.2",
# Using a mix of models for different tasks
# RESEARCH_MODEL = "llama3.2"           # Llama 3.2 for research
# BACKTEST_MODEL = "deepseek-reasoner"  # DeepSeek API for backtesting
# DEBUG_MODEL = "deepseek-r1"           # Ollama DeepSeek-R1 for debugging
# PACKAGE_MODEL = "llama3.2"            # Llama 3.2 for package optimization

# AI Prompts

RESEARCH_PROMPT = """
You are Moon Dev's Research AI 🌙

IMPORTANT NAMING RULES:
1. Create a UNIQUE TWO-WORD NAME for this specific strategy
2. The name must be DIFFERENT from any generic names like "TrendFollower" or "MomentumStrategy"
3. First word should describe the main approach (e.g., Adaptive, Neural, Quantum, Fractal, Dynamic)
4. Second word should describe the specific technique (e.g., Reversal, Breakout, Oscillator, Divergence)
5. Make the name SPECIFIC to this strategy's unique aspects

Examples of good names:
- "AdaptiveBreakout" for a strategy that adjusts breakout levels
- "FractalMomentum" for a strategy using fractal analysis with momentum
- "QuantumReversal" for a complex mean reversion strategy
- "NeuralDivergence" for a strategy focusing on divergence patterns

BAD names to avoid:
- "TrendFollower" (too generic)
- "SimpleMoving" (too basic)
- "PriceAction" (too vague)

Output format must start with:
STRATEGY_NAME: [Your unique two-word name]

Then analyze the trading strategy content and create detailed instructions.
Focus on:
1. Key strategy components
2. Entry/exit rules
3. Risk management
4. Required indicators

Your complete output must follow this format:
STRATEGY_NAME: [Your unique two-word name]

STRATEGY_DETAILS:
[Your detailed analysis]

Remember: The name must be UNIQUE and SPECIFIC to this strategy's approach!
"""

BACKTEST_PROMPT = """
You are Moon Dev's Backtest AI 🌙 ONLY SEND BACK CODE, NO OTHER TEXT.
Create a backtesting.py implementation for the strategy.
USE BACKTESTING.PY
Include:
1. All necessary imports
2. Strategy class with indicators
3. Entry/exit logic
4. Risk management
5. your size should be 1,000,000
6. If you need indicators use TA lib or pandas TA.

IMPORTANT DATA HANDLING:
1. Clean column names by removing spaces: data.columns = data.columns.str.strip().str.lower()
2. Drop any unnamed columns: data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
3. Ensure proper column mapping to match backtesting requirements:
   - Required columns: 'Open', 'High', 'Low', 'Close', 'Volume'
   - Use proper case (capital first letter)

FOR THE PYTHON BACKTESTING LIBRARY USE BACKTESTING.PY AND SEND BACK ONLY THE CODE, NO OTHER TEXT.

INDICATOR CALCULATION RULES:
1. ALWAYS use self.I() wrapper for ANY indicator calculations
2. Use talib functions instead of pandas operations:
   - Instead of: self.data.Close.rolling(20).mean()
   - Use: self.I(talib.SMA, self.data.Close, timeperiod=20)
3. For swing high/lows use talib.MAX/MIN:
   - Instead of: self.data.High.rolling(window=20).max()
   - Use: self.I(talib.MAX, self.data.High, timeperiod=20)

BACKTEST EXECUTION ORDER:
1. Run initial backtest with default parameters first
2. Print full stats using print(stats) and print(stats._strategy)
3. no optimization code needed, just print the final stats, make sure full stats are printed, not just part or some. stats = bt.run() print(stats) is an example of the last line of code. no need for plotting ever.

do not creeate charts to plot this, just print stats. no charts needed.

CRITICAL POSITION SIZING RULE:
When calculating position sizes in backtesting.py, the size parameter must be either:
1. A fraction between 0 and 1 (for percentage of equity)
2. A whole number (integer) of units

The common error occurs when calculating position_size = risk_amount / risk, which results in floating-point numbers. Always use:
position_size = int(round(position_size))

Example fix:
❌ self.buy(size=3546.0993)  # Will fail
✅ self.buy(size=int(round(3546.0993)))  # Will work

RISK MANAGEMENT:
1. Always calculate position sizes based on risk percentage
2. Use proper stop loss and take profit calculations
4. Print entry/exit signals with Moon Dev themed messages

If you need indicators use TA lib or pandas TA. 

Use this data path: {PROJECT_ROOT}/src/data/rbi/BTC-USD-15m.csv (use absolute path in code)
the above data head looks like below
datetime, open, high, low, close, volume,
2023-01-01 00:00:00, 16531.83, 16532.69, 16509.11, 16510.82, 231.05338022,
2023-01-01 00:15:00, 16509.78, 16534.66, 16509.11, 16533.43, 308.12276951,

Always add plenty of Moon Dev themed debug prints with emojis to make debugging easier! 🌙 ✨ 🚀

FOR THE PYTHON BACKTESTING LIBRARY USE BACKTESTING.PY AND SEND BACK ONLY THE CODE, NO OTHER TEXT.
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

DEBUG_PROMPT = """
You are Moon Dev's Debug AI 🌙
Fix technical issues in the backtest code WITHOUT changing the strategy logic.

CRITICAL BACKTESTING REQUIREMENTS:
1. Position Sizing Rules:
   - Must be either a fraction (0 < size < 1) for percentage of equity
   - OR a positive whole number (round integer) for units
   - Example: size=0.5 (50% of equity) or size=100 (100 units)
   - NEVER use floating point numbers for unit-based sizing

2. Common Fixes Needed:
   - Round position sizes to whole numbers if using units
   - Convert to fraction if using percentage of equity
   - Ensure stop loss and take profit are price levels, not distances

Focus on:
1. Syntax errors (like incorrect string formatting)
2. Import statements and dependencies
3. Class and function definitions
4. Variable scoping and naming
5. Print statement formatting

DO NOT change:
1. Strategy logic
2. Entry/exit conditions
3. Risk management rules
4. Parameter values (unless fixing technical issues)

Return the complete fixed code with Moon Dev themed debug prints! 🌙 ✨
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

PACKAGE_PROMPT = """
You are Moon Dev's Package AI 🌙
Your job is to ensure the backtest code NEVER uses ANY backtesting.lib imports or functions.

❌ STRICTLY FORBIDDEN:
1. from backtesting.lib import *
2. import backtesting.lib
3. from backtesting.lib import crossover
4. ANY use of backtesting.lib

✅ REQUIRED REPLACEMENTS:
1. For crossover detection:
   Instead of: backtesting.lib.crossover(a, b)
   Use: (a[-2] < b[-2] and a[-1] > b[-1])  # for bullish crossover
        (a[-2] > b[-2] and a[-1] < b[-1])  # for bearish crossover

2. For indicators:
   - Use talib for all standard indicators (SMA, RSI, MACD, etc.)
   - Use pandas-ta for specialized indicators
   - ALWAYS wrap in self.I()

3. For signal generation:
   - Use numpy/pandas boolean conditions
   - Use rolling window comparisons with array indexing
   - Use mathematical comparisons (>, <, ==)

Example conversions:
❌ from backtesting.lib import crossover
❌ if crossover(fast_ma, slow_ma):
✅ if fast_ma[-2] < slow_ma[-2] and fast_ma[-1] > slow_ma[-1]:

❌ self.sma = self.I(backtesting.lib.SMA, self.data.Close, 20)
✅ self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20)

IMPORTANT: Scan the ENTIRE code for any backtesting.lib usage and replace ALL instances!
Return the complete fixed code with proper Moon Dev themed debug prints! 🌙 ✨
ONLY SEND BACK CODE, NO OTHER TEXT.
"""

def get_model_id(model):
    """Get DR/DC identifier based on model"""
    return "DR" if model == "deepseek-reasoner" else "DC"

import os
import time
import re
from datetime import datetime
import requests
from io import BytesIO
import openai
from termcolor import cprint
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    cprint("⚠️ Anthropic SDK not installed. Claude models will be unavailable. (Moon Dev note)", "yellow")
from pathlib import Path
import threading
import itertools
import sys
import hashlib  # Added for idea hashing
from src.config import *  # Import config settings including AI_MODEL
from src.models import model_factory

# DeepSeek Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Get today's date for organizing outputs
TODAY_DATE = datetime.now().strftime("%m_%d_%Y")

# Update data directory paths
PROJECT_ROOT = Path(__file__).parent.parent  # Points to src/
DATA_DIR = PROJECT_ROOT / "data/rbi"
TODAY_DIR = DATA_DIR / TODAY_DATE  # Today's date folder
RESEARCH_DIR = TODAY_DIR / "research"
BACKTEST_DIR = TODAY_DIR / "backtests"
PACKAGE_DIR = TODAY_DIR / "backtests_package"
FINAL_BACKTEST_DIR = TODAY_DIR / "backtests_final"
CHARTS_DIR = TODAY_DIR / "charts"  # New directory for HTML charts
PROCESSED_IDEAS_LOG = DATA_DIR / "processed_ideas.log"  # New file to track processed ideas

# Create main directories if they don't exist
for directory in [DATA_DIR, TODAY_DIR, RESEARCH_DIR, BACKTEST_DIR, PACKAGE_DIR, FINAL_BACKTEST_DIR, CHARTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

cprint(f"📂 Using RBI data directory: {DATA_DIR}")
cprint(f"📅 Today's date folder: {TODAY_DATE}")
cprint(f"📂 Research directory: {RESEARCH_DIR}")
cprint(f"📂 Backtest directory: {BACKTEST_DIR}")
cprint(f"📂 Package directory: {PACKAGE_DIR}")
cprint(f"📂 Final backtest directory: {FINAL_BACKTEST_DIR}")
cprint(f"📈 Charts directory: {CHARTS_DIR}")

def init_deepseek_client():
    """Initialize DeepSeek client with proper error handling"""
    try:
        deepseek_key = os.getenv("DEEPSEEK_KEY")
        if not deepseek_key:
            cprint("⚠️ DEEPSEEK_KEY not found - DeepSeek models will not be available", "yellow")
            return None
            
        print("🔑 Initializing DeepSeek client...")
        print("🌟 Moon Dev's RBI AI is connecting to DeepSeek...")
        
        client = openai.OpenAI(
            api_key=deepseek_key,
            base_url=DEEPSEEK_BASE_URL
        )
        
        print("✅ DeepSeek client initialized successfully!")
        print("🚀 Moon Dev's RBI AI ready to roll!")
        return client
    except Exception as e:
        print(f"❌ Error initializing DeepSeek client: {str(e)}")
        print("💡 Will fall back to Claude model from config.py")
        return None

def init_anthropic_client():
    """Initialize Anthropic client for Claude models"""
    try:
        if Anthropic is None:
            cprint("⚠️ Anthropic client unavailable (package not installed)", "yellow")
            return None
        anthropic_key = os.getenv("ANTHROPIC_KEY")
        if not anthropic_key:
            cprint("⚠️ ANTHROPIC_KEY not found in env. Skipping Claude init.", "yellow")
            return None
        return Anthropic(api_key=anthropic_key)
    except Exception as e:
        print(f"❌ Error initializing Anthropic client: {str(e)}")
        return None

def chat_with_model(system_prompt, user_content, model_config):
    """Chat with AI model using model factory"""
    try:
        # Initialize model using factory with specific config
        model = model_factory.get_model(model_config["type"], model_config["name"])
        if not model:
            raise ValueError(f"🚨 Could not initialize {model_config['type']} {model_config['name']} model!")

        cprint(f"🤖 Using {model_config['type']} model: {model_config['name']}", "cyan")
        cprint("🌟 Moon Dev's RBI AI is thinking...", "yellow")
        
        # Debug prints for prompt lengths
        cprint(f"📝 System prompt length: {len(system_prompt)} chars", "cyan")
        cprint(f"📝 User content length: {len(user_content)} chars", "cyan")
        # If model returned a wrapper, normalize early
        if hasattr(model, 'model_name') and model.model_type == 'openai':
            cprint(f"🧪 OpenAI model in use: {model.model_name}", "cyan")

        # For Ollama models, handle response differently
        if model_config["type"] == "ollama":
            response = model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=AI_TEMPERATURE
            )
            # Handle string response from Ollama
            if isinstance(response, str):
                return response
            # Handle object response
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        else:
            # For other models, use standard parameters
            response = model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=AI_TEMPERATURE,
                max_tokens=AI_MAX_TOKENS
            )
            if response is None:
                cprint("❌ Model returned None response", "red")
                return None

            # Coerce response into text content
            content = None
            try:
                from src.models.base_model import ModelResponse
                if isinstance(response, ModelResponse):
                    content = response.content
            except Exception:
                pass

            if content is None:
                if isinstance(response, str):
                    content = response
                elif hasattr(response, 'content'):
                    content = response.content
                else:
                    cprint(f"❌ Response missing content attribute. Response type: {type(response)}", "red")
                    try:
                        cprint(f"Response attributes: {dir(response)}", "yellow")
                    except Exception:
                        pass
                    return None

            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            if not content or len(content.strip()) == 0:
                cprint("❌ Model returned empty content", "red")
                return None

            return content

    except Exception as e:
        cprint(f"❌ Error in AI chat: {str(e)}", "red")
        cprint(f"🔍 Error type: {type(e).__name__}", "yellow")
        if hasattr(e, 'response'):
            cprint(f"🔍 Response error: {getattr(e, 'response', 'No response details')}", "yellow")
        if hasattr(e, '__dict__'):
            cprint("🔍 Error attributes:", "yellow")
            for attr in dir(e):
                if not attr.startswith('_'):
                    cprint(f"  ├─ {attr}: {getattr(e, attr)}", "yellow")
        return None

def get_youtube_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            cprint("⚠️ youtube-transcript-api not installed. Skipping YouTube transcript fetch.", "yellow")
            return None
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en'])
        
        # Get the full transcript text
        transcript_text = ' '.join([t['text'] for t in transcript.fetch()])
        
        # Print the transcript with nice formatting
        cprint("\n📝 YouTube Transcript:", "cyan")
        cprint("=" * 80, "yellow")
        print(transcript_text)
        cprint("=" * 80, "yellow")
        cprint(f"📊 Transcript length: {len(transcript_text)} characters", "cyan")
        
        return transcript_text
    except Exception as e:
        cprint(f"❌ Error fetching transcript: {e}", "red")
        return None

def get_pdf_text(url):
    """Extract text from PDF URL"""
    try:
        try:
            import PyPDF2
        except ImportError:
            cprint("⚠️ PyPDF2 not installed. Skipping PDF extraction.", "yellow")
            return None
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        reader = PyPDF2.PdfReader(BytesIO(response.content))
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        cprint("📚 Successfully extracted PDF text!", "green")
        return text
    except Exception as e:
        cprint(f"❌ Error reading PDF: {e}", "red")
        return None

def animate_progress(agent_name, stop_event):
    """Fun animation while AI is thinking"""
    spinners = ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘']
    messages = [
        "brewing coffee ☕️",
        "studying charts 📊",
        "checking signals 📡",
        "doing math 🔢",
        "reading docs 📚",
        "analyzing data 🔍",
        "making magic ✨",
        "trading secrets 🤫",
        "Moon Dev approved 🌙",
        "to the moon! 🚀"
    ]
    
    spinner = itertools.cycle(spinners)
    message = itertools.cycle(messages)
    
    while not stop_event.is_set():
        sys.stdout.write(f'\r{next(spinner)} {agent_name} is {next(message)}...')
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write('\r' + ' ' * 50 + '\r')
    sys.stdout.flush()

def run_with_animation(func, agent_name, *args, **kwargs):
    """Run a function with a fun loading animation"""
    stop_animation = threading.Event()
    animation_thread = threading.Thread(target=animate_progress, args=(agent_name, stop_animation))
    
    try:
        animation_thread.start()
        result = func(*args, **kwargs)
        return result
    finally:
        stop_animation.set()
        animation_thread.join()

def clean_model_output(output, content_type="text"):
    """Clean model output by removing thinking tags and extracting code from markdown
    
    Args:
        output (str): Raw model output
        content_type (str): Type of content to extract ('text', 'code')
        
    Returns:
        str: Cleaned output
    """
    cleaned_output = output
    
    # Step 1: Remove thinking tags if present
    if "<think>" in output and "</think>" in output:
        cprint(f"🧠 Detected DeepSeek-R1 thinking tags, cleaning...", "yellow")
        
        # First try: Get everything after the last </think> tag
        clean_content = output.split("</think>")[-1].strip()
        
        # If that doesn't work, try removing all <think>...</think> blocks
        if not clean_content:
            import re
            clean_content = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
            
        if clean_content:
            cleaned_output = clean_content
            cprint("✅ Successfully removed thinking tags", "green")
    
    # Step 2: If code content, extract from markdown code blocks
    if content_type == "code" and "```" in cleaned_output:
        cprint("🔍 Extracting code from markdown blocks...", "yellow")
        
        try:
            import re
            # First look for python blocks
            code_blocks = re.findall(r'```python\n(.*?)\n```', cleaned_output, re.DOTALL)
            
            # If no python blocks, try any code blocks
            if not code_blocks:
                code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', cleaned_output, re.DOTALL)
                
            if code_blocks:
                # Join multiple code blocks with newlines between them
                cleaned_output = "\n\n".join(code_blocks)
                cprint("✅ Successfully extracted code from markdown", "green")
            else:
                cprint("⚠️ No code blocks found in markdown", "yellow")
        except Exception as e:
            cprint(f"❌ Error extracting code: {str(e)}", "red")
    
    return cleaned_output

def research_strategy(content):
    """Research AI: Analyzes and creates trading strategy"""
    cprint("\n🔍 Starting Research AI...", "cyan")
    cprint("🤖 Time to discover some alpha!", "yellow")
    
    output = run_with_animation(
        chat_with_model,
        "Research AI",
        RESEARCH_PROMPT, 
        content,
        RESEARCH_CONFIG  # Pass research-specific model config
    )
    
    if output:
        # Clean the output to remove thinking tags
        output = clean_model_output(output, "text")
        
        # Guard against non-string responses from model wrappers
        if not isinstance(output, str):
            try:
                from src.models.base_model import ModelResponse
                if isinstance(output, ModelResponse):
                    output = output.content or ""
                else:
                    output = str(output)
            except Exception:
                output = str(output)
        
        # Extract strategy name from output
        strategy_name = "UnknownStrategy"  # Default name
        if "STRATEGY_NAME:" in output:
            try:
                # Split by the STRATEGY_NAME: marker and get the text after it
                name_section = output.split("STRATEGY_NAME:")[1].strip()
                # Take the first line or up to the next section marker
                if "\n\n" in name_section:
                    strategy_name = name_section.split("\n\n")[0].strip()
                else:
                    strategy_name = name_section.split("\n")[0].strip()
                    
                # Clean up strategy name to be file-system friendly
                strategy_name = re.sub(r'[^\w\s-]', '', strategy_name)
                strategy_name = re.sub(r'[\s]+', '', strategy_name)
                
                cprint(f"✅ Successfully extracted strategy name: {strategy_name}", "green")
            except Exception as e:
                cprint(f"⚠️ Error extracting strategy name: {str(e)}", "yellow")
                cprint(f"🔄 Using default name: {strategy_name}", "yellow")
        else:
            cprint("⚠️ No STRATEGY_NAME found in output, using default", "yellow")
            
            # Try to generate a name based on key terms in the output
            import random
            adjectives = ["Adaptive", "Dynamic", "Quantum", "Neural", "Fractal", "Momentum", "Harmonic", "Volatility"]
            nouns = ["Breakout", "Oscillator", "Reversal", "Momentum", "Divergence", "Scalper", "Crossover", "Arbitrage"]
            strategy_name = f"{random.choice(adjectives)}{random.choice(nouns)}"
            cprint(f"🎲 Generated random strategy name: {strategy_name}", "yellow")
        
        # Save research output
        filepath = RESEARCH_DIR / f"{strategy_name}_strategy.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"📝 Research AI found something spicy! Saved to {filepath} 🌶️", "green")
        cprint(f"🏷️ Generated strategy name: {strategy_name}", "yellow")
        return output, strategy_name
    return None, None

def create_backtest(strategy, strategy_name="UnknownStrategy"):
    """Backtest AI: Creates backtest implementation"""
    cprint("\n📊 Starting Backtest AI...", "cyan")
    cprint("💰 Let's turn that strategy into profits!", "yellow")
    
    output = run_with_animation(
        chat_with_model,
        "Backtest AI",
        BACKTEST_PROMPT,
        f"Create a backtest for this strategy:\n\n{strategy}",
        BACKTEST_CONFIG  # Pass backtest-specific model config
    )
    
    if output:
        # Clean the output and extract code from markdown
        output = clean_model_output(output, "code")
        if not isinstance(output, str):
            try:
                from src.models.base_model import ModelResponse
                if isinstance(output, ModelResponse):
                    output = output.content or ""
                else:
                    output = str(output)
            except Exception:
                output = str(output)
        
        filepath = BACKTEST_DIR / f"{strategy_name}_BT.py"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"🔥 Backtest AI cooked up some heat! Saved to {filepath} 🚀", "green")
        return output
    return None

def debug_backtest(backtest_code, strategy=None, strategy_name="UnknownStrategy"):
    """Debug AI: Fixes technical issues in backtest code"""
    cprint("\n🔧 Starting Debug AI...", "cyan")
    cprint("🔍 Time to squash some bugs!", "yellow")
    
    context = f"Here's the backtest code to debug:\n\n{backtest_code}"
    if strategy:
        context += f"\n\nOriginal strategy for reference:\n{strategy}"
    
    output = run_with_animation(
        chat_with_model,
        "Debug AI",
        DEBUG_PROMPT,
        context,
        DEBUG_CONFIG  # Pass debug-specific model config
    )
    
    if output:
        # Clean the output and extract code from markdown
        output = clean_model_output(output, "code")
        if not isinstance(output, str):
            try:
                from src.models.base_model import ModelResponse
                if isinstance(output, ModelResponse):
                    output = output.content or ""
                else:
                    output = str(output)
            except Exception:
                output = str(output)
            
        filepath = FINAL_BACKTEST_DIR / f"{strategy_name}_BTFinal.py"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"🔧 Debug AI fixed the code! Saved to {filepath} ✨", "green")
        return output
    return None

def package_check(backtest_code, strategy_name="UnknownStrategy"):
    """Package AI: Ensures correct indicator packages are used"""
    cprint("\n📦 Starting Package AI...", "cyan")
    cprint("🔍 Checking for proper indicator imports!", "yellow")
    
    output = run_with_animation(
        chat_with_model,
        "Package AI",
        PACKAGE_PROMPT,
        f"Check and fix indicator packages in this code:\n\n{backtest_code}",
        PACKAGE_CONFIG  # Pass package-specific model config
    )
    
    if output:
        # Clean the output and extract code from markdown
        output = clean_model_output(output, "code")
        if not isinstance(output, str):
            try:
                from src.models.base_model import ModelResponse
                if isinstance(output, ModelResponse):
                    output = output.content or ""
                else:
                    output = str(output)
            except Exception:
                output = str(output)
            
        filepath = PACKAGE_DIR / f"{strategy_name}_PKG.py"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        cprint(f"📦 Package AI optimized the imports! Saved to {filepath} ✨", "green")
        return output
    return None

def get_idea_content(idea_url: str) -> str:
    """Extract content from a trading idea URL or text"""
    print("\n📥 Extracting content from idea...")
    
    try:
        if "youtube.com" in idea_url or "youtu.be" in idea_url:
            # Extract video ID from URL
            if "v=" in idea_url:
                video_id = idea_url.split("v=")[1].split("&")[0]
            else:
                video_id = idea_url.split("/")[-1].split("?")[0]
            
            print("🎥 Detected YouTube video, fetching transcript...")
            transcript = get_youtube_transcript(video_id)
            if transcript:
                print("✅ Successfully extracted YouTube transcript!")
                return f"YouTube Strategy Content:\n\n{transcript}"
            else:
                raise ValueError("Failed to extract YouTube transcript")
                
        elif idea_url.endswith(".pdf"):
            print("📚 Detected PDF file, extracting text...")
            pdf_text = get_pdf_text(idea_url)
            if pdf_text:
                print("✅ Successfully extracted PDF content!")
                return f"PDF Strategy Content:\n\n{pdf_text}"
            else:
                raise ValueError("Failed to extract PDF text")
                
        else:
            print("📝 Using raw text input...")
            return f"Text Strategy Content:\n\n{idea_url}"
            
    except Exception as e:
        print(f"❌ Error extracting content: {str(e)}")
        raise

def get_idea_hash(idea: str) -> str:
    """Generate a unique hash for an idea to track processing status"""
    # Create a hash of the idea to use as a unique identifier
    return hashlib.md5(idea.encode('utf-8')).hexdigest()

def is_idea_processed(idea: str) -> bool:
    """Check if an idea has already been processed"""
    if not PROCESSED_IDEAS_LOG.exists():
        return False
        
    idea_hash = get_idea_hash(idea)
    
    with open(PROCESSED_IDEAS_LOG, 'r', encoding='utf-8') as f:
        processed_hashes = [line.strip().split(',')[0] for line in f if line.strip()]
        
    return idea_hash in processed_hashes

def log_processed_idea(idea: str, strategy_name: str = "Unknown") -> None:
    """Log an idea as processed with timestamp and strategy name"""
    idea_hash = get_idea_hash(idea)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the log file if it doesn't exist
    if not PROCESSED_IDEAS_LOG.exists():
        PROCESSED_IDEAS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_IDEAS_LOG, 'w', encoding='utf-8') as f:
            f.write("# Moon Dev's RBI AI - Processed Ideas Log\n")
            f.write("# Format: hash,timestamp,strategy_name,idea_snippet\n")

    # Append the processed idea to the log
    with open(PROCESSED_IDEAS_LOG, 'a', encoding='utf-8') as f:
        # Truncate idea if too long for the log
        idea_snippet = idea[:100] + ('...' if len(idea) > 100 else '')
        f.write(f"{idea_hash},{timestamp},{strategy_name},{idea_snippet}\n")
    
    cprint(f"📝 Idea logged as processed: {idea_hash}", "green")

def process_trading_idea(idea: str) -> None:
    """Process a single trading idea completely independently"""
    print("\n🚀 Moon Dev's RBI AI Processing New Idea!")
    print("🌟 Let's find some alpha in the chaos!")
    print(f"📝 Processing idea: {idea[:100]}...")
    print(f"📅 Saving results to today's folder: {TODAY_DATE}")
    
    try:
        # Step 1: Extract content from the idea
        idea_content = get_idea_content(idea)
        if not idea_content:
            print("❌ Failed to extract content from idea!")
            return
            
        print(f"📄 Extracted content length: {len(idea_content)} characters")
        
        # Phase 1: Research with isolated content
        print("\n🧪 Phase 1: Research")
        strategy, strategy_name = research_strategy(idea_content)
        
        if not strategy:
            print("❌ Research phase failed!")
            return
            
        print(f"🏷️ Strategy Name: {strategy_name}")
        
        # Log the idea as processed once we have a strategy name
        log_processed_idea(idea, strategy_name)
        
        # Save research output
        research_file = RESEARCH_DIR / f"{strategy_name}_strategy.txt"
        with open(research_file, 'w', encoding='utf-8') as f:
            f.write(strategy)

        # Phase 2: Backtest using only the research output
        print("\n📈 Phase 2: Backtest")
        backtest = create_backtest(strategy, strategy_name)

        if not backtest:
            print("❌ Backtest phase failed!")
            return

        # Save backtest output
        backtest_file = BACKTEST_DIR / f"{strategy_name}_BT.py"
        with open(backtest_file, 'w', encoding='utf-8') as f:
            f.write(backtest)

        # Phase 3: Package Check using only the backtest code
        print("\n📦 Phase 3: Package Check")
        package_checked = package_check(backtest, strategy_name)

        if not package_checked:
            print("❌ Package check failed!")
            return

        # Save package check output
        package_file = PACKAGE_DIR / f"{strategy_name}_PKG.py"
        with open(package_file, 'w', encoding='utf-8') as f:
            f.write(package_checked)

        # Phase 4: Debug using only the package-checked code
        print("\n🔧 Phase 4: Debug")
        final_backtest = debug_backtest(package_checked, strategy, strategy_name)

        if not final_backtest:
            print("❌ Debug phase failed!")
            return

        # Save final backtest
        final_file = FINAL_BACKTEST_DIR / f"{strategy_name}_BTFinal.py"
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(final_backtest)
            
        print("\n🎉 Mission Accomplished!")
        print(f"🚀 Strategy '{strategy_name}' is ready to make it rain! 💸")
        print(f"✨ Final backtest saved at: {final_file}")
        
    except Exception as e:
        print(f"\n❌ Error processing idea: {str(e)}")
        raise

def main():
    """Main function to process ideas from file"""
    # We keep ideas.txt in the main RBI directory, not in the date folder
    ideas_file = DATA_DIR / "ideas.txt"
    
    if not ideas_file.exists():
        cprint("❌ ideas.txt not found! Creating template...", "red")
        ideas_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ideas_file, 'w') as f:
            f.write("# Add your trading ideas here (one per line)\n")
            f.write("# Can be YouTube URLs, PDF links, or text descriptions\n")
        return
        
    with open(ideas_file, 'r', encoding='utf-8', errors='ignore') as f:
        ideas = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
    total_ideas = len(ideas)
    cprint(f"\n🎯 Found {total_ideas} trading ideas to process", "cyan")
    
    # Count how many ideas have already been processed
    already_processed = sum(1 for idea in ideas if is_idea_processed(idea))
    new_ideas = total_ideas - already_processed
    
    cprint(f"🔍 Status: {already_processed} already processed, {new_ideas} new ideas", "cyan")
    
    # Optional: limit number of ideas via env var (for quick debugging)
    max_ideas_env = os.getenv("RBI_MAX_IDEAS")
    max_ideas = int(max_ideas_env) if max_ideas_env and max_ideas_env.isdigit() else None
    processed_count = 0

    for i, idea in enumerate(ideas, 1):
        # Check if this idea has already been processed
        if is_idea_processed(idea):
            cprint(f"\n{'='*50}", "red")
            cprint(f"⏭️  SKIPPING idea {i}/{total_ideas} - ALREADY PROCESSED", "red", attrs=['reverse'])
            idea_snippet = idea[:100] + ('...' if len(idea) > 100 else '')
            cprint(f"📝 Idea: {idea_snippet}", "red")
            cprint(f"{'='*50}\n", "red")
            continue
            
        cprint(f"\n{'='*50}", "yellow")
        cprint(f"🌙 Processing idea {i}/{total_ideas}", "cyan")
        cprint(f"📝 Idea content: {idea[:100]}{'...' if len(idea) > 100 else ''}", "yellow")
        cprint(f"{'='*50}\n", "yellow")
        
        try:
            # Process each idea in complete isolation
            process_trading_idea(idea)
            
            # Clear separator between ideas
            cprint(f"\n{'='*50}", "green")
            cprint(f"✅ Completed idea {i}/{total_ideas}", "green")
            cprint(f"{'='*50}\n", "green")
            
            # Break between ideas
            if i < total_ideas:
                cprint("😴 Taking a break before next idea...", "yellow")
                time.sleep(5)
            processed_count += 1
            if max_ideas and processed_count >= max_ideas:
                cprint("🛑 Reached RBI_MAX_IDEAS limit, exiting after quick debug run.", "yellow")
                break
                
        except Exception as e:
            cprint(f"\n❌ Error processing idea {i}: {str(e)}", "red")
            cprint("🔄 Continuing with next idea...\n", "yellow")
            continue

if __name__ == "__main__":
    try:
        cprint(f"\n🌟 Moon Dev's RBI AI Starting Up!", "green")
        cprint(f"📅 Today's Date: {TODAY_DATE} - All outputs will be saved in this folder", "magenta")
        cprint(f"🧠 DeepSeek-R1 thinking tags will be automatically removed from outputs", "magenta")
        cprint(f"📋 Processed ideas log: {PROCESSED_IDEAS_LOG}", "magenta")
        cprint("\n🤖 Model Configurations:", "cyan")
        cprint(f"📚 Research: {RESEARCH_CONFIG['type']} - {RESEARCH_CONFIG['name']}", "cyan")
        cprint(f"📊 Backtest: {BACKTEST_CONFIG['type']} - {BACKTEST_CONFIG['name']}", "cyan")
        cprint(f"🔧 Debug: {DEBUG_CONFIG['type']} - {DEBUG_CONFIG['name']}", "cyan")
        cprint(f"📦 Package: {PACKAGE_CONFIG['type']} - {PACKAGE_CONFIG['name']}", "cyan")
        main()
    except KeyboardInterrupt:
        cprint("\n👋 Moon Dev's RBI AI shutting down gracefully...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
