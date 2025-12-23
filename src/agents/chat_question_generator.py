"""
🌙 Moon Dev's Chat Question Generator
Built with love by Moon Dev 🚀

This agent analyzes recent chat messages and generates relevant questions
that could be asked based on the conversation context.
"""

import sys
from pathlib import Path
# Add project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import os
import time
from datetime import datetime, timedelta
from termcolor import cprint
from dotenv import load_dotenv
import pandas as pd
from src.config import *
from src.models import model_factory
import threading

# Load environment variables from the project root
env_path = Path(project_root) / '.env'
if not env_path.exists():
    raise ValueError(f"🚨 .env file not found at {env_path}")

load_dotenv(dotenv_path=env_path)

# Configuration
MINUTES_TO_ANALYZE = 10  # How many minutes of chat history to analyze
REFRESH_INTERVAL = 30  # How often to generate new questions (seconds)
MAX_QUESTIONS = 5  # Maximum number of questions to generate
MODEL_TYPE = "claude"  # Using Claude for question generation
MODEL_NAME = "claude-3-haiku-20240307"  # Fast model for quick responses

# Question generation prompt
QUESTION_GENERATION_PROMPT = """You are an AI that analyzes live stream chat conversations and generates short, readable questions.

Given the following recent chat messages from the last {minutes} minutes, generate up to {max_questions} questions that are:
1. MAXIMUM 10 WORDS EACH
2. Clear and easy to read
3. Based on what people are discussing
4. Relevant to the conversation

Examples of good questions:
- "What's the best Python library for backtesting strategies?"
- "How do you handle API rate limits?"
- "Which VPS provider do you recommend for bots?"
- "What risk management rules do you follow?"
- "How to start with algo trading?"

Recent chat messages:
{chat_messages}

Generate SHORT questions (10 words max!) that viewers might ask. Format each question on a new line starting with "Q:".
"""

class ChatQuestionGenerator:
    def __init__(self):
        """Initialize the Question Generator"""
        cprint("\n🤔 Initializing Moon Dev's Chat Question Generator...", "cyan")
        
        # Set up data directory and file paths
        self.data_dir = Path(project_root) / "src" / "data" / "chat_agent"
        self.chat_log_path = self.data_dir / "chat_history.csv"
        
        # Verify chat history exists
        if not self.chat_log_path.exists():
            raise ValueError(f"🚨 Chat history not found at {self.chat_log_path}")
        
        # Initialize model
        self.model_factory = model_factory
        self.model = self.model_factory.get_model(MODEL_TYPE, MODEL_NAME)
        
        if not self.model:
            raise ValueError(f"🚨 Could not initialize {MODEL_TYPE} {MODEL_NAME} model!")
        
        cprint(f"✅ Using model: {MODEL_TYPE} - {MODEL_NAME}", "green")
        cprint("🎯 Chat Question Generator initialized!", "green")
        
        self.running = False
        
    def get_recent_messages(self, minutes=MINUTES_TO_ANALYZE):
        """Get chat messages from the last X minutes"""
        try:
            # Read chat history
            df = pd.read_csv(self.chat_log_path)
            
            if df.empty:
                return []
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get current time and calculate cutoff
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=minutes)
            
            # Filter messages from the last X minutes
            recent_df = df[df['timestamp'] > cutoff_time]
            
            # Format messages as "username: message"
            messages = []
            for _, row in recent_df.iterrows():
                # Skip negative messages (score -1)
                if 'score' in df.columns and row.get('score', 0) == -1:
                    continue
                messages.append(f"{row['user']}: {row['message']}")
            
            return messages
            
        except Exception as e:
            cprint(f"❌ Error reading recent messages: {str(e)}", "red")
            return []
    
    def generate_questions(self, messages):
        """Generate questions based on recent chat messages"""
        if not messages:
            return []
            
        try:
            # Format messages for the prompt
            chat_messages = "\n".join(messages[-50:])  # Limit to last 50 messages
            
            # Create the prompt
            prompt = QUESTION_GENERATION_PROMPT.format(
                minutes=MINUTES_TO_ANALYZE,
                max_questions=MAX_QUESTIONS,
                chat_messages=chat_messages
            )
            
            # Generate questions using AI
            response = self.model.generate_response(
                system_prompt="You are a helpful AI that generates relevant questions based on chat conversations.",
                user_content=prompt,
                temperature=0.8,
                max_tokens=300
            )
            
            # Extract questions from response
            questions = []
            for line in response.content.strip().split('\n'):
                if line.strip().startswith('Q:'):
                    question = line[2:].strip()
                    if question:
                        questions.append(question)
            
            return questions[:MAX_QUESTIONS]
            
        except Exception as e:
            cprint(f"❌ Error generating questions: {str(e)}", "red")
            return []
    
    def display_questions(self, questions, username_context=None):
        """Display generated questions with nice formatting"""
        if not questions:
            cprint("📭 No questions generated (might need more chat activity)", "yellow")
            return
            
        # Header
        print("\n" + "="*60)
        cprint("🎯 SUGGESTED QUESTIONS BASED ON RECENT CHAT", "cyan", attrs=['bold'])
        print("="*60)
        
        if username_context:
            cprint(f"Context: Recent messages from viewers", "green")
            print()
        
        # Display questions
        for i, question in enumerate(questions, 1):
            print(f"  {i}. ", end="")
            cprint(question, "white", attrs=['bold'])
            print()  # Add line break between questions
            
        print("="*60 + "\n")
    
    def run_continuous(self):
        """Run continuously, generating questions at intervals"""
        self.running = True
        cprint(f"\n🚀 Starting continuous question generation...", "cyan")
        cprint(f"📊 Analyzing last {MINUTES_TO_ANALYZE} minutes of chat", "cyan")
        cprint(f"⏰ Refreshing every {REFRESH_INTERVAL} seconds", "cyan")
        print()
        
        while self.running:
            try:
                # Get recent messages
                messages = self.get_recent_messages()
                
                if messages:
                    cprint(f"📝 Analyzing {len(messages)} recent messages...", "cyan")
                    
                    # Generate questions
                    questions = self.generate_questions(messages)
                    
                    # Display questions
                    self.display_questions(questions)
                else:
                    cprint("⏳ Waiting for more chat messages...", "yellow")
                
                # Wait before next generation
                time.sleep(REFRESH_INTERVAL)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"❌ Error in continuous run: {str(e)}", "red")
                time.sleep(REFRESH_INTERVAL)
    
    def stop(self):
        """Stop the continuous generation"""
        self.running = False

def main():
    """Main entry point"""
    try:
        generator = ChatQuestionGenerator()
        generator.run_continuous()
    except KeyboardInterrupt:
        cprint("\n👋 Question Generator shutting down...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")

if __name__ == "__main__":
    main()