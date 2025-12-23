#!/usr/bin/env python
'''
🧹 Moon Dev's Ideas Cleaner 🧹
Cleans up the ideas.txt file by removing prefixes, extracting core ideas,
and ensuring consistent formatting.

Created with ❤️ by Moon Dev
'''

import re
import csv
import time
import random
import sys
from pathlib import Path
import pandas as pd
from termcolor import cprint, colored

# Fun emojis for animation
EMOJIS = ["🚀", "💫", "✨", "🌟", "💎", "🔮", "🌙", "⭐", "🌠", "💰", "📈", "🧠"]
CLEANING_EMOJIS = ["🧹", "🧼", "🧽", "✨", "💫", "🌪️", "🌀", "🔮", "🧿", "🌟"]

def animate_text(text, color="white", bg_color="on_blue", delay=0.03):
    """Animate text with a typewriter effect"""
    for char in text:
        print(colored(char, color, bg_color), end='', flush=True)
        time.sleep(delay)
    print()

def animate_progress(total, message="Cleaning", emoji="🧹"):
    """Show a fun progress bar animation"""
    for i in range(total + 1):
        percent = i * 100 // total
        bar_length = 30
        filled_length = bar_length * i // total
        
        # Choose random colors for fun
        colors = ["cyan", "magenta", "blue", "green", "yellow"]
        bg_colors = ["on_blue", "on_magenta", "on_cyan"]
        color = colors[i % len(colors)]
        bg = bg_colors[i % len(bg_colors)]
        
        # Create the progress bar
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print with random emoji
        if i % 5 == 0:
            emoji = random.choice(CLEANING_EMOJIS)
            
        print(f"\r{colored(f' {emoji} {message} [{bar}] {percent}% ', 'white', bg)}", end="", flush=True)
        time.sleep(0.05)
    print()

def animate_moon_dev():
    """Show a fun Moon Dev animation"""
    moon_dev = [
        "  __  __                         ____                 ",
        " |  \\/  |  ___    ___   _ __   |  _ \\   ___  __   __ ",
        " | |\\/| | / _ \\  / _ \\ | '_ \\  | | | | / _ \\ \\ \\ / / ",
        " | |  | || (_) || (_) || | | | | |_| ||  __/  \\ V /  ",
        " |_|  |_| \\___/  \\___/ |_| |_| |____/  \\___|   \\_/   "
    ]
    
    bg_colors = ["on_blue", "on_cyan", "on_magenta"]
    
    for i, line in enumerate(moon_dev):
        bg = bg_colors[i % len(bg_colors)]
        cprint(line, "white", bg)
        time.sleep(0.1)
    
    # Add some sparkles
    for _ in range(5):
        emoji = random.choice(EMOJIS)
        position = random.randint(0, 50)
        print(" " * position + emoji)
        time.sleep(0.05)

def clean_idea(idea):
    """Clean up a single idea text"""
    # Remove thinking tags if present (for DeepSeek-R1)
    if "<think>" in idea and "</think>" in idea:
        idea = re.sub(r'<think>.*?</think>', '', idea, flags=re.DOTALL).strip()
    
    # Extract content from markdown bold/quotes if present
    bold_match = re.search(r'\*\*\"?(.*?)\"?\*\*', idea)
    if bold_match:
        idea = bold_match.group(1).strip()
    
    # Handle common prefixes
    prefixes_to_remove = [
        'Sure', 'Sure,', 'Here\'s', 'Here is', 'I\'ll', 'I will',
        'A unique', 'One unique', 'Here\'s a', 'Here is a',
        'Trading strategy:', 'Strategy idea:', 'Trading idea:'
    ]
    
    for prefix in prefixes_to_remove:
        if idea.lower().startswith(prefix.lower()):
            idea = idea[len(prefix):].strip()
            # Remove any leading punctuation after prefix removal
            idea = idea.lstrip(',:;.- ')
    
    # Remove any markdown formatting
    idea = idea.replace('```', '').replace('#', '')
    
    # Remove any "Strategy:" or similar prefixes
    prefixes = ["Strategy:", "Idea:", "Trading Strategy:", "Trading Idea:"]
    for prefix in prefixes:
        if idea.startswith(prefix):
            idea = idea[len(prefix):].strip()
    
    # Remove quotes if they wrap the entire idea
    if (idea.startswith('"') and idea.endswith('"')) or (idea.startswith("'") and idea.endswith("'")):
        idea = idea[1:-1].strip()
    
    # Ensure it's a single line
    idea = ' '.join(idea.split())
    
    # Truncate if too long (aim for 1-2 sentences)
    sentences = re.split(r'[.!?]+', idea)
    if len(sentences) > 2:
        idea = '.'.join(sentences[:2]).strip() + '.'
    
    # Ensure first letter is capitalized
    if idea and not idea[0].isupper():
        idea = idea[0].upper() + idea[1:]
    
    return idea

def clean_ideas_file():
    """Clean up the ideas.txt and strategy_ideas.csv files"""
    # Fancy startup animation
    animate_moon_dev()
    
    # Colorful header
    print("\n" + "=" * 60)
    cprint(" 🧹 MOON DEV'S IDEA CLEANER 🧹 ", "white", "on_magenta")
    print("=" * 60 + "\n")
    
    # Define paths
    DATA_DIR = Path(__file__).parent.parent / "data" / "rbi"
    IDEAS_TXT = DATA_DIR / "ideas.txt"
    IDEAS_CSV = DATA_DIR / "strategy_ideas.csv"
    
    cprint(f"🔍 SCANNING FOR IDEAS FILES...", "white", "on_blue")
    time.sleep(0.5)
    
    if not IDEAS_TXT.exists():
        cprint(f"❌ IDEAS.TXT NOT FOUND AT {IDEAS_TXT}", "white", "on_red")
        return
    
    cprint(f"✅ FOUND IDEAS.TXT!", "white", "on_green")
    time.sleep(0.3)
    
    if IDEAS_CSV.exists():
        cprint(f"✅ FOUND STRATEGY_IDEAS.CSV!", "white", "on_green")
    else:
        cprint(f"⚠️ CSV FILE NOT FOUND - WILL ONLY CLEAN TXT FILE", "yellow", "on_blue")
    
    time.sleep(0.5)
    
    # Read existing content from ideas.txt
    cprint(f"\n📂 READING IDEAS FROM {IDEAS_TXT}", "white", "on_blue")
    with open(IDEAS_TXT, 'r') as f:
        content = f.read()
    
    # Clean up the content
    lines = content.split('\n')
    cleaned_lines = []
    
    # Show progress animation
    cprint(f"\n🧹 INITIATING CLEANING SEQUENCE...", "white", "on_cyan")
    time.sleep(0.5)
    
    # Count non-empty, non-comment lines
    valid_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    
    if not valid_lines:
        cprint(f"⚠️ NO IDEAS FOUND TO CLEAN!", "yellow", "on_red")
        return
    
    cprint(f"🔍 FOUND {len(valid_lines)} IDEAS TO PROCESS", "white", "on_blue")
    time.sleep(0.5)
    
    # Process each line with animation
    animate_progress(len(valid_lines), "Cleaning ideas")
    
    for line in lines:
        # Skip empty lines or comment lines
        if not line.strip() or line.strip().startswith('#'):
            continue
            
        # Handle YouTube URL line
        if 'youtube.com' in line:
            # Try to extract the idea part after the URL
            youtube_match = re.search(r'youtube\.com.*?%3D%3D(.*?)$', line)
            if youtube_match:
                idea = youtube_match.group(1).strip()
                cleaned_lines.append(idea)
            else:
                # If we can't extract cleanly, just skip this line
                cprint(f"⚠️ Skipping malformed YouTube URL line: {line[:50]}...", "yellow", "on_red")
            continue
            
        # Clean the idea
        cleaned_idea = clean_idea(line)
        cleaned_lines.append(cleaned_idea)
    
    # Write back the cleaned content to ideas.txt
    cprint(f"\n💾 SAVING CLEANED IDEAS...", "white", "on_blue")
    
    with open(IDEAS_TXT, 'w') as f:
        f.write('# Moon Dev\'s Trading Strategy Ideas 🌙\n')
        f.write('# One idea per line - Generated by Research Agent 🤖\n')
        f.write('# Format: Strategy idea text (1-2 sentences)\n\n')
        for line in cleaned_lines:
            f.write(f'{line}\n')
    
    # Clean up the CSV file if it exists
    if IDEAS_CSV.exists():
        cprint(f"\n📊 PROCESSING CSV FILE...", "white", "on_magenta")
        try:
            # Read the CSV file
            df = pd.read_csv(IDEAS_CSV)
            
            # Clean up the ideas in the CSV
            if 'idea' in df.columns:
                cprint(f"🔍 Found {len(df)} entries in CSV", "cyan")
                
                # Show progress animation
                animate_progress(len(df), "Cleaning CSV entries")
                
                df['idea'] = df['idea'].apply(clean_idea)
                
                # Write back the cleaned CSV
                df.to_csv(IDEAS_CSV, index=False)
                cprint(f"✅ CSV FILE CLEANED SUCCESSFULLY!", "white", "on_green")
        except Exception as e:
            cprint(f"❌ ERROR CLEANING CSV FILE: {str(e)}", "white", "on_red")
    
    # Success message with animation
    cprint("\n✨ CLEANING COMPLETE! ✨", "white", "on_green")
    
    # Show the cleaned ideas with fancy formatting
    cprint(f"\n📝 CLEANED IDEAS ({len(cleaned_lines)}):", "white", "on_blue")
    
    for i, line in enumerate(cleaned_lines, 1):
        # Alternate background colors
        bg_color = "on_blue" if i % 2 == 0 else "on_cyan"
        cprint(f" {i:2d}. {line}", "yellow", bg_color)
        
        # Add random emoji every few lines
        if i % 3 == 0:
            emoji = random.choice(EMOJIS)
            position = random.randint(0, 30)
            print(" " * position + emoji)
        
        # Small delay for visual effect
        time.sleep(0.1)
    
    # Final celebration
    print("\n" + "★" * 60)
    animate_text(" 🌟 MOON DEV'S IDEA CLEANER COMPLETED SUCCESSFULLY! 🌟 ", "white", "on_magenta", 0.01)
    print("★" * 60)
    
    # Show some celebratory emojis
    for _ in range(10):
        position = random.randint(0, 50)
        emoji = random.choice(EMOJIS)
        print(" " * position + emoji)
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        clean_ideas_file()
    except KeyboardInterrupt:
        cprint("\n👋 CLEANING PROCESS INTERRUPTED", "white", "on_yellow")
        cprint("🌙 Thank you for using Moon Dev's Idea Cleaner! 🌙", "white", "on_magenta")
    except Exception as e:
        cprint(f"\n❌ ERROR: {str(e)}", "white", "on_red")
        import traceback
        cprint(traceback.format_exc(), "red") 