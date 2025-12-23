#!/usr/bin/env python
'''
🌙 Moon Dev's Countdown Demo 🌙
A simple demonstration of the colorful countdown animation used in the Research Agent.

Created with ❤️ by Moon Dev
'''

import time
import random
from termcolor import cprint, colored

def demo_countdown(seconds=15):
    """Demonstrate the colorful countdown animation"""
    # Fun waiting animation
    cprint(f"\n⏱️ COOLDOWN PERIOD ACTIVATED", "white", "on_blue")
    
    # Show a colorful countdown for the entire interval
    moon_emojis = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    bg_colors = ["on_blue", "on_magenta", "on_cyan", "on_green"]
    text_colors = ["white", "yellow", "cyan"]
    
    print("\nCountdown with changing colors and moon phases:")
    for i in range(seconds):
        # Cycle through colors and emojis
        bg_color = bg_colors[i % len(bg_colors)]
        text_color = text_colors[i % len(text_colors)]
        left_emoji = moon_emojis[i % len(moon_emojis)]
        right_emoji = moon_emojis[(i + 4) % len(moon_emojis)]
        
        # Display countdown with changing colors
        remaining = seconds - i
        cprint(f"\r{left_emoji} Next idea in: {remaining} seconds {right_emoji}", text_color, bg_color, end="", flush=True)
        time.sleep(1)
    
    print("\n\nCountdown complete!")
    cprint("✨ MOON DEV'S COUNTDOWN DEMO FINISHED! ✨", "white", "on_green")

if __name__ == "__main__":
    print("🌙 Moon Dev's Countdown Demo 🌙")
    print("This script demonstrates the colorful countdown animation used in the Research Agent.")
    
    # Run the demo
    demo_countdown(15) 