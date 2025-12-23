#!/usr/bin/env python3
"""
🌙 Moon Dev's Real-Time Clips Agent
Built with love by Moon Dev 🚀

AI-Powered OBS Clip Creator with Model Factory Integration
Automatically finds the best moments from your live streams and names your clips!

Works with ALL models through model_factory: Claude, GPT, DeepSeek, Groq, Grok, Ollama
"""

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

# AUTONOMOUS MODE: Set to True to auto-clip every N minutes
AUTONOMOUS = True

# TWITTER AUTO-POST: Set to True to open Twitter compose with title after each clip
TWITTER = True

# OBS Recording Folder
OBS_FOLDER = '/Volumes/Moon 26/OBS'

# Clips Output Folder (will be created in src/data/)
CLIPS_BASE_FOLDER = 'realtime_clips'

# Autonomous Settings
AUTO_CLIP_INTERVAL = 120  # Check every 2 minutes (120 seconds)
AUTO_CLIP_LENGTH = 2  # Analyze the last 2 minutes

# AI Model Configuration (via Model Factory)
# Available types: 'groq', 'openai', 'claude', 'deepseek', 'xai', 'ollama'
# Groq is recommended for speed! (Default: llama-3.3-70b-versatile)
AI_MODEL_TYPE = 'xai'
AI_MODEL_NAME = None  # None = use default for model type, or specify: 'llama-3.3-70b-versatile', 'gpt-4o', etc.

# AI Decision Prompt - Should we clip this?
DECISION_PROMPT = """You are analyzing a video transcript segment that has ALREADY been trimmed to the best content by another AI.

Your job: Rate if this trimmed segment is worth saving as a clip (1-5):

## Rating Scale:

**5 - Excellent Clip**
- Starts immediately with interesting content (no rambling intro)
- Teaching something valuable with clear insights
- Telling an engaging story or example
- Funny, entertaining, or highly engaging throughout
- Example: "so here's why this strategy failed spectacularly... [explains specific lesson]"

**4 - Good Clip**
- Strong content but maybe takes a moment to get going
- Interesting insights or useful information
- Engaging but could be tighter
- Example: "okay so... [brief pause] here's what I learned about order flow..."

**3 - Mediocre**
- Some useful content but mixed with too much filler
- Takes too long to get to the point
- Interesting topic but poorly delivered
- Example: "um, so yeah, I was thinking about... well... the thing is..."

**2 - Weak**
- Mostly filler with minimal content
- Vague or unclear points
- More "um" and "uh" than actual information
- Example: "so uh... yeah... I mean... you know what I'm saying?"

**1 - Not Worth Clipping**
- Dead air, silence, or pure filler
- No actual content or insight
- Random off-topic chit-chat
- Technical difficulties or interruptions
- Example: "[silence]... uh... sorry what was I saying?... [more silence]"

## What Makes a Great Clip:
- Starts with someone SPEAKING about something interesting
- Gets to the point quickly
- Informal, conversational, but substantive
- Funny, insightful, or educational
- Makes you think "I'd want to share this"

## Output Format
Return ONLY valid JSON with no markdown:
{"score": 4, "reason": "brief explanation of why this score"}

ONLY give scores of 4 or 5 to clips that are genuinely worth keeping. Be strict - most random segments should be 1-3."""

# AI Trim Analysis Prompt
TRIM_PROMPT = """You are a video editor analyzing a transcript to find the best content.

## Goal
Find the single best continuous segment that:
- Avoids dead pauses, long silences, excessive "um", "uh" filler words
- Avoids random off-topic chit-chat at the start/end
- Keeps the core valuable content (whatever the topic is)
- Is at least 30 seconds long (minimum)
- Has a clear start and end point

## Important
- DO NOT reject content for being off-topic - keep whatever is being discussed
- If the entire clip is good content, return the full duration
- Only trim if there's genuinely bad dead air or filler at start/end
- MAKE SURE YOU START AT A GOOD TALKING PART.

## Output Format
Return ONLY valid JSON with no markdown formatting:
{"start_time": 123.5, "end_time": 456.7, "reason": "brief explanation"}

Times must be in seconds (float). The reason should be 1 short sentence.

ENSURE YOU ONLY KEEP PARTS OF THE CLIP THAT ARE GOOD CONTENT. IT MUST START AT A GOOD TALKING PART.
"""

# AI Clip Title Prompt
TITLE_PROMPT = """You are a video title writer for algo trading bootcamp clips.

Based on the video transcript, write 1 sentence that describes what happens in this clip.

Rules:
- keep it casual and lowercase
- no punctuation at the end
- specific to what's actually happening in the video
- make it searchable/descriptive

Output ONLY the title sentence, nothing else."""

# ============================================================================
# END CONFIGURATION
# ============================================================================

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from termcolor import cprint
import re
import whisper
import time
import threading
import webbrowser
from urllib.parse import quote

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_factory import model_factory

# Setup paths relative to src/data/
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / CLIPS_BASE_FOLDER
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load Whisper model (runs locally, free!)
cprint("🔧 Loading Whisper model (this only happens once)...", "cyan")
whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
cprint("✅ Whisper model loaded!", "green")

class RealtimeClipsAgent:
    """AI-powered real-time clip creator using Moon Dev's Model Factory"""

    def __init__(self):
        self.obs_folder = Path(OBS_FOLDER)
        self.base_clips_folder = DATA_DIR
        self.base_clips_folder.mkdir(exist_ok=True)

        # Initialize AI model via model factory
        cprint(f"\n🤖 Initializing AI model: {AI_MODEL_TYPE}", "cyan")
        self.model = model_factory.get_model(AI_MODEL_TYPE, AI_MODEL_NAME)

        if not self.model:
            cprint(f"❌ Failed to initialize {AI_MODEL_TYPE} model!", "red")
            cprint("Available models:", "yellow")
            for model_type in model_factory._models.keys():
                cprint(f"  - {model_type}", "yellow")
            sys.exit(1)

        cprint(f"✅ Using model: {self.model.model_name}", "green")

    def get_todays_folder(self):
        """Get today's date folder (MM-DD-YYYY format)."""
        date_str = datetime.now().strftime("%m-%d-%Y")
        today_folder = self.base_clips_folder / date_str
        today_folder.mkdir(exist_ok=True)
        return today_folder

    @property
    def clips_folder(self):
        """Always return today's date folder."""
        return self.get_todays_folder()

    def find_current_recording(self):
        """Find the most recent .mov file (currently recording)."""
        cprint("🔍 Looking for current OBS recording...", "cyan")
        mov_files = list(self.obs_folder.glob('*.mov'))

        if not mov_files:
            cprint("❌ No .mov files found in OBS folder", "red")
            return None

        # Sort by modification time, most recent first
        mov_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        recording = mov_files[0]
        cprint(f"✅ Found recording: {recording.name}", "green")
        return recording

    def get_video_duration(self, video_path):
        """Get video duration in seconds using ffprobe."""
        cprint("⏱️  Getting video duration...", "cyan")
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            cprint(f"✅ Video duration: {duration/60:.1f} minutes ({duration:.1f} seconds)", "green")
            return duration
        except Exception as e:
            cprint(f"❌ Error getting duration: {e}", "red")
            return None

    def extract_initial_clip(self, recording, minutes):
        """Extract initial clip of requested length."""
        cprint(f"\n{'='*80}", "yellow")
        cprint(f"STEP 1: EXTRACTING {minutes}-MINUTE CLIP", "yellow")
        cprint(f"{'='*80}", "yellow")

        # Get video duration
        duration = self.get_video_duration(recording)

        if duration is None:
            cprint("❌ Cannot proceed without video duration", "red")
            return None

        if duration < minutes * 60:
            cprint(f"❌ Recording too short!", "red")
            cprint(f"   Need: {minutes} minutes ({minutes*60} seconds)", "red")
            cprint(f"   Have: {duration/60:.1f} minutes ({duration:.1f} seconds)", "red")
            return None

        # Calculate start time (last N minutes)
        start_time = duration - (minutes * 60)
        cprint(f"📍 Clip start time: {start_time:.1f} seconds from beginning", "cyan")
        cprint(f"📍 Clip end time: {duration:.1f} seconds (end of recording)", "cyan")

        # Generate temp output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = self.clips_folder / f"temp_{minutes}min_{timestamp}.mov"

        cprint(f"💾 Saving initial clip to: {temp_file.name}", "cyan")
        cprint("⚙️  Running ffmpeg (this may take a minute)...", "cyan")

        # Extract clip using ffmpeg
        # Use codec copy for speed (no re-encoding)
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Seek before input
            '-i', str(recording),
            '-t', str(minutes * 60),
            '-c', 'copy',  # Copy codec (fast, no re-encoding)
            '-y',
            str(temp_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            cprint(f"✅ Initial clip extracted successfully!", "green")
            return temp_file
        except subprocess.CalledProcessError as e:
            cprint(f"❌ FFmpeg error: {e}", "red")
            cprint(f"   stderr: {e.stderr}", "red")
            return None
        except Exception as e:
            cprint(f"❌ FFmpeg error: {e}", "red")
            return None

    def transcribe_clip(self, video_file):
        """Transcribe audio with timestamps using local Whisper."""
        cprint(f"\n{'='*80}", "yellow")
        cprint(f"STEP 2: TRANSCRIBING AUDIO (LOCAL WHISPER)", "yellow")
        cprint(f"{'='*80}", "yellow")
        cprint("🎤 Running Whisper locally (free, no API costs)...", "cyan")
        cprint("   (This may take 30-60 seconds depending on clip length)", "cyan")

        try:
            # Transcribe using local Whisper model
            result = whisper_model.transcribe(
                str(video_file),
                verbose=False,
                word_timestamps=True
            )

            cprint(f"✅ Transcription complete!", "green")
            cprint(f"📝 Total segments: {len(result['segments'])}", "cyan")
            cprint(f"📝 Full text length: {len(result['text'])} characters", "cyan")

            # Show first few words
            preview = result['text'][:150]
            cprint(f"\n📄 Transcript preview (first 150 chars):", "cyan")
            cprint(f"   \"{preview}...\"", "yellow")

            return result

        except Exception as e:
            cprint(f"❌ Transcription failed: {e}", "red")
            return None

    def get_segment_text(self, transcript_obj, start_time, end_time):
        """Extract text from transcript for a specific time segment."""
        segment_texts = []
        for seg in transcript_obj['segments']:
            # Include segments that overlap with our time range
            if seg['end'] >= start_time and seg['start'] <= end_time:
                segment_texts.append(seg['text'])
        return " ".join(segment_texts)

    def chat_with_ai(self, system_prompt, user_content):
        """Send prompt to AI model via model factory"""
        try:
            response = self.model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.7,
                max_tokens=1024
            )

            if hasattr(response, 'content'):
                return response.content
            return str(response)

        except Exception as e:
            cprint(f"❌ AI model error: {e}", "red")
            return None

    def decide_if_worth_clipping(self, transcript_text):
        """Use AI to rate if this segment is worth clipping (1-5 scale)."""
        cprint(f"\n{'='*80}", "yellow")
        cprint(f"STEP 4: AI RATING - HOW GOOD IS THIS CLIP? (1-5)", "yellow")
        cprint(f"{'='*80}", "yellow")

        # Show transcript preview
        preview = transcript_text[:300]
        cprint(f"📄 Transcript preview:", "cyan")
        cprint(f"   \"{preview}...\"", "yellow")
        cprint(f"\n🤖 Asking {self.model.model_name} to rate this clip (1-5)...", "cyan")

        try:
            result_text = self.chat_with_ai(
                DECISION_PROMPT,
                f"Video transcript:\n\n{transcript_text}\n\nRate this clip 1-5:"
            )

            if not result_text:
                cprint("❌ AI returned no response", "red")
                return False, "AI decision failed"

            cprint(f"\n🤖 AI Response:", "cyan")
            cprint(f"   {result_text}", "yellow")

            # Parse JSON response (remove markdown if present)
            clean_result = result_text.strip()
            if clean_result.startswith('```'):
                clean_result = re.sub(r'```json?\n?', '', clean_result)
                clean_result = re.sub(r'```', '', clean_result)
                clean_result = clean_result.strip()

            result = json.loads(clean_result)
            score = result['score']
            reason = result['reason']

            # Generate star rating visualization
            stars = "⭐" * score + "☆" * (5 - score)

            cprint(f"\n📊 SCORE: {score}/5 {stars}", "cyan")
            cprint(f"   Reason: {reason}", "yellow")

            # Only clip if score is 4 or 5
            should_clip = score >= 4

            if should_clip:
                cprint(f"\n✅ DECISION: CLIP IT! (Score {score}/5)", "green")
            else:
                cprint(f"\n❌ DECISION: SKIP IT (Score {score}/5 - need 4+)", "red")

            return should_clip, reason

        except json.JSONDecodeError as e:
            cprint(f"❌ AI returned invalid JSON: {e}", "red")
            cprint(f"   Raw response: {result_text}", "red")
            # Default to skipping if AI fails (safer than clipping everything)
            cprint(f"⚠️  Defaulting to SKIP (AI error)", "yellow")
            return False, "AI decision failed"
        except Exception as e:
            cprint(f"❌ AI decision failed: {e}", "red")
            # Default to skipping if AI fails
            cprint(f"⚠️  Defaulting to SKIP (AI error)", "yellow")
            return False, "AI decision failed"

    def find_best_segment(self, transcript_obj):
        """Use AI to find the best segment to keep."""
        cprint(f"\n{'='*80}", "yellow")
        cprint(f"STEP 3: AI ANALYSIS - FINDING BEST SEGMENT", "yellow")
        cprint(f"{'='*80}", "yellow")

        # Format transcript with timestamps
        formatted = []
        for seg in transcript_obj['segments']:
            formatted.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

        transcript_text = "\n".join(formatted)

        cprint(f"🤖 Sending transcript to {self.model.model_name} for analysis...", "cyan")
        cprint(f"📊 Total duration: {transcript_obj['segments'][-1]['end']:.1f} seconds", "cyan")

        try:
            result_text = self.chat_with_ai(
                TRIM_PROMPT,
                f"Transcript with timestamps:\n\n{transcript_text}\n\nFind the best segment:"
            )

            if not result_text:
                cprint("❌ AI returned no response", "red")
                return None, None

            cprint(f"\n🤖 AI Response:", "cyan")
            cprint(f"   {result_text}", "yellow")

            # Parse JSON response (remove markdown if present)
            clean_result = result_text.strip()
            if clean_result.startswith('```'):
                clean_result = re.sub(r'```json?\n?', '', clean_result)
                clean_result = re.sub(r'```', '', clean_result)
                clean_result = clean_result.strip()

            result = json.loads(clean_result)

            start = result['start_time']
            end = result['end_time']
            reason = result['reason']

            duration = end - start
            cprint(f"\n✅ Best segment identified:", "green")
            cprint(f"   Start: {start:.1f}s", "cyan")
            cprint(f"   End: {end:.1f}s", "cyan")
            cprint(f"   Duration: {duration:.1f}s ({duration/60:.1f} minutes)", "cyan")
            cprint(f"   Reason: {reason}", "yellow")

            return start, end

        except json.JSONDecodeError as e:
            cprint(f"❌ AI returned invalid JSON: {e}", "red")
            cprint(f"   Raw response: {result_text}", "red")
            return None, None
        except Exception as e:
            cprint(f"❌ AI analysis failed: {e}", "red")
            return None, None

    def generate_title(self, transcript_text):
        """Generate a short title for the clip."""
        cprint(f"\n{'='*80}", "yellow")
        cprint(f"STEP 6: GENERATING CLIP TITLE", "yellow")
        cprint(f"{'='*80}", "yellow")

        # Show transcript preview
        preview = transcript_text[:200]
        cprint(f"📄 Transcript preview for title generation:", "cyan")
        cprint(f"   \"{preview}...\"", "yellow")
        cprint(f"\n🤖 Asking {self.model.model_name} for a short title...", "cyan")

        try:
            title = self.chat_with_ai(
                TITLE_PROMPT,
                f"Video transcript:\n\n{transcript_text}\n\nWrite a short title:"
            )

            if not title:
                cprint("❌ Title generation failed", "red")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return f"clip_{timestamp}"

            title = title.strip()
            cprint(f"✅ AI-generated title: \"{title}\"", "green")

            # Convert to filename-safe format (replace spaces with underscores, remove special chars)
            safe_title = re.sub(r'[^\w\s-]', '', title.lower())
            safe_title = re.sub(r'[\s_]+', '_', safe_title)
            safe_title = safe_title.strip('_')

            cprint(f"📝 Filename-safe version: \"{safe_title}\"", "cyan")

            return safe_title

        except Exception as e:
            cprint(f"❌ Title generation failed: {e}", "red")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"clip_{timestamp}"

    def trim_clip(self, temp_file, start_time, end_time, output_name):
        """Create final trimmed clip."""
        cprint(f"\n{'='*80}", "yellow")
        cprint(f"STEP 7: CREATING FINAL TRIMMED CLIPS (NORMAL + TALL)", "yellow")
        cprint(f"{'='*80}", "yellow")

        duration = end_time - start_time
        output_file = self.clips_folder / f"{output_name}.mov"
        output_file_tall = self.clips_folder / f"tall_{output_name}.mov"

        cprint(f"✂️  Trimming video:", "cyan")
        cprint(f"   From: {start_time:.1f}s", "cyan")
        cprint(f"   To: {end_time:.1f}s", "cyan")
        cprint(f"   Duration: {duration:.1f}s ({duration/60:.1f} minutes)", "cyan")

        # Create normal version
        cprint(f"\n📹 Creating normal version...", "cyan")
        cprint(f"💾 Output: {output_file.name}", "cyan")
        cprint("⚙️  Running ffmpeg...", "cyan")

        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', str(temp_file),
            '-t', str(duration),
            '-c', 'copy',
            '-y',
            str(output_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            cprint(f"✅ Normal clip created successfully!", "green")
        except subprocess.CalledProcessError as e:
            cprint(f"❌ Normal clip failed: {e}", "red")
            cprint(f"   stderr: {e.stderr}", "red")
            temp_file.unlink()
            return None
        except Exception as e:
            cprint(f"❌ Normal clip failed: {e}", "red")
            temp_file.unlink()
            return None

        # Create tall version (9:16 vertical format for Twitter/TikTok/Reels)
        cprint(f"\n📱 Creating TALL version (9:16 vertical - 1080x1920)...", "cyan")
        cprint(f"💾 Output: {output_file_tall.name}", "cyan")
        cprint("⚙️  Running ffmpeg with 9:16 split + stack...", "cyan")

        # FFmpeg filter: Split horizontally, zoom to fill, stack vertically to 9:16
        # Right half on top, left half on bottom - zoomed to fill width with no black bars
        filter_complex = (
            "[0:v]split=2[left][right];"
            "[left]crop=iw/2:ih:0:0,scale=1080:-1,crop=1080:960[left_crop];"
            "[right]crop=iw/2:ih:ow:0,scale=1080:-1,crop=1080:960[right_crop];"
            "[right_crop][left_crop]vstack,setsar=1[out]"
        )

        cmd_tall = [
            'ffmpeg',
            '-i', str(output_file),
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-map', '0:a?',
            '-c:v', 'h264_videotoolbox',
            '-b:v', '5M',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            str(output_file_tall)
        ]

        try:
            result = subprocess.run(cmd_tall, capture_output=True, text=True, check=True)
            cprint(f"✅ TALL clip created successfully!", "green")

            # Check file size
            file_size = output_file_tall.stat().st_size
            cprint(f"📦 File size: {file_size / (1024*1024):.2f} MB", "cyan")

            # Delete temp file
            temp_file.unlink()
            cprint(f"🗑️  Deleted temporary file", "cyan")

            return output_file

        except subprocess.CalledProcessError as e:
            cprint(f"❌ TALL clip failed: {e}", "red")
            cprint(f"📄 STDERR: {e.stderr}", "red")
            cprint(f"⚠️  Normal clip still saved, continuing...", "yellow")
            temp_file.unlink()
            return output_file
        except Exception as e:
            cprint(f"❌ TALL clip failed: {e}", "red")
            cprint(f"⚠️  Normal clip still saved, continuing...", "yellow")
            temp_file.unlink()
            return output_file

    def create_clip(self, minutes):
        """Main workflow: extract, transcribe, analyze, trim, name."""
        cprint(f"\n{'='*80}", "green")
        cprint(f"🎬 STARTING CLIP CREATION WORKFLOW", "green")
        cprint(f"{'='*80}", "green")
        cprint(f"Requested length: {minutes} minutes", "cyan")
        cprint(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan")

        # Step 1: Find and extract initial clip
        recording = self.find_current_recording()
        if not recording:
            return

        temp_file = self.extract_initial_clip(recording, minutes)
        if not temp_file:
            return

        # Step 2: Transcribe
        transcript_obj = self.transcribe_clip(temp_file)
        if not transcript_obj:
            temp_file.unlink()
            return

        # Step 3: Find best segment FIRST
        start_time, end_time = self.find_best_segment(transcript_obj)
        if start_time is None or end_time is None or start_time == end_time or (end_time - start_time) < 10:
            cprint("⚠️  AI returned invalid times, using full clip", "yellow")
            start_time = 0
            end_time = transcript_obj['segments'][-1]['end']

        # Step 4: Extract just the trimmed segment's text for rating
        trimmed_text = self.get_segment_text(transcript_obj, start_time, end_time)

        # Step 5: Rate the TRIMMED segment (not the full transcript)
        should_clip, reason = self.decide_if_worth_clipping(trimmed_text)

        if not should_clip:
            cprint(f"\n{'='*80}", "red")
            cprint(f"⏭️  SKIPPING THIS SEGMENT", "red")
            cprint(f"{'='*80}", "red")
            cprint(f"Reason: {reason}", "yellow")
            cprint(f"{'='*80}\n", "red")
            temp_file.unlink()
            return

        # Step 6: Generate title
        title = self.generate_title(trimmed_text)

        # Step 7: Create final clip
        final_file = self.trim_clip(temp_file, start_time, end_time, title)

        if final_file:
            cprint(f"\n{'='*80}", "green")
            cprint(f"✅ CLIP CREATION COMPLETE!", "green")
            cprint(f"{'='*80}", "green")
            cprint(f"📁 Normal: {final_file}", "cyan")
            cprint(f"📱 Tall: {self.clips_folder / f'tall_{title}.mov'}", "cyan")
            cprint(f"🎬 Title: {title}", "cyan")
            cprint(f"⏱️  Duration: {(end_time - start_time)/60:.1f} minutes", "cyan")
            cprint(f"{'='*80}\n", "green")

            # Open Twitter if enabled
            if TWITTER:
                self.open_twitter_compose(title, final_file)

            return final_file

        return None

    def open_twitter_compose(self, title, video_file):
        """Open Twitter compose window with title pre-filled."""
        cprint(f"\n🐦 TWITTER AUTO-COMPOSE", "cyan")
        cprint(f"{'='*80}", "cyan")

        # Convert underscores to spaces for readable title
        tweet_text = title.replace('_', ' ')

        cprint(f"📝 Tweet text: \"{tweet_text}\"", "yellow")
        cprint(f"🎬 Video file: {video_file.name}", "yellow")

        # Encode for URL
        encoded_text = quote(tweet_text)
        twitter_url = f"https://twitter.com/intent/tweet?text={encoded_text}"

        cprint(f"🌐 Opening Twitter in browser...", "cyan")
        webbrowser.open(twitter_url)

        cprint(f"✅ Twitter opened!", "green")
        cprint(f"💡 Drag and drop this file into the tweet:", "yellow")
        cprint(f"   {video_file}", "yellow")
        cprint(f"{'='*80}\n", "cyan")


def print_help():
    """Print help message."""
    help_text = """
╔════════════════════════════════════════════════════════════════════════════════╗
║             🌙 MOON DEV'S REAL-TIME CLIPS AGENT 🌙                            ║
╚════════════════════════════════════════════════════════════════════════════════╝

Commands:
  3 or clip 3   Create a clip from the last 3 minutes
  5 or clip 5   Create a clip from the last 5 minutes
  10 or clip 10 Create a clip from the last 10 minutes

  /help         Show this help
  /quit         Exit

How it works:
  1. Extracts the last N minutes from your OBS recording
  2. Transcribes audio with Whisper (with timestamps)
  3. AI finds the best segment to keep
  4. AI rates that segment 1-5 (only proceeds if 4+)
  5. If 4+: AI generates a descriptive title
  6. If 4+: Creates final trimmed clip
  7. If 4+: Saves with AI-generated name

Example:
  💬 You: clip 5

  🎬 STARTING CLIP CREATION WORKFLOW
  ✂️  Extracting 5-minute clip...
  🎤 Transcribing audio...
  🤖 AI finding best segment...
  🤖 AI Rating: 5/5 ⭐⭐⭐⭐⭐ - Great content!
  ✅ DECISION: CLIP IT!
  📝 Generating title...
  ✅ CLIP COMPLETE: building_order_flow_features_for_hft_bots.mov
  ⏱️  Duration: 3.2 minutes (trimmed from 5.0)

The AI first finds the best segment, then rates it (1-5).
Only clips scoring 4+ are saved with AI-generated names!

Model Factory Integration:
  - Works with ALL Moon Dev models: Claude, GPT, DeepSeek, Groq, Grok, Ollama
  - Configure AI_MODEL_TYPE at the top of the file
  - Default: Groq (fast and cheap!)
"""
    print(help_text)


def autonomous_mode(agent):
    """Run autonomous clipping mode - checks every N minutes, AI decides if worth clipping."""
    cprint("\n" + "="*80, "green")
    cprint("🤖 AUTONOMOUS MODE ACTIVATED", "green")
    cprint("="*80, "green")
    cprint(f"⏰ Checking every {AUTO_CLIP_INTERVAL} seconds ({AUTO_CLIP_INTERVAL/60:.0f} minutes)", "cyan")
    cprint(f"✂️  Analyzing last {AUTO_CLIP_LENGTH} minutes each time", "cyan")
    cprint(f"🧠 AI decides if segment is worth clipping", "cyan")
    cprint(f"📁 Recording folder: {agent.obs_folder}", "cyan")
    cprint(f"💾 Clips saved to: {agent.base_clips_folder}", "cyan")
    cprint(f"📅 Today's folder: {agent.clips_folder.name}", "cyan")
    cprint(f"🤖 AI Model: {agent.model.model_name}", "cyan")
    cprint("\n💡 Press Ctrl+C to stop\n", "yellow")

    check_count = 0
    clips_created = 0
    clips_skipped = 0

    while True:
        try:
            check_count += 1
            check_start_time = time.time()

            cprint(f"\n{'='*80}", "green")
            cprint(f"🔄 AUTO-CHECK #{check_count} - {datetime.now().strftime('%H:%M:%S')}", "green")
            cprint(f"📊 Stats: {clips_created} clipped | {clips_skipped} skipped", "cyan")
            cprint(f"{'='*80}", "green")

            # This will either create a clip or skip it based on AI decision
            result = agent.create_clip(AUTO_CLIP_LENGTH)

            # Track if we created or skipped (based on whether result was None or not)
            if result is None:
                clips_skipped += 1
            else:
                clips_created += 1

            cprint(f"\n⏳ Sleeping for {AUTO_CLIP_INTERVAL} seconds...", "yellow")
            cprint(f"   (Processing time was ~{(time.time() - check_start_time):.0f}s)", "yellow")
            cprint(f"   Next check: {datetime.fromtimestamp(time.time() + AUTO_CLIP_INTERVAL).strftime('%H:%M:%S')}", "yellow")

            time.sleep(AUTO_CLIP_INTERVAL)

        except KeyboardInterrupt:
            cprint("\n\n🛑 Autonomous mode stopped", "red")
            cprint(f"📊 Final Stats:", "cyan")
            cprint(f"   Total checks: {check_count}", "cyan")
            cprint(f"   Clips created: {clips_created}", "green")
            cprint(f"   Clips skipped: {clips_skipped}", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Error in autonomous mode: {e}", "red")
            cprint(f"⏳ Waiting {AUTO_CLIP_INTERVAL} seconds before retrying...", "yellow")
            time.sleep(AUTO_CLIP_INTERVAL)


def interactive_loop(agent):
    """Interactive command loop."""
    while True:
        try:
            user_input = input("💬 You: ").strip()

            if not user_input:
                continue

            if user_input.startswith('/'):
                cmd = user_input.lower()

                if cmd == '/quit' or cmd == '/exit':
                    cprint("\n👋 Later, Doctor Data Dawg! 🌙\n", "cyan")
                    break

                elif cmd == '/help':
                    print_help()

                else:
                    cprint(f"⚠️  Unknown command: {cmd}", "yellow")

            elif user_input.lower().startswith('clip'):
                parts = user_input.split()
                if len(parts) == 2 and parts[1].isdigit():
                    minutes = int(parts[1])
                    agent.create_clip(minutes)
                else:
                    cprint("⚠️  Usage: clip <minutes>  (e.g., 'clip 5')", "yellow")

            elif user_input.isdigit():
                # Just a number = clip that many minutes
                minutes = int(user_input)
                cprint(f"📝 Interpreting as: clip {minutes} minutes", "cyan")
                agent.create_clip(minutes)

            else:
                cprint("⚠️  Unknown command. Try '5' or 'clip 5' or /help", "yellow")

        except KeyboardInterrupt:
            cprint("\n\n👋 Later, Doctor Data Dawg! 🌙\n", "cyan")
            break
        except Exception as e:
            cprint(f"❌ Error: {e}", "red")


def main():
    """Main entry point."""
    cprint("\n" + "="*80, "cyan")
    cprint("🌙 MOON DEV'S REAL-TIME CLIPS AGENT 🌙", "cyan")
    cprint("="*80, "cyan")

    agent = RealtimeClipsAgent()

    if AUTONOMOUS:
        # Run in autonomous mode
        autonomous_mode(agent)
    else:
        # Run in interactive mode
        cprint("\n" + "="*80, "cyan")
        cprint("✂️  AI-POWERED OBS CLIP CREATOR", "cyan")
        cprint("="*80, "cyan")
        cprint("\nCommands:", "cyan")
        cprint("  Just type a number: 3, 5, or 10  - Create AI-trimmed clips", "yellow")
        cprint("  /help                             - Show full help", "yellow")
        cprint("  /quit                             - Exit", "yellow")
        cprint(f"\nRecording folder: {agent.obs_folder}", "cyan")
        cprint(f"Clips saved to: {agent.clips_folder}", "cyan")
        cprint(f"AI Model: {agent.model.model_name}\n", "cyan")

        interactive_loop(agent)


if __name__ == "__main__":
    main()
