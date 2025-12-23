"""
🌙 Moon Dev's Stream Agent
Built with love by Moon Dev 🚀

This agent continuously listens to voice, generates titles and thumbnails for livestreams.
"""

import sys
from pathlib import Path
# Add project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables from the project root
env_path = Path(project_root) / '.env'
if not env_path.exists():
    raise ValueError(f"🚨 .env file not found at {env_path}")

import os
import time
from datetime import datetime
import pyaudio
import openai
from termcolor import cprint
from dotenv import load_dotenv
import base64
from src.models import model_factory
import wave
import tempfile
import whisper

# Load .env file explicitly from project root
load_dotenv(dotenv_path=env_path)

# Configuration
RECORDING_DURATION = 120  # Duration to record for title generation
WAIT_BETWEEN_BATCHES = 600 # 15 minutes wait between batches
AUDIO_CHUNK_SIZE = 2048
SAMPLE_RATE = 16000
NUM_TITLES = 10  # Number of titles to generate
NUM_THUMBNAILS = 3  # Number of thumbnails to generate

# Output to Downloads folder
DOWNLOADS_DIR = Path.home() / "Downloads"

# Master prompts
TITLE_MASTER_PROMPT = """You are Gilfoyle from Silicon Valley generating stream titles. Based on Moon Dev's livestream transcript, generate exactly {num_titles} self-deprecating, sarcastic titles.

Rules:
- Each title MUST be 10 words or less
- Make fun of Moon Dev (the streamer) in a funny way
- Be sarcastic like Gilfoyle would be
- Self-deprecating humor about coding/trading attempts
- Short, punchy, and slightly mean but funny
- NO NUMBERING, just the titles
- Act like Moon Dev is trying too hard or missing obvious things

Example format:
Watch Me Lose Money With Bad Code
Another Failed Trading Bot Live
Overcomplicated Simple Thing Again Stream
Why Use 5 Lines When 500 Work
My Code Works (Narrator: It Didn't)
Professional Bug Creator Makes Trading Bots

TRANSCRIPT:
{transcript}"""

THUMBNAIL_MASTER_PROMPT = """Create a YouTube thumbnail for Moon Dev's coding livestream. The stream is about: {context}

IMPORTANT REQUIREMENTS:
- LANDSCAPE orientation (16:9 aspect ratio)
- Wide YouTube thumbnail format
- THERE MUST BE SOME SORT OF LINE CHART IN THE THUMBNAIL GOING UP 
- NO MORE THAN 3 ELEMENTS IN THE THUMBNAIL TO KEEP IT SIMPLE

The thumbnail should visually represent Moon Dev building trading algorithms live."""

class StreamAgent:
    def __init__(self):
        """Initialize the Stream Agent"""
        # Silent initialization
        self.downloads_dir = DOWNLOADS_DIR
        
        # Verify environment variables
        if not os.getenv("OPENAI_KEY"):
            raise ValueError(f"🚨 OPENAI_KEY not found!")
        
        # Initialize OpenAI client
        openai_key = os.getenv("OPENAI_KEY")
        self.openai_client = openai.OpenAI(api_key=openai_key)
        
        # Initialize local Whisper model
        self.whisper_model = whisper.load_model("base")
        
        # Initialize model for title generation
        self.model = model_factory.get_model("openai", "gpt-4o")
        
        self.is_recording = False
        self.current_transcript = []
        
    def record_audio(self):
        """Record audio and transcribe using Whisper API"""
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE
        )
        
        frames = []
        start_time = time.time()
        
        try:
            self.is_recording = True
            
            while time.time() - start_time < RECORDING_DURATION:
                data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
            
        except Exception as e:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            self.is_recording = False
        
        # Save audio to temporary file and transcribe
        if frames:
            try:
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_filename = temp_audio.name
                    
                    # Write WAV file
                    with wave.open(temp_filename, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(b''.join(frames))
                
                # Transcribe using local Whisper
                result = self.whisper_model.transcribe(
                    temp_filename,
                    language="en",
                    fp16=False
                )
                
                # Store transcript
                self.current_transcript = [result["text"]]
                
                # Clean up temporary file
                os.unlink(temp_filename)
                
            except Exception as e:
                if 'temp_filename' in locals():
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
    
    def generate_titles(self, transcript):
        """Generate stream titles based on transcript"""
        try:
            prompt = TITLE_MASTER_PROMPT.format(
                num_titles=NUM_TITLES,
                transcript=transcript
            )
            
            response = self.model.generate_response(
                system_prompt="You are a YouTube title generator. Generate exactly the requested number of titles, one per line.",
                user_content=prompt,
                temperature=0.8,
                max_tokens=500
            )
            
            # Extract titles from response
            titles = [line.strip() for line in response.content.split('\n') if line.strip()]
            titles = titles[:NUM_TITLES]  # Ensure we have exactly NUM_TITLES
            
            return titles
            
        except Exception as e:
            return []
    
    def generate_thumbnails(self, transcript, titles):
        """Generate thumbnail images based on transcript and titles"""
        thumbnails_generated = []
        
        try:
            # Create context from transcript and titles
            context = f"Moon Dev is coding trading algorithms. Topics discussed: {transcript[:500]}... Key themes from titles: {', '.join(titles[:3])}"
            
            for i in range(NUM_THUMBNAILS):
                try:
                    # Create unique prompt for each thumbnail
                    prompt = THUMBNAIL_MASTER_PROMPT.format(context=context)
                    if i > 0:  # Add variation for subsequent thumbnails
                        prompt += f"\n\nVariation {i+1}: Create a different visual approach while maintaining the theme."
                    
                    # Generate image
                    result = self.openai_client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size="1792x1024",  # Wide format for YouTube thumbnails
                        quality="standard",
                        response_format="b64_json"
                    )
                    
                    # Save thumbnail directly to Downloads
                    image_base64 = result.data[0].b64_json
                    image_bytes = base64.b64decode(image_base64)
                    
                    # Create unique filename with timestamp
                    now = datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"stream_thumbnail_{timestamp}_{i+1}.png"
                    filepath = self.downloads_dir / filename
                    
                    with open(filepath, "wb") as f:
                        f.write(image_bytes)
                    
                    thumbnails_generated.append(str(filepath))
                    
                except Exception as e:
                    pass
                    
                # Small delay between generations
                if i < NUM_THUMBNAILS - 1:
                    time.sleep(2)
            
            return thumbnails_generated
            
        except Exception as e:
            return thumbnails_generated
    
    def display_results(self, titles):
        """Display only the generated titles"""
        print()  # Empty line
        for title in titles:
            print(title)
    
    def run_once(self):
        """Run one iteration of recording and generation"""
        # Record audio
        self.record_audio()
        
        # Process transcript
        if self.current_transcript:
            full_transcript = ' '.join(self.current_transcript)
            
            if full_transcript.strip():
                # Generate titles
                titles = self.generate_titles(full_transcript)
                
                # Generate thumbnails
                thumbnails = self.generate_thumbnails(full_transcript, titles)
                
                # Display results (only titles)
                if titles:
                    self.display_results(titles)
    
    def run(self):
        """Main execution flow - runs continuously"""
        while True:
            try:
                # Run one iteration
                self.run_once()
                
                # Wait before next batch (silently)
                time.sleep(WAIT_BETWEEN_BATCHES)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                time.sleep(30)  # Short wait on error

if __name__ == "__main__":
    try:
        agent = StreamAgent()
        agent.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass