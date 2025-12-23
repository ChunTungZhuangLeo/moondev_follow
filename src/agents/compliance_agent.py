'''
🌙 Moon Dev's Compliance Agent 🌙
This agent analyzes ad images and videos against Facebook guidelines to ensure compliance.
It extracts frames from videos, transcribes audio, and provides a compliance rating.

Created with ❤️ by Moon Dev's AI Assistant
'''

import os
import sys
import cv2
import time
import json
import shutil
import whisper
import numpy as np
import base64
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from termcolor import colored, cprint
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model factory
from src.models import model_factory

# Configuration
MODEL_CONFIG = {
    "type": "openai",
    "name": "gpt-4o-mini",  # Using OpenAI's GPT-4o-mini model for analysis
    "reasoning_effort": "high"  # Maximum reasoning for compliance checks
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "compliance"
GUIDELINES_PATH = DATA_DIR / "fb_guidelines.txt"
VIDEOS_DIR = Path("/Users/md/Dropbox/dev/github/search-arbitrage/bots/compliance/tiktok_ads")
OUTPUT_DIR = DATA_DIR / "analysis"
FRAMES_DIR = OUTPUT_DIR / "frames"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# System prompt for compliance analysis
COMPLIANCE_PROMPT = """
You are Moon Dev's Compliance Agent 🌙, an expert in analyzing ads for compliance with Facebook's advertising guidelines.

Your task is to analyze the provided ad frames and transcript to determine if they comply with Facebook's advertising guidelines.

For each ad, you will:
1. Analyze each frame for visual compliance issues
2. Analyze the transcript for text/audio compliance issues
3. Check for specific violations of Facebook's guidelines
4. Provide an overall compliance rating (0-100%)
5. Identify specific issues that need to be fixed
6. Provide recommendations for improving compliance

Common compliance issues to check for:
- Personal attributes (assuming characteristics about the viewer)
- Sensational content (shocking, scary, or violent imagery)
- Adult content or nudity
- Misleading claims or false information
- Health claims without proper disclaimers
- Before/after images that imply unrealistic results
- Targeting sensitive categories (race, religion, etc.)
- Prohibited products or services
- Text-to-image ratio issues

Your analysis should be thorough but concise. Focus on actionable feedback.

Your response MUST follow this format:
{
  "compliance_status": "compliant" or "non-compliant",
  "overall_assessment": "Brief overall assessment",
  "moon_dev_message": "A fun message mentioning Moon Dev 🌙"
}

Remember to be thorough but fair in your assessment. The goal is to help improve ad compliance, not to reject ads unnecessarily.
"""

class ComplianceAgent:
    """Agent to analyze ad compliance with Facebook guidelines"""
    
    def __init__(self, guidelines_path: Path = None):
        """Initialize the compliance agent"""
        self.guidelines = self._load_guidelines(guidelines_path)
        self.model = self._init_model()
        self._whisper_model = None  # Lazy-loaded
        
        # Create output directories if they don't exist
        for dir_path in [OUTPUT_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, REPORTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        cprint("🌙 Moon Dev's Compliance Agent initialized! 🌙", "magenta")
        
    def _load_guidelines(self, guidelines_path: Optional[Path] = None) -> str:
        """Load Facebook advertising guidelines"""
        if not guidelines_path:
            guidelines_path = GUIDELINES_PATH
            
        try:
            with open(guidelines_path, 'r') as f:
                guidelines = f.read()
            cprint(f"✅ Loaded guidelines from {guidelines_path}", "green")
            return guidelines
        except Exception as e:
            cprint(f"❌ Error loading guidelines: {str(e)}", "red")
            return "Facebook advertising guidelines not available."
            
    def _init_model(self):
        """Initialize AI model for compliance analysis"""
        try:
            cprint(f"🤖 Initializing {MODEL_CONFIG['type']} model: {MODEL_CONFIG['name']}...", "cyan")
            
            model = model_factory.get_model(MODEL_CONFIG["type"], MODEL_CONFIG["name"])
            if not model:
                cprint(f"❌ Failed to initialize {MODEL_CONFIG['type']} model", "red")
                cprint("⚠️ Will attempt to use any available model", "yellow")
                
                # Try to use any available model
                for model_type in model_factory.MODEL_IMPLEMENTATIONS.keys():
                    if model_factory.is_model_available(model_type):
                        model = model_factory.get_model(model_type)
                        cprint(f"✅ Using alternative model: {model_type}", "green")
                        break
                        
            if not model:
                cprint("❌ No AI models available. Please check your API keys.", "red")
                raise ValueError("No AI models available")
                
            cprint(f"✅ Successfully initialized AI model", "green")
            return model
            
        except Exception as e:
            cprint(f"❌ Error initializing model: {str(e)}", "red")
            raise
    
    def _lazy_load_whisper(self):
        """Lazy-load the Whisper model when needed"""
        if self._whisper_model is None:
            try:
                cprint("🔊 Loading Whisper model for transcription...", "cyan")
                self._whisper_model = whisper.load_model("base")
                cprint("✅ Whisper model loaded successfully!", "green")
            except Exception as e:
                cprint(f"❌ Error loading Whisper model: {str(e)}", "red")
                cprint("⚠️ Transcription will not be available", "yellow")
    
    def extract_frames(self, video_path: Path, output_folder: Path) -> List[Path]:
        """Extract exactly 10 frames evenly distributed throughout the video"""
        try:
            # Create output folder if it doesn't exist
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Open the video
            video = cv2.VideoCapture(str(video_path))
            
            # Check if video opened successfully
            if not video.isOpened():
                cprint(f"❌ Error: Could not open video {video_path}", "red")
                return []
            
            # Get video properties
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cprint(f"🎬 Processing video: {video_path.name}", "cyan")
            cprint(f"🖼️ Total frames: {frame_count}", "cyan")
            
            # Calculate intervals to capture 10 frames
            frame_interval = max(1, frame_count // 10)
            saved_count = 0
            frame_paths = []
            
            for frame_number in range(0, frame_count, frame_interval):
                # Set the video position to the frame number
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = video.read()
                
                if not success:
                    break
                
                # Save the frame
                frame_path = output_folder / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
                saved_count += 1
                
                # Stop after saving 10 frames
                if saved_count >= 10:
                    break
                
                # Show progress
                if saved_count % 10 == 0:
                    cprint(f"💫 Extracted {saved_count} frames... Moon Dev would be proud!", "green")
            
            # Release the video
            video.release()
            
            cprint(f"✅ Extraction complete! Saved {saved_count} frames to {output_folder}", "green")
            return frame_paths
        except Exception as e:
            cprint(f"❌ Error extracting frames: {str(e)}", "red")
            return []
    
    def transcribe_video(self, video_path: Path, output_path: Path) -> str:
        """Transcribe video audio using Whisper"""
        try:
            # Lazy-load the Whisper model
            self._lazy_load_whisper()
            
            if self._whisper_model is None:
                return "Transcription not available - Whisper model failed to load."
            
            cprint(f"🎤 Transcribing audio from {video_path.name}...", "cyan")
            
            # Transcribe the audio
            result = self._whisper_model.transcribe(str(video_path))
            transcript = result["text"]
            
            # Save the transcript
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            cprint(f"✅ Transcription complete! Saved to {output_path}", "green")
            cprint(f"📝 Transcript preview: {transcript[:100]}...", "cyan")
            
            return transcript
        except Exception as e:
            cprint(f"❌ Error transcribing video: {str(e)}", "red")
            return "Error transcribing video."
    
    def analyze_compliance(self, ad_name: str, frames_dir: Path, transcript: str) -> Dict:
        """Analyze ad compliance using AI model"""
        try:
            cprint(f"🔍 Analyzing compliance for ad: {ad_name}...", "cyan")
            
            # Get list of frames
            frames = sorted(list(frames_dir.glob("*.jpg")))
            if not frames:
                cprint(f"❌ No frames found in {frames_dir}", "red")
                return {"error": "No frames found for analysis"}
            
            # Create base64 encoded images instead of URLs
            image_contents = []
            for frame in frames[:5]:  # Limit to 5 frames to avoid token limits
                try:
                    with open(frame, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        })
                    cprint(f"✅ Encoded frame: {frame.name}", "green")
                except Exception as e:
                    cprint(f"❌ Error encoding frame {frame.name}: {str(e)}", "red")
            
            # Prepare user content
            user_content = [
                {"type": "text", "text": "Please analyze this ad for compliance with Facebook's advertising guidelines."},
                {"type": "text", "text": f"Ad Name: {ad_name}"},
                {"type": "text", "text": f"Analyzing {len(image_contents)} frames from the video:"},
                *image_contents,
                {"type": "text", "text": "Transcript:"},
                {"type": "text", "text": transcript[:2000]},
                {"type": "text", "text": "Guidelines:"},
                {"type": "text", "text": self.guidelines[:5000]}
            ]
            
            # Get analysis from model
            cprint("🧠 Asking AI to analyze compliance...", "cyan")
            cprint("🌙 Moon Dev's Compliance Agent is thinking deeply... 🌙", "magenta")
            
            response = self.model.generate_response(
                system_prompt=COMPLIANCE_PROMPT,
                user_content=user_content,
                temperature=0.7,
                max_tokens=2000
            )
            
            if not response or not hasattr(response, 'content'):
                cprint("❌ Model returned empty response", "red")
                return {"error": "Model returned empty response"}
            
            content = response.content
            
            # Try to parse JSON from response
            try:
                # Find JSON in the response (it might be wrapped in markdown code blocks)
                json_match = None
                if "```json" in content:
                    json_match = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_match = content.split("```")[1].split("```")[0].strip()
                else:
                    json_match = content.strip()
                
                analysis = json.loads(json_match)
                cprint("✅ Successfully parsed compliance analysis!", "green")
                
                # Print the response with a red or blue background based on compliance
                if analysis.get('compliance_status', 'non-compliant') == 'compliant':
                    cprint(f"\nCompliance Status: {analysis.get('compliance_status', 'Unknown')}", "white", "on_blue")
                else:
                    cprint(f"\nCompliance Status: {analysis.get('compliance_status', 'Unknown')}", "white", "on_red")
                cprint(f"Overall Assessment: {analysis.get('overall_assessment', 'No assessment provided.')}", "white", "on_blue")
                cprint(f"Moon Dev Message: {analysis.get('moon_dev_message', 'Moon Dev says: Keep those ads compliant! 🌙')}", "white", "on_blue")
                
                return analysis
            except Exception as json_error:
                cprint(f"❌ Error parsing JSON from model response: {str(json_error)}", "red")
                cprint(f"Raw response: {content[:500]}...", "yellow")
                return {
                    "error": "Failed to parse JSON from model response",
                    "raw_response": content
                }
            
        except Exception as e:
            cprint(f"❌ Error analyzing compliance: {str(e)}", "red")
            return {"error": f"Error analyzing compliance: {str(e)}"}
    
    def generate_report(self, ad_name: str, analysis: Dict, output_path: Path) -> None:
        """Generate a detailed compliance report in JSON format only"""
        try:
            cprint(f"📊 Generating compliance report for {ad_name}...", "cyan")
            
            # Save the raw JSON analysis
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            cprint(f"✅ JSON report generated and saved to {json_path}", "green")
        except Exception as e:
            cprint(f"❌ Error generating JSON report: {str(e)}", "red")
    
    def process_video(self, video_path: Path) -> Dict:
        """Process a single video for compliance analysis"""
        try:
            cprint(f"\n{'='*50}", "yellow")
            cprint(f"🎬 Processing video: {video_path.name}", "cyan")
            cprint(f"{'='*50}\n", "yellow")
            
            # Create output directories for this video
            ad_name = video_path.stem
            video_frames_dir = FRAMES_DIR / ad_name
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Extract frames
            cprint("🖼️ Step 1: Extracting frames...", "cyan")
            frames = self.extract_frames(video_path, video_frames_dir)
            if not frames:
                cprint("❌ Failed to extract frames", "red")
                return {"error": "Failed to extract frames"}
            
            # Step 2: Transcribe audio
            cprint("🎤 Step 2: Transcribing audio...", "cyan")
            transcript_path = TRANSCRIPTS_DIR / f"{ad_name}_transcript.json"
            transcript = self.transcribe_video(video_path, transcript_path)
            
            # Step 3: Analyze compliance
            cprint("🔍 Step 3: Analyzing compliance...", "cyan")
            analysis = self.analyze_compliance(ad_name, video_frames_dir, transcript)
            
            # Step 4: Generate report
            cprint("📊 Step 4: Generating report...", "cyan")
            report_path = REPORTS_DIR / f"{ad_name}_report.json"
            self.generate_report(ad_name, analysis, report_path)
            
            cprint(f"\n✅ Completed processing for {ad_name}!", "green")
            cprint(f"📝 Report saved to: {report_path}", "green")
            
            # Return the analysis results
            return analysis
        except Exception as e:
            cprint(f"❌ Error processing video {video_path.name}: {str(e)}", "red")
            return {"error": f"Error processing video: {str(e)}"}
    
    def process_all_videos(self) -> None:
        """Process all videos in the videos directory"""
        try:
            # Get all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(list(VIDEOS_DIR.glob(f"*{ext}")))
            
            if not video_files:
                cprint(f"⚠️ No video files found in {VIDEOS_DIR}", "yellow")
                cprint(f"🌙 Moon Dev says: Add some videos and try again! 🌙", "magenta")
                return
            
            # Process each video
            cprint(f"🎥 Found {len(video_files)} videos to process", "cyan")
            
            results = []
            for i, video_path in enumerate(video_files, 1):
                cprint(f"\n🔍 Processing video {i}/{len(video_files)}: {video_path.name}", "cyan")
                start_time = time.time()
                
                result = self.process_video(video_path)
                results.append({
                    "video": video_path.name,
                    "result": result
                })
                
                end_time = time.time()
                cprint(f"⏱️ Processing time: {end_time - start_time:.2f} seconds", "cyan")
            
            # Generate summary report
            self._generate_summary_report(results)
            
            cprint("\n🎉 All videos processed successfully!", "green")
            cprint(f"📂 Reports saved to: {REPORTS_DIR}", "green")
            cprint(f"🌟 Moon Dev's Compliance Agent has completed all tasks! 🌟", "magenta")
        except Exception as e:
            cprint(f"❌ Error processing videos: {str(e)}", "red")
    
    def _generate_summary_report(self, results: List[Dict]) -> None:
        """Generate a summary report of all processed videos in JSON format"""
        try:
            cprint("📊 Generating summary JSON report...", "cyan")
            
            summary_path = REPORTS_DIR / "summary_report.json"
            
            # Prepare summary data
            summary_data = {
                "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_videos_processed": len(results),
                "compliant_videos": sum(1 for r in results if r.get('result', {}).get('is_compliant', False)),
                "non_compliant_videos": sum(1 for r in results if not r.get('result', {}).get('is_compliant', False)),
                "video_results": []
            }
            
            for i, result in enumerate(results, 1):
                video_name = result.get('video', 'Unknown')
                analysis = result.get('result', {})
                
                video_summary = {
                    "video": video_name,
                    "score": analysis.get('compliance_score', 'N/A'),
                    "status": "PASS" if analysis.get('is_compliant', False) else "FAIL",
                    "top_issue": "None"
                }
                
                if 'error' in analysis:
                    video_summary["error"] = analysis.get('error', 'Unknown error')
                else:
                    # Get top issue
                    visual_issues = analysis.get('visual_issues', [])
                    text_issues = analysis.get('text_issues', [])
                    all_issues = visual_issues + text_issues
                    
                    if all_issues:
                        # Sort by severity
                        severity_map = {"high": 3, "medium": 2, "low": 1}
                        sorted_issues = sorted(
                            all_issues, 
                            key=lambda x: severity_map.get(x.get('severity', 'low'), 0),
                            reverse=True
                        )
                        if sorted_issues:
                            video_summary["top_issue"] = sorted_issues[0].get('issue', 'Unknown issue')
                
                summary_data["video_results"].append(video_summary)
            
            # Save the summary JSON
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            cprint(f"✅ Summary JSON report generated and saved to {summary_path}", "green")
        except Exception as e:
            cprint(f"❌ Error generating summary JSON report: {str(e)}", "red")

def main():
    """Main function to run the compliance agent"""
    print("🚀 Starting Moon Dev's Compliance Agent 🚀")
    
    try:
        # Initialize the compliance agent
        agent = ComplianceAgent()
        
        # Process all videos
        agent.process_all_videos()
        
    except KeyboardInterrupt:
        cprint("\n👋 Moon Dev's Compliance Agent shutting down gracefully...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
        import traceback
        cprint(traceback.format_exc(), "red")

if __name__ == "__main__":
    main()