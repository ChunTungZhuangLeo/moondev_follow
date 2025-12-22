"""
🌙 Moon Dev's Gemini Model Implementation
Built with love by Moon Dev 🚀

Uses direct REST API instead of SDK for better Python 3.9 compatibility.
"""

import requests
from termcolor import cprint
from .base_model import BaseModel, ModelResponse


class GeminiModel(BaseModel):
    """Implementation for Google's Gemini models using REST API"""

    AVAILABLE_MODELS = {
        "gemini-2.0-flash": "Fast Gemini 2.0 model",
        "gemini-1.5-flash": "Fast Gemini 1.5 model",
        "gemini-1.5-pro": "Advanced Gemini 1.5 model",
        "gemini-2.5-flash": "Fast Gemini 2.5 model (preview)",
        "gemini-2.5-pro": "Advanced Gemini 2.5 model (preview)",
    }

    # API endpoint
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash", **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self._available = False
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the Gemini client (verify API key works)"""
        try:
            # Test API key with a simple request
            url = f"{self.BASE_URL}/{self.model_name}:generateContent?key={self.api_key}"
            test_payload = {
                "contents": [{"parts": [{"text": "Say hi"}]}],
                "generationConfig": {"maxOutputTokens": 10}
            }

            response = requests.post(url, json=test_payload, timeout=30)

            if response.status_code == 200:
                self._available = True
                self.client = True  # Placeholder for compatibility
                cprint(f"✨ Initialized Gemini model: {self.model_name}", "green")
            else:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', response.text)

                # Try fallback models if primary fails
                fallback_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
                for fallback in fallback_models:
                    if fallback != self.model_name:
                        fallback_url = f"{self.BASE_URL}/{fallback}:generateContent?key={self.api_key}"
                        fallback_resp = requests.post(fallback_url, json=test_payload, timeout=30)
                        if fallback_resp.status_code == 200:
                            self.model_name = fallback
                            self._available = True
                            self.client = True
                            cprint(f"✨ Initialized Gemini model: {self.model_name} (fallback)", "green")
                            return

                cprint(f"❌ Gemini API error: {error_msg}", "red")
                self.client = None

        except Exception as e:
            cprint(f"❌ Failed to initialize Gemini model: {str(e)}", "red")
            self.client = None

    def generate_response(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using Gemini REST API"""
        try:
            url = f"{self.BASE_URL}/{self.model_name}:generateContent?key={self.api_key}"

            # Build the request payload
            # Gemini supports system instructions in newer models
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_content}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.95,
                    "topK": 40
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            }

            response = requests.post(url, json=payload, timeout=120)

            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', response.text)
                raise Exception(f"Gemini API error ({response.status_code}): {error_msg}")

            data = response.json()

            # Extract the response text
            candidates = data.get('candidates', [])
            if not candidates:
                block_reason = data.get('promptFeedback', {}).get('blockReason', 'UNKNOWN')
                raise Exception(f"Gemini returned no candidates - blocked: {block_reason}")

            content = candidates[0].get('content', {})
            parts = content.get('parts', [])

            if not parts:
                finish_reason = candidates[0].get('finishReason', 'UNKNOWN')
                raise Exception(f"Gemini returned empty response - finish_reason: {finish_reason}")

            text = parts[0].get('text', '').strip()

            # Get usage info if available
            usage_metadata = data.get('usageMetadata', {})
            usage = {
                'prompt_tokens': usage_metadata.get('promptTokenCount', 0),
                'completion_tokens': usage_metadata.get('candidatesTokenCount', 0),
                'total_tokens': usage_metadata.get('totalTokenCount', 0)
            }

            return ModelResponse(
                content=text,
                raw_response=data,
                model_name=self.model_name,
                usage=usage
            )

        except requests.exceptions.Timeout:
            cprint(f"❌ Gemini request timed out", "red")
            raise Exception("Gemini API timeout")
        except Exception as e:
            cprint(f"❌ Gemini generation error: {str(e)}", "red")
            raise

    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self._available and self.client is not None

    @property
    def model_type(self) -> str:
        return "gemini"
