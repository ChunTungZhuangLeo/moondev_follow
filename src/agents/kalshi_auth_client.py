"""
Kalshi Authentication Client - Working Implementation
Uses direct requests with RSA-PSS signature (bypasses broken SDK)
"""
import os, time, base64, requests
from pathlib import Path
from typing import Dict, Any, Optional
from termcolor import cprint
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

class KalshiAuthClient:
    """Authenticated Kalshi API client with working RSA-PSS signatures"""
    
    def __init__(self, api_key_id: str, private_key: rsa.RSAPrivateKey, environment: str = "prod"):
        self.key_id = api_key_id
        self.private_key = private_key
        self.host = "https://api.elections.kalshi.com"
        self.session = requests.Session()
        cprint(f"Kalshi Auth Client initialized (prod environment)", "green")
    
    @classmethod
    def from_env(cls, environment: Optional[str] = None):
        """Create client from environment variables"""
        key_id = os.getenv("KALSHI_API_KEY")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        
        if not key_id:
            raise ValueError("KALSHI_API_KEY environment variable not set")
        if not private_key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH environment variable not set")
        
        key_path = Path(private_key_path).expanduser()
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file not found: {private_key_path}")
        
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        
        return cls(key_id, private_key)
    
    def _sign_message(self, message: str) -> str:
        """Sign message with RSA-PSS"""
        signature_bytes = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature_bytes).decode("utf-8")
    
    def _build_headers(self, method: str, path: str) -> Dict[str, str]:
        """Build authenticated headers"""
        timestamp_ms = int(time.time() * 1000)
        timestamp_str = str(timestamp_ms)
        message = f"{timestamp_str}{method}{path}"
        signature = self._sign_message(message)
        
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
            "Content-Type": "application/json"
        }
    
    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request"""
        full_path = f"/trade-api/v2{path}"
        headers = self._build_headers("GET", full_path)
        
        resp = self.session.get(f"{self.host}{full_path}", headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()
    
    def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST request"""
        full_path = f"/trade-api/v2{path}"
        headers = self._build_headers("POST", full_path)
        
        resp = self.session.post(f"{self.host}{full_path}", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
    
    def place_order(self, ticker: str, side: str, action: str, count: int, 
                    type: str = "limit", yes_price: Optional[int] = None,
                    no_price: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Place an order"""
        body = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action.lower(),
            "count": count,
            "type": type.lower()
        }
        
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        
        return self.post("/portfolio/orders", body)

# Compatibility exports
class KalshiEnvironment:
    DEMO = "demo"
    PROD = "prod"

class KalshiConfig:
    pass

def cents_to_dollars(cents: int) -> float:
    return cents / 100.0

def dollars_to_cents(dollars: float) -> int:
    return int(dollars * 100)

def calculate_payout(contracts: int, price_cents: int, side: str) -> int:
    if side.lower() == "yes":
        return contracts * 100
    else:
        return contracts * (100 - price_cents)
