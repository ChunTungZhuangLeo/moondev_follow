"""
Kalshi RSA-PSS Authentication Client
====================================
Implements Kalshi's RSA-PSS with SHA-256 authentication for secure API access.

This module provides:
- RSA-PSS signature generation for request authentication
- Secure key management and loading from environment/files
- HTTP client with automatic request signing
- Rate limiting and error handling

Author: Moon Dev
Reference: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
"""

import os
import sys
import time
import base64
import requests
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from termcolor import cprint
from dotenv import load_dotenv

# Cryptography imports for RSA-PSS signing
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

load_dotenv()


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class KalshiEnvironment(Enum):
    """
    Kalshi API environment selection.

    DEMO: Use for testing (no real money)
    PROD: Live trading environment
    """
    DEMO = "demo"
    PROD = "prod"


@dataclass
class KalshiConfig:
    """
    Configuration container for Kalshi API settings.

    Attributes:
        key_id: Kalshi API key identifier
        private_key_path: Path to RSA private key PEM file
        environment: Trading environment (DEMO or PROD)
        rate_limit_ms: Minimum milliseconds between API calls
    """
    key_id: str
    private_key_path: str
    environment: KalshiEnvironment = KalshiEnvironment.DEMO
    rate_limit_ms: int = 100  # 10 requests per second max


# API Base URLs by environment
API_BASE_URLS = {
    KalshiEnvironment.DEMO: "https://demo-api.kalshi.com/trade-api/v2",
    KalshiEnvironment.PROD: "https://trading-api.kalshi.com/trade-api/v2",
}

# WebSocket URLs by environment (for future streaming support)
WS_BASE_URLS = {
    KalshiEnvironment.DEMO: "wss://demo-api.kalshi.com/trade-api/ws/v2",
    KalshiEnvironment.PROD: "wss://trading-api.kalshi.com/trade-api/ws/v2",
}


# ==============================================================================
# RSA KEY MANAGEMENT
# ==============================================================================

def load_private_key_from_file(path: str) -> rsa.RSAPrivateKey:
    """
    Load RSA private key from PEM file.

    Args:
        path: Path to PEM file containing private key

    Returns:
        RSAPrivateKey object for signing requests

    Raises:
        FileNotFoundError: If key file doesn't exist
        ValueError: If key file is invalid
    """
    key_path = Path(path).expanduser()

    if not key_path.exists():
        raise FileNotFoundError(f"Private key file not found: {path}")

    with open(key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,  # Keys from Kalshi are not password protected
            backend=default_backend()
        )

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ValueError(f"Invalid key type. Expected RSA private key, got {type(private_key)}")

    return private_key


def load_private_key_from_string(key_string: str) -> rsa.RSAPrivateKey:
    """
    Load RSA private key from PEM-formatted string.

    Args:
        key_string: PEM-formatted private key string

    Returns:
        RSAPrivateKey object for signing requests
    """
    # Ensure proper PEM formatting (handle single-line keys)
    if "-----BEGIN" not in key_string:
        key_string = f"-----BEGIN RSA PRIVATE KEY-----\n{key_string}\n-----END RSA PRIVATE KEY-----"

    private_key = serialization.load_pem_private_key(
        key_string.encode('utf-8'),
        password=None,
        backend=default_backend()
    )

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ValueError(f"Invalid key type. Expected RSA private key")

    return private_key


# ==============================================================================
# KALSHI HTTP CLIENT WITH RSA-PSS AUTHENTICATION
# ==============================================================================

class KalshiAuthClient:
    """
    Authenticated HTTP client for Kalshi API.

    Implements RSA-PSS with SHA-256 signing as required by Kalshi's API.
    Handles rate limiting, retries, and error responses.

    Usage:
        # From environment variables
        client = KalshiAuthClient.from_env()

        # From explicit config
        config = KalshiConfig(
            key_id="your-key-id",
            private_key_path="~/.kalshi/private_key.pem"
        )
        client = KalshiAuthClient(config)

        # Make authenticated requests
        markets = client.get("/markets")
        order = client.post("/orders", {"ticker": "...", "side": "yes"})
    """

    def __init__(self, config: KalshiConfig, private_key: Optional[rsa.RSAPrivateKey] = None):
        """
        Initialize Kalshi authenticated client.

        Args:
            config: KalshiConfig with API credentials
            private_key: Optional pre-loaded private key (loads from config if not provided)
        """
        self.config = config
        self.key_id = config.key_id
        self.environment = config.environment
        self.base_url = API_BASE_URLS[config.environment]
        self.ws_url = WS_BASE_URLS[config.environment]
        self.rate_limit_ms = config.rate_limit_ms

        # Load private key if not provided
        if private_key:
            self.private_key = private_key
        else:
            self.private_key = load_private_key_from_file(config.private_key_path)

        # Rate limiting state
        self._last_request_time = 0.0

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        cprint(f"Kalshi Auth Client initialized ({self.environment.value} environment)", "green")

    @classmethod
    def from_env(cls, environment: Optional[KalshiEnvironment] = None) -> "KalshiAuthClient":
        """
        Create client from environment variables.

        Expects:
            KALSHI_API_KEY: API key identifier
            KALSHI_PRIVATE_KEY_PATH: Path to private key PEM file
            KALSHI_PRIVATE_KEY: (Alternative) Private key as string
            KALSHI_ENVIRONMENT: (Optional) 'demo' or 'prod'

        Args:
            environment: Override environment from env var

        Returns:
            Configured KalshiAuthClient
        """
        key_id = os.getenv("KALSHI_API_KEY")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        private_key_string = os.getenv("KALSHI_PRIVATE_KEY")
        env_str = os.getenv("KALSHI_ENVIRONMENT", "demo").lower()

        if not key_id:
            raise ValueError("KALSHI_API_KEY environment variable not set")

        if not private_key_path and not private_key_string:
            raise ValueError("Either KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY must be set")

        # Determine environment
        if environment is None:
            environment = KalshiEnvironment.PROD if env_str == "prod" else KalshiEnvironment.DEMO

        # Load private key
        if private_key_string:
            private_key = load_private_key_from_string(private_key_string)
        else:
            private_key = load_private_key_from_file(private_key_path)

        config = KalshiConfig(
            key_id=key_id,
            private_key_path=private_key_path or "",
            environment=environment
        )

        return cls(config, private_key=private_key)

    def _sign_pss(self, message: str) -> str:
        """
        Sign message using RSA-PSS with SHA-256.

        This is Kalshi's required signature algorithm:
        - RSA-PSS padding with MGF1(SHA-256)
        - Salt length = digest length (32 bytes for SHA-256)
        - Result is base64 encoded

        Args:
            message: String message to sign

        Returns:
            Base64-encoded signature
        """
        message_bytes = message.encode('utf-8')

        try:
            signature = self.private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH  # 32 bytes
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8')

        except InvalidSignature as e:
            raise ValueError(f"RSA-PSS signing failed: {e}") from e

    def _build_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Build authentication headers for Kalshi API request.

        The signature is computed over: timestamp_ms + method + path_without_params

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path (without base URL)

        Returns:
            Dict with KALSHI-ACCESS-* headers
        """
        # Current timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        timestamp_str = str(timestamp_ms)

        # Strip query parameters from path for signature
        # (query params are NOT included in the signature message)
        path_without_params = path.split('?')[0]

        # Build message to sign: timestamp + method + path
        message = f"{timestamp_str}{method.upper()}{path_without_params}"

        # Sign the message
        signature = self._sign_pss(message)

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        }

    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        current_time = time.time() * 1000  # ms
        elapsed = current_time - self._last_request_time

        if elapsed < self.rate_limit_ms:
            sleep_time = (self.rate_limit_ms - elapsed) / 1000.0
            time.sleep(sleep_time)

        self._last_request_time = time.time() * 1000

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Process API response and handle errors.

        Args:
            response: requests.Response object

        Returns:
            Parsed JSON response

        Raises:
            requests.HTTPError: For 4xx/5xx responses
        """
        try:
            response.raise_for_status()

            if response.status_code == 204:  # No content
                return {}

            return response.json()

        except requests.HTTPError as e:
            # Try to extract error message from response
            try:
                error_body = response.json()
                error_msg = error_body.get("message", str(e))
                cprint(f"Kalshi API Error: {error_msg}", "red")
            except Exception:
                cprint(f"Kalshi API Error: {response.status_code} - {response.text}", "red")
            raise

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform authenticated GET request.

        Args:
            path: API path (e.g., "/markets", "/portfolio/balance")
            params: Optional query parameters

        Returns:
            Parsed JSON response
        """
        self._rate_limit()

        # Build full path with query params for request
        full_path = path
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query_string:
                full_path = f"{path}?{query_string}"

        headers = self._build_auth_headers("GET", path)

        response = self.session.get(
            f"{self.base_url}{full_path}",
            headers=headers,
            timeout=30
        )

        return self._handle_response(response)

    def post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform authenticated POST request.

        Args:
            path: API path (e.g., "/orders")
            body: Request body as dict

        Returns:
            Parsed JSON response
        """
        self._rate_limit()

        headers = self._build_auth_headers("POST", path)

        response = self.session.post(
            f"{self.base_url}{path}",
            json=body or {},
            headers=headers,
            timeout=30
        )

        return self._handle_response(response)

    def delete(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform authenticated DELETE request.

        Args:
            path: API path (e.g., "/orders/{order_id}")
            body: Optional request body

        Returns:
            Parsed JSON response
        """
        self._rate_limit()

        headers = self._build_auth_headers("DELETE", path)

        response = self.session.delete(
            f"{self.base_url}{path}",
            json=body if body else None,
            headers=headers,
            timeout=30
        )

        return self._handle_response(response)

    # ==========================================================================
    # CONVENIENCE METHODS FOR COMMON API ENDPOINTS
    # ==========================================================================

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance and portfolio value."""
        return self.get("/portfolio/balance")

    def get_positions(self) -> Dict[str, Any]:
        """Get all open positions."""
        return self.get("/portfolio/positions")

    def get_orders(self, status: Optional[str] = None) -> Dict[str, Any]:
        """
        Get orders with optional status filter.

        Args:
            status: Filter by status ('resting', 'pending', 'executed', etc.)
        """
        params = {"status": status} if status else None
        return self.get("/portfolio/orders", params)

    def get_fills(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get order fills/executions.

        Args:
            ticker: Filter by market ticker
        """
        params = {"ticker": ticker} if ticker else None
        return self.get("/portfolio/fills", params)

    def get_markets(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        status: str = "open",
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        tickers: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get markets with pagination and filters.

        Args:
            limit: Max markets per page (max 200)
            cursor: Pagination cursor
            status: Market status filter ('open', 'closed', 'settled')
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker
            tickers: Comma-separated list of specific tickers
        """
        params = {
            "limit": limit,
            "cursor": cursor,
            "status": status,
            "event_ticker": event_ticker,
            "series_ticker": series_ticker,
            "tickers": tickers
        }
        return self.get("/markets", {k: v for k, v in params.items() if v is not None})

    def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get single market by ticker."""
        return self.get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get orderbook for a market.

        Args:
            ticker: Market ticker
            depth: Number of price levels (default 10)
        """
        return self.get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_trades(self, ticker: str, limit: int = 100) -> Dict[str, Any]:
        """Get recent trades for a market."""
        return self.get(f"/markets/{ticker}/trades", {"limit": limit})

    def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,  # Number of contracts
        type: str = "limit",  # "limit" or "market"
        yes_price: Optional[int] = None,  # Price in cents (1-99)
        no_price: Optional[int] = None,  # Price in cents (1-99)
        expiration_ts: Optional[int] = None,  # Unix timestamp for expiration
        client_order_id: Optional[str] = None  # Custom order ID
    ) -> Dict[str, Any]:
        """
        Place an order on Kalshi.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no' - which outcome to trade
            action: 'buy' or 'sell'
            count: Number of contracts
            type: Order type ('limit' or 'market')
            yes_price: Limit price for YES in cents (1-99)
            no_price: Limit price for NO in cents (1-99)
            expiration_ts: Order expiration timestamp
            client_order_id: Custom order ID for tracking

        Returns:
            Order response with order_id
        """
        order = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action.lower(),
            "count": count,
            "type": type.lower(),
        }

        if yes_price is not None:
            order["yes_price"] = yes_price
        if no_price is not None:
            order["no_price"] = no_price
        if expiration_ts is not None:
            order["expiration_ts"] = expiration_ts
        if client_order_id is not None:
            order["client_order_id"] = client_order_id

        return self.post("/portfolio/orders", order)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel
        """
        return self.delete(f"/portfolio/orders/{order_id}")

    def batch_create_orders(self, orders: list) -> Dict[str, Any]:
        """
        Create multiple orders atomically.

        Args:
            orders: List of order dicts
        """
        return self.post("/portfolio/orders/batched", {"orders": orders})

    def batch_cancel_orders(self, order_ids: list) -> Dict[str, Any]:
        """
        Cancel multiple orders atomically.

        Args:
            order_ids: List of order IDs
        """
        return self.delete("/portfolio/orders/batched", {"order_ids": order_ids})


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def cents_to_dollars(cents: int) -> float:
    """Convert Kalshi cents (1-99) to dollar price (0.01-0.99)."""
    return cents / 100.0


def dollars_to_cents(dollars: float) -> int:
    """Convert dollar price (0.01-0.99) to Kalshi cents (1-99)."""
    return int(round(dollars * 100))


def calculate_payout(price_cents: int, contracts: int) -> float:
    """
    Calculate potential payout for a position.

    Args:
        price_cents: Entry price in cents
        contracts: Number of contracts

    Returns:
        Payout in dollars if position wins
    """
    cost = (price_cents / 100.0) * contracts
    payout = contracts * 1.0  # Each contract pays $1 on win
    profit = payout - cost
    return profit


# ==============================================================================
# STANDALONE TESTING
# ==============================================================================

def test_client():
    """Test the Kalshi auth client with demo environment."""
    try:
        # Try to create client from environment
        client = KalshiAuthClient.from_env()

        cprint("\nTesting Kalshi Auth Client...", "cyan", attrs=['bold'])

        # Test getting balance
        cprint("\n1. Getting account balance...", "yellow")
        balance = client.get_balance()
        cprint(f"   Balance: ${balance.get('balance', 0) / 100:.2f}", "green")

        # Test getting markets
        cprint("\n2. Fetching markets...", "yellow")
        markets_response = client.get_markets(limit=5)
        markets = markets_response.get("markets", [])
        cprint(f"   Found {len(markets)} markets", "green")

        for m in markets[:3]:
            ticker = m.get("ticker", "")
            title = m.get("title", "")[:50]
            cprint(f"   - {ticker}: {title}...", "white")

        cprint("\nKalshi Auth Client test successful!", "green", attrs=['bold'])

    except ValueError as e:
        cprint(f"\nConfiguration Error: {e}", "yellow")
        cprint("\nTo use the Kalshi Auth Client, set these environment variables:", "cyan")
        cprint("  KALSHI_API_KEY=your-api-key-id", "white")
        cprint("  KALSHI_PRIVATE_KEY_PATH=/path/to/private_key.pem", "white")
        cprint("  KALSHI_ENVIRONMENT=demo  (or 'prod' for live trading)", "white")

    except Exception as e:
        cprint(f"\nError testing client: {e}", "red")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_client()
