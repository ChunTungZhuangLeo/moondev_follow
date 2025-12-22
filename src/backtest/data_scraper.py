"""
Kalshi Data Scraper
===================
Historical data collection from Kalshi API for backtesting.

Features:
- Fetches ALL markets (open, closed, settled) with pagination
- Collects hourly candlestick (OHLCV) data for each market
- Full history from Kalshi's beginning (2020) to today
- Checkpoint/resume support for interrupted scrapes
- Rate limiting to respect API limits
- Parquet storage for efficient data access

Author: Moon Dev
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "agents"))

load_dotenv()

# Configure logging - create data directory first
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(data_dir / "scraper.log")
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Kalshi launched in July 2020, but most markets started late 2020/early 2021
KALSHI_ORIGIN_DATE = "2020-07-01"

# Categories to exclude (sports has hundreds of thousands of markets)
EXCLUDED_CATEGORIES = ["Sports"]

# Ticker patterns to exclude (catches sports markets without category field)
EXCLUDED_TICKER_PATTERNS = [
    # Sports category markers
    "SPORT", "MULTIGAME", "SINGLEGAME", "MULTILEG",
    # Major leagues
    "NFL", "NBA", "MLB", "NHL", "NCAA", "MLS", "PGA", "WNBA",
    # Sports types
    "SOCCER", "TENNIS", "GOLF", "MMA", "UFC", "BOXING", "CRICKET",
    "FOOTBALL", "BASEBALL", "BASKETBALL", "HOCKEY",
    # Event patterns (fantasy/betting)
    "KXMVE",  # Kalshi sports event prefix
    "FLSINGLEGAME", "BASINGLEGAME", "NFLPTS", "NHLPTS", "NBAPTS",
]

@dataclass
class ScraperConfig:
    """
    Configuration for the Kalshi data scraper.

    Attributes:
        start_date: Start date for historical data (YYYY-MM-DD)
        end_date: End date for historical data (YYYY-MM-DD)
        rate_limit_ms: Minimum milliseconds between API calls
        markets_per_page: Number of markets to fetch per page
        max_pages: Maximum number of pages to fetch (0 = unlimited)
        categories: List of category filters (empty = all)
        include_open: Whether to include open markets
        include_settled: Whether to include settled/closed markets
        checkpoint_interval: Save checkpoint every N markets
        data_dir: Directory for storing scraped data
        candlestick_resolution: Candlestick resolution (1m, 5m, 1h, 1d)
    """
    start_date: str = KALSHI_ORIGIN_DATE  # From beginning of Kalshi
    end_date: str = ""  # Empty means today
    rate_limit_ms: int = 100  # 10 requests per second
    markets_per_page: int = 200
    max_pages: int = 0  # 0 = unlimited
    categories: List[str] = field(default_factory=list)
    exclude_sports: bool = True  # Exclude sports markets (huge volume)
    include_open: bool = True  # Include open markets
    include_settled: bool = True  # Include settled/closed markets
    checkpoint_interval: int = 50
    data_dir: str = ""  # Empty means default
    candlestick_resolution: str = "1h"  # Hourly data

    def __post_init__(self):
        """Set defaults after initialization."""
        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
        if not self.data_dir:
            self.data_dir = str(project_root / "data" / "raw")


# ==============================================================================
# KALSHI DATA SCRAPER
# ==============================================================================

class KalshiDataScraper:
    """
    Comprehensive data scraper for Kalshi prediction markets.

    Collects:
    - Market metadata (title, category, resolution rules, etc.)
    - Candlestick/OHLCV price data (hourly)
    - Event information

    Features:
    - Scrapes ALL market statuses (open, closed, settled)
    - Full history from Kalshi's beginning
    - Checkpoint/resume for long scrapes
    - Rate limiting
    - Parquet storage
    """

    # API endpoints
    API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, config: Optional[ScraperConfig] = None, use_auth: bool = False):
        """
        Initialize the scraper with optional configuration.

        Args:
            config: ScraperConfig instance (uses defaults if None)
            use_auth: Whether to use authenticated client (needed for some endpoints)
        """
        self.config = config or ScraperConfig()
        self.use_auth = use_auth
        self.auth_client = None

        # Try to use authenticated client for better rate limits
        if use_auth:
            try:
                from kalshi_auth_client import KalshiAuthClient, KalshiEnvironment
                self.auth_client = KalshiAuthClient.from_env(environment=KalshiEnvironment.PROD)
                logger.info("Using authenticated Kalshi client")
            except Exception as e:
                logger.warning(f"Could not initialize auth client: {e}")
                logger.info("Falling back to unauthenticated requests")

        # Session for unauthenticated requests
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        # Rate limiting state
        self._last_request_time = 0.0

        # Create data directories
        self.data_dir = Path(self.config.data_dir)
        self.markets_dir = self.data_dir / "markets"
        self.candlesticks_dir = self.data_dir / "candlesticks"
        self.checkpoint_file = self.data_dir / "scraper_checkpoint.json"

        for d in [self.data_dir, self.markets_dir, self.candlesticks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Checkpoint state
        self.checkpoint: Dict[str, Any] = {}
        self.scraped_tickers: Set[str] = set()

        logger.info("="*60)
        logger.info("KalshiDataScraper initialized")
        logger.info(f"  Date range: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"  Include open markets: {self.config.include_open}")
        logger.info(f"  Include settled markets: {self.config.include_settled}")
        logger.info(f"  Exclude sports: {self.config.exclude_sports}")
        logger.info(f"  Candlestick resolution: {self.config.candlestick_resolution}")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info("="*60)

    def _is_sports_market(self, market: Dict[str, Any]) -> bool:
        """
        Check if a market is a sports market (should be excluded).

        Args:
            market: Market dict from API

        Returns:
            True if this is a sports market
        """
        # Check category
        category = market.get("category", "")
        if category in EXCLUDED_CATEGORIES:
            return True

        # Check ticker patterns
        ticker = market.get("ticker", "").upper()
        for pattern in EXCLUDED_TICKER_PATTERNS:
            if pattern in ticker:
                return True

        return False

    def _rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        current_time = time.time() * 1000  # ms
        elapsed = current_time - self._last_request_time

        if elapsed < self.config.rate_limit_ms:
            sleep_time = (self.config.rate_limit_ms - elapsed) / 1000.0
            time.sleep(sleep_time)

        self._last_request_time = time.time() * 1000

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Make rate-limited API request with retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            retries: Number of retry attempts

        Returns:
            JSON response dict or None on failure
        """
        self._rate_limit()

        # Use authenticated client if available
        if self.auth_client:
            try:
                if method.upper() == "GET":
                    return self.auth_client.get(endpoint, params)
                else:
                    return self.auth_client.post(endpoint, params)
            except Exception as e:
                logger.debug(f"Auth request failed, falling back: {e}")

        # Fallback to unauthenticated request
        url = f"{self.API_BASE}{endpoint}"

        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=30)
                else:
                    response = self.session.request(method, url, json=params, timeout=30)

                if response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code == 404:
                    logger.debug(f"Not found: {endpoint}")
                    return None

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue

        return None

    def fetch_all_markets(
        self,
        status: Optional[str] = None,
        cursor: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all markets with pagination, filtering sports during fetch.

        Args:
            status: Filter by status ('open', 'closed', 'settled')
            cursor: Pagination cursor

        Returns:
            List of market dicts (sports markets excluded if config.exclude_sports)
        """
        all_markets = []
        page_count = 0
        sports_filtered = 0

        logger.info(f"Fetching markets (status={status or 'all'}, exclude_sports={self.config.exclude_sports})...")

        while True:
            params = {
                "limit": self.config.markets_per_page,
            }
            if cursor:
                params["cursor"] = cursor
            if status:
                params["status"] = status

            response = self._request("GET", "/markets", params)

            if not response:
                logger.error("Failed to fetch markets")
                break

            markets = response.get("markets", [])
            cursor = response.get("cursor")

            # Filter sports markets DURING fetch to save memory
            if self.config.exclude_sports:
                filtered_markets = []
                for m in markets:
                    if self._is_sports_market(m):
                        sports_filtered += 1
                    else:
                        filtered_markets.append(m)
                markets = filtered_markets

            all_markets.extend(markets)
            page_count += 1

            if page_count % 100 == 0:
                logger.info(f"  Page {page_count}: kept {len(all_markets)} markets, filtered {sports_filtered} sports")
            elif page_count <= 10 or page_count % 50 == 0:
                logger.info(f"  Page {page_count}: {len(markets)} markets (total: {len(all_markets)}, sports filtered: {sports_filtered})")

            # Check pagination limits
            if not cursor:
                break
            if self.config.max_pages > 0 and page_count >= self.config.max_pages:
                logger.info(f"  Reached max pages limit ({self.config.max_pages})")
                break

        logger.info(f"Fetched {len(all_markets)} {status or 'all'} markets (filtered {sports_filtered} sports)")
        return all_markets

    def fetch_candlesticks(
        self,
        ticker: str,
        series_ticker: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        resolution: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch candlestick/OHLCV data for a market.

        Args:
            ticker: Market ticker
            series_ticker: Event/series ticker (for candlestick endpoint)
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            resolution: Candle resolution (1m, 5m, 1h, 1d)

        Returns:
            List of candlestick dicts with structure:
            - end_period_ts: Unix timestamp for end of candle
            - volume: Number of contracts traded
            - open_interest: Total open positions
            - price: {open, high, low, close} in cents
            - yes_bid/yes_ask: Bid/ask OHLC
        """
        resolution = resolution or self.config.candlestick_resolution
        period_interval = self._resolution_to_minutes(resolution)

        # IMPORTANT: start_ts and end_ts are REQUIRED by Kalshi API
        if start_ts is None:
            start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            start_ts = int(start_dt.timestamp())
        if end_ts is None:
            end_ts = int(time.time())  # Use current time, not config end_date

        # Need series_ticker for candlestick endpoint
        if not series_ticker:
            logger.debug(f"  No series_ticker for {ticker}, skipping candlesticks")
            return []

        all_candles = []

        # Kalshi API can return up to ~1000 candles per request
        # For hourly data over years, we need to chunk by month
        current_start = start_ts
        chunk_days = 30  # Fetch 30 days at a time

        while current_start < end_ts:
            chunk_end = min(current_start + (chunk_days * 86400), end_ts)

            params = {
                "period_interval": period_interval,
                "start_ts": current_start,
                "end_ts": chunk_end,
            }

            endpoint = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
            response = self._request("GET", endpoint, params)

            if response and response.get("candlesticks"):
                candles = response.get("candlesticks", [])
                all_candles.extend(candles)
                logger.debug(f"  Got {len(candles)} candles for chunk")

            # Move to next chunk
            current_start = chunk_end

            # Small delay between chunks
            time.sleep(0.05)

        return all_candles

    def _resolution_to_minutes(self, resolution: str) -> int:
        """Convert resolution string to minutes for API."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping.get(resolution, 60)  # Default to 1h

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint from file if exists.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint found, starting fresh")
            return False

        try:
            with open(self.checkpoint_file, "r") as f:
                self.checkpoint = json.load(f)

            self.scraped_tickers = set(self.checkpoint.get("scraped_tickers", []))

            logger.info(f"Loaded checkpoint: {len(self.scraped_tickers)} markets already scraped")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        self.checkpoint = {
            "scraped_tickers": list(self.scraped_tickers),
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
        }

        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.checkpoint, f, indent=2)

            logger.debug(f"Checkpoint saved: {len(self.scraped_tickers)} markets")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def scrape_all_markets(self, resume: bool = True) -> Dict[str, Any]:
        """
        Scrape ALL market data (open + settled).

        This is the main entry point for data collection. It:
        1. Fetches all markets by status (open, closed, settled)
        2. Filters by date range and categories
        3. Collects hourly candlestick data for each market
        4. Saves to parquet files with checkpointing

        Args:
            resume: Whether to resume from checkpoint

        Returns:
            Summary dict with statistics
        """
        start_date = self.config.start_date
        end_date = self.config.end_date

        logger.info("="*60)
        logger.info("KALSHI DATA SCRAPER - Full History Scrape")
        logger.info("="*60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Resolution: {self.config.candlestick_resolution}")

        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        # Fetch all markets by status
        all_markets = []

        # Get open markets
        if self.config.include_open:
            open_markets = self.fetch_all_markets(status="open")
            all_markets.extend(open_markets)
            logger.info(f"Open markets: {len(open_markets)}")

        # Get settled and closed markets
        if self.config.include_settled:
            settled_markets = self.fetch_all_markets(status="settled")
            all_markets.extend(settled_markets)
            logger.info(f"Settled markets: {len(settled_markets)}")

            closed_markets = self.fetch_all_markets(status="closed")
            all_markets.extend(closed_markets)
            logger.info(f"Closed markets: {len(closed_markets)}")

        # Deduplicate by ticker
        markets_by_ticker = {m["ticker"]: m for m in all_markets}
        unique_markets = list(markets_by_ticker.values())

        logger.info(f"Total unique markets: {len(unique_markets)}")
        # Note: Sports markets already filtered during fetch if exclude_sports=True

        # Filter by categories if specified
        if self.config.categories:
            filtered = [
                m for m in unique_markets
                if m.get("category", "") in self.config.categories
            ]
            logger.info(f"Filtered to {len(filtered)} markets by category")
            unique_markets = filtered

        # Filter by date range - include markets active during the period
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        date_filtered = []
        for m in unique_markets:
            # Check if market was active during date range
            created_str = m.get("created_time") or m.get("open_time") or ""
            close_str = m.get("close_time") or m.get("expiration_time") or ""

            try:
                if created_str:
                    created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    created_dt = created_dt.replace(tzinfo=None)
                else:
                    created_dt = datetime.min

                if close_str:
                    close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    close_dt = close_dt.replace(tzinfo=None)
                else:
                    close_dt = datetime.max

                # Include if market overlaps with date range
                if created_dt <= end_dt and close_dt >= start_dt:
                    date_filtered.append(m)

            except Exception:
                # Include if we can't parse dates
                date_filtered.append(m)

        logger.info(f"Markets in date range: {len(date_filtered)}")

        # Save market metadata
        if date_filtered:
            markets_df = pd.DataFrame(date_filtered)
            markets_file = self.markets_dir / "all_markets.parquet"
            markets_df.to_parquet(markets_file, index=False)
            logger.info(f"Saved market metadata to {markets_file}")

            # Also save as CSV for easy inspection
            markets_csv = self.markets_dir / "all_markets.csv"
            markets_df.to_csv(markets_csv, index=False)

        # Scrape candlestick data for each market
        stats = {
            "total_markets": len(date_filtered),
            "scraped_markets": 0,
            "skipped_markets": 0,
            "failed_markets": 0,
            "total_candles": 0,
            "markets_with_candles": 0,
        }

        for i, market in enumerate(date_filtered):
            ticker = market["ticker"]
            event_ticker = market.get("event_ticker", "")

            # Skip if already scraped (checkpoint)
            if ticker in self.scraped_tickers:
                stats["skipped_markets"] += 1
                continue

            logger.info(f"[{i+1}/{len(date_filtered)}] Scraping {ticker}...")

            try:
                # Get market-specific date range for candlesticks
                created_str = market.get("created_time") or market.get("open_time") or ""
                close_str = market.get("close_time") or market.get("expiration_time") or ""

                market_start_ts = None
                market_end_ts = None

                if created_str:
                    try:
                        created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                        market_start_ts = int(created_dt.timestamp())
                    except Exception:
                        pass

                if close_str:
                    try:
                        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        market_end_ts = int(close_dt.timestamp())
                    except Exception:
                        pass

                # Fetch candlestick data
                candles = self.fetch_candlesticks(
                    ticker=ticker,
                    series_ticker=event_ticker,
                    start_ts=market_start_ts,
                    end_ts=market_end_ts
                )

                if candles:
                    # Save candlestick data
                    candles_df = pd.DataFrame(candles)
                    candle_file = self.candlesticks_dir / f"{ticker.replace('-', '_')}.parquet"
                    candles_df.to_parquet(candle_file, index=False)

                    stats["total_candles"] += len(candles)
                    stats["markets_with_candles"] += 1
                    logger.info(f"  Saved {len(candles)} candles")
                else:
                    logger.debug(f"  No candlestick data available")

                stats["scraped_markets"] += 1
                self.scraped_tickers.add(ticker)

            except Exception as e:
                logger.error(f"  Failed: {e}")
                stats["failed_markets"] += 1

            # Save checkpoint periodically
            if (i + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                logger.info(f"  Checkpoint saved at market {i + 1}")

        # Final checkpoint
        self.save_checkpoint()

        # Summary
        logger.info("="*60)
        logger.info("SCRAPING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total markets: {stats['total_markets']}")
        logger.info(f"Scraped: {stats['scraped_markets']}")
        logger.info(f"Skipped (checkpoint): {stats['skipped_markets']}")
        logger.info(f"Failed: {stats['failed_markets']}")
        logger.info(f"Markets with candles: {stats['markets_with_candles']}")
        logger.info(f"Total candles: {stats['total_candles']}")
        logger.info(f"Data saved to: {self.data_dir}")

        return stats

    def clear_checkpoint(self) -> None:
        """Clear the checkpoint to start fresh."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")
        self.scraped_tickers = set()
        self.checkpoint = {}


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    """Command-line entry point for the data scraper."""
    parser = argparse.ArgumentParser(
        description="Kalshi Data Scraper - Collect historical market data for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape ALL data from Kalshi's beginning to today (recommended)
  python -m src.backtest.data_scraper

  # Scrape specific date range
  python -m src.backtest.data_scraper --start 2024-01-01 --end 2024-12-31

  # Scrape specific categories
  python -m src.backtest.data_scraper --categories economics politics

  # Resume interrupted scrape
  python -m src.backtest.data_scraper --resume

  # Start fresh (clear checkpoint)
  python -m src.backtest.data_scraper --fresh

  # Only settled markets (for backtesting known outcomes)
  python -m src.backtest.data_scraper --only-settled
        """
    )

    parser.add_argument(
        "--start", "-s",
        type=str,
        default=KALSHI_ORIGIN_DATE,
        help=f"Start date (YYYY-MM-DD). Default: {KALSHI_ORIGIN_DATE} (Kalshi launch)"
    )
    parser.add_argument(
        "--end", "-e",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="*",
        default=[],
        help="Category filters (space-separated)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        default="1h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Candlestick resolution. Default: 1h (hourly)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh (clear checkpoint)"
    )
    parser.add_argument(
        "--only-open",
        action="store_true",
        help="Only scrape open markets"
    )
    parser.add_argument(
        "--only-settled",
        action="store_true",
        help="Only scrape settled/closed markets"
    )
    parser.add_argument(
        "--include-sports",
        action="store_true",
        help="Include sports markets (excluded by default due to huge volume)"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        help="Rate limit in milliseconds between requests"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Data output directory"
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Use authenticated API client (recommended for production)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which market types to include
    include_open = True
    include_settled = True

    if args.only_open:
        include_settled = False
    elif args.only_settled:
        include_open = False

    # Create config from arguments
    config = ScraperConfig(
        start_date=args.start,
        end_date=args.end,
        categories=args.categories or [],
        exclude_sports=not args.include_sports,  # Default: exclude sports
        candlestick_resolution=args.resolution,
        include_open=include_open,
        include_settled=include_settled,
        rate_limit_ms=args.rate_limit,
        data_dir=args.data_dir or "",
    )

    # Run scraper
    scraper = KalshiDataScraper(config, use_auth=args.auth)

    # Clear checkpoint if starting fresh
    if args.fresh:
        scraper.clear_checkpoint()

    stats = scraper.scrape_all_markets(resume=args.resume and not args.fresh)

    print(f"\nScraping complete. Stats: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
