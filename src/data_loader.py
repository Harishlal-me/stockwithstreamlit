import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import Config


# TODO: paste your own EODHD API key here
# You get this from your EODHD account dashboard.
EODHD_API_KEY = " 6942667c267a86.58820444"

# EODHD base URL for end-of-day prices
EODHD_BASE_URL = "https://eodhd.com/api/eod"


def ensure_dirs() -> None:
    """Ensure data directories exist."""
    Config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _eodhd_symbol(symbol: str) -> str:
    """
    Convert plain ticker (AAPL) to EODHD format.
    For US stocks, EODHD uses 'AAPL.US', 'MSFT.US', etc.
    """
    symbol = symbol.upper()
    return f"{symbol}.US"


def fetch_stock_data(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = Config.INTERVAL,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a symbol using EODHD REST API.

    - Uses explicit start/end dates (15+ years).
    - Returns DataFrame with index 'Date' and columns:
      Open, High, Low, Close, Adj Close, Volume
    """
    if EODHD_API_KEY == "YOUR_EODHD_API_KEY_HERE":
        raise RuntimeError(
            "Please set EODHD_API_KEY in src/data_loader.py "
            "to your actual API key from eodhd.com."
        )

    ensure_dirs()

    if start is None:
        start = Config.START_DATE
    if end is None:
        end = Config.today_str()

    # EODHD expects YYYY-MM-DD strings
    start_date = start
    end_date = end

    # EODHD ticker format (e.g., AAPL.US)
    eod_symbol = _eodhd_symbol(symbol)

    params = {
        "api_token": EODHD_API_KEY,
        "from": start_date,
        "to": end_date,
        "fmt": "json",   # JSON response
    }

    # For daily bars, 'period' can be 'd'; EODHD daily is default.
    # Many examples use '/eod/{symbol}', but here we use query param 's'.
    url = f"{EODHD_BASE_URL}/{eod_symbol}"

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"EODHD request failed for {symbol} "
            f"with HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse EODHD JSON for {symbol}: {exc}") from exc

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"No data returned from EODHD for {symbol} between {start_date} and {end_date}"
        )

    df = pd.DataFrame(data)

    # EODHD fields typically: date, open, high, low, close, adjusted_close, volume
    # Normalize column names and ensure expected structure.
    rename_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjusted_close": "Adj Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing required column '{col}' in EODHD data for {symbol}"
            )

    # Some accounts might not have adjusted_close; handle gracefully
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    # Parse dates and sort ascending
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Keep only expected columns in consistent order
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    if df.empty:
        raise ValueError(
            f"EODHD returned empty DataFrame for {symbol} "
            f"between {start_date} and {end_date}"
        )

    # Cache raw data to CSV
    csv_path = Config.DATA_RAW_DIR / f"{symbol}_raw.csv"
    df.to_csv(csv_path)

    return df


def load_stock_data(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """
    Load cached raw data for a symbol, or fetch from EODHD if missing/refresh.
    """
    ensure_dirs()
    csv_path = Config.DATA_RAW_DIR / f"{symbol}_raw.csv"

    if csv_path.exists() and not refresh:
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        if not df.empty:
            return df

    return fetch_stock_data(symbol)
