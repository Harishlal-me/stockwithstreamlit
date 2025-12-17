from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config


def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create standard technical indicators on OHLCV data.
    Assumes columns: Open, High, Low, Close, Adj Close, Volume
    """
    df = df.copy()

    # Daily returns
    df["ret_1d"] = df["Close"].pct_change()

    # Moving averages
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    # Exponential moving averages
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Volatility (rolling std of returns)
    df["volatility_20"] = df["ret_1d"].rolling(20).std()

    # Volume features
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_sma_20"] + 1e-9)

    df = df.dropna()
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary direction targets and continuous log-return targets.
    - target_tomorrow_dir: 1 if Close[t+1] > Close[t] else 0
    - target_week_dir: 1 if Close[t+7] > Close[t] else 0
    - target_tomorrow_ret: log(C[t+1] / C[t])
    - target_week_ret: log(C[t+7] / C[t])
    """
    df = df.copy()
    close = df["Close"]

    df["target_tomorrow_dir"] = (close.shift(-1) > close).astype(int)
    df["target_week_dir"] = (close.shift(-7) > close).astype(int)

    df["target_tomorrow_ret"] = np.log(close.shift(-1) / close)
    df["target_week_ret"] = np.log(close.shift(-7) / close)

    df = df.dropna()
    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Build scaled feature matrix and target arrays.
    Returns:
        X_scaled, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, scaler
    """
    df = df.copy()

    target_cols = [
        "target_tomorrow_dir",
        "target_week_dir",
        "target_tomorrow_ret",
        "target_week_ret",
    ]
    feature_cols = [c for c in df.columns if c not in target_cols]

    X_raw = df[feature_cols].values.astype("float32")
    y_tom_dir = df["target_tomorrow_dir"].values.astype("int32")
    y_week_dir = df["target_week_dir"].values.astype("int32")
    y_tom_ret = df["target_tomorrow_ret"].values.astype("float32")
    y_week_ret = df["target_week_ret"].values.astype("float32")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, scaler


def make_sequences(X: np.ndarray, y1: np.ndarray, y2: np.ndarray, y3: np.ndarray, y4: np.ndarray, 
                  seq_len: int = 60) -> tuple:
    """Create sequences for LSTM input."""
    Xs, y1s, y2s, y3s, y4s = [], [], [], [], []
    
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i + seq_len)])
        y1s.append(y1[i + seq_len])
        y2s.append(y2[i + seq_len])
        y3s.append(y3[i + seq_len])
        y4s.append(y4[i + seq_len])
    
    return (np.array(Xs), np.array(y1s), np.array(y2s), np.array(y3s), np.array(y4s))

    """
    Turn flat arrays into rolling sequences of length seq_len.
    """
    if seq_len is None:
        seq_len = Config.SEQUENCE_LENGTH

    seq_X = []
    seq_tom_dir = []
    seq_week_dir = []
    seq_tom_ret = []
    seq_week_ret = []

    for i in range(seq_len, len(X)):
        seq_X.append(X[i - seq_len : i])
        seq_tom_dir.append(y_tom_dir[i])
        seq_week_dir.append(y_week_dir[i])
        seq_tom_ret.append(y_tom_ret[i])
        seq_week_ret.append(y_week_ret[i])

    return (
        np.asarray(seq_X, dtype="float32"),
        np.asarray(seq_tom_dir, dtype="int32"),
        np.asarray(seq_week_dir, dtype="int32"),
        np.asarray(seq_tom_ret, dtype="float32"),
        np.asarray(seq_week_ret, dtype="float32"),
    )
