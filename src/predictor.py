from typing import Dict
import numpy as np
import tensorflow as tf
import requests
import pandas as pd
from pathlib import Path

from config import Config
from src.data_loader import load_stock_data, EODHD_API_KEY, EODHD_BASE_URL, _eodhd_symbol
from src.feature_engineer import (
    create_technical_indicators,
    create_targets,
    build_feature_matrix,
    make_sequences,
)
from src.decision_engine import make_trading_decision, PredictionResult

# Global validation accuracies (set by training: 59.7%, 67.4%)
_VAL_ACC_TOMORROW: float = 0.597
_VAL_ACC_WEEK: float = 0.674

def set_validation_accuracies(val_tom: float, val_week: float):
    global _VAL_ACC_TOMORROW, _VAL_ACC_WEEK
    _VAL_ACC_TOMORROW = val_tom
    _VAL_ACC_WEEK = val_week

def _load_model():
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}. Run: python train.py")
    return tf.keras.models.load_model(Config.MODEL_PATH)

def _latest_sequence_for_symbol(symbol: str):
    """Build latest 60-day sequence - FIXED shape handling"""
    df = load_stock_data(symbol)
    df = create_technical_indicators(df)
    df = create_targets(df)
    
    X_scaled, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, _ = build_feature_matrix(df)
    X_seq, _, _, _, _ = make_sequences(
        X_scaled, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, 
        seq_len=Config.SEQUENCE_LENGTH,
    )
    
    if len(X_seq) == 0:
        raise ValueError(f"Insufficient data for {symbol} (need 60+ days)")
    
    return X_seq[-1:]  # Shape: (1, 60, features)

def _current_price(symbol: str) -> float:
    """Get current price - EODHD or cache fallback"""
    try:
        # Try EODHD recent data
        if EODHD_API_KEY != "YOUR_EODHD_API_KEY_HERE":
            eod_symbol = _eodhd_symbol(symbol)
            params = {"api_token": EODHD_API_KEY, "fmt": "json", "from": "2025-11-01"}
            resp = requests.get(f"{EODHD_BASE_URL}/{eod_symbol}", params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return float(pd.DataFrame(data)["close"].iloc[-1])
    except:
        pass  # Silent fallback
    
    # Fallback to cached historical data
    df = load_stock_data(symbol)
    if len(df) > 0:
        return float(df["Close"].iloc[-1])
    
    raise ValueError(f"No price data for {symbol}")

def predict_for_symbol(symbol: str) -> Dict:
    """Main prediction - FIXED model output parsing"""
    symbol = symbol.upper()
    if symbol not in Config.SUPPORTED_STOCKS:
        raise ValueError(f"{symbol} not supported. Use: {Config.SUPPORTED_STOCKS}")
    
    print(f"🔄 Loading model for {symbol}...")
    model = _load_model()
    
    print(f"📊 Building latest 60-day sequence...")
    X_last = _latest_sequence_for_symbol(symbol)
    
    print("🤖 Running prediction...")
    predictions = model.predict(X_last, verbose=0)  # List of 4 outputs
    
    # FIXED: Handle single prediction outputs correctly
    # Each output is shape (1,1) - extract scalar values
    tomorrow_cls = predictions[0][0, 0]  # Sigmoid output: probability of UP
    week_cls = predictions[1][0, 0]      # Sigmoid output: probability of UP
    tomorrow_ret = predictions[2][0, 0]  # Log return
    week_ret = predictions[3][0, 0]      # Log return
    
    current_price = _current_price(symbol)
    
    print(f"📈 Raw outputs:")
    print(f"   P(Tomorrow UP): {tomorrow_cls:.1%}")
    print(f"   P(Week UP):     {week_cls:.1%}")
    print(f"   Current price:  ${current_price:.2f}")
    
    result = make_trading_decision(
        prob_tomorrow_up=tomorrow_cls,
        prob_week_up=week_cls,
        log_ret_tomorrow=tomorrow_ret,
        log_ret_week=week_ret,
        current_price=current_price,
        val_acc_tomorrow=_VAL_ACC_TOMORROW,
        val_acc_week=_VAL_ACC_WEEK,
    )
    result.symbol = symbol
    
    return result_to_dict(result)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = predict_for_symbol(sys.argv[1])
        print(result["prediction"])
