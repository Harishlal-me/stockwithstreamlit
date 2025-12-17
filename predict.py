#!/usr/bin/env python3
"""
PERFECT Stock Predictor - Professional Trading Signals
ALL bugs fixed + MSFT neutral zone + documented thresholds
"""

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import Config
from src.data_loader import load_stock_data
from src.feature_engineer import (
    create_technical_indicators,
    create_targets,
    build_feature_matrix,
    make_sequences,
)

# YOUR ACTUAL validation accuracies from training
_VAL_ACC_TOMORROW = 0.597  # 59.7%
_VAL_ACC_WEEK = 0.674      # 67.4%

# OFFICIAL THRESHOLDS (documented)
NEUTRAL_LOW = 0.45
NEUTRAL_HIGH = 0.55
HIGH_EDGE = 0.15
MEDIUM_EDGE = 0.08

def predict_for_symbol(symbol: str):
    """Perfect probability → direction → action logic"""
    symbol = symbol.upper()
    print(f"🔄 {symbol} Analysis...")
    
    # Load model
    model_path = Path("models/lstm_stock_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run: python train.py")
    model = tf.keras.models.load_model(model_path)
    
    # Build sequence
    df = load_stock_data(symbol)
    df = create_technical_indicators(df)
    df = create_targets(df)
    
    X_scaled, _, _, _, _, _ = build_feature_matrix(df)
    X_seq, _, _, _, _ = make_sequences(X_scaled, *[np.zeros(len(X_scaled))]*4, seq_len=Config.SEQUENCE_LENGTH)
    
    if len(X_seq) == 0:
        raise ValueError(f"Insufficient data for {symbol}")
    
    # Model prediction
    predictions = model.predict(X_seq[-1:], verbose=0)
    p_tom_up = float(predictions[0][0, 0])
    p_week_up = float(predictions[1][0, 0])
    
    current_price = float(df["Close"].iloc[-1])
    
    print(f"📊 Raw Model Probabilities:")
    print(f"   P(Tomorrow UP): {p_tom_up:.1%}")
    print(f"   P(Week UP):     {p_week_up:.1%}")
    print(f"   Current Price:  ${current_price:.2f}")
    
    # FIXED: Proper direction logic with neutral zone
    if p_week_up >= NEUTRAL_HIGH:
        week_direction = "UP"
    elif p_week_up <= NEUTRAL_LOW:
        week_direction = "DOWN"
    else:
        week_direction = "HOLD"
    
    tom_direction = "UP" if p_tom_up >= 0.50 else "DOWN"
    
    # Edge = distance from neutral (50%)
    week_edge = abs(p_week_up - 0.5)
    
    # Strict action rules matching direction thresholds
    if p_week_up >= NEUTRAL_HIGH:
        action = "BUY"
    elif p_week_up <= NEUTRAL_LOW:
        action = "SELL"
    else:
        action = "HOLD"
    
    # Consistent signal strength thresholds
    if week_edge >= HIGH_EDGE:
        signal_strength = "HIGH"
    elif week_edge >= MEDIUM_EDGE:
        signal_strength = "MEDIUM"
    else:
        signal_strength = "LOW"
    
    # PROFESSIONAL OUTPUT
    print("\n" + "="*80)
    print(f"📈 {symbol} PROFESSIONAL TRADING SIGNAL")
    print(f"📅 Tomorrow: {tom_direction} (P={p_tom_up:.1%}) - Informational only")
    print(f"📈 Week:     {week_direction} ({signal_strength}) | P(UP)={p_week_up:.1%}")
    print(f"🎯 ACTION:   {action}")
    print(f"📊 EDGE:     {week_edge*100:.1f}% from neutral (50%)")
    print(f"💰 PRICE:    ${current_price:.2f}")
    print("\n📈 MODEL PERFORMANCE (Historical Validation)")
    print(f"   Tomorrow Direction: {int(_VAL_ACC_TOMORROW*100)}% accuracy (global)")
    print(f"   Weekly Direction:   {int(_VAL_ACC_WEEK*100)}% accuracy (global)")
    print("\n📏 THRESHOLDS USED:")
    print(f"   UP signal:     ≥ {NEUTRAL_HIGH*100:.0f}%")
    print(f"   DOWN signal:   ≤ {NEUTRAL_LOW*100:.0f}%")
    print(f"   HIGH strength: ≥ {HIGH_EDGE*100:.0f}% edge")
    print(f"   MEDIUM:        {MEDIUM_EDGE*100:.0f}-{HIGH_EDGE*100:.0f}% edge")
    print("="*80)
    print("✅ Prediction complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Professional LSTM Stock Signals")
    parser.add_argument("--stock", "-s", required=True, help="Stock symbol (AAPL, MSFT, etc.)")
    args = parser.parse_args()
    
    try:
        predict_for_symbol(args.stock)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
