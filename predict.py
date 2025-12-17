#!/usr/bin/env python3
"""
PERFECT Stock Predictor - Professional Trading Signals
CLI + Streamlit UI support
"""

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
sys.path.append(str(Path(__file__).parent))

from config import Config
from src.data_loader import load_stock_data
from src.feature_engineer import (
    create_technical_indicators,
    create_targets,
    build_feature_matrix,
    make_sequences,
)

# Validation accuracies
_VAL_ACC_TOMORROW = 0.597
_VAL_ACC_WEEK = 0.674

# Thresholds
NEUTRAL_LOW = 0.45
NEUTRAL_HIGH = 0.55
HIGH_EDGE = 0.15
MEDIUM_EDGE = 0.08

@dataclass
class UIMetrics:
    symbol: str
    current_price: float
    p_tom_up: float
    p_week_up: float
    tom_direction: str
    week_direction: str
    action: str
    signal_strength: str
    val_acc_tom: float
    val_acc_week: float

def set_validation_accuracies(val_tom: float, val_week: float):
    global _VAL_ACC_TOMORROW, _VAL_ACC_WEEK
    _VAL_ACC_TOMORROW = float(val_tom)
    _VAL_ACC_WEEK = float(val_week)

def _get_prediction(symbol: str):
    """Core prediction logic - returns all metrics"""
    symbol = symbol.upper()
    
    model_path = Path("models/lstm_stock_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run: python train.py")
    model = tf.keras.models.load_model(model_path)
    
    df = load_stock_data(symbol)
    df = create_technical_indicators(df)
    df = create_targets(df)
    
    X_scaled, _, _, _, _, _ = build_feature_matrix(df)
    X_seq, _, _, _, _ = make_sequences(X_scaled, *[np.zeros(len(X_scaled))]*4, seq_len=Config.SEQUENCE_LENGTH)
    
    if len(X_seq) == 0:
        raise ValueError(f"Insufficient data for {symbol}")
    
    predictions = model.predict(X_seq[-1:], verbose=0)
    p_tom_up = float(predictions[0][0, 0])
    p_week_up = float(predictions[1][0, 0])
    current_price = float(df["Close"].iloc[-1])
    
    # Direction logic
    if p_week_up >= NEUTRAL_HIGH:
        week_direction = "UP"
        action = "BUY"
    elif p_week_up <= NEUTRAL_LOW:
        week_direction = "DOWN"
        action = "SELL"
    else:
        week_direction = "HOLD"
        action = "HOLD"
    
    tom_direction = "UP" if p_tom_up >= 0.50 else "DOWN"
    
    # Signal strength
    week_edge = abs(p_week_up - 0.5)
    if week_edge >= HIGH_EDGE:
        signal_strength = "HIGH"
    elif week_edge >= MEDIUM_EDGE:
        signal_strength = "MEDIUM"
    else:
        signal_strength = "LOW"
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'p_tom_up': p_tom_up,
        'p_week_up': p_week_up,
        'tom_direction': tom_direction,
        'week_direction': week_direction,
        'action': action,
        'signal_strength': signal_strength,
        'week_edge': week_edge
    }

def predict_for_symbol(symbol: str):
    """CLI version - prints formatted output"""
    print(f"🔄 {symbol.upper()} Analysis...")
    
    data = _get_prediction(symbol)
    
    print(f"📊 Raw Model Probabilities:")
    print(f"   P(Tomorrow UP): {data['p_tom_up']:.1%}")
    print(f"   P(Week UP):     {data['p_week_up']:.1%}")
    print(f"   Current Price:  ${data['current_price']:.2f}")
    print("\n" + "="*80)
    print(f"📈 {data['symbol']} PROFESSIONAL TRADING SIGNAL")
    print(f"📅 Tomorrow: {data['tom_direction']} (P={data['p_tom_up']:.1%}) - Informational only")
    print(f"📈 Week:     {data['week_direction']} ({data['signal_strength']}) | P(UP)={data['p_week_up']:.1%}")
    print(f"🎯 ACTION:   {data['action']}")
    print(f"📊 EDGE:     {data['week_edge']*100:.1f}% from neutral (50%)")
    print(f"💰 PRICE:    ${data['current_price']:.2f}")
    print("\n📈 MODEL PERFORMANCE (Historical Validation)")
    print(f"   Tomorrow Direction: {int(_VAL_ACC_TOMORROW*100)}% accuracy")
    print(f"   Weekly Direction:   {int(_VAL_ACC_WEEK*100)}% accuracy")
    print("\n📏 THRESHOLDS USED:")
    print(f"   UP signal:     ≥ {NEUTRAL_HIGH*100:.0f}%")
    print(f"   DOWN signal:   ≤ {NEUTRAL_LOW*100:.0f}%")
    print(f"   HIGH strength: ≥ {HIGH_EDGE*100:.0f}% edge")
    print(f"   MEDIUM:        {MEDIUM_EDGE*100:.0f}-{HIGH_EDGE*100:.0f}% edge")
    print("="*80)
    print("✅ Prediction complete.\n")

def ui_predict_for_symbol(symbol: str) -> UIMetrics:
    """Streamlit UI version - returns structured data"""
    data = _get_prediction(symbol)
    
    return UIMetrics(
        symbol=data['symbol'],
        current_price=data['current_price'],
        p_tom_up=data['p_tom_up'],
        p_week_up=data['p_week_up'],
        tom_direction=data['tom_direction'],
        week_direction=data['week_direction'],
        action=data['action'],
        signal_strength=data['signal_strength'],
        val_acc_tom=_VAL_ACC_TOMORROW,
        val_acc_week=_VAL_ACC_WEEK,
    )

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