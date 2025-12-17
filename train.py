#!/usr/bin/env python3
"""
Train Multi-Task LSTM Stock Prediction Model
15+ years data → 60-day sequences → 4 outputs (tomorrow/week direction + returns)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.trainer import train_and_save_model, set_validation_accuracies
from src.predictor import set_validation_accuracies as set_pred_accuracies

def main():
    print("🚀 Training Multi-Task LSTM Stock Model...")
    print("📊 15+ years data → 60-day sequences → Realistic confidence calibration")
    
    val_tom, val_week = train_and_save_model()
    
    # Update both trainer and predictor with real validation accuracies
    set_validation_accuracies(val_tom, val_week)
    set_pred_accuracies(val_tom, val_week)
    
    print(f"\n✅ Training Complete!")
    print(f"📈 Validation Accuracy → Tomorrow: {val_tom:.1%} | Week: {val_week:.1%}")
    print(f"💾 Model saved: {Path('models/lstm_stock_model.h5').absolute()}")
    print("\n🔥 Run predictions: python predict.py --stock AAPL")

if __name__ == "__main__":
    main()
