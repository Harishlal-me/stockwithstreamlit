from typing import Dict, NamedTuple, Tuple
from dataclasses import dataclass
import numpy as np
from config import Config

@dataclass
class PredictionResult:
    symbol: str
    current_price: float
    tomorrow_direction: str
    tomorrow_confidence: float  # 0-1 realistic edge
    tomorrow_pct_change: float
    tomorrow_price: float
    week_direction: str
    week_confidence: float     # 0-1 realistic edge  
    week_pct_change: float
    week_price: float
    action: str
    reason: str
    model_tomorrow_acc: float  # Raw validation accuracy
    model_week_acc: float      # Raw validation accuracy

class RealisticConfidence:
    """Convert raw model probabilities to realistic trading edges"""
    
    @staticmethod
    def calibrate_edge(raw_prob: float, val_accuracy: float) -> float:
        """
        Calibrate raw model probability to realistic trading edge.
        raw_prob=0.97 → calibrated_edge=0.12 (12% edge over random)
        """
        raw_edge = abs(raw_prob - 0.5) * 2  # Normalize to 0-1 edge
        return raw_edge * val_accuracy  # Scale by actual validation accuracy
    
    @staticmethod
    def get_confidence_level(edge: float) -> str:
        """Convert edge to qualitative confidence"""
        if edge >= 0.15:
            return "HIGH"
        elif edge >= 0.08:
            return "MEDIUM"
        else:
            return "LOW"

def make_trading_decision(
    prob_tomorrow_up: float,
    prob_week_up: float,
    log_ret_tomorrow: float,
    log_ret_week: float,
    current_price: float,
    val_acc_tomorrow: float,  # From training (e.g. 0.55)
    val_acc_week: float,      # From training (e.g. 0.77)
) -> PredictionResult:
    """Professional trading decision with realistic confidence"""
    
    # Calculate realistic edges (not raw model probabilities)
    tom_edge = RealisticConfidence.calibrate_edge(prob_tomorrow_up, val_acc_tomorrow)
    week_edge = RealisticConfidence.calibrate_edge(prob_week_up, val_acc_week)
    
    # Convert log returns to %
    tomorrow_pct = (np.exp(log_ret_tomorrow) - 1) * 100
    week_pct = (np.exp(log_ret_week) - 1) * 100
    
    # Predicted prices
    tomorrow_price = current_price * np.exp(log_ret_tomorrow)
    week_price = current_price * np.exp(log_ret_week)
    
    # Directions
    tomorrow_dir = "UP" if log_ret_tomorrow > 0 else "DOWN"
    week_dir = "UP" if log_ret_week > 0 else "DOWN"
    
    # Trading decision (week-dominant, realistic thresholds)
    week_conf_level = RealisticConfidence.get_confidence_level(week_edge)
    
    if week_conf_level == "HIGH":
        action = "BUY" if week_dir == "UP" else "SELL"
    elif week_conf_level == "MEDIUM":
        action = "BUY" if week_dir == "UP" else "SELL"
    else:
        action = "HOLD"
    
    reason = (
        f"{week_conf_level} weekly {week_dir} signal "
        f"(model edge {week_edge*100:.0f}%). "
        f"Week accuracy: {val_acc_week:.0%}"
    )
    
    return PredictionResult(
        symbol="AAPL",  # Will be set by caller
        current_price=current_price,
        tomorrow_direction=tomorrow_dir,
        tomorrow_confidence=tom_edge,
        tomorrow_pct_change=tomorrow_pct,
        tomorrow_price=tomorrow_price,
        week_direction=week_dir,
        week_confidence=week_edge,
        week_pct_change=week_pct,
        week_price=week_price,
        action=action,
        reason=reason,
        model_tomorrow_acc=val_acc_tomorrow,
        model_week_acc=val_acc_week,
    )

def result_to_dict(result: PredictionResult) -> Dict:
    """Format professional output"""
    tom_conf_level = RealisticConfidence.get_confidence_level(result.tomorrow_confidence)
    week_conf_level = RealisticConfidence.get_confidence_level(result.week_confidence)
    
    return {
        "symbol": result.symbol,
        "current_price": f"${result.current_price:.2f}",
        "prediction": f"""📈 {result.symbol} PREDICTION
📅 Tomorrow: {result.tomorrow_direction} ({tom_conf_level}) | {result.tomorrow_pct_change:+.2f}% → ${result.tomorrow_price:.2f}
📅 1 Week:   {result.week_direction} ({week_conf_level}) | {result.week_pct_change:+.2f}% → ${result.week_price:.2f}
💰 CURRENT: {result.current_price}
🎯 ACTION: {result.action}
🧠 {result.reason}
📊 Model: Tomorrow {result.model_tomorrow_acc:.0%} | Week {result.model_week_acc:.0%}
✅ Prediction complete.""",
    }
