from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol, e.g., AAPL")


class PredictionResponse(BaseModel):
    tomorrow_direction: Literal["UP", "DOWN"]
    tomorrow_confidence: float
    week_direction: Literal["UP", "DOWN"]
    week_confidence: float
    action: Literal["BUY", "SELL", "HOLD"]
    reason: str
    current_price: float
