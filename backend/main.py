from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models import PredictionRequest, PredictionResponse
from backend.utils import validate_symbol
from src.predictor import predict_for_symbol

app = FastAPI(
    title="Stock Prediction API",
    description="ML/DL-based short-term and 1-week stock direction predictions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Stock Prediction API running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    symbol = validate_symbol(request.symbol)
    result = predict_for_symbol(symbol)
    return PredictionResponse(**result)
