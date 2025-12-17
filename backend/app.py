# Optional extra module; main logic lives in backend.main
from fastapi import FastAPI

app = FastAPI(title="Stock Prediction Service")

@app.get("/health")
async def health():
    return {"status": "ok"}
