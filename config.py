from dataclasses import dataclass
from pathlib import Path
import datetime as dt

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # Stocks
    SUPPORTED_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]

    # Data window: ~15 years of daily data
    START_DATE = "2010-01-01"
    END_DATE = None  # use today's date if None
    INTERVAL = "1d"

    # Paths
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODEL_PATH = BASE_DIR / "models" / "lstm_stock_model.h5"

    # Sequence settings: use last 60 days to predict
    SEQUENCE_LENGTH = 60

    # Model hyperparameters
    LSTM_UNITS_1 = 64
    LSTM_UNITS_2 = 32
    DENSE_UNITS_1 = 64
    DENSE_UNITS_2 = 32
    DROPOUT_1 = 0.2
    DROPOUT_2 = 0.2
    DROPOUT_3 = 0.1

    # Training
    EPOCHS = 40
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42

    # Decision logic thresholds (on confidence 0–1)
    WEEKLY_CONFIDENCE_STRONG = 0.60
    WEEKLY_CONFIDENCE_MILD = 0.55

    # Utility
    @staticmethod
    def today_str() -> str:
        return dt.datetime.today().strftime("%Y-%m-%d")
