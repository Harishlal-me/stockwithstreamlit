from fastapi import HTTPException

from config import Config


def validate_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol not in Config.SUPPORTED_STOCKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported symbol {symbol}. "
                   f"Supported: {', '.join(Config.SUPPORTED_STOCKS)}",
        )
    return symbol
