def generate_label(candle: dict) -> int:
    """
    Generate a label for a candlestick.

    Args:
        candle (dict): Dict with keys ['open', 'close']

    Returns:
        int: 1 for bullish (green candle), 0 for bearish (red candle)
    """
    return int(candle['close'] > candle['open'])
