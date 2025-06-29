def format_candle_to_text(candle: dict) -> str:
    """
    Convert a single candlestick row to a formatted text string.

    Args:
        candle (dict): Dict with keys ['open', 'high', 'low', 'close', 'volume']

    Returns:
        str: Natural language representation
    """
    return (
        f"Open: {candle['open']:.2f}, High: {candle['high']:.2f}, "
        f"Low: {candle['low']:.2f}, Close: {candle['close']:.2f}, "
        f"Volume: {candle['volume']:.2f}"
    )