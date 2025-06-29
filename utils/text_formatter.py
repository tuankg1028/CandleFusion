def format_candle_to_text(candle_dict):
    """
    Convert OHLCV candle data to text description.
    
    Args:
        candle_dict (dict): Dictionary with open, high, low, close, volume keys
        
    Returns:
        str: Formatted text description of the candle
    """
    return f"Open: {candle_dict['open']:.2f}, High: {candle_dict['high']:.2f}, Low: {candle_dict['low']:.2f}, Close: {candle_dict['close']:.2f}, Volume: {candle_dict['volume']:.0f}"