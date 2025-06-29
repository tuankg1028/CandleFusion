def generate_label(candle_dict):
    """
    Generate binary label based on candle data.
    
    Args:
        candle_dict (dict): Dictionary with open, high, low, close, volume keys
        
    Returns:
        int: 0 for bearish (close < open), 1 for bullish (close >= open)
    """
    return 1 if candle_dict['close'] >= candle_dict['open'] else 0
