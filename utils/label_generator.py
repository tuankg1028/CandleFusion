def generate_label(candle_dict):
    """
    Generate enhanced 3-class label based on candle data.
    
    Args:
        candle_dict (dict): Dictionary with open, high, low, close, volume keys
        
    Returns:
        int: 0 for bearish (< -0.5%), 1 for neutral (-0.5% to 0.5%), 2 for bullish (> 0.5%)
    """
    price_change = (candle_dict['close'] - candle_dict['open']) / candle_dict['open'] * 100
    
    if price_change < -0.5:
        return 0  # bearish
    elif price_change > 0.5:
        return 2  # bullish
    else:
        return 1  # neutral