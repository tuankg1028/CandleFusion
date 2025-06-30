def format_candle_to_text(candle_dict):
    """
    Convert OHLCV candle data to enhanced text description.
    """
    # Calculate additional features
    price_change = candle_dict['close'] - candle_dict['open']
    price_change_pct = (price_change / candle_dict['open']) * 100
    high_low_range = candle_dict['high'] - candle_dict['low']
    upper_shadow = candle_dict['high'] - max(candle_dict['open'], candle_dict['close'])
    lower_shadow = min(candle_dict['open'], candle_dict['close']) - candle_dict['low']
    body_size = abs(candle_dict['close'] - candle_dict['open'])
    
    # Determine candle type
    if price_change > 0:
        candle_type = "bullish"
    elif price_change < 0:
        candle_type = "bearish"
    else:
        candle_type = "doji"
    
    # Volume description
    volume_desc = "high" if candle_dict['volume'] > candle_dict.get('avg_volume', candle_dict['volume']) else "normal"
    
    return f"Candle: {candle_type}, Price change: {price_change_pct:.2f}%, Open: {candle_dict['open']:.2f}, High: {candle_dict['high']:.2f}, Low: {candle_dict['low']:.2f}, Close: {candle_dict['close']:.2f}, Volume: {volume_desc}, Range: {high_low_range:.2f}, Body size: {body_size:.2f}, Upper shadow: {upper_shadow:.2f}, Lower shadow: {lower_shadow:.2f}"