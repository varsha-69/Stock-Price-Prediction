import pandas as pd

def compute_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def compute_macd(data: pd.DataFrame,
                 short_window: int = 12,
                 long_window: int = 26,
                 signal_window: int = 9) -> pd.DataFrame:
    exp1 = data['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    data['MACD'] = macd
    data['Signal_Line'] = signal
    return data
