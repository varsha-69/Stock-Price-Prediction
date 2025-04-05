import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

@st.cache_data
def load_multiple_stocks(tickers, start_date, end_date):
    combined_data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        combined_data[ticker] = df
    return combined_data
