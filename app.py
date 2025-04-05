import streamlit as st
import base64

def set_background_color(dark_mode):
    if dark_mode:
        st.markdown(
            """
            <style>
            body {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            </style>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #FFFFFF;
                color: #000000;
            }
            </style>
            """, unsafe_allow_html=True
        )

import streamlit as st
import pandas as pd
from datetime import date

from helpers.data_loader import load_stock_data, load_multiple_stocks
from helpers.indicators import compute_rsi, compute_macd
from helpers.model import scale_data, create_sequences, build_model, predict_future
from helpers.utils import plot_predictions, plot_rsi_macd, build_future_df

import numpy as np
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="📈 Modular LSTM Stock Predictor", layout="wide")

st.title("📊 Stock Price Predictor")
st.markdown("Built using Streamlit, Bidirectional LSTM, and Technical Indicators")

# ---------------- Sidebar ----------------
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True)
set_background_color(dark_mode)
st.sidebar.header("🔧 Configuration")

ticker_input = st.sidebar.text_input("Enter Ticker(s) (comma-separated)", value="AAPL")
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"),max_value=date.today())

predict_days = st.sidebar.slider("🔮 Days to Predict", 1, 30, 7)
sequence_length = 60

# ---------------- Load & Display Data ----------------
if tickers:
    if len(tickers) == 1:
        df = load_stock_data(tickers[0], start_date, end_date)
        st.subheader(f"📁 Data Preview for {tickers[0]}")
        st.dataframe(df.tail())

        # Indicators
        df = compute_rsi(df)
        df = compute_macd(df)

        st.subheader("📈 Technical Indicators")
        st.pyplot(plot_rsi_macd(df))

        # LSTM PREP
        data = df[['Close']]
        scaled_data, scaler = scale_data(data.values)
        x, y = create_sequences(scaled_data, seq_len=sequence_length)

        # Split manually
        split = int(len(x) * 0.8)
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        x_train = x_train.reshape((-1, sequence_length, 1))
        x_test = x_test.reshape((-1, sequence_length, 1))

        # Build & Train Model
        model = build_model((x_train.shape[1], 1))
        with st.spinner("🔁 Training LSTM model..."):
            model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

        # Predict on test
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        valid = df.iloc[-len(predictions):].copy()
        valid['Predictions'] = predictions

        train = df.iloc[:len(df) - len(predictions)].copy()

        st.subheader("📊 Model Prediction vs Actual")
        st.pyplot(plot_predictions(train, valid))

        # Future Prediction
        last_60 = scaled_data[-sequence_length:]
        future_preds = predict_future(model, last_60, predict_days, scaler)
        future_df = build_future_df(df['Date'].iloc[-1], future_preds)

        st.subheader(f"🔮 {predict_days}-Day Forecast")
        st.dataframe(future_df)

        st.download_button("⬇️ Download Forecast CSV", data=future_df.to_csv(index=False).encode(),
                           file_name=f"{tickers[0]}_forecast.csv", mime='text/csv')

    elif len(tickers) > 1:
        st.subheader("📊 Multi-Stock Price Comparison")
        all_data = load_multiple_stocks(tickers, start_date, end_date)

        close_prices = pd.DataFrame()
        for ticker, df in all_data.items():
            df.set_index('Date', inplace=True)
            close_prices[ticker] = df['Close']

        st.line_chart(close_prices)
else:
    st.info("👈 Enter at least one ticker to get started!")
