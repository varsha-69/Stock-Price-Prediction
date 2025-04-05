import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot actual vs predicted prices
def plot_predictions(train_data, valid_data, predicted_column='Predictions'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_data['Date'], train_data['Close'], label='Training Data', color='blue')
    ax.plot(valid_data['Date'], valid_data['Close'], label='Actual Price', color='green')
    ax.plot(valid_data['Date'], valid_data[predicted_column], label='Predicted Price', color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig

# Plot technical indicators
def plot_rsi_macd(data):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # RSI
    axs[0].plot(data['Date'], data['RSI'], label='RSI', color='purple')
    axs[0].axhline(70, linestyle='--', alpha=0.5, color='red')
    axs[0].axhline(30, linestyle='--', alpha=0.5, color='green')
    axs[0].set_title('RSI')
    axs[0].legend()

    # MACD
    axs[1].plot(data['Date'], data['MACD'], label='MACD', color='blue')
    axs[1].plot(data['Date'], data['Signal_Line'], label='Signal Line', color='orange')
    axs[1].set_title('MACD')
    axs[1].legend()

    plt.tight_layout()
    return fig

# Rebuild date-based DataFrame for prediction results
def build_future_df(last_date, predictions):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions))
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predictions.flatten()
    })
    return future_df
