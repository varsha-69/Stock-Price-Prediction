import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional

# Scaling
def scale_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

# Sequence builder for LSTM
def create_sequences(data, seq_len=60):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Model Builder
def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(units=50)))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict on test + future
def predict_future(model, last_sequence, days, scaler):
    future_predictions = []
    current_input = last_sequence.reshape(1, -1, 1)

    for _ in range(days):
        next_pred = model.predict(current_input, verbose=0)[0][0]
        future_predictions.append(next_pred)

        # shift input sequence window
        current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
