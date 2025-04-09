from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Function to create and train LSTM model
def train_lstm_model(data):
    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training data
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)

    return model, scaler

# Endpoint to train and predict stock prices
@app.route('/api/predict', methods=['POST'])
def train_and_predict():
    symbol = request.args.get('symbol')
    period = request.args.get('period', '1y')
    data = yf.download(symbol, period=period)
    close_prices=data['Close'].values.reshape(-1,1)

    # Train model
    model, scaler = train_lstm_model(close_prices)

    # Prepare test data
    test_data = close_prices[-60:]
    test_data_scaled = scaler.transform(test_data)
    X_test = np.array([test_data_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make prediction
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    return jsonify({'predicted_price': predicted_price.flatten().tolist()})

# Endpoint to fetch historical stock data
@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    symbol = request.args.get('symbol')
    period = request.args.get('period', '1y')
    data = yf.download(symbol, period=period)
    return data.to_json()

# Endpoint to calculate simple moving average
@app.route('/api/technical', methods=['GET'])
def get_technical_indicators():
    symbol = request.args.get('symbol')
    data = yf.download(symbol, period='1y')
    data['SMA'] = data['Close'].rolling(window=20).mean()
    return data[['Close', 'SMA']].dropna().to_json()

# Endpoint for sentiment analysis (dummy implementation)
@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text')
    # Dummy sentiment analysis
    sentiment_score = len(text) % 3 - 1  # Just a placeholder
    return jsonify({'sentiment_score': sentiment_score})

# Endpoint for risk assessment
@app.route('/api/risk', methods=['GET'])
def assess_risk():
    symbol = request.args.get('symbol')
    data = yf.download(symbol, period='1y')
    close_prices = data['Close']  # Tetap sebagai Pandas Series
    returns = close_prices.pct_change().dropna()  # Menghitung return harian & hapus NaN
    volatility = returns.std() * np.sqrt(252)  # Menghitung volatilitas tahunan
    return jsonify({'volatility': float(volatility)})  # Convert ke float sebelum jsonify


if __name__ == '__main__':
    app.run(debug=True)