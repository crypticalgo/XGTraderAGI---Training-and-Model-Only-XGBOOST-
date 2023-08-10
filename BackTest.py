import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import ta

class TradingSystem:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for no position

    def long(self, price, volume):
        if self.position >= 0:
            cost = price * volume
            if cost <= self.balance:
                self.balance -= cost
                self.position += volume
                return True
        return False

    def short(self, price, volume):
        if self.position <= 0:
            self.balance += price * volume
            self.position -= volume
            return True
        return False

    def close_position(self, price):
        if self.position != 0:
            self.balance += price * abs(self.position)
            self.position = 0
            return True
        return False

    def plot(self, data):
        plt.figure(figsize=(10,5))
        plt.plot(data['Close'], label='Close Price', color='blue', alpha=0.35)
        plt.scatter(data.loc[data['Signal'] == 1].index, data.loc[data['Signal'] == 1]['Close'], color='green', label='Buy Signal', marker='^', alpha=1)
        plt.scatter(data.loc[data['Signal'] == -1].index, data.loc[data['Signal'] == -1]['Close'], color='red', label='Sell Signal', marker='v', alpha=1)
        plt.title('Close Price and Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.show()

def generate_features_and_labels(data):
    # Ensure that the index is a datetime
    data.index = pd.to_datetime(data.index)

    # Calculate EMA
    data['EMA_5'] = ta.trend.ema_indicator(data['Close'], window=5)
    
    # Calculate lagged features
    data['Close_lag1'] = data['Close'].shift(1)
    data['Volume_lag1'] = data['Volume'].shift(1)

    # Calculate MACD
    macd_indicator = ta.trend.MACD(data['Close'], window_fast=3, window_slow=9)
    data['MACD_line'] = macd_indicator.macd()
    data['Signal_line'] = macd_indicator.macd_signal()

    # Calculate RSI
    data['RSI_6'] = ta.momentum.RSIIndicator(data['Close'], window=6).rsi()

    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'], window=5, window_dev=2)
    data['Upper_BB'] = bollinger.bollinger_hband()
    data['Lower_BB'] = bollinger.bollinger_lband()

    # Calculate VWAP (Volume Weighted Average Price)
    data['Volume'] = data['Volume'].astype(float)  # Ensure Volume is float
    data['VWAP'] = (data['Volume'] * data['Close']).cumsum() / data['Volume'].cumsum()

    # Calculate ATR and Momentum
    data['ATR_14'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
    data['Momentum_3'] = ta.momentum.roc(data['Close'], window=3)

    # Simple head-and-shoulders pattern
    data['Head-Shoulders'] = 0
    for i in range(2, len(data) - 2):
        if (data['Close'].iloc[i-2] < data['Close'].iloc[i-1]) and (data['Close'].iloc[i-1] > data['Close'].iloc[i]) and (data['Close'].iloc[i] < data['Close'].iloc[i+1]) and (data['Close'].iloc[i+1] < data['Close'].iloc[i+2]):
            data['Head-Shoulders'].iloc[i] = 1

    # Define label (1 when MACD crosses above signal line and Close crosses above EMA, 0 otherwise)
    data['Label'] = 0
    data.loc[(data['MACD_line'] > data['Signal_line']) & (data['Close'] > data['EMA_5']), 'Label'] = 1
    data.loc[(data['MACD_line'] < data['Signal_line']) & (data['Close'] < data['EMA_5']), 'Label'] = 0

    # Drop rows with NaN values
    data = data.dropna()

    # Split data into features and labels
    features = data[['Close', 'EMA_5', 'MACD_line', 'Signal_line', 'RSI_6', 'Upper_BB', 'Lower_BB', 'VWAP', 'Head-Shoulders', 'Close_lag1', 'Volume_lag1', 'ATR_14', 'Momentum_3']]
    labels = data['Label']

    return features, labels

def preprocess_data(data):
    data = data.dropna()
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])
    scaler = StandardScaler()
    
    # Select only the numeric columns (excluding 'Open time')
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Apply scaling only to the numeric columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Load the previously trained model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Function to get historical data
def get_historical_data(csv_path):
    # Load data from CSV file
    data = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)

    # Convert columns to appropriate data types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col])

    return data

def make_prediction(model, data):
    # Preprocess data
    data = preprocess_data(data)

    # Generate features
    features, _ = generate_features_and_labels(data)

    # Use the latest data point for prediction
    latest_features = features.iloc[-1]

    # Convert the series to a 1-row DataFrame
    latest_features_df = pd.DataFrame([latest_features])

    # Make prediction
    prediction = model.predict(latest_features_df)
    return prediction[0]



# Load model
model_path = '5momo101.pkl'
model = load_model(model_path)

# Get historical data
csv_path = './data/XRPUSDT_5m.csv'
data = get_historical_data(csv_path)

# Preprocess data
data = preprocess_data(data)

# Generate features
features, _ = generate_features_and_labels(data)

# Initialize trading system
trading_system = TradingSystem()


# Initialize trading system
trading_system = TradingSystem()

# Iterate through data
for i in range(len(features)):
    # Make prediction
    prediction = model.predict(features.iloc[[i]])

    # Based on the prediction, decide what to do
    if prediction == 1:
        # If prediction is 1 (bullish), then go long
        if trading_system.position <= 0:
            trading_system.close_position(data.iloc[i]['Close'])
        trading_system.long(data.iloc[i]['Close'], 1)
        data.at[data.iloc[[i]].index.values[0], 'Signal'] = 1
    elif prediction == 0:
        # If prediction is 0 (bearish), then go short
        if trading_system.position >= 0:
            trading_system.close_position(data.iloc[i]['Close'])
        trading_system.short(data.iloc[i]['Close'], 1)
        data.at[data.iloc[[i]].index.values[0], 'Signal'] = -1

# Print final balance
print("Final balance:$", trading_system.balance)

# Plot the data
trading_system.plot(data)
