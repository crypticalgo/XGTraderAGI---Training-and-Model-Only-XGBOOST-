import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib
import ta
from skopt import BayesSearchCV
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
def load_data(filename):
    try:
        data = pd.read_csv(filename)
        logging.info(f'Successfully loaded data from {filename}')
    except Exception as e:
        logging.error(f'Error loading data from {filename}: {e}')
        data = None
    return data


# Load the data Training
def load_dataT(filenames):
    dataframes = []
    for filename in filenames:
        try:
            data = pd.read_csv(filename)
            dataframes.append(data)
            logging.info(f'Successfully loaded data from {filename}')
        except Exception as e:
            logging.error(f'Error loading data from {filename}: {e}')
    return pd.concat(dataframes, ignore_index=True)


# Preprocess the data
def preprocess_data(data):
    data = data.dropna()
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data


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


def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Define the sample weights
    sample_weights = np.zeros(len(y_train))
    sample_weights[y_train == 1] = 1  # Give weight of 1 to the positive class
    sample_weights[y_train == 0] = 0.5  # Give smaller weight to the negative class

    params = {
        'max_depth': (3, 10),
        'gamma': (0, 5),
        'colsample_bytree': (0.5, 1),
        'learning_rate': (0.01, 0.05),
        'n_estimators': (500, 1000),
        'min_child_weight': (1, 10),
        'subsample': (0.5, 1),
        'reg_lambda': (0.01, 1),  # Added L2 regularization
        'reg_alpha': (0.01, 1),  # Added L1 regularization
        'tree_method': ['gpu_hist'],
        'n_jobs': [-1]
    }

    model = xgb.XGBClassifier()
    bayes_search = BayesSearchCV(model, search_spaces=params, 
                                  n_iter=150, scoring='roc_auc', n_jobs=-1, cv=7, verbose=1, random_state=42)
    
    bayes_search.fit(X_train, y_train, sample_weight=sample_weights)

    model = bayes_search.best_estimator_

    joblib.dump(model, '5momo101.pkl')

    # Save the model's parameters
    with open('model_params.txt', 'w') as file:
        file.write(str(bayes_search.best_params_))

    # Generate signals and evaluate the strategy
    signals = generate_signals(model, X_test)
    accuracy, precision, recall, roc_auc = evaluate_strategy(signals, y_test)

    # Save the model's performance metrics
    with open('model_metrics.txt', 'w') as file:
        file.write(
            f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, AUC-ROC: {roc_auc}')

    return model, X_test, y_test


# Generate signals
def generate_signals(model, X_test):
    predictions = model.predict(X_test)
    signals = ['Buy' if prediction == 1 else 'Sell' for prediction in predictions]
    
    # Ensure that the strategy is "Buy-Hold-Sell"
    last_action = None
    for i in range(len(signals)):
        if last_action == signals[i]:
            signals[i] = 'Hold'
        else:
            last_action = signals[i]

    return signals

# Evaluate the strategy
def evaluate_strategy(signals, y_test):
    labels = [1 if signal == 'Buy' else 0 for signal in signals]
    accuracy = accuracy_score(y_test, labels)
    precision = precision_score(y_test, labels, zero_division=1)
    recall = recall_score(y_test, labels)
    roc_auc = roc_auc_score(y_test, labels)
    return accuracy, precision, recall, roc_auc

# Visualize the results
def visualize_results(signals, data):
    plt.plot(data.index, data['Close'])
    buy_signals = [i for i, signal in enumerate(signals) if signal == 'Buy']
    plt.scatter(data.index[buy_signals], data['Close']
                [buy_signals], color='green')
    sell_signals = [i for i, signal in enumerate(signals) if signal == 'Sell']
    plt.scatter(data.index[sell_signals], data['Close']
                [sell_signals], color='red')
    plt.show()


# Main function for generating signals
def main_generate_signals():
    # Load and preprocess data
    data = load_data('./data/BTCUSDT_5m.csv')
    data = preprocess_data(data)
    features, labels = generate_features_and_labels(data)

    # Load model
    model = load_model('./models/5momo101.pkl')

    window_size = 20  # Set the size of the rolling window
    signals = []

    # Generate signals
    for i in range(window_size, len(features)):
        X_test = features.iloc[i-window_size:i]
        signal = generate_signals(model, X_test)
        signals.append(signal[-1])  # Store the most recent signal

    # Transform list into a Pandas Series
    signals = pd.Series(signals, index=features.index[window_size:])

    # Ensure the first action is a 'Buy'
    first_buy_index = signals[signals == 'Buy'].first_valid_index()
    signals = signals.loc[first_buy_index:]

    # Trim the labels to match the signals length
    labels = labels.loc[signals.index]

    # Evaluate strategy
    accuracy, precision, recall, roc_auc = evaluate_strategy(signals, labels)
    print(
        f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, AUC-ROC: {roc_auc}')

    # Visualize results
    visualize_results(signals, data)



# Load the model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# # Main function for training
def main_train():
    data = load_dataT(['./data/BTCUSDT_3m.csv', './data/ETHUSDT_3m.csv', './data/BNBUSDT_3m.csv', './data/ADAUSDT_3m.csv', './data/LTCUSDT_3m.csv', './data/XRPUSDT_3m.csv'])  # Add more paths as needed
    data = preprocess_data(data)
    features, labels = generate_features_and_labels(data)
    model, X_test, y_test = train_model(features, labels)
    print('Model trained and saved successfully.')



if __name__ == '__main__':
    choice = input("Enter '1' to train the model or '2' to generate signals: ")
    if choice.lower() == '1':
        main_train()
    elif choice.lower() == '2':
        main_generate_signals()
    else:
        print("Invalid choice. Please enter either '1' or '2'.")
