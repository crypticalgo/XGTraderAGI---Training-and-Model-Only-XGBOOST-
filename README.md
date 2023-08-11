# XGTraderAGI Training and Model Only XGBOOST

This repository contains a trading strategy script that leverages the XGBoost machine learning algorithm to predict trading signals (Buy, Sell, or Hold) for various cryptocurrencies based on a series of technical indicators.

## Learn How to use this
[Blog Link]([https://choosealicense.com/licenses/mit/](https://www.crypticalgo.com/article/money-printing-trading-bot-for-cryptocurrency-using-xgboost-machine-learning))

## Features:

1. **Data Loading**: Functions to load data from single or multiple CSV files.
2. **Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
3. **Feature Generation**: Creates various technical indicators like EMA, MACD, RSI, and more.
4. **Model Training**: Uses Bayesian optimization for hyperparameter tuning and trains an XGBoost classifier.
5. **Signal Generation**: Predicts trading signals based on the trained model.
6. **Evaluation**: Computes various performance metrics such as accuracy, precision, recall, and ROC-AUC.
7. **Visualization**: Plots the stock's close prices and overlays Buy and Sell signals.

## Prerequisites:

Ensure you have the following libraries installed:

- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- joblib
- ta
- skopt
- logging

You can install these using `pip`:

```bash
pip install pandas numpy xgboost scikit-learn matplotlib joblib ta[all] scikit-optimize logging
```

## Usage:

1. **Data Format**:
    - Make sure your data is in CSV format with columns like 'Close', 'High', 'Low', 'Volume', etc.
    - For training, you can use multiple CSV files representing data from different cryptocurrencies.

2. **Training the Model**:
    - Place your CSV files in a folder named `data`.
    - Run the script and choose the training option:
    ```bash
    python trading_script.py
    ```
    When prompted, choose:
    ```
    Enter '1' to train the model or '2' to generate signals: 1
    ```

3. **Generating Trading Signals**:
    - Make sure you've trained the model at least once.
    - Place the new data CSV in the `data` folder.
    - Run the script and choose the signal generation option:
    ```bash
    python trading_script.py
    ```
    When prompted, choose:
    ```
    Enter '1' to train the model or '2' to generate signals: 2
    ```

4. **Visualizing Results**:
    - After generating signals, the script will automatically display a plot of the cryptocurrency's close prices with Buy and Sell signals.

## Contributing:

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License:

[MIT](https://choosealicense.com/licenses/mit/)

---
