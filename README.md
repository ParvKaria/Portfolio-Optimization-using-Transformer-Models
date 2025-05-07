****📊 Stock Price Movement Prediction Using Transformers****

This repository contains two Transformer-based deep learning models for binary classification of stock price movements — with and without the use of technical indicators. The goal is to predict whether a stock's price will go up (BUY) or go down (SELL) the next day, using historical data from a 5-year dataset.

**📁 Files in This Repository:-**

🔹 Model_without_Technical-Indicators.py
A Transformer model trained using only the raw closing price (scaled) of stocks.

🔹 Model_with_Technical-Indicators.py
An enhanced Transformer model that uses technical indicators (e.g., RSI, MACD, Bollinger Bands) along with the price to make more informed predictions.

**🧠 Key Concepts Used:-**

Transformer Encoder (using PyTorch)
MinMaxScaler for scaling price data
Binary Classification (Buy/Sell)
ROC-AUC and Accuracy Metrics
Monte Carlo Simulations for price forecasting
Sharpe Ratio calculation for risk-adjusted returns
Technical Indicators (in the enhanced model)

**📝 Requirements:-**

Ensure you have the following installed:
pip install pandas numpy matplotlib scikit-learn seaborn torch

**📌 Dataset:-**

Make sure the dataset file 5yrdataset.csv is present in the same directory. The dataset should have the following columns:
date
Name (stock ticker)
close
open, high, low, volume (used in technical indicator model)

**▶️ How to Run:-**

1. Clone this repo:
git clone https://github.com/yourusername/stock-transformer-prediction.git
cd stock-transformer-prediction

2. Run either of the two scripts:
python Model_without_Technical-Indicators.py
or
python Model_with_Technical-Indicators.py

3. Enter the stock symbol when prompted (e.g., AAPL).

**🔍 What Each Model Does:-**
✅ Common Features
Takes the past 30 days of data to predict the next day’s movement.
Outputs:
Accuracy and ROC-AUC score
ROC Curve
BUY/SELL signal graph
Model’s verdict (BUY/SELL tendency)
Sharpe Ratio calculation
Monte Carlo simulation of 100 future days over 500 simulations

🧩 Model 1: Without Technical Indicators
Uses only the scaled closing price as input to the Transformer.
Basic but effective for trend-based prediction.

🧩 Model 2: With Technical Indicators
Adds RSI, MACD, EMA, SMA, Bollinger Bands, Momentum, etc., to the features.
Designed to capture nuanced market signals.
More complex and often more accurate.

**🧪 Output Example:-**

✅ Accuracy: 0.7350
✅ ROC AUC:  0.8142

📊 ROC Curve shows how well the model separates Buy vs Sell decisions.

🧠 Verdict for AAPL:
✅ Predominantly showing BUY signals recently.
Reason: The model identifies more upward movements following historical trends.

📈 Sharpe Ratio: 1.2305
Alpha (Risk-Free Rate): 5.00%
🔎 Sharpe Ratio > 1 suggests a good risk-adjusted return.

📉 Monte Carlo simulates 500 possible price paths based on historical returns and volatility.


**📊 Visualizations:-**

ROC Curve: Performance evaluation.
Buy/Sell Signal Plot: Indicates model decisions on historical price.
Monte Carlo Simulation: Simulated future price paths.

**📚 Future Improvements:-**

Integration with live data APIs (like Yahoo Finance or Alpha Vantage)
Use GPU acceleration for faster training
Convert into Flask app or Streamlit dashboard
Add cross-validation and hyperparameter tuning

**💡 Notes:-**

The Transformer model captures temporal dependencies in price trends.
Adding technical indicators provides feature richness to the model.
Always use financial models as decision support tools, not as guarantees.

**📬 Contact:-**
Have questions or suggestions? Feel free to raise an issue or reach out via GitHub!
