import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import seaborn as sns

# Load and preprocess data
df = pd.read_csv("5yrdataset.csv")
df.dropna(inplace=True)
stock_list = sorted(df['Name'].unique())
print("✅ All Available Stock Symbols:")
print(stock_list)
print(f"\nTotal Stocks Available: {len(stock_list)}")

# Ask user for stock selection
while True:
    selected_stock = input("\nEnter stock symbol to analyze (e.g., AAPL): ").strip().upper()
    if selected_stock in stock_list:
        break
    else:
        print("❌ Invalid stock symbol. Please re-enter from the available list.")

# Prepare data
stock_df = df[df['Name'] == selected_stock].copy()
stock_df['date'] = pd.to_datetime(stock_df['date'], format='%d-%m-%Y')
stock_df.sort_values('date', inplace=True)
stock_df.set_index('date', inplace=True)

scaler = MinMaxScaler()
stock_df['scaled_close'] = scaler.fit_transform(stock_df[['close']])
stock_df['target'] = (stock_df['scaled_close'].shift(-1) > stock_df['scaled_close']).astype(int)
stock_df.dropna(inplace=True)

# Sequence creation
def create_binary_sequences(data, seq_len=30):
    X, y = [], []
    prices = data['scaled_close'].values
    targets = data['target'].values
    for i in range(len(data) - seq_len):
        X.append(prices[i:i+seq_len])
        y.append(targets[i+seq_len])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

X, y = create_binary_sequences(stock_df)

# Split data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define Transformer
class TransformerBinaryClassifier(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).squeeze(1)

# Train the model
device = torch.device("cpu")
model = TransformerBinaryClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):  # ⬆️ Increased epochs to improve learning
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int().numpy()
    true = y_test.int().numpy()

acc = accuracy_score(true, preds)
auc = roc_auc_score(true, probs.numpy())
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ ROC AUC:  {auc:.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(true, probs.numpy())
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
print("📊 ROC Curve shows how well the model separates Buy vs Sell decisions. Closer to top-left is better.")

# Buy/Sell Graph
signal_dates = stock_df.index[-len(y_test):]
plt.figure(figsize=(12, 5))
plt.plot(signal_dates, stock_df['close'][-len(y_test):], label="Close Price")
plt.scatter(signal_dates[preds == 1], stock_df['close'][-len(y_test):][preds == 1], label="Buy", marker="^", color="green")
plt.scatter(signal_dates[preds == 0], stock_df['close'][-len(y_test):][preds == 0], label="Sell", marker="v", color="red")
plt.legend()
plt.title(f"Buy/Sell Prediction - {selected_stock}")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.grid()
plt.show()

# Verdict
buy_signals = sum(preds)
sell_signals = len(preds) - buy_signals
print(f"\n🧠 Verdict for {selected_stock}:")
if buy_signals > sell_signals:
    print("✅ Predominantly showing **BUY** signals recently.")
    print("Reason: The model identifies more upward movements following historical trends.")
else:
    print("⚠️ Predominantly showing **SELL** signals recently.")
    print("Reason: The model detects more price declines, suggesting caution.")

# Sharpe Ratio
returns = stock_df['close'].pct_change().dropna()
alpha = 0.05  # Risk-free rate
mean_return = returns.mean()
std_dev = returns.std()
sharpe_ratio = (mean_return - alpha) / std_dev
print(f"\n📈 Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Alpha (Risk-Free Rate): {alpha:.2%}")
print("🔎 Sharpe Ratio > 1 suggests a good risk-adjusted return; < 1 implies risk might outweigh rewards.")

# Monte Carlo simulation
simulations = 500
days = 100
last_price = stock_df['close'].iloc[-1]
sim_results = np.zeros((days, simulations))

for sim in range(simulations):
    price = last_price
    for day in range(days):
        daily_return = np.random.normal(mean_return, std_dev)
        price *= (1 + daily_return)
        sim_results[day, sim] = price

plt.figure(figsize=(12, 5))
plt.plot(sim_results, color='skyblue', alpha=0.1)
plt.title(f"Monte Carlo Simulation of Future Price - {selected_stock}")
plt.xlabel("Day")
plt.ylabel("Simulated Price")
plt.grid()
plt.show()
print("📉 Monte Carlo simulates 500 possible price paths based on historical returns and volatility.")
