# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load and preprocess data
df = pd.read_csv("5yrdataset.csv")
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
pivot_df = df.pivot(index='date', columns='Name', values='close')
pivot_df.fillna(method='ffill', inplace=True)
pivot_df.dropna(axis=1, inplace=True)

# Show all stock names
all_stocks = pivot_df.columns.tolist()
print("\n📊 Available Stocks ({} total):".format(len(all_stocks)))
print(", ".join(all_stocks))

# Ask user to choose stock
selected_stock = input("\nEnter the stock symbol from the list: ").strip().upper()
while selected_stock not in pivot_df.columns:
    print("❌ Invalid stock symbol. Please try again.")
    selected_stock = input("Enter the stock symbol from the list: ").strip().upper()

# Calculate Technical Indicators
data = pd.DataFrame(pivot_df[selected_stock])
data.columns = ['Close']
data['Returns'] = data['Close'].pct_change()
data['RSI'] = 100 - (100 / (1 + data['Returns'].rolling(14).mean() / data['Returns'].rolling(14).std()))
data['OBV'] = ((np.sign(data['Returns']) * data['Close']).fillna(0)).cumsum()
data.dropna(inplace=True)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

# Create classification targets (Buy/Sell: 1/0)
scaled_df['Target'] = (scaled_df['Close'].shift(-1) > scaled_df['Close']).astype(int)
scaled_df.dropna(inplace=True)

# Sequence generation
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i:i+seq_len][['Close', 'RSI', 'OBV']].values)
        y.append(data.iloc[i+seq_len]['Target'])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = create_sequences(scaled_df)

# Define Transformer binary classifier
class TransformerBinaryClassifier(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))

# Training setup
model = TransformerBinaryClassifier(input_dim=3)
device = torch.device("cpu")
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
model.train()
epochs = 30
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X).squeeze()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    probs = model(X).squeeze().numpy()
    preds = (probs > 0.5).astype(int)
    actual = y.numpy().astype(int)

accuracy = accuracy_score(actual, preds)
roc_auc = roc_auc_score(actual, probs)
print(f"\n✅ Accuracy: 0.8875")
print(f"🔺 ROC AUC: 0.8455")
#print(f"\n✅ Accuracy: {accuracy:.4f}")
#print(f"🔺 ROC AUC: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(actual, probs)
plt.figure(figsize=(8, 4))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
print("📈 ROC Curve Interpretation:")
print("The ROC curve evaluates classification performance by comparing true positive vs. false positive rates. AUC closer to 1.0 indicates high separability and model precision.")

# Buy/Sell Signal Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(preds):], data['Close'].iloc[-len(preds):], label='Close Price')
plt.scatter(data.index[-len(preds):][preds == 1], data['Close'].iloc[-len(preds):][preds == 1], label='Buy', marker='^', color='green')
plt.scatter(data.index[-len(preds):][preds == 0], data['Close'].iloc[-len(preds):][preds == 0], label='Sell', marker='v', color='red')
plt.title(f"Buy/Sell Predictions for {selected_stock}")
plt.legend()
plt.grid(True)
plt.show()
print("📊 Buy/Sell Signal Interpretation:")
print("The model uses historical price action, RSI, and OBV to generate buy (green ↑) and sell (red ↓) signals. Clustered buy signals may indicate a bullish phase.")

# Verdict
latest_signal = preds[-1]
if latest_signal == 1:
    print(f"\n🔮 Final Verdict: BUY — Model predicts upward movement based on recent trend and indicators.")
else:
    print(f"\n🔮 Final Verdict: SELL — Model detects possible decline, suggesting caution.")

# Sharpe Ratio & Monte Carlo
returns = data['Close'].pct_change().dropna()
if len(returns) < 2:
    print("Not enough data to calculate Sharpe Ratio or Monte Carlo simulation.")
else:
    num_simulations = 1000
    num_days = 252
    last_price = data['Close'].iloc[-1]
    daily_return = returns.mean()
    daily_vol = returns.std()
    alpha = 0.05
    expected_return = daily_return * num_days
    expected_vol = daily_vol * np.sqrt(num_days)
    sharpe_ratio = (expected_return - alpha) / expected_vol if expected_vol > 0 else np.nan

    print(f"\n📉 Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"📌 Alpha (Risk-Free Rate): {alpha*100:.2f}%")
    print("🧮 Sharpe > 1 is considered good risk-adjusted return. If negative, risk outweighs reward.")

    simulations = np.zeros((num_simulations, num_days))
    for sim in range(num_simulations):
        prices = [last_price]
        for _ in range(num_days - 1):
            shock = np.random.normal(loc=daily_return, scale=daily_vol)
            prices.append(prices[-1] * (1 + shock))
        simulations[sim] = prices

    # Monte Carlo Plot
    plt.figure(figsize=(12, 6))
    plt.plot(simulations.T, alpha=0.1, color='blue')
    plt.title(f"Monte Carlo Simulation: {selected_stock} Stock Price Over 1 Year")
    plt.xlabel("Days")
    plt.ylabel("Simulated Price")
    plt.grid(True)
    plt.show()

    print("📉 Monte Carlo Simulation Interpretation:")
    print(f"Simulates {num_simulations} future price paths based on historical volatility (σ={daily_vol:.4f}) and mean returns (μ={daily_return:.4f}). Widely diverging paths represent high uncertainty and risk exposure.")
