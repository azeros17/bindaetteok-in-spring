import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path (for DataProcessor class)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lstm_predictor import DataProcessor, OulringLSTM, SEQ_LENGTH

DEVICE = torch.device('cpu')

def load_data(data_dir, station_id="SJ_00400"):
    # Load all CSV files (train + test) for full historical view
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and "이용 현황" in f]
    dfs = []
    processor = DataProcessor()
    for f in files:
        path = os.path.join(data_dir, f)
        dfs.append(processor.process_file(path, station_id))
    full = pd.concat(dfs).sort_index()
    return full

def prepare_sequences(df, scaler):
    # Select all feature columns except the target column
    feature_cols = [c for c in df.columns if c != 'target']
    df[feature_cols] = scaler.transform(df[feature_cols])
    X = []
    for i in range(len(df) - SEQ_LENGTH):
        X.append(df[feature_cols].values[i:i+SEQ_LENGTH])
    X = np.array(X, dtype=np.float32)
    return torch.tensor(X)

def train_model(train_df):
    # Simple train on provided dataframe (reuse same hyper‑params as lstm_predictor)
    processor = DataProcessor()
    scaler = StandardScaler()
    feature_cols = [c for c in train_df.columns if c != 'target']
    scaler.fit(train_df[feature_cols])
    X_train = prepare_sequences(train_df, scaler)
    # Prepare training targets
    y_train = torch.tensor(train_df['target'].values[SEQ_LENGTH:], dtype=torch.float32).unsqueeze(1)
    # Prepare test/validation targets
    # (Later code uses same column name)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = OulringLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(30):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # print progress (optional)
    return model, scaler

def simulate_relocation(model, scaler, df, buffer=5):
    # Predict next 12‑hour net flow for the most recent timestamp
    # Use the same feature columns as during training (exclude target)
    feature_cols = [c for c in df.columns if c != 'target']
    recent = df.iloc[-SEQ_LENGTH:]
    recent_scaled = scaler.transform(recent[feature_cols])
    seq = torch.tensor(recent_scaled[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred_net = model(seq).item()
    # Positive net_flow means more returns than rentals (surplus), negative means deficit
    current_inventory = 5  # assume we have 5 bikes at station now
    projected_inventory = current_inventory + pred_net
    action = None
    if projected_inventory < buffer:
        needed = buffer - projected_inventory
        action = f"Dispatch {int(np.ceil(needed))} bikes to station (deficit)."
    else:
        action = f"No relocation needed (projected inventory {projected_inventory:.1f})."
    return pred_net, projected_inventory, action

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Use all data for training (2023‑2024) and keep 2025 for final check
    full_df = load_data(data_dir)
    # Split by year for a quick train/validation split
    train_df = full_df[full_df.index.year.isin([2023, 2024])]
    test_df = full_df[full_df.index.year == 2025]
    model, scaler = train_model(train_df)
    # Simulate for the latest timestamp in 2025 data
    pred_net, proj_inv, action = simulate_relocation(model, scaler, test_df)
    print("=== LSTM‑Based Relocation Simulation ===")
    print(f"Predicted 12‑h net flow: {pred_net:.2f} bikes")
    print(f"Projected inventory after 12 h (assuming 5 now): {proj_inv:.2f}")
    print(f"Recommended action: {action}")
