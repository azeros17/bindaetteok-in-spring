import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path (for DataProcessor class)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformer_predictor import DataProcessor, TransformerSeq2One, SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS

DEVICE = torch.device('cpu')

def load_data(data_dir, station_id="SJ_00400"):
    # Load all CSV files (train + test) for full historical view
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and "이용 현황" in f]
    dfs = []
    processor = DataProcessor()
    for f in files:
        path = os.path.join(data_dir, f)
        dfs.append(processor.process_file(path, station_id)) # This uses the UPDATED DataProcessor (14 feats)
    
    if not dfs: return pd.DataFrame()
    full = pd.concat(dfs).sort_index()
    return full

def prepare_sequences(df, scaler):
    feature_cols = [c for c in df.columns if c != 'target']
    df_copy = df.copy() # Safe copy
    df_copy[feature_cols] = scaler.transform(df[feature_cols])
    X = []
    
    data_x = df_copy[feature_cols].values
    
    for i in range(len(df_copy) - SEQ_LENGTH):
        X.append(data_x[i:i+SEQ_LENGTH])
        
    X = np.array(X, dtype=np.float32)
    return torch.tensor(X)

def train_model(train_df):
    processor = DataProcessor()
    scaler = StandardScaler()
    feature_cols = [c for c in train_df.columns if c != 'target']
    
    # Fit scaler
    scaler.fit(train_df[feature_cols])
    
    X_train = prepare_sequences(train_df, scaler)
    y_train = torch.tensor(train_df['target'].values[SEQ_LENGTH:], dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Updated to Transformer
    feature_dim = len(feature_cols) # Should be 14
    model = TransformerSeq2One(feature_dim=feature_dim, d_model=HIDDEN_SIZE, nhead=8, num_layers=NUM_LAYERS, dropout=0.1).to(DEVICE)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print("Training Transformer for Single Station Relocation Sim...")
    for epoch in range(10): # Quick demo training
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred.squeeze(), yb.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")
    
    return model, scaler

def simulate_relocation(model, scaler, df, buffer=5):
    feature_cols = [c for c in df.columns if c != 'target']
    
    # We need the last SEQ_LENGTH rows
    if len(df) < SEQ_LENGTH:
        print("Not enough data for sequence.")
        return 0, 5, "Insufficient Data"
        
    recent = df.iloc[-SEQ_LENGTH:].copy()
    recent[feature_cols] = scaler.transform(recent[feature_cols])
    
    seq = torch.tensor(recent[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        pred_net = int(round(model(seq).item() * 10)) # Round to integer
    
    # Logic
    current_inventory = 5 # Mock current
    projected_inventory = int(current_inventory + pred_net)
    
    action = None
    if projected_inventory < buffer:
        needed = buffer - projected_inventory
        action = f"Dispatch {needed} bikes (deficit)."
    else:
        action = f"No action needed (Proj: {projected_inventory})."
        
    return pred_net, projected_inventory, action

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_df = load_data(data_dir)
    
    if full_df.empty:
        print("No CSV data found.")
    else:
        # Train on 2023-2024
        train_df = full_df[full_df.index.year.isin([2023, 2024])]
        # Test on 2025
        test_df = full_df[full_df.index.year == 2025]
        
        if not train_df.empty:
            model, scaler = train_model(train_df)
            if not test_df.empty:
                pred_net, proj_inv, action = simulate_relocation(model, scaler, test_df)
                print("=== Transformer-Based Relocation Simulation (Single Station) ===")
                print(f"Predicted 12‑h Net Flow: {pred_net:+d} bikes")
                print(f"Projected Inventory: {proj_inv}")
                print(f"Recommended Action: {action}")
            else:
                print("No 2025 Test Data found.")
        else:
            print("No Training Data found.")
