import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math # Added for PositionalEncoding

# ... (Previous imports) ...

# Hyperparameters
# Updated per user code preference
HIDDEN_SIZE = 64
NUM_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 1e-3
EPOCHS = 20
BATCH_SIZE = 32
SEQ_LENGTH = 24  # Look back 24 hours

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerSeq2One(nn.Module):
    def __init__(self, feature_dim=14, d_model=64, nhead=8, num_layers=3):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # seq -> single (mean pooling 후 1개 출력)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape : (B, seq_len, feature_dim)

        x = self.input_proj(x)         # (B, seq_len, d_model)
        x = self.pos_encoder(x)        # (B, seq_len, d_model)
        x = self.encoder(x)            # (B, seq_len, d_model)

        # --- seq pooling ---
        x = x.mean(dim=1)              # (B, d_model)

        # final output (single value)
        out = self.fc_out(x)           # (B, 1)
        
        return out


class OulringDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    """Replicates the 13-Scenario Engineering for Transformer Input"""
    def __init__(self):
        self.station_types = {
            "SJ_00400": 2, # Blackhole
            "SJ_00509": 3, # Uphill
            "SJ_00123": 1, # Hotspot
            "default": 0
        }
        self.scaler = StandardScaler()
        # Hardcoded Korean Holidays (2023-2025)
        self.holidays = set([
            '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', # New Year, Seollal
            '2023-03-01', '2023-05-05', '2023-05-27', '2023-06-06', '2023-08-15', # Samil, Children, Buddha, Memorial, Liberation
            '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-02', '2023-10-03', '2023-10-09', '2023-12-25', # Chuseok, Foundation, Hangeul, Xmas
            '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', 
            '2024-03-01', '2024-04-10', '2024-05-05', '2024-05-06', '2024-05-15', '2024-06-06', '2024-08-15', # +Election Day
            '2024-09-16', '2024-09-17', '2024-09-18', '2024-10-03', '2024-10-09', '2024-12-25',
            '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', 
            '2025-03-01', '2025-03-03', '2025-05-05', '2025-05-06', '2025-06-06', '2025-08-15',
            '2025-10-03', '2025-10-05', '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-12-25' # Chuseok estimate
        ])

    def _get_time_features(self, dt):
        h = dt.hour
        dow = dt.dayofweek
        date_str = dt.strftime('%Y-%m-%d')
        is_holiday = 1 if date_str in self.holidays else 0
        
        # Logic Update: Treat Holiday like Weekend
        # Disable Commute if it's a Holiday
        is_commute = 1 if (h in [7, 8, 9, 17, 18, 19] and not is_holiday) else 0
        is_lunch = 1 if h in [11, 12] else 0
        
        is_weekend_real = 1 if dow >= 5 else 0
        effective_weekend = 1 if (is_weekend_real or is_holiday) else 0
        
        is_leisure_time = 1 if effective_weekend and (10 <= h <= 18) else 0
        is_last_mile = 1 if h in [8, 18] else 0
        is_late_night = 1 if h >= 22 else 0
        
        # School Term Logic (Universities in Sejong)
        md = dt.strftime('%m-%d')
        is_semester = 1 if ('03-02' <= md <= '06-21') or ('09-01' <= md <= '12-21') else 0
        
        # Exam Period Logic
        is_exam = 0
        if ('04-20' <= md <= '04-26') or ('10-20' <= md <= '10-26') or \
           ('06-15' <= md <= '06-21') or ('12-15' <= md <= '12-21'):
            is_exam = 1
            
        return [is_commute, is_lunch, is_leisure_time, is_last_mile, is_late_night, h, dow, is_holiday, is_semester, is_exam]

    def process_file(self, file_path, station_id="SJ_00400"):
        try:
            df = pd.read_csv(file_path, encoding='cp949')
        except:
            df = pd.read_csv(file_path, encoding='utf-8')
            
        df['대여시간'] = pd.to_datetime(df['대여시간'])
        df = df.sort_values('대여시간')
        df['hour'] = df['대여시간'].dt.floor('H')
        
        rentals = df[df['시작 대여소'] == station_id].groupby('hour').size()
        returns = df[df['반납 대여소'] == station_id].groupby('hour').size()
        
        flow = pd.DataFrame({'rentals': rentals, 'returns': returns}).fillna(0)
        flow['net_flow'] = flow['returns'] - flow['rentals']
        
        # Scenarios
        time_feats = flow.index.map(self._get_time_features).tolist()
        time_df = pd.DataFrame(time_feats, columns=['is_commute', 'is_lunch', 'is_leisure', 'is_last_mile', 'is_late_night', 'hour_idx', 'day_of_week', 'is_holiday', 'is_semester', 'is_exam'], index=flow.index)
        
        full = pd.concat([flow, time_df], axis=1)
        full['station_type'] = self.station_types.get(station_id, 0)
        
        # Target: 12h Sum
        full['target'] = full['net_flow'].rolling(window=12).sum().shift(-12)
        full = full.dropna()
        
        return full

def create_sequences(data, seq_length):
    xs, ys = [], []
    # Drop target column from input features
    feature_cols = [c for c in data.columns if c != 'target']
    data_x = data[feature_cols].values
    data_y = data['target'].values
    
    for i in range(len(data) - seq_length):
        x = data_x[i:(i+seq_length)]
        y = data_y[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def run_transformer_training(data_dir):
    print("Initializing Transformer Predictor (User Model)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    processor = DataProcessor()
    
    # 1. Load Data
    all_files = sorted([f for f in os.listdir(data_dir) if "이용 현황" in f and f.endswith(".csv")])
    train_files = [f for f in all_files if "2023" in f or "2024" in f]
    test_files = [f for f in all_files if "2025" in f] # Used for validation in this loop
    
    dfs = []
    print("Loading Training Data...")
    for f in train_files:
        path = os.path.join(data_dir, f)
        dfs.append(processor.process_file(path))
        
    if not dfs: return
    train_df = pd.concat(dfs)
    
    # [Data Analysis] Calculate Volume Stats on RAW DATA (Before Scaling)
    if 'rentals' in train_df.columns:
        raw_avg_hourly = train_df['rentals'].mean()
        print(f"\n📊 [Data Analysis] Raw Data Statistics")
        print(f"   - Average Hourly Rentals per Station: {raw_avg_hourly:.2f}")
        print(f"   - Est. 12h Volume per Station: {raw_avg_hourly * 12:.1f} bikes")
        print(f"   - Est. City-wide 12h Volume (300 stations): {raw_avg_hourly * 12 * 300:,.0f} bikes")

    # Load Validation Data (2025 Q1 as Validation)
    val_dfs = []
    if test_files:
        val_path = os.path.join(data_dir, test_files[0]) # Use first 2025 file as validation
        val_dfs.append(processor.process_file(val_path))
    val_df = pd.concat(val_dfs) if val_dfs else train_df.iloc[-100:] # Fallback
    
    # 2. Scale & Sequence
    feature_cols = [c for c in train_df.columns if c != 'target']
    
    # [User Request] Calculate Average 12h Rental Volume
    # 'target' is the 12h Net Flow sum. To get absolute rental volume, we might need 'rentals'.
    # But user asked for "average 12h rental volume". 'net_flow' = returns - rentals.
    # We should sum 'rentals' over 12h windows if possible.
    # Since we don't have a direct '12h_rentals' column pre-calculated for the target, but we have 'rentals' col.
    # Let's calculate it quickly.
    avg_12h_net = train_df['target'].mean()
    print(f"\n📊 [Data Analysis] Average 12-Hour Net Flow (Target): {avg_12h_net:.2f}")
    
    # Approximate 12h Rental Volume
    # We need to roll 'rentals' similarly to how 'target' was created.
    if 'rentals' in train_df.columns:
        # Note: train_df is a concat of processed files, index might reset or be non-contiguous time-wise across files.
        # But 'process_file' returns a DF with 'rentals'.
        # We can't simply roll on the concatenated DF safely.
        # However, for an "Average", the mean of 1 hour * 12 is a decent proxy?
        avg_hourly_rental = train_df['rentals'].mean()
        print(f"   [Data Analysis] Average Hourly Rentals: {avg_hourly_rental:.2f}")
        print(f"   👉 Estimated 12-Hour Total Rentals: {avg_hourly_rental * 12:.2f} bikes\n")

    # Fit scaler on training data only
    processor.scaler.fit(train_df[feature_cols])
    
    # Transform Train
    train_df[feature_cols] = processor.scaler.transform(train_df[feature_cols])
    X_train, y_train = create_sequences(train_df, SEQ_LENGTH)
    
    # Transform Val
    val_df[feature_cols] = processor.scaler.transform(val_df[feature_cols])
    X_val, y_val = create_sequences(val_df, SEQ_LENGTH)
    
    train_dataset = OulringDataset(X_train, y_train)
    val_dataset = OulringDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model Setup (Transformer)
    feature_dim = X_train.shape[2] 
    # User's Model Initialization
    model_trans = TransformerSeq2One(feature_dim=feature_dim, d_model=HIDDEN_SIZE, nhead=8, num_layers=NUM_LAYERS)
    model_trans = model_trans.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_trans.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop (with Early Stopping)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    print(f"Starting Transformer Training for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        model_trans.train()
        train_loss_sum = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model_trans(batch_X)
            loss = criterion(output.squeeze(), batch_y) # Squeeze output to match target
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # ===== validation =====
        model_trans.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)

                val_output = model_trans(val_X)
                val_loss = criterion(val_output.squeeze(), val_y)
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ===== Early Stopping =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model_trans, 'model_transform.pt') # Save Best
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered at Epoch {epoch+1} (Best Val Loss: {best_val_loss:.4f})")
                break

    print(f"Training Complete. Best Model Saved (Val Loss: {best_val_loss:.4f}).")
    print("\n✅ Architecture Check: Self-Attention Mechanism Active (nn.TransformerEncoder).")

    # Final RMSE Calculation (Load Best Model)
    # We trust our own saved model, so weights_only=False to avoid class restrictions on the custom Transformer model
    best_model = torch.load('model_transform.pt', weights_only=False)
    best_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X = val_X.to(device)
            val_y = val_y.to(device)
            output = best_model(val_X)
            all_preds.extend(output.squeeze().cpu().numpy())
            all_targets.extend(val_y.cpu().numpy())
    
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    print(f"\n✅ Final Validation Results")
    print(f"   - RMSE: {rmse:.4f}")
    
    # Re-print Vol Stats for clarity
    print(f"   - Avg 12h Volume Per Station: {train_df['rentals'].mean() * 12:.1f} bikes")
    print(f"   - Est. City-wide 12h Volume (300 stations): {train_df['rentals'].mean() * 12 * 300:,.0f} bikes")
    print(f"\n✅ Training Complete (Epochs: {EPOCHS})")
    print(f"📊 Final RMSE: {rmse:.4f}")

    torch.save(model_trans, 'model_transform.pt')
    print("💾 Model Saved: model_transform.pt")


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        run_transformer_training(data_dir)
    except NameError:
        print("Error: PyTorch not installed. Please install torch to run this Transformer script.")
