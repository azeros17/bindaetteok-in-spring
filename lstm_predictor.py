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

# Hyperparameters
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
SEQ_LENGTH = 24  # Look back 24 hours

class OulringLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(OulringLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1) # Predict scalar (Target 12h Sum)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Take last time step
        out = self.fc(out)
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
    """Replicates the 13-Scenario Engineering for LSTM Input"""
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
        
        # Leisure logic applies to Holidays too
        is_leisure_time = 1 if effective_weekend and (10 <= h <= 18) else 0
        is_last_mile = 1 if h in [8, 18] else 0
        is_late_night = 1 if h >= 22 else 0
        
        # Added is_holiday to features, now adding Academic Calendar
        
        # School Term Logic (Universities in Sejong)
        # Sem 1: Mar 02 - Jun 21
        # Sem 2: Sep 01 - Dec 21
        md = dt.strftime('%m-%d')
        is_semester = 1 if ('03-02' <= md <= '06-21') or ('09-01' <= md <= '12-21') else 0
        
        # Exam Period Logic (Approximate)
        # Mid: Apr 20-26, Oct 20-26
        # Final: Jun 15-21, Dec 15-21
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

def run_lstm_training(data_dir):
    print("Initializing LSTM Scenario Predictor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    processor = DataProcessor()
    
    # 1. Load Data
    all_files = sorted([f for f in os.listdir(data_dir) if "이용 현황" in f and f.endswith(".csv")])
    train_files = [f for f in all_files if "2023" in f or "2024" in f]
    
    dfs = []
    print("Loading Training Data...")
    for f in train_files:
        path = os.path.join(data_dir, f)
        dfs.append(processor.process_file(path))
        
    if not dfs: return
    train_df = pd.concat(dfs)
    
    # 2. Scale & Sequence
    feature_cols = [c for c in train_df.columns if c != 'target']
    # Fit scaler on training data only
    processor.scaler.fit(train_df[feature_cols])
    
    # Transform
    train_df[feature_cols] = processor.scaler.transform(train_df[feature_cols])
    X_train, y_train = create_sequences(train_df, SEQ_LENGTH)
    
    train_dataset = OulringDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model Setup
    input_size = X_train.shape[2] 
    model = OulringLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Train
    model.train()
    print(f"Starting Training for {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")
            
    # 5. Evaluate
    print("Evaluating on 2025 Data...")
    test_files = [f for f in all_files if "2025" in f]
    model.eval()
    
    for f in test_files:
        test_df = processor.process_file(os.path.join(data_dir, f))
        test_df[feature_cols] = processor.scaler.transform(test_df[feature_cols])
        X_test, y_test = create_sequences(test_df, SEQ_LENGTH)
        
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().flatten()
            
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Test File: {f} -> RMSE: {rmse:.2f}")

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        run_lstm_training(data_dir)
    except NameError:
        print("Error: PyTorch not installed. Please install torch to run this LSTM script.")
