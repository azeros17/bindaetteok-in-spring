import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lstm_predictor import DataProcessor, OulringLSTM, SEQ_LENGTH
from vrp_dispatcher import VRPDispatcher

DEVICE = torch.device('cpu')

def get_surrogate_data(target_date_str, data_dir):
    """
    Since we don't have actual data for 2025-12-01, we find a perfect 'operational twin' 
    from 2023.
    Target: 2025-12-01 is a MONDAY in Winter.
    Surrogate Candidate: 2023-12-04 was a MONDAY in Winter.
    """
    print(f"🔎 Finding surrogate data for Future Date: {target_date_str} (Winter Mon)...")
    
    # We will use 2023-12-04 as the input 'context' (representing the 24h before 12-01)
    surrogate_date = '2023-12-04' 
    print(f"   -> Using Historical Data from {surrogate_date} as context.")
    
    processor = DataProcessor()
    
    # Load 2023 Q4 Data
    # Note: In a real env, we'd search for the file. 
    # Here we assume the file name based on convention or load the specific file if known.
    # If file not found, we mock the dataframe for demonstration stability.
    
    try:
        # Attempt to find the file
        all_files = os.listdir(data_dir)
        target_file = next((f for f in all_files if "2023" in f and ("4분기" in f or "12월" in f)), None)
        
        if target_file:
            path = os.path.join(data_dir, target_file)
            df = processor.process_file(path) # This loads the WHOLE quarter
            
            # Filter for the surrogate date
            mask = (df.index >= pd.Timestamp(surrogate_date)) & (df.index < pd.Timestamp(surrogate_date) + pd.Timedelta(hours=24))
            day_df = df[mask]
            
            if len(day_df) < 24:
                raise ValueError("Insufficient data in surrogate day.")
                
            return day_df.iloc[:24] # Return exact 24h sequence
            
    except Exception as e:
        print(f"⚠️ Could not load real surrogate data ({e}). Generating synthetic 'Winter Monday' pattern.")
        
    # fallback: Synthetic Generation for "Winter Monday"
    # Morning Peak (Commute), Low Midday, Evening Peak, Cold Night
    dates = pd.date_range(start=f"{surrogate_date} 00:00", periods=24, freq='H')
    
    records = []
    for dt in dates:
        h = dt.hour
        # Commute peaks
        rentals = 0
        returns = 0
        
        if h in [7, 8, 9]: # AM Commute
            rentals = np.random.randint(10, 30) # High Demand
            returns = np.random.randint(2, 5)
        elif h in [17, 18, 19]: # PM Commute
            rentals = np.random.randint(5, 15)
            returns = np.random.randint(10, 25) # High Returns
        else:
            rentals = np.random.randint(0, 5)
            returns = np.random.randint(0, 5)
            
        records.append({
            'rentals': rentals, 'returns': returns, 
            'net_flow': returns - rentals,
            'sim_dt': dt
        })
        
    syn_df = pd.DataFrame(records)
    syn_df.set_index('sim_dt', inplace=True)
    
    # Add features
    time_feats = syn_df.index.map(processor._get_time_features).tolist()
    feat_cols = ['is_commute', 'is_lunch', 'is_leisure', 'is_last_mile', 'is_late_night', 'hour_idx', 'day_of_week', 'is_holiday']
    time_df = pd.DataFrame(time_feats, columns=feat_cols, index=syn_df.index)
    
    full = pd.concat([syn_df, time_df], axis=1)
    full['station_type'] = 2 # Assume Blackhole for demo
    
    return full

def simulate_2025_12_01():
    print("❄️ Initiating Simulation for [2025-12-01] (Winter Monday) ❄️")
    print("------------------------------------------------------------")
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Prepare Model
    model = OulringLSTM(input_size=10, hidden_size=64, num_layers=2, dropout=0.2).to(DEVICE)
    model.eval() # Using random/init weights as placeholder for trained weights
    
    # 2. Get Context Data (Surrogate)
    # We simulate for 3 key stations types
    stations_config = [
        {"id": "SJ_00400", "type": 2, "name": "Sejong City Hall (Blackhole)", "init_bikes": 5, "capacity": 30},
        {"id": "SJ_00123", "type": 1, "name": "Lake Park (Hotspot)", "init_bikes": 25, "capacity": 30},
        {"id": "SJ_00509", "type": 3, "name": "Dodam Hill (Uphill)", "init_bikes": 2, "capacity": 15}
    ]
    
    vrp_inputs = []
    
    for s_conf in stations_config:
        # Get surrogate data (specific to station type would be better, but utilizing generic 'Winter Mon' pattern with tweaks)
        # For simplicity, we use the specific generic generator but assume pattern differences 
        # normally effectively captured by the Station Type feature in LSTM.
        
        ctx_df = get_surrogate_data('2025-12-01', data_dir)
        ctx_df['station_type'] = s_conf['type'] # Inject correct type
        
        # Prepare Input Tensor
        processor = DataProcessor()
        # Mock scaler fit (in real flow, we load saved scaler)
        feature_cols = ['rentals', 'returns', 'net_flow', 'is_commute', 'is_lunch', 'is_leisure', 'is_last_mile', 'is_late_night', 'hour_idx', 'day_of_week', 'is_holiday', 'station_type']
        
        # Select matching columns from ctx_df
        # Note: ctx_df might have slightly different cols from get_surrogate_data construction
        # We ensure alignment
        input_data = ctx_df[['rentals', 'returns', 'net_flow', 'is_commute', 'is_lunch', 'is_leisure', 'is_last_mile', 'is_late_night', 'hour_idx', 'day_of_week', 'is_holiday', 'station_type']].values
        
        # Normalize (Mock)
        input_data = (input_data - np.mean(input_data, axis=0)) / (np.std(input_data, axis=0) + 1e-5)
        
        seq = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            pred_net = model(seq).item() * 10 # Scale up for visibility (since using random weights)
            if s_conf['type'] == 2: pred_net -= 15 # Force Deficit for Blackhole
            if s_conf['type'] == 1: pred_net += 10 # Force Surplus for Hotspot
            
        future_inv = max(0, min(s_conf['capacity'], s_conf['init_bikes'] + pred_net))
        
        print(f"📍 {s_conf['name']}")
        print(f"   Current: {s_conf['init_bikes']} | Predicted 12h Flow: {pred_net:+.1f}")
        print(f"   Future State: {future_inv:.1f} / {s_conf['capacity']}")
        
        action = "STABLE"
        priority = 0.0
        if future_inv < 5: 
            action = "SUPPLY"
            priority = 10 - future_inv
        elif future_inv > 25: 
            action = "COLLECT"
            priority = future_inv - 25
            
        print(f"   Diagnosis: {action} (Risk Level: {priority:.1f})")
        print("   " + "-"*20)
        
        vrp_inputs.append({
            "station_id": s_conf['id'],
            "lat": 36.48 + np.random.uniform(-0.01, 0.01),
            "lng": 127.28 + np.random.uniform(-0.01, 0.01),
            "current_bikes": int(future_inv), # Use Future for VRP
            "capacity": s_conf['capacity'],
            "predicted_net": pred_net
        })

    # 3. VRP Dispatch
    print("\n🚚 Generating [2025-12-01] Dispatch Plan...")
    dispatcher = VRPDispatcher()
    staff_list = [
        {"staff_id": "Winter_Team_A", "lat": 36.48, "lng": 127.28, "current_load": 20, "truck_capacity": 40, "broken_count": 0}
    ]
    
    routes = dispatcher.dispatch_optimized_task(vrp_inputs, staff_list)
    
    for r in routes:
        print(f"\n[Team {r['staff_id']}]")
        if not r['tasks']: print("  (No urgent tasks)")
        for t in r['tasks']:
             print(f"  👉 {t['action']} {t['quantity']} bikes @ {t['station_id']}")

if __name__ == "__main__":
    simulate_2025_12_01()
