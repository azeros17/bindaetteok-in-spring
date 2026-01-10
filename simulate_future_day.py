import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformer_predictor import DataProcessor, TransformerSeq2One, SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS
from vrp_dispatcher import VRPDispatcher

DEVICE = torch.device('cpu')

def get_surrogate_data(target_date_str, data_dir):
    """
    Finds 'operational twin' data.
    Target: 2025-03-10 (Monday, Spring)
    Decision Time: 15:00
    Surrogate: 2024-03-11 (Monday, Spring) -> Fetch 2024-03-10 15:00 ~ 2024-03-11 15:00 (24h context)
    """
    print(f"🔎 Finding surrogate data for Future Date: {target_date_str} (Spring Mon)...")
    
    # 2024-03-11 was a Monday. We need context ending at 15:00 on Monday.
    # So we need data from Sunday 15:00 (Mar 10) to Monday 15:00 (Mar 11).
    surrogate_target = '2024-03-11'
    start_dt = pd.Timestamp('2024-03-10 15:00')
    end_dt = pd.Timestamp('2024-03-11 15:00')
    
    print(f"   -> Using Historical Data Window: {start_dt} ~ {end_dt} (24h Context)")
    
    processor = DataProcessor()
    
    try:
        # Attempt to find the file (Looking for 2024 1Q or March)
        all_files = os.listdir(data_dir)
        target_file = next((f for f in all_files if "2024" in f and ("1분기" in f or "03월" in f)), None)
        
        if target_file:
            path = os.path.join(data_dir, target_file)
            df = processor.process_file(path) # Loads whole period
            
            # Filter for window
            mask = (df.index >= start_dt) & (df.index < end_dt)
            day_df = df[mask]
            
            if len(day_df) < 24:
                # If we can't find exact window (e.g. file split), fallback to simple day
                print("   ⚠️ Exact 15:00 window not found/complete. Falling back to simple daily 00:00-23:00.")
                fallback_dt = pd.Timestamp(surrogate_target)
                mask = (df.index >= fallback_dt) & (df.index < fallback_dt + pd.Timedelta(hours=24))
                day_df = df[mask]
                
            if len(day_df) >= 24:
                return day_df.iloc[:24]
            else: 
                 print("   ⚠️ Data found but insufficient for 24h.")
                
    except Exception as e:
        print(f"⚠️ Could not load real surrogate data ({e}). Generating synthetic 'Spring Monday' pattern.")
        
    # fallback: Synthetic Generation
    # If using fallback, we generate starting from 15:00
    dates = pd.date_range(start=f"2025-03-10 15:00", periods=24, freq='h')
    
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
    # Now _get_time_features returns 10 items
    feat_cols = ['is_commute', 'is_lunch', 'is_leisure', 'is_last_mile', 'is_late_night', 'hour_idx', 'day_of_week', 'is_holiday', 'is_semester', 'is_exam']
    time_df = pd.DataFrame(time_feats, columns=feat_cols, index=syn_df.index)
    
    full = pd.concat([syn_df, time_df], axis=1)
    full['station_type'] = 2 # Assume Blackhole for demo
    
    return full

def simulate_2025_03_10(num_trucks=3):
    print(f"🌸 Initiating Simulation for [2025-03-10] (Spring Monday) [TRANSFORMER EDITION] 🌸")
    print(f"🕒 Decision Time: 15:00 (3:00 PM) | Target Window: 15:00 ~ Next Day 03:00")
    print(f"🚚 Fleet Size: {num_trucks} Trucks Configured")
    print("-----------------------------------------------------------------------------------")
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Prepare Model
    # Feature dim derived from DataProcessor: 14
    model = TransformerSeq2One(feature_dim=14, d_model=HIDDEN_SIZE, nhead=8, num_layers=NUM_LAYERS).to(DEVICE)
    model.eval() 
    
    # 2. Get Context Data (Surrogate)
    # We simulate for 3 key stations types
    stations_config = [
        {"id": "SJ_00400", "type": 2, "name": "Sejong City Hall (Blackhole)", "init_bikes": 5, "capacity": 30},
        {"id": "SJ_00123", "type": 1, "name": "Lake Park (Hotspot)", "init_bikes": 25, "capacity": 30},
        {"id": "SJ_00509", "type": 3, "name": "Dodam Hill (Uphill)", "init_bikes": 2, "capacity": 15}
    ]
    
    vrp_inputs = []
    
    for s_conf in stations_config:
        # Note: We assume get_surrogate_data fetches the 24h prior to 15:00.
        ctx_df = get_surrogate_data('2025-03-10', data_dir) 
        ctx_df['station_type'] = s_conf['type'] # Inject correct type
        
        # Prepare Input Tensor
        processor = DataProcessor()
        # Ensure column order matches DataProcessor.process_file
        # rentals, returns, net_flow (3) + time_feats (10) + station_type (1) = 14
        input_cols = ['rentals', 'returns', 'net_flow', 
                      'is_commute', 'is_lunch', 'is_leisure', 'is_last_mile', 'is_late_night', 'hour_idx', 'day_of_week', 'is_holiday', 'is_semester', 'is_exam',
                      'station_type']
        
        # Select and reorder
        input_data = ctx_df[input_cols].values
        
        # Normalize (Mock)
        input_data = (input_data - np.mean(input_data, axis=0)) / (np.std(input_data, axis=0) + 1e-5)
        
        seq = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Predict
        # Predict
        with torch.no_grad():
            pred_net = int(round(model(seq).item() * 10))
            # Adjust mock logic for PM window
            if s_conf['type'] == 2: pred_net -= 25 # Blackhole: Huge Outflow in PM Commute (15:00-19:00)
            if s_conf['type'] == 1: pred_net += 5  # Hotspot: Balanced or slight inflow
            
        future_inv = int(max(0, min(s_conf['capacity'], s_conf['init_bikes'] + pred_net)))
        
        print(f"📍 {s_conf['name']}")
        print(f"   Current (15:00): {s_conf['init_bikes']} | Predicted 12h Flow (until 03:00): {pred_net:+d}")
        print(f"   Future State: {future_inv} / {s_conf['capacity']}")
        
        action = "STABLE"
        priority = 0.0
        if future_inv < 5: 
            action = "SUPPLY"
            priority = 10.0 - future_inv
        elif future_inv > 25: 
            action = "COLLECT"
            priority = future_inv - 25.0
            
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

    # 3. VRP Dispatch (Fixed Fleet)
    print(f"\n🚚 Generating [2025-03-10 15:00] Dispatch Plan for {num_trucks} Trucks...")
    dispatcher = VRPDispatcher()
    
    # Dynamically correct staff list
    staff_list = []
    for i in range(num_trucks):
        staff_list.append({
            "staff_id": f"Truck_{i+1}", 
            "lat": 36.48, # Depot Start
            "lng": 127.28, 
            "current_load": 20, 
            "truck_capacity": 40, 
            "broken_count": 0
        })
    
    routes = dispatcher.dispatch_optimized_task(vrp_inputs, staff_list)
    
    for r in routes:
        print(f"\n[Team {r['staff_id']}]")
        if not r['tasks']: print("  (No urgent tasks)")
        for t in r['tasks']:
             print(f"  👉 {t['action']} {t['quantity']} bikes @ {t['station_id']}")

if __name__ == "__main__":
    simulate_2025_03_10(num_trucks=3)
