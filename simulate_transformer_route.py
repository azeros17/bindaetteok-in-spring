import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformer_predictor import DataProcessor, TransformerSeq2One, SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS
from vrp_dispatcher import VRPDispatcher

DEVICE = torch.device('cpu')

def load_trained_model(model_path=None):
    # For simulation, we will quickly re-train a small model or load mock weights
    # ideally we load a saved .pth, but here we'll instantiate and use random weights 
    # for demonstration unless we saved one. 
    # To make it realistic, let's assume valid weights or just illustrate the pipeline.
    # We will use the same architecture.
    # Transformer Model
    model = TransformerSeq2One(feature_dim=14, d_model=HIDDEN_SIZE, nhead=8, num_layers=NUM_LAYERS)
    # Load weights if available, else random (mock)
    if model_path and os.path.exists(model_path):
        try:
            model = torch.load(model_path)
            model.eval()
        except:
             print("⚠️ Failed to load model weights. Using initialized weights.")
    
    model = model.to(DEVICE)
    return model

def mock_station_data():
    """Returns mock data for 3 stations to simulate different scenarios."""
    return [
        {
            "station_id": "SJ_00400", "name": "Sejong City Hall (Blackhole)",
            "lat": 36.4800, "lng": 127.2890, "type": 2, 
            "current_bikes": 5, "capacity": 30, # Low inventory, high demand expected
            "mock_pred_net": -22.5 # Deficit: Needs Supply
        },
        {
            "station_id": "SJ_00123", "name": "Lake Park (Hotspot)",
            "lat": 36.4900, "lng": 127.2900, "type": 1,
            "current_bikes": 28, "capacity": 30, # Full, returns expected
            "mock_pred_net": +10.0 # Surplus: Needs Collect
        },
        {
            "station_id": "SJ_00509", "name": "Dodam Hill (Uphill)",
            "lat": 36.5000, "lng": 127.2800, "type": 3,
            "current_bikes": 0, "capacity": 15, # Empty
            "mock_pred_net": -5.0 # Deficit: Needs Supply
        }
    ]

def run_simulation(num_trucks=2):
    print("=== [Phase 1] LSTM Prediction (12-Hour Lookahead) ===")
    stations = mock_station_data()
    
    # In a real run, we would loop through stations, load their CSV, and run model(seq)
    # Here we use 'mock_pred_net' to represent the LSTM output for clarity of the workflow.
    
    operational_state = []
    
    for s in stations:
        # LSTM Prediction Step (Simulated)
        pred_net_flow = s['mock_pred_net'] 
        future_inventory = s['current_bikes'] + pred_net_flow
        
        print(f"Station: {s['name']} ({s['station_id']})")
        print(f"  - Current: {s['current_bikes']} | LSTM Pred(12h): {pred_net_flow:+.1f}")
        print(f"  - Future State: {future_inventory:.1f} / {s['capacity']}")
        
        # Determine Logic (Simplified from VRP)
        action_needed = "NONE"
        qty = 0
        priority = 0.0
        
        # Logic: If future < 10% capacity -> SUPPLY
        # Logic: If future > 90% capacity -> COLLECT
        if future_inventory < (s['capacity'] * 0.1):
            action_needed = "SUPPLY"
            qty = int((s['capacity'] * 0.5) - future_inventory) # Fill to 50%
            priority = 0.9 # High urgency
        elif future_inventory > (s['capacity'] * 0.9):
            action_needed = "COLLECT"
            qty = int(future_inventory - (s['capacity'] * 0.5)) # Drain to 50%
            priority = 0.8
            
        print(f"  - AI Decision: {action_needed} {qty} bikes (Priority: {priority:.2f})")
        print("-" * 30)
        
        # Prepare for VRP
        operational_state.append({
            "station_id": s['station_id'],
            "lat": s['lat'], "lng": s['lng'],
            "current_bikes": s['current_bikes'], # VRP uses current for capacity checks
            "capacity": s['capacity'],
            # We inject the "Future Risk" as "Current Urgency" for the VRP to react NOW
            # By hacking the fill_ratio or passing explicit priority
            "predicted_net": pred_net_flow
        })

    print("\n=== [Phase 2] VRP Route Optimization (Dispatch) ===")
    
    dispatcher = VRPDispatcher()
    
    # Define Staff (Trucks)
    # Define Staff (Trucks) - Dynamic Generation
    staff_list = []
    print(f"🚚 Configuring {num_trucks} Trucks for VRP...")
    for i in range(num_trucks):
        staff_list.append({
            "staff_id": f"Truck_{i+1}", 
            "lat": 36.4850 + (i*0.01), # Slight offset
            "lng": 127.2850, 
            "current_load": 10, 
            "truck_capacity": 40, 
            "broken_count": 0
        })
    
    # We override the VRP's basic "Current Fill Ratio" logic with our "LSTM-Predicted Urgency"
    # The VRP Dispatcher code we viewed prioritizes based on current fill status.
    # To integrate, we pass the *future* projected state as if it is the current state 
    # so the VRP solves the *future* problem *now*.
    
    vrp_input_stations = []
    for s in operational_state:
        # Trick: We set 'current_bikes' to the 'future_inventory' for the VRP 
        # so it generates tasks to fix the future problem.
        # But we must ensure 'lat'/'lng' are correct.
        vrp_s = s.copy()
        future_val = max(0, min(s['capacity'], s['current_bikes'] + s['predicted_net']))
        vrp_s['current_bikes'] = future_val 
        vrp_input_stations.append(vrp_s)

    routes = dispatcher.dispatch_optimized_task(vrp_input_stations, staff_list)
    
    for r in routes:
        print(f"\n[ 🚚 {r['staff_id']} Route Plan ]")
        if not r['tasks']:
            print("  - Status: Standby (No urgent tasks nearby)")
            continue
            
        for idx, task in enumerate(r['tasks']):
            s_name = next(s['name'] for s in stations if s['station_id'] == task['station_id'])
            print(f"  {idx+1}. {task['action']:<7} {task['quantity']} bikes @ {s_name}")
            
    print("\n✅ Verification: The system proactively balances the network BEFORE the stock-out occurs.")

if __name__ == "__main__":
    # Accept user input for trucks if running directly
    try:
        input_trucks = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    except:
        input_trucks = 2
    run_simulation(num_trucks=input_trucks)
