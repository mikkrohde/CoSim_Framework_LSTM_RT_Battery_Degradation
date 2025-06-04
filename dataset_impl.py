import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns 

# -----------------------------------------------------------------------------
# Processor + Dataset
# -----------------------------------------------------------------------------

class AngleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        """Convert angles to [sin(theta), cos(theta)]"""
        return np.hstack([np.sin(X), np.cos(X)])
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for transformed angles"""
        if input_features is None:
            return ['sin', 'cos']
        return [f"{col}_{trig}" for col in input_features for trig in ['sin', 'cos']]

def make_transformer(unnormalised_cols:list[str], numeric_cols:list[str], categorical_cols:list[str], age_levels = [1, 5, 20]):
    ohe_age = OneHotEncoder(
        categories=[age_levels],    
        sparse_output=False,
        handle_unknown="ignore"
    )

    ohe_traffic = OneHotEncoder(
        categories=[[0, 1]],  # Define expected traffic light states
        handle_unknown="ignore"
    )
    
    angle_cols = [col for col in numeric_cols if col in ["carla_roll", "carla_pitch", "carla_yaw"]]
    #degradation_col = [col for col in numeric_cols if col == "sim_battery_degradation"]
    numeric_cols = [col for col in numeric_cols if col not in (angle_cols)]#+ degradation_col

    return ColumnTransformer([
        #("pass", 'passthrough', unnormalised_cols),
        ("age",   ohe_age, [categorical_cols[0]]),
        #("deg", RobustScaler(), degradation_col),
        ("num", StandardScaler(), numeric_cols),
        ("ang", AngleTransformer(), angle_cols),
        ("cat", ohe_traffic, [categorical_cols[1]]),
    ])

def global_state_transformer(csv_dir, unnormalised_cols, numeric_cols, categorical_cols):
    # Read all CSVs
    all_dfs = [pd.read_csv(p) for p in Path(csv_dir).glob("*.csv")]
    full_df = pd.concat(all_dfs, ignore_index=True)
    transformers = make_transformer(unnormalised_cols, numeric_cols, categorical_cols)
    return transformers.fit(full_df)

class TBPTTVehicleDataset(Dataset):
    def __init__(self, sequences):
        """
        sequences: list of tuples (s_seq, a_seq, ns_seq, r_seq, d_seq)
           where each s_seq is np.ndarray [seq_len, state_dim], etc.
        """
        if len(sequences) == 0:
            raise ValueError("TBPTTDataset received 0 sequences!")
        
        self.seq_len = sequences[0][0].shape[0]
        self.state_dim  = sequences[0][0].shape[1]
        self.action_dim = sequences[0][1].shape[1]
        self.samples = sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state_seq, action_seq, next_state_seq, reward_seq, done_seq = self.samples[idx]
        return (
            torch.from_numpy(state_seq).float(),        # [seq_len, state_dim]
            torch.from_numpy(action_seq).float(),       # [seq_len, action_dim]
            torch.from_numpy(next_state_seq).float(),   # [seq_len, state_dim]
            torch.from_numpy(reward_seq).unsqueeze(-1), # [seq_len, 1]
            torch.from_numpy(done_seq).unsqueeze(-1),   # [seq_len, 1]
        )

# -----------------------------------------------------------------------------
# prepare_dataset function
# -----------------------------------------------------------------------------
def prepare_dataset(csv_dir, unnormalised_cols:list[str], numeric_cols:list[str], categorical_cols:list[str], action_cols:list[str], seq_len:int=64, step:int=1, transformer: Optional[ColumnTransformer] = None, is_test:bool=False):
    csv_path = Path(csv_dir)
    
    if csv_path.is_file():
        all_dfs = [pd.read_csv(csv_path)]
    elif csv_path.is_dir():
        all_dfs = [pd.read_csv(p) for p in csv_path.glob("*.csv")]
    else:
        raise ValueError(f"Invalid path: {csv_dir}")

    if len(all_dfs) == 1:
        full_df = all_dfs[0]
    else:
        full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Transformer handling
    if not is_test and transformer is None:
        # Fit on training data
        ct = make_transformer(unnormalised_cols, numeric_cols, categorical_cols)
        ct.fit(full_df)  # Fit once on ALL training scenarios
        feature_names = ct.get_feature_names_out().tolist()
    else:
        # Use pre-fitted transformer for test
        ct = transformer  
        feature_names = ct.get_feature_names_out().tolist() if ct else []
    
    # Transform the full dataset
    X_state = ct.transform(full_df)
    X_action = full_df[action_cols].values.astype(np.float32)

    degradation_scaled = X_state[:, feature_names.index("num__sim_battery_degradation")]
    print(f"Degradation stats - Mean: {degradation_scaled.mean():.2f}, Std: {degradation_scaled.std():.2f}")

    #Build windows in memory
    windows = []
    for start in range(0, len(full_df) - seq_len, step):
        end = start + seq_len
        s_seq  = X_state[start:end]
        a_seq  = X_action[start:end]
        ns_seq = X_state[start+1:end+1]
        r_seq  = np.zeros(seq_len, dtype=np.float32)
        d_seq  = np.zeros(seq_len, dtype=np.float32)
        windows.append((s_seq, a_seq, ns_seq, r_seq, d_seq))

    #Return as torch dataset
    dataset = TBPTTVehicleDataset(windows)
    return dataset, feature_names, ct

"""df       = pd.read_csv(csv_path) 
        state_cols = unnormalised_cols + numeric_cols + categorical_cols
        if is_test or transformer is not None:
            X_state = ct.transform(df[state_cols])  # Test: use pre-fit
        else:
            X_state = ct.fit_transform(df[state_cols])  # Train: fit"""


def prepare_dataset_raw(csv_dir, 
                        unnormalised_cols:list[str], 
                        numeric_cols:list[str], 
                        categorical_cols:list[str], 
                        action_cols:list[str], 
                        seq_len:int=64, 
                        step:int=1, 
                        transformer: Optional[ColumnTransformer] = None, 
                        is_test:bool=False):
    
    # Load all CSV files into single DataFrame
    all_dfs = [pd.read_csv(p) for p in Path(csv_dir).glob("*.csv")]
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Directly extract raw values without transformation
    state_cols = numeric_cols + categorical_cols
    X_state = full_df[state_cols].values.astype(np.float32)
    X_action = full_df[action_cols].values.astype(np.float32)

    # Get raw degradation stats
    raw_degradation = full_df['sim_battery_degradation'].values
    print(f"Raw degradation stats - Mean: {raw_degradation.mean():.6f}, "
          f"Std: {raw_degradation.std():.6f}")

    # Build windows directly from raw data
    windows = []
    for start in range(0, len(full_df) - seq_len, step):
        end = start + seq_len
        state_seq = X_state[start:end]
        action_seq = X_action[start:end] 
        nextstate_seq = X_state[start+1:end+1]
        rew_seq = np.zeros(seq_len, dtype=np.float32)
        done_seq = np.zeros(seq_len, dtype=np.float32)
        #battery_soh = full_df['sim_battery_soh'].iloc[start + seq_len - 1]  # Last SOH value in the sequence
        #done_seq[-1] = 1.0 if battery_soh <= 0 else 0.0 # Set last done flag based on battery state of health
        windows.append((state_seq, action_seq, nextstate_seq, rew_seq, done_seq))

    # Return dataset with empty transformer and original column names
    dataset = TBPTTVehicleDataset(windows)
    return dataset, state_cols, None

def save_fig(save_dir, fig, name):
    path = save_dir / f"{name}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Sanity‐check code
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    unnormalised_cols = ['sim_simulation_time', 'sim_dt_physics', 'sim_battery_time_accel']
    numeric_cols = ['sim_speed', 'sim_front_rpm', 'sim_rear_rpm', 'sim_range', 
                  'sim_battery_degradation', 'sim_battery_soh', 'sim_battery_temp', 
                  'sim_battery_temp_std', 'sim_battery_soc', 'sim_battery_curr',
                  'sim_battery_cooling_temp', 'carla_roll', 'carla_pitch', 'carla_yaw', 
                  'carla_env_temp'
                  ]
    categorical_cols = ['sim_battery_age_factor', 'carla_traffic_light']
    action_cols = ['sim_throttle', 'carla_steering']
    state_cols = unnormalised_cols + numeric_cols + categorical_cols

    print(f"State_cols length: {len(state_cols)}")

    # Initialize dataset with explicit columns
    ds, feature_cols, ct = prepare_dataset_raw(
        csv_dir="F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/train",
        unnormalised_cols=unnormalised_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        action_cols=action_cols,
        seq_len=64,
        step=1
    )
    print(f"number of features: {len(feature_cols)}")
    print(f"features: {feature_cols}")

    
    #Degradation dataset distribution
    fig = plt.figure(figsize=(8,5))
    deg_idx = list(feature_cols).index('sim_battery_degradation')
    sns.histplot(ds[deg_idx], kde=True)
    plt.title("Distribution: Battery Degradation")
    plt.xlabel("RMSE")
    save_fig("F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/", fig, "dist_degradation_data")
    
    fig = plt.figure(figsize=(8,5))
    deg_idx = list(feature_cols).index('sim_speed')
    sns.histplot(ds[deg_idx], kde=True)
    plt.title("Distribution: Battery Speed")
    plt.xlabel("RMSE")
    save_fig("F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/", fig, "dist_degradation_data")

    fig = plt.figure(figsize=(8,5))
    deg_idx = list(feature_cols).index('sim_range')
    sns.histplot(ds[deg_idx], kde=True)
    plt.title("Distribution: Battery Range")
    plt.xlabel("RMSE")
    save_fig("F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/", fig, "dist_degradation_data")

    fig = plt.figure(figsize=(8,5))
    deg_idx = list(feature_cols).index('sim_battery_curr')
    sns.histplot(ds[deg_idx], kde=True)
    plt.title("Distribution: Battery Current")
    plt.xlabel("RMSE")
    save_fig("F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/", fig, "dist_degradation_data")

    fig = plt.figure(figsize=(8,5))
    deg_idx = list(feature_cols).index('sim_battery_temp')
    sns.histplot(ds[deg_idx], kde=True)
    plt.title("Distribution: Battery Temp")
    plt.xlabel("RMSE")
    save_fig("F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/", fig, "dist_degradation_data")

    #csv_folder = "F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/train"
    #files = os.listdir(csv_folder)
    #print("Found CSVs:", files)
    #
    ## Load the first one
    #sample = pd.read_csv(os.path.join(csv_folder, files[0]))
    #print(" - Rows:", len(sample))
    #print(" - Columns:", list(sample.columns))
    #print(sample.head())
    
    #ds = prepare_dataset(
    #    csv_dir = "F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/train",
    #    state_cols=state_cols,
    #    action_cols=action_cols,
    #    seq_len=100,
    #    step=1
    #)

    print(f"→ total windows: {len(ds)}")
    for idx in range(5):
        state_seq, action_seq, next_state_seq, reward_seq, done_seq = ds[idx]
     
        df_s  = pd.DataFrame(state_seq.numpy(),      columns=feature_cols)
        df_a  = pd.DataFrame(action_seq.numpy(),     columns=action_cols)
        df_ns = pd.DataFrame(next_state_seq.numpy(), columns=feature_cols)
        df_r  = pd.DataFrame(reward_seq.numpy(),     columns=['reward'])
        df_d  = pd.DataFrame(done_seq.numpy(),       columns=['done'])
    
        print(f"\n=== Sample {idx} ===")
        print("State sequence (first 5 rows):")
        print(f"sample{idx}_state\n", df_s.head())
        #
        print("Action sequence (first 5 rows):")
        print(f"sample{idx}_action\n", df_a.head())
        
        print("Next-state sequence (first 5 rows):")
        print(f"sample{idx}_next_state", df_ns.head())
        
        print("Reward sequence (first 5 rows):")
        print(f"sample{idx}_reward\n", df_r.head())
        
        print("Done sequence (first 5 rows):")
        print(f"sample{idx}_done\n", df_d.head())
        
    #    #qgrid.show_grid(df_s.head())
    #