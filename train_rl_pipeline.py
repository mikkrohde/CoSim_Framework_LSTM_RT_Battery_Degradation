import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from torch.distributions import Normal
import pandas as pd
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from dataset_impl import prepare_dataset, prepare_dataset_raw
from tqdm import trange
from types import SimpleNamespace
from typing import Optional
from pathlib import Path
import uuid
import csv
import json
from datetime import datetime
from pprint import pprint
import joblib
import seaborn as sns
from sklearn.compose import ColumnTransformer
from scipy.signal import savgol_filter

# Get the directory of THIS script file
SCRIPT_DIR = Path(__file__).parent.absolute()

# Create checkpoints directory INSIDE the script's folder
MODEL_DIR = SCRIPT_DIR / "checkpoints"  # No extra "Scripts/RL_network" needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Creates "checkpoints" if missing


LOG_DIR = SCRIPT_DIR / "logs"          # sibling to checkpoints/
LOG_DIR.mkdir(exist_ok=True)

# ---------- CONSTANTS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#All
BATCH_SIZE = 256
SEQ_LENGTH = 64
VAL_FRAC = 0.2                # 20 % for validation
VAL_BATCH_SIZE = 1024
EPOCH_WINDOWS = 2_500       # 780k samples of ~64 timesteps each = ~50k sequences.. 60 million state samples

#Dynamics model
HIDDEN_SIZE_DYN = 128
NUM_LAYERS_DYN = 3
LR_DYNAMICS = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
FACTOR = 0.5
DROPOUT = 0.2
NUM_DYNAMIC_EPOCHS = 100
EARLY_STOP_PATIENCE = 10     # stop if no val-loss improvement for 30 epochs
EARLY_STOP_DELTA = 1e-5
#lambda_state = 1.0
#lambda_deg   = 10.0     # put more emphasis on battery degradation
#lambda_rng   = 1.0
#lambda_spd   = 1.0

#SAC Model 
HIDDEN_SIZE_SAC = 128
NUM_LAYERS_ACT = 3
NUM_LAYERS_CRI = NUM_LAYERS_ACT + 1
LR_ACTOR = 3e-4
LR_CRITIC = 1e-4
NUM_SAC_EPOCHS = 100
GAMMA = 0.99
TAU = 0.002 #update frequency for target networks 
REWARD_SCALE = 1.0

Transition = namedtuple('Transition', ['state_seq', 'action_seq', 'next_state_seq', 'reward_seq', 'done_seq', 'hidden_init'])

def init_logger(fname, header):
    """Create <LOG_DIR>/<fname>.csv with header if it does not exist."""
    fpath = LOG_DIR / f"{fname}.csv"
    if not fpath.exists():
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + header)
    return fpath

def log_row(fpath, values):
    """Append one row to a CSV."""
    with open(fpath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat()] + values)

def make_loaders(csv_dir,
                 unnormalised_cols,
                 numeric_cols,
                 categorical_cols,
                 action_cols,
                 seq_len        = SEQ_LENGTH,
                 step           = 1,
                 batch_size     = BATCH_SIZE,
                 num_workers    = 0,
                 epoch_windows  = None,
                 pin_memory     = True,
                 transformer: Optional[ColumnTransformer] = None,
                 test : bool = False):

    full_ds, feature_cols, ct = prepare_dataset(csv_dir, unnormalised_cols, numeric_cols, categorical_cols, action_cols, seq_len=seq_len, step=step, transformer=transformer, is_test=test)

    # Chronological split
    n_total = len(full_ds)
    n_val   = int(n_total * VAL_FRAC)
    idx     = np.arange(n_total)
    #train_idx, val_idx = idx[:-n_val], idx[-n_val:]

    val_idx = np.random.choice(idx, size=n_val, replace=False)
    train_idx = np.setdiff1d(idx, val_idx)

    # train-sampler: either the whole set or a capped random sample
    if epoch_windows is not None and epoch_windows < len(train_idx):
        train_sampler = RandomSampler(train_idx, replacement=True, num_samples=epoch_windows)
    else: # full epoch every time
        train_sampler = SubsetRandomSampler(train_idx)

    val_sampler = SubsetRandomSampler(val_idx)   # always full val split
    train_loader = DataLoader(full_ds,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              drop_last=True,
                              persistent_workers=num_workers>0)

    val_loader   = DataLoader(full_ds,
                              batch_size=VAL_BATCH_SIZE,
                              sampler=val_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              drop_last=False,
                              persistent_workers=num_workers>0)

    return train_loader, val_loader, feature_cols, ct

def plot_sac_training_results(log_file_path, save_dir=None):
    """
    Plot all important SAC training metrics from a log file
    
    Args:
        log_file_path: Path to the CSV log file
        save_dir: Directory to save plots (None shows them instead)
    """
    # Load training log data
    df = pd.read_csv(log_file_path)
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create subplots
    fig, axs = plt.subplots(4, 2, figsize=(20, 18))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # 1. Reward progression
    ax = axs[0, 0]
    ax.plot(df['epoch'], df['reward_mean'], label='Mean Reward', lw=2)
    #ax.fill_between(
    #    df['epoch'],
    #    df['reward_mean'] - df['reward_std'],
    #    df['reward_mean'] + df['reward_std'],
    #    alpha=0.2
    #)
    ax.set_title("Reward Progression")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.legend()
    
    # 2. Loss components
    ax = axs[0, 1]
    ax.plot(df['epoch'], df['total_loss'], label='Total Loss', lw=2.5, color='black')
    ax.plot(df['epoch'], df['critic_loss'], label='Critic Loss', lw=2, linestyle='--')
    ax.plot(df['epoch'], df['actor_loss'], label='Actor Loss', lw=2, linestyle='-.')
    ax.plot(df['epoch'], df['alpha_loss'], label='Alpha Loss', lw=2, linestyle=':')
    ax.set_title("Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend()
    
    # 3. Alpha value and entropy
    ax = axs[1, 0]
    ax.plot(df['epoch'], df['alpha'], label='Alpha', color='tab:blue', lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Alpha", color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax.twinx()
    ax2.plot(df['epoch'], df['entropy'], label='Entropy', color='tab:red', lw=2)
    ax2.set_ylabel("Entropy", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.set_title("Alpha Value and Policy Entropy")
    
    # 4. Reward distribution statistics
    ax = axs[1, 1]
    ax.plot(df['epoch'], df['reward_mean'], label='Mean', lw=2)
    ax.plot(df['epoch'], df['reward_min'], label='Min', linestyle=':', lw=1.5)
    ax.plot(df['epoch'], df['reward_max'], label='Max', linestyle=':', lw=1.5)
    ax.set_title("Reward Distribution Statistics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.legend()
    
    # 5. Loss correlations
    ax = axs[2, 0]
    corr_matrix = df[['total_loss', 'critic_loss', 'actor_loss', 'alpha_loss']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Loss Component Correlations")
    
    # 6. Training progress overview
    ax = axs[2, 1]
    markers = ['o', 's', 'D', '^']
    for i, col in enumerate(['total_loss', 'reward_mean', 'alpha', 'entropy']):
        ax.plot(df['epoch'], (df[col]-df[col].min())/(df[col].max()-df[col].min()), 
                label=col, marker=markers[i], markevery=20)
    ax.set_title("Normalized Training Progress")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Value")
    ax.legend()

    ax = axs[3,0]          # use a free cell or create a new figure
    ax.plot(df['epoch'], df['q1'], label='Q1 mean')
    ax.plot(df['epoch'], df['q2'], label='Q2 mean')
    ax.fill_between(df['epoch'],
                    df['q1']-df['q_abs'],
                    df['q1']+df['q_abs'],
                    alpha=0.2, label='|Q1-Q2|')
    ax.set_title("Critic Estimates")
    ax.legend()

    ax = axs[3, 1]
    ax.plot(df['epoch'], df['td_error'])
    ax.set_yscale('log')
    ax.set_title("|TD error|")
    
    # Save or show plots
    if save_dir:
        plt.savefig(save_dir / "sac_training_summary.png", bbox_inches='tight')
        print(f"Saved training summary to {save_dir / 'sac_training_summary.png'}")
    else:
        plt.tight_layout()
        plt.show()

def _thin(arr, step):
    """Return every *step*-th element along the first axis (works for list / np / tensor)."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()[::step]
    return np.asarray(arr)[::step]

def populate_experience_buffer_from_windows(train_loader, capacity=50_000):
    buffer = ExperienceBuffer(capacity)
    
    print(f"Populating ExperienceBuffer from DataLoader...")
    count = 0
    
    for batch_idx, (s_seq, a_seq, ns_seq, r_seq, d_seq) in enumerate(train_loader):
        batch_size, seq_len, state_dim = s_seq.shape
        action_dim = a_seq.shape[-1]

        for b in range(batch_size):
            if count >= capacity:
                return buffer
                
            # Extract sequences
            state_seq = s_seq[b].numpy()
            action_seq = a_seq[b].numpy() 
            next_state_seq = ns_seq[b].numpy()
            reward_seq = r_seq[b].squeeze(-1).numpy()
            done_seq = d_seq[b].squeeze(-1).numpy()

            done_seq = np.zeros(seq_len)
            done_seq[-1] = 1
            
            hidden_init = (
                torch.zeros((NUM_LAYERS_ACT, HIDDEN_SIZE_SAC), dtype=torch.float32),  # h0
                torch.zeros((NUM_LAYERS_ACT, HIDDEN_SIZE_SAC), dtype=torch.float32)   # c0
            )
            
            buffer.push(state_seq, action_seq, next_state_seq, reward_seq, done_seq, hidden_init)
            count += 1
            
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx} batches, {count} sequences added")
            
        if count >= capacity:
            break
    
    print(f"✅ ExperienceBuffer populated with {len(buffer)} sequences")
    return buffer

class ExperienceBuffer:
    """
    A replay buffer that stores fixed-length sequences of transitions along with
    the LSTM initial hidden state for truncated BPTT. 
    
    Only for online training i.e rolling window for online deployment
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        #self.reward_mean = 0
        #self.reward_std = 1

    def push(
        self,
        state_seq,
        action_seq,
        next_state_seq,
        reward_seq,
        done_seq,
        hidden_init
    ):
        """
        Add a sequence of transitions and the corresponding initial hidden state.

        Args:
            state_seq      : np.array of shape [seq_len, state_dim]
            action_seq     : np.array of shape [seq_len, action_dim]
            next_state_seq : np.array of shape [seq_len, state_dim]
            reward_seq     : np.array of shape [seq_len]
            done_seq       : np.array of shape [seq_len]
            hidden_init    : tuple (h0, c0) each of shape [num_layers, hidden_dim]
        """
        transition = Transition(
            state_seq,
            action_seq,
            next_state_seq,
            reward_seq,
            done_seq,
            hidden_init
        )
        self.buffer.append(transition)

    def sample(self, batch_size, device):
        """
        Sample a batch of sequences.

        Returns:
            state_batch      : torch.FloatTensor [B, seq_len, state_dim]
            action_batch     : torch.FloatTensor [B, seq_len, action_dim]
            next_state_batch : torch.FloatTensor [B, seq_len, state_dim]
            reward_batch     : torch.FloatTensor [B, seq_len, 1]
            done_batch       : torch.FloatTensor [B, seq_len, 1]
            hidden_batch     : tuple of two torch.FloatTensors [num_layers, B, hidden_dim]
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Unzip the batch
        states, actions, next_states, rewards, dones, hidden_inits = zip(*batch)

        # Convert to torch tensors
        state_batch = torch.FloatTensor(np.stack(states)).to(device)
        action_batch = torch.FloatTensor(np.stack(actions)).to(device)
        next_state_batch = torch.FloatTensor(np.stack(next_states)).to(device)
        reward_batch = (torch.FloatTensor(np.stack(rewards)).unsqueeze(-1).to(device))
        done_batch = (torch.FloatTensor(np.stack(dones)).unsqueeze(-1).to(device))

        #done_mask = done_batch[:, -1, 0].bool()  # [B]

        # hidden_inits is a tuple list: [(h0, c0), ...]
        # Stack separately for h0 and c0: each becomes [num_layers, batch_size, hidden_dim]
        h0_batch = torch.stack([h for (h, _) in hidden_inits], dim=1).to(device)
        c0_batch = torch.stack([c for (_, c) in hidden_inits], dim=1).to(device)

        #h0_batch[:, ~done_mask] = 0  # Reset hidden states for terminated sequences
        #c0_batch[:, ~done_mask] = 0

        return (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            done_batch,
            (h0_batch, c0_batch)
        )

    def __len__(self):
        return len(self.buffer)

class EarlyStopper:
    """
    Stop training when the monitored metric hasn't improved for N epochs.
    """
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        """
        Args
        ----
        patience   : #epochs with no improvement before stopping
        min_delta  : minimum change (improvement) to qualify as new best
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.epochs_bad = 0

    def step(self, current_score: float) -> bool:
        """
        Call after each validation epoch.

        Returns True if training should stop early.
        """
        if self.best_score is None or current_score < (self.best_score - self.min_delta):
            self.best_score = current_score
            self.epochs_bad = 0
            return False  # keep going
        else:
            self.epochs_bad += 1
            return self.epochs_bad >= self.patience

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers, hidden_size):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.lstm = nn.LSTM(
            state_dim + action_dim, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=DROPOUT if NUM_LAYERS_DYN > 1 else 0.0
        )
        self.ln   = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(DROPOUT)

        #residual projection
        self.res_proj = nn.Linear(state_dim + action_dim, hidden_size)
        #self.state_skip = nn.Linear(state_dim, hidden_size)

        # Multiple prediction heads
        self.state_head = nn.Linear(hidden_size, state_dim)

        # Shared base network for auxiliary predictions
        self.shared_base = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
   
        self.deg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            self.dropout,
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

        self.range_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            self.dropout,
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

        self.speed_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            self.dropout,
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize residual layers near zero
        for m in [self.res_proj]: #, self.state_skip
            nn.init.normal_(m.weight, mean=0, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, states, actions):
        x0 = torch.cat([states, actions], dim=-1)
        residual = self.res_proj(x0)
        lstm_out, hidden = self.lstm(x0)
        out = self.ln(lstm_out + residual)  #  + residual Add residual connection (First residual + norm)

        shared_features = self.shared_base(out)
        state_pred = self.state_head(shared_features) + states  #Added residual skip connection

        deg_pred = self.deg_head(out)   # Degradation
        range_pred = self.range_head(out)   # Range prediction
        speed_pred = self.speed_head(out)   # speed loss

        #hidden layers are here if we want to enable full TBPTT and Temporal difference learning.
        return state_pred, deg_pred, range_pred, speed_pred, hidden

# ------------------- Actor-Critic Networks -------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers, hidden_size):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm = nn.LSTM(state_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.fc1 = nn.Linear(hidden_size, hidden_size)

        self.register_buffer("action_scale",torch.tensor([0.5, 1.0])) # mult tanh in [-1, 1] to get [0, 1] for throttle 
        self.register_buffer("action_bias",torch.tensor([0.5, 0.0])) # add tanh in and [-1, 1] for steering

    def forward(self, states, hidden=None):
        if hidden is None:
            lstm_out, hidden = self.lstm(states)
        else:
            lstm_out, hidden = self.lstm(states, hidden)
        
        out = F.silu(self.fc1(lstm_out))
        mean = self.mean(out) #torch.tanh() odd forcing squize its done in #self.sample() anyway
        log_std = torch.clamp(self.log_std(out), -20, 2)
        return mean, log_std, hidden

    def sample(self, states, hidden=None):
        mean, log_std, _ = self.forward(states, hidden)
        std     = log_std.exp() #lets try and let it explore torch.clamp(,0.01, 0.2) # Ensure std is not too small or too large
        normal  = Normal(mean, std)
        x_t     = normal.rsample()
        actions = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        actions = actions * self.action_scale + self.action_bias 
        return actions, log_prob, _
    
    @torch.no_grad()
    def step(self, state, hidden=None):
        """
        One-timestep inference for test.

        Args
        ----
        state  :  [1, state_dim]  torch tensor
        hidden :  (h, c) or None

        Returns
        -------
        action :  [1, action_dim]  (already in env scale 0‒1 or ‒1‒1)
        logp   :  [1, 1]
        hidden :  next hidden state tuple
        """
        state = state.unsqueeze(0).unsqueeze(0)          # → [B=1, T=1, S]
        action, logp, hidden = self.sample(state, hidden)
        return action.squeeze(0).squeeze(0), \
               logp.squeeze(0).squeeze(0), \
               hidden

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers, hidden_size):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Twin Q-networks (Double Q-learning on single LSTM output)
        self.lstm = nn.LSTM(state_dim + action_dim, num_layers=num_layers, hidden_size=hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)

        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )

        self.q1 = nn.Linear(hidden_size, 1)  # Output single Q-value
        self.q2 = nn.Linear(hidden_size, 1)  # Output single Q-value

    def forward(self, states, actions):
        sa = torch.cat([states, actions], -1)
        
        lstm_out, _ = self.lstm(sa)
        enc_out = self.encoder(lstm_out)
        q1 = self.q1(enc_out)
        q2 = self.q2(enc_out)
    
        return q1, q2
    
    def train(self, mode=True):
        """Explicitly set LSTMs to training/eval mode."""
        super().train(mode)
        self.lstm.train(mode)
        #self.lstm2.train(mode)
        return self

class RewardNormalizer:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mean = 0.0          # Scalar, not tensor
        self.std = 1.0           # Scalar, not tensor
        
    def update(self, batch_rewards):
        """Update running statistics with EMA"""
        all_rewards = batch_rewards.view(-1)  # Flatten to 1D
        batch_mean = torch.mean(all_rewards).item()
        batch_std = torch.std(all_rewards).item() + 1e-8
        
        # EMA update (scalars)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
        self.std = (1 - self.alpha) * self.std + self.alpha * batch_std

    def normalize(self, rewards, scale):
        return (rewards - self.mean) / (self.std + 1e-8) * scale

# ------------------- SAC Network -------------------
class SequenceSACNetwork():
    def __init__(self, state_columns, action_columns):
        self.state_dim = len(state_columns)
        self.action_dim = len(action_columns)

        self.actor = Actor(self.state_dim, self.action_dim, num_layers=NUM_LAYERS_ACT, hidden_size=HIDDEN_SIZE_SAC).to(DEVICE)
        self.critic = Critic(self.state_dim, self.action_dim, num_layers=NUM_LAYERS_CRI, hidden_size=HIDDEN_SIZE_SAC).to(DEVICE)
        self.critic_target = Critic(self.state_dim, self.action_dim, num_layers=NUM_LAYERS_CRI, hidden_size=HIDDEN_SIZE_SAC).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        #self.log_alpha = nn.Parameter(torch.log(torch.tensor([0.3], dtype=torch.float32, device=DEVICE))) #torch.zeros(1, requires_grad=True  device=DEVICE) 
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)  # Log alpha parameter for entropy regularization
        self.alpha = self.log_alpha.exp().detach()  # torch.clamp(, 0.01, 0.3)Initial alpha value, clamped to avoid too high values
        self.target_entropy = -float(self.action_dim) #-self.action_dim * 1.0 # Target entropy for SAC (negative because we want to maximize entropy)
        self.reward_normalizer = RewardNormalizer()  # Normalizing rewards

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4) # low learning rate for alpha
        
        self.speed_idx = state_columns.index('num__sim_speed')
        self.deg_idx = state_columns.index('num__sim_battery_degradation')

    def update(self, epoch, batch, dynamics_model):
        self.actor.train()
        self.critic.train()
        self.critic_target.eval()
        """
        Do one SAC gradient step from a batch of *single-timestep* transitions:
            state:      [B, S]
            action:     [B, A]
            next_state: [B, S]
            reward:     [B, 1]
            done:       [B, 1]
        Returns: total_loss (critic+actor+alpha) as a Python scalar.
        """

        # Extract batch data - ensure proper dimensions
        curr_states = batch.states[:, :-1]        # [B, 64, state_dim]  
        curr_actions = batch.actions[:, :-1]      # [B, 64, action_dim]
        next_states = batch.next_states[:, 1:]  # [B, 64, state_dim]

        B, T, S = curr_states.shape
        assert T > 0, f"Sequence length is {T}, should be > 0"

        # Predict next states and degradation
        with torch.no_grad():
            next_states_pred, deg_pred, range_pred, speed_pred, _ = dynamics_model(curr_states, curr_actions)

            # Extract relevant features
            current_speed = curr_states[:, :, self.speed_idx].unsqueeze(-1)
            gt_deg = next_states[:, :, self.deg_idx].unsqueeze(-1)  # Ground truth degradation
            gt_next_states_speed = next_states[:, :, self.speed_idx].unsqueeze(-1)  # Reasonable target speed (km/h)
           
            #if epoch % 50 == 0:                                  # log occasionally
            #    per_timestep_deg_err = (gt_deg - deg_pred).abs()  # Absolute error in degradation
            #    rmse = torch.sqrt((per_timestep_deg_err ** 2).mean()).item()
            #    mae  = per_timestep_deg_err.abs().mean().item()
            #    bias = per_timestep_deg_err.mean().item()
            #    print(f"[diag] epoch {epoch}  deg RMSE={rmse:.3f}  MAE={mae:.3f}  bias={bias:+.3f}")

            deg_reward = torch.exp(1.0 / (1.0 + deg_pred)) #5.0* * 25.0  # Degradation prediction penalty
            #deg_penalty = torch.clamp(deg_penalty, -10.0, 0.0)

            # Speed performance reward (encourage maintaining reasonable speed)
            speed_reward = torch.abs(gt_next_states_speed - speed_pred) # Encourage speed to match predicted speed 
            speed_reward = torch.exp(-speed_reward)

            #throttle_jerk = torch.tanh(curr_actions[..., 0])  # Throttle action difference
            #throttle_jerk = torch.diff(throttle_jerk, dim=1).abs()
            #action_penalty = -throttle_jerk.unsqueeze(-1)*2.0 #-torch.exp(throttle_diff).unsqueeze(-1) #10.0  throttle_diff * 5.0 # Penalize large throttle changes
            #action_penalty = F.pad(action_penalty, (0, 0, 0, 1), value=0.0)
            
            #if epoch % 20 == 0:
            #    print(f"[R] deg {deg_penalty.mean():+.3f}  speed {speed_reward.mean():+.3f}  smooth {action_penalty.mean():+.3f}")

            reward_components = torch.stack([
                    deg_reward.squeeze(-1), 
                    speed_reward.squeeze(-1),
            #        #action_penalty.squeeze(-1),
                ], dim=-1)
            #
            self.reward_normalizer.update(reward_components)  # Update normalizer with the raw reward components
            normalized_reward = self.reward_normalizer.normalize(reward_components, REWARD_SCALE)#.unsqueeze(1)  # [B, T, 1]

            reward_weights = torch.tensor([0.8, 0.2], device=DEVICE)
            normalized_reward = (normalized_reward @ reward_weights).unsqueeze(-1)  # [B, T, 1] 

            #print(f"Epoch {epoch} - Normalized Reward Mean: {normalized_reward.mean().item():+.3f}, Std: {normalized_reward.std().item():+.3f}")
            #Normalize reward components
            #total_reward = torch.clamp(raw_reward, -REWARD_SCALE, REWARD_SCALE)

            #Target Q-values
            pred_actions, next_log_prob, _ = self.actor.sample(next_states_pred)
            q1_next, q2_next = self.critic_target(next_states_pred, pred_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob

            # Shift target Q-values towards known next state
            target_q = normalized_reward + GAMMA * q_next #.detach() #Shift towards known (Ground Truth) next state

            #print(f"deg_penalty.mean() / total_reward.mean(): {deg_penalty.mean().item()} / {total_reward.mean().item()}")

            #print("Target Q-values shape:", target_q.shape)
        
        #self.reward_normalizer.update(reward_components) #Reward normalizer update to keep track of the mean and std of the reward components

        # ----------------  Critic loss ----------------
        q1, q2 = self.critic(curr_states, curr_actions)
        critic_loss = (F.mse_loss(q1, target_q.detach()) + F.mse_loss(q2, target_q.detach()))

        q1_mean = q1.detach().mean().item()
        q2_mean = q2.detach().mean().item()
        q_abs_diff = (q1.detach() - q2.detach()).abs().mean().item()
        td_error = (q1.detach() - target_q.detach()).abs().mean().item()  # TD error for logging
        
        # ----------------  Critic optimizer ----------------
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---------------- Actor loss ----------------
        # π(s_{t+1})
        actions_pred, log_prob, _ = self.actor.sample(curr_states)  # [B,A]
        q1_pred, q2_pred = self.critic(curr_states, actions_pred)
        q_val = torch.min(q1_pred, q2_pred)
        # Calculate behavior cloning loss
        #bc_loss = self.calculate_throttle_smoothing_loss(actions_pred, curr_actions)  # [B, T, A]
        actor_loss = (self.alpha.detach() * log_prob - q_val).mean() #+ bc_loss  # Add BC loss to actor loss behavior cloning
        
        # ---------------- Actor optimizer ----------------
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # ---------------- Alpha loss ----------------
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha = self.log_alpha.exp()#.detach()#.clamp(0.01, 1.0)
        
        # ---------------- Actor optimizer ----------------
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        self.alpha_optimizer.step()

        # ---------------- Target network update ----------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        total_loss = critic_loss + actor_loss + alpha_loss
        entropy = -log_prob.mean().item() # Policy entropy
        

        return (total_loss.item(), critic_loss.item(), actor_loss.item(), alpha_loss.item(), normalized_reward, entropy, (q1_mean, q2_mean, q_abs_diff, td_error))

    def calculate_throttle_smoothing_loss(self, pred_actions, gt_actions):
        """Encourage policy to stay close to ground truth patterns"""
        # Behavioral cloning component
        bc_loss = F.mse_loss(torch.diff(pred_actions, dim=1), torch.diff(gt_actions, dim=1)) # * 0.5 #Punish Extreme deviations
        
        # Action smoothness penalty
        action_diff = torch.diff(pred_actions, dim=1).abs()
        smooth_loss = action_diff.mean() * 0.5
        
        return bc_loss + smooth_loss

    def calculate_action_jerk_penalty(self, actions):
        """Calculate jerk penalty based on action changes"""
        # Calculate the difference between consecutive actions
        action_diff = torch.diff(actions, dim=1).abs()

        # Calculate the mean jerk across the sequence
        jerk_penalty = action_diff.mean(dim=2, keepdim=True)  # [B, 1, 1]

        # Scale the penalty
        jerk_penalty = -5.0 * torch.abs(jerk_penalty).mean(dim=2, keepdim=True)  # Negative because we want to minimize jerk

        # Pad to match sequence length
        jerk_penalty = F.pad(jerk_penalty, (0, 0, 0, 1), value=0.0)  # [B, T, 1]
        
        return jerk_penalty

    def calculate_state_jerk_penalty(self, states):
        """State jerk penalty for smooth battery state transitions [B, T, 1]"""
        if states.shape[1] < 3:
            return torch.zeros(states.shape[0], states.shape[1], 1, device=states.device)
        
        # Focus on critical battery states for jerk calculation
        # Extract key state indices (modify based on your state_columns)
        soc_idx = self.state_columns.index('num__sim_battery_soc') if hasattr(self, 'state_columns') else 10
        temp_idx = self.state_columns.index('num__sim_battery_temp') if hasattr(self, 'state_columns') else 9
        curr_idx = self.state_columns.index('num__sim_battery_curr') if hasattr(self, 'state_columns') else 11
        
        # Extract critical states [B, T, 3]
        critical_states = torch.stack([
            states[:, :, soc_idx],      # SoC
            states[:, :, temp_idx],     # Temperature  
            states[:, :, curr_idx]      # Current
        ], dim=2)
        
        # First derivative (state velocity) [B, T-1, 3]
        state_vel = torch.diff(critical_states, dim=1)
        
        # Second derivative (state jerk) [B, T-2, 3]
        state_jerk = torch.diff(state_vel, dim=1).abs()
        
        # Weighted jerk penalty - different weights for different states
        soc_jerk = state_jerk[:, :, 0] * 20.0      # High penalty for SoC jumps
        temp_jerk = state_jerk[:, :, 1] * 15.0     # High penalty for temp jumps
        curr_jerk = state_jerk[:, :, 2] * 10.0     # Moderate penalty for current jumps
        
        # Combined state jerk penalty [B, T-2, 1]
        combined_jerk = -(soc_jerk + temp_jerk + curr_jerk).unsqueeze(-1)
        
        # Pad to match original sequence length [B, T, 1]
        state_jerk_penalty = F.pad(combined_jerk, (0, 0, 0, 2), value=0.0)
        
        return state_jerk_penalty

    def calculate_battery_health_reward(self, next_states_pred, curr_actions):
        """Health-aware reward based on battery research"""
        
        # Extract battery states from predictions
        next_soc = next_states_pred[:, :, self.soc_idx].unsqueeze(-1)
        next_temp = next_states_pred[:, :, self.temp_idx].unsqueeze(-1)
        next_deg = next_states_pred[:, :, self.deg_idx].unsqueeze(-1)
        
        # Primary: Degradation minimization
        degradation_reward = -torch.abs(next_deg) * 20.0
        
        # Health-conscious SOC management (70% optimal for Li-ion)
        soc_health_reward = torch.where(
            torch.logical_and(next_soc > 0.2, next_soc < 0.8),
            0.0,  # No penalty in healthy range
            -15.0 * torch.abs(next_soc - 0.5)  # Penalty outside range
        )
        
        # Thermal management (critical for battery life)
        temp_reward = torch.where(
            next_temp > 35.0,
            -50.0 * torch.exp((next_temp - 35.0) / 10.0),  # Exponential penalty
            -0.5 * torch.abs(next_temp - 25.0)  # Optimize around 25°C
        )
        
        # Action smoothness (prevent battery stress)
        action_diff = torch.diff(curr_actions, dim=1).abs()
        smoothness_reward = -10.0 * action_diff.mean(dim=2, keepdim=True)
        smoothness_reward = F.pad(smoothness_reward, (0, 0, 0, 1), value=0.0)
        
        return degradation_reward + soc_health_reward + temp_reward + smoothness_reward

    def smooth_sequence(self, sequence):
        """Apply smoothing along sequence dimension"""
        # Use conv1d for efficient sequence smoothing
        Batch, T, Features = sequence.shape
        sequence_flat = sequence.view(Batch*Features, 1, T)
        
        # Apply 1D convolution for smoothing
        kernel = torch.ones(1, 1, 5, device=sequence.device) / 5
        smoothed = F.conv1d(sequence_flat, kernel, padding=2)
        
        return smoothed.view(Batch, T, Features)

# ------------------- Training Workflow -------------------
def pretrain_dynamics_model(data_loader, state_feature_dims, action_dim, val_loader, deg_idx, rng_idx, speed_idx, ct):
    # generate a unique run ID
    run_id = uuid.uuid4().hex
    
    #Create folder for run
    run_id = run_id[:4]
    model_save_dir = MODEL_DIR / f"Model_{run_id}"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model ID: {run_id}")

    # prepare filenames
    model_fname = model_save_dir / f"dynamic_model_pretrain_{run_id}.pth"
    params_fname = model_save_dir / f"dynamic_model_pretrain_{run_id}.csv"

    #Save transformer used
    joblib.dump(ct, model_save_dir / "transformer.joblib")

    config = {
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_LENGTH,
        "optimizer": "AdamW",
        "created_at": datetime.now().isoformat()
    }

    with open(model_save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # log hyperparameters 
    hyperparams = {
        "run_id": run_id,  
        "state_dim": state_feature_dims,
        "action_dim": action_dim,
        "num_layers": NUM_LAYERS_DYN,
        "hidden_size": HIDDEN_SIZE_DYN,
        "learning_rate": LR_DYNAMICS,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": {
            "factor": FACTOR,
            "patience": PATIENCE
        },
        "training": {
            "num_epochs": NUM_DYNAMIC_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_delta": EARLY_STOP_DELTA
        }
    }
    
    # Save as JSON with pretty print
    with open(params_fname, 'w') as f:
        json.dump(hyperparams, f, indent=4)  # Proper JSON serialization

    dynamic_model = DynamicsModel(state_feature_dims, action_dim, num_layers=NUM_LAYERS_DYN, hidden_size=HIDDEN_SIZE_DYN).to(DEVICE)
    dynamic_model.train()
    dyn_optimizer = optim.AdamW(dynamic_model.parameters(), lr=LR_DYNAMICS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dyn_optimizer, 'min', factor=FACTOR, patience=PATIENCE)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dyn_optimizer, T_max=50, eta_min=1e-6)
    early_stop = EarlyStopper(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_DELTA)

    csv_path = init_logger("dynamics_pretrain", 
                           ["timestamp",
                            "epoch",
                            "train_loss",
                            "val_loss",
                            "learning_rate",
                            "avg_deg_pred",
                            "avg_speed_pred"])


    print("Pre-training dynamics model…")
    all_val_losses = []
    all_train_losses = []
    #Add if using full-TBPTT
    #h_dyn = torch.zeros(NUM_LAYERS_DYN, BATCH_SIZE, HIDDEN_SIZE_DYN, device=DEVICE)
    #c_dyn = torch.zeros(NUM_LAYERS_DYN, BATCH_SIZE, HIDDEN_SIZE_DYN, device=DEVICE)
    #hidden_dyn = (h_dyn, c_dyn)
    deg_loss = 1.0
    for epoch in range(NUM_DYNAMIC_EPOCHS):
        # ----- Training pass -----
        dynamic_model.train()
        total_loss = 0
        for batch_idx, (states, actions, next_states, _, _) in enumerate(data_loader):
            #States:    [B, seq, S], Actions:   [B, seq, A], Next_states: [B, S]
            states, actions, next_states = [t.to(DEVICE) for t in (states, actions, next_states)]
            gt_state = next_states[:, 1:, :]                     # [B, S]
            gt_deg   = gt_state[:, :, deg_idx].unsqueeze(-1)      # [B, 1]
            gt_rng   = gt_state[:, :, rng_idx].unsqueeze(-1)      # [B, 1]
            gt_speed = gt_state[:, :, speed_idx].unsqueeze(-1)    # [B, 1]

            pred_state, pred_degr, pred_range, pred_speed, _ = dynamic_model(states[:, :-1], actions[:, :-1])
            
            #hidden_dyn = (hidden_dyn[0].detach(), hidden_dyn[1].detach()) #Only back propogate to the previous sliding window
            
            loss = (
                F.huber_loss(pred_state,  gt_state) +
                deg_loss*F.huber_loss(pred_degr, gt_deg) + #prioritise degradation a lot!
                F.huber_loss(pred_range,  gt_rng) +
                F.huber_loss(pred_speed,  gt_speed)
            )

            dyn_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dynamic_model.parameters(), max_norm=0.5)
            dyn_optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(data_loader)

        # ----- validation pass -----
        dynamic_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            deg_preds = []
            speed_preds = []
            for states, actions, next_states, _, _ in val_loader:
                states, actions, next_states = [t.to(DEVICE) for t in (states, actions, next_states)]
                pred_state, pred_degr, pred_range, pred_speed, _ = dynamic_model(states[:, :-1], actions[:, :-1])
                
                gt_state = next_states[:, 1:, :]                     # [B, S]
                gt_deg   = gt_state[:, :, deg_idx].unsqueeze(-1)      # [B, 1]
                gt_rng   = gt_state[:, :, rng_idx].unsqueeze(-1)      # [B, 1]
                gt_speed = gt_state[:, :, speed_idx].unsqueeze(-1)    # [B, 1]

                # Store predictions
                deg_preds.append(pred_degr.detach().cpu())
                speed_preds.append(pred_speed.detach().cpu())
                        
                val_batch_loss = (F.mse_loss(pred_state,  gt_state) + deg_loss*F.mse_loss(pred_degr,     gt_deg) +
                                  F.mse_loss(pred_range,  gt_rng)   + F.mse_loss(pred_speed,    gt_speed))
                
                val_loss += val_batch_loss.item()

        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        all_val_losses.append(avg_val)
        all_train_losses.append(avg_train)
        
        if early_stop.step(avg_val):
            print(f"🛑 Early-stopping triggered at epoch {epoch}. "
                  f"Best val-loss = {early_stop.best_score:.4f}")
            break

        avg_deg_pred = torch.cat(deg_preds).mean().item()
        avg_speed_pred = torch.cat(speed_preds).mean().item()
        lr_now = dyn_optimizer.param_groups[0]["lr"]
        
        log_row(csv_path, [
            datetime.now().isoformat(),  # Timestamp
            epoch,
            avg_train,
            avg_val,
            lr_now,
            avg_deg_pred,      # Track degradation predictions
            avg_speed_pred     # Track speed predictions
        ])
    
        #if epoch % 2 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_DYNAMIC_EPOCHS} train={avg_train:.4f} val={avg_val:.4f}")
    
    plt.figure(figsize=(10,6))
    plt.plot(all_val_losses, label=f"Val Loss")
    plt.plot(all_train_losses, label=f"Train Loss")
    plt.title(f"Dynamics Pretrain Loss (Run {run_id})")
    plt.legend(loc="upper left", ncol=2, fontsize="small")
    plt.savefig(model_save_dir / "loss_curve.png")  # Save plot to run folder
    plt.close()

    # Save dynamics model and hyperparameters
    torch.save(dynamic_model.state_dict(), model_fname)
    print(f"✔️  Saved dynamics model to {model_fname}")
    print(f"✔️  Saved hyperparams to {params_fname}")

    return dynamic_model, all_val_losses, all_train_losses

def train_sac_with_experience_buffer(dynamics_model:DynamicsModel, feature_cols, action_columns, num_epochs=NUM_SAC_EPOCHS):
    """
    Train SAC using ExperienceBuffer instead of DataLoader
    """
    # Generate run ID and setup directories
    run_id = uuid.uuid4().hex[:4]
    save_dir = MODEL_DIR / f"SAC_run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Populate experience buffer from your existing train_loader
    print("Converting DataLoader to ExperienceBuffer...")
    experience_buffer = populate_experience_buffer_from_windows(
        train_loader, 
        capacity=780_000  # Adjust based on memory constraints
    )
    
    # Initialize SAC agent
    agent = SequenceSACNetwork(feature_cols, action_columns)
    dynamics_model.eval()

    # Create reward logger
    reward_log = init_logger(
        f"sac_rewards_{run_id}",
        ["epoch", 
         "total_loss","critic_loss","actor_loss","alpha_loss", 
         "q1","q2","q_abs","td_error", 
         "reward_mean","reward_std","reward_min","reward_max", 
         "entropy","alpha"]
    )
    
    # Training loop
    print("Starting SAC training with ExperienceBuffer...")
    all_rewards = []
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        critic_vals   = []
        actor_vals    = []
        alpha_vals    = []
        entropy_vals  = []
        q1_means      = []
        q2_means      = []
        q_diffs       = []
        td_errors     = []
        epoch_rewards = []
        num_updates = min(500, len(experience_buffer) // BATCH_SIZE)  # Number of updates per epoch

        for update in range(num_updates):
            try:
                # Sample random batch of sequences from buffer
                states, actions, next_states, rewards, dones, hidden_batch = experience_buffer.sample(BATCH_SIZE, DEVICE)
                
                # Create batch object (same format as your current implementation)
                batch = SimpleNamespace(
                    states=states,          # [B, 64, S]
                    actions=actions,        # [B, 64, A]
                    next_states=next_states, # [B, 64, S]
                    rewards=rewards,        # [B, 64, 1]
                    dones=dones,           # [B, 64, 1]
                    hidden=hidden_batch    # (h0, c0) tuple - if needed
                )
                
                # SAC update
                loss, critic_loss, actor_loss, alpha_loss, rewards_pred, entropy, (q1_mean, q2_mean, q_abs_diff, td_error) = agent.update(epoch, batch, dynamics_model)
                
                epoch_loss += loss
                critic_vals.append(critic_loss)
                actor_vals.append(actor_loss)
                alpha_vals.append(alpha_loss)
                entropy_vals.append(entropy)
                q1_means.append(q1_mean)
                q2_means.append(q2_mean)
                q_diffs.append(q_abs_diff)
                td_errors.append(td_error)

                # insert rewards into the buffer rewards

                #print(f"Update {update+1}/{num_updates} | "
                #      f"Loss: {loss:.4f} | "
                #      f"Critic Loss: {critic_loss:.4f} | "
                #      f"Actor Loss: {actor_loss:.4f} | "
                #      f"Alpha Loss: {alpha_loss:.4f} | "
                #      f"Entropy: {entropy:.4f} | "
                #      f"Reward: {rewards_pred.mean().item():.4f} | "
                #      f"Alpha: {agent.alpha.item():.4f}")
                epoch_rewards.append(rewards_pred.detach().cpu().numpy().flatten())  # Store rewards
                #print(f"Update {update+1}/{num_updates} loss={loss:.4f}, reward={rewards_pred.mean():.4f}, alpha={agent.alpha.item():.4f}")
                
            except Exception as e:
                print(f"Error in update {update}: {e}")
                continue
        
        #if epoch_rewards:
        rewards_array = np.concatenate(epoch_rewards)
        avg_reward = rewards_array.mean()
        std_reward = rewards_array.std()
        min_reward = rewards_array.min()
        max_reward = rewards_array.max()

        avg_loss        = epoch_loss / num_updates
        avg_critic_loss = np.mean(critic_vals)
        avg_actor_loss  = np.mean(actor_vals)
        avg_alpha_loss  = np.mean(alpha_vals)
        avg_entropy     = np.mean(entropy_vals)

        avg_q1          = np.mean(q1_means)
        avg_q2          = np.mean(q2_means)
        avg_qdiff       = np.mean(q_diffs)
        avg_td          = np.mean(td_errors)
        
        # log loss and rewards
        #avg_loss = epoch_loss / max(1, num_updates)
        #avg_critic_loss = critic_loss / max(1, num_updates)
        #avg_actor_loss = actor_loss / max(1, num_updates)
        #avg_alpha_loss = alpha_loss / max(1, num_updates)
        #avg_entropy = entropy / max(1, num_updates)
        current_alpha = agent.alpha.item()
        
        log_row(reward_log, [
            epoch,
            avg_loss, avg_critic_loss, avg_actor_loss, avg_alpha_loss,
            avg_q1, avg_q2, avg_qdiff, avg_td,
            avg_reward, std_reward, min_reward, max_reward,
            avg_entropy, current_alpha
        ])

        all_rewards.append(avg_reward)    
        #avg_loss = epoch_loss / max(1, num_updates)
        print(f"Epoch {epoch}/{num_epochs} | " 
              f"avg_loss: {avg_loss:.4f}  | " 
              f"Critic Loss: {avg_critic_loss:.4f}  | " 
              f"Actor Loss: {avg_actor_loss:.4f}  | " 
              f"Alpha Loss: {avg_alpha_loss:.4f}  | " 
              f"Reward: {avg_reward:.2f} +- {std_reward:.2f} | " 
              f"Range: [{min_reward:.2f}, {max_reward:.2f}] | " 
              f"alpha: {agent.alpha.item():.4f}")
        
        # Save checkpoints periodically
        if epoch % 20 == 0:
            torch.save(agent.actor.state_dict(), save_dir / f"actor_epoch_{epoch}.pth")
            torch.save(agent.critic.state_dict(), save_dir / f"critic_epoch_{epoch}.pth")
    
    # Save final models
    torch.save(agent.actor.state_dict(), save_dir / "sac_actor.pth")
    torch.save(agent.critic.state_dict(), save_dir / "sac_critic.pth")
    torch.save(dynamics_model.state_dict(), save_dir / "dynamic_model.pth")
    
    reward_stds = [np.std(r) for r in all_rewards]

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(all_rewards, label='Average Reward')
    plt.fill_between(
        range(len(all_rewards)),
        [r - s for r,s in zip(all_rewards, reward_stds)],
        [r + s for r,s in zip(all_rewards, reward_stds)],
        alpha=0.2
    )
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.savefig(save_dir / "reward_progress.png")

    plot_sac_training_results(reward_log, save_dir)

    return agent
    

def test_agent(agent_path, dynamics_model_path, test_csv_path, transformer_path,
              state_cols, action_cols, categorical_cols, device="cuda"):
    """
    Comprehensive testing of SAC agent with statistical validation
    """
    # 1. Load components with proper architecture
    # --------------------------------------------------
    # Load transformer
    ct = joblib.load(transformer_path)
    
    # Load dynamics model with CORRECT architecture
    state_dim = len(state_cols)
    action_dim = len(action_cols)
    dynamics_model = DynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        num_layers=128,  # MUST match saved model
        hidden_size=128  # MUST match saved model
    ).to(device)
    dynamics_model.load_state_dict(torch.load(dynamics_model_path))
    dynamics_model.eval()

    # Load SAC agent
    agent = SequenceSACAgent(state_cols, action_cols)
    agent.actor.load_state_dict(torch.load(agent_path))
    agent.actor.eval()

    test_loader, _, feature_cols, _ = make_loaders(
        csv_dir=test_csv_path,
        unnormalised_cols=['sim_simulation_time', 'sim_dt_physics', 'sim_battery_time_accel'],
        numeric_cols=state_cols,
        categorical_cols=categorical_cols,
        action_cols=action_cols,
        transformer=ct,
        test=True
    )

    deg_idx = feature_cols.index('num__sim_battery_degradation')
    speed_idx = feature_cols.index('num__sim_speed')
    throttle_idx = action_cols.index('sim_throttle')

    metrics = {
        'gt_degradation': [],
        'pred_degradation': [],
        'throttle_actions': [],
        'gt_speed': [],
        'pred_speed': []
    }

    with torch.no_grad():
        for states, _, next_states, _, _ in test_loader:
            states = states.to(device)
            
            # Get policy actions (only final action in sequence)
            pol_act, _, _ = agent.actor.sample(states)
            final_throttle = pol_act[:, -1, throttle_idx].cpu().numpy()
            metrics['throttle_actions'].extend(final_throttle)
            
            # Dynamics predictions
            _, deg_pred, _, speed_pred, _ = dynamics_model(states, pol_act)
            
            # Store final timestep metrics
            metrics['gt_degradation'].extend(next_states[:, -1, deg_idx].cpu().numpy())
            metrics['pred_degradation'].extend(deg_pred[:, -1, 0].cpu().numpy())
            metrics['gt_speed'].extend(next_states[:, -1, speed_idx].cpu().numpy())
            metrics['pred_speed'].extend(speed_pred[:, -1, 0].cpu().numpy())

    # 4. Calculate performance metrics
    # --------------------------------------------------
    gt_deg = np.array(metrics['gt_degradation'])
    pred_deg = np.array(metrics['pred_degradation'])
    gt_speed = np.array(metrics['gt_speed'])
    pred_speed = np.array(metrics['pred_speed'])
    
    return {
        'degradation_rmse': np.sqrt(np.mean((gt_deg - pred_deg)**2)),
        'degradation_mae': np.mean(np.abs(gt_deg - pred_deg)),
        'speed_mae': np.mean(np.abs(gt_speed - pred_speed)),
        'avg_throttle': np.mean(metrics['throttle_actions']),
        'throttle_std': np.std(metrics['throttle_actions'])
    }

def statistical_validation():
    """Run comprehensive statistical testing"""
    # Test SAC agent across multiple seeds
    sac_results = []
    for seed in [42, 123, 456, 789, 1011]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Your existing training code here...
        # train_sac_with_experience_buffer(...)
        
        # Test and collect metrics
        metrics = test_agent(
            agent_path="path/to/sac_actor.pth",
            dynamics_model_path="path/to/dynamics_model.pth",
            test_csv_path="path/to/test/data",
            transformer_path="path/to/transformer.joblib",
            state_cols=state_cols,
            action_cols=action_cols,
            categorical_cols=categorical_cols
        )
        sac_results.append(metrics)

    # Compare against random policy baseline
    random_results = test_random_policy()  # Implement similar to test_agent

    # Perform Welch's t-test
    def welch_test(sac_values, random_values):
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(sac_values, random_values, equal_var=False)
        effect_size = (np.mean(sac_values) - np.mean(random_values)) / np.sqrt(
            (np.var(sac_values, ddof=1) + np.var(random_values, ddof=1)) / 2)
        return t_stat, p_value, effect_size

    print("\nStatistical Results:")
    print(f"Degradation RMSE: {welch_test([r['degradation_rmse'] for r in sac_results], [r['degradation_rmse'] for r in random_results])}")
    print(f"Speed MAE: {welch_test([r['speed_mae'] for r in sac_results], [r['speed_mae'] for r in random_results])}")

def sac_policy_windowed_inference(
    test_csv_path,
    state_cols,
    action_cols,
    actor,
    dynamics,
    transformer,
    seq_len=64,
    device="cuda",
    smooth_win=21,
    run_id="default_run",
    scenario_num=1
):
    """
    Run windowed inference for SAC policy and plot battery degradation, throttle actions, and speed predictions.
    """
    # Prepare test dataset with sliding windows
    test_ds, feature_cols, _ = prepare_dataset(
        csv_dir=test_csv_path,
        unnormalised_cols=[],
        numeric_cols=state_cols,
        categorical_cols=[],
        action_cols=action_cols,
        seq_len=seq_len,
        transformer=transformer,
        is_test=True
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    actor = actor.to(device).eval()
    dynamics = dynamics.to(device).eval()
 
    # Get feature indices
    deg_idx = feature_cols.index('num__sim_battery_degradation')
    speed_idx = feature_cols.index('num__sim_speed')
    #throtte_idx_state = feature_cols.index("num__sim_throttle")
    throttle_idx = action_cols.index('sim_throttle')

    # Storage for analysis
    gt_deg, pred_deg = [], []
    throttle_actions, gt_throttle = [], []
    gt_speed, pred_speed = [], []

    with torch.no_grad():
        for states, actions, next_states, _, _ in test_loader:
            states = states.to(device)
            
            # Get policy actions
            pol_act, _, _ = actor.sample(states)
            
            # Store FINAL throttle action of the sequence
            throttle_actions.append(pol_act[0, -1, throttle_idx].item())
            
            # Dynamics prediction
            next_states_pred, deg_pred, _, speed_pred, _ = dynamics(states, pol_act)
            
            # Store final timestep metrics
            gt_throttle.append(actions[0, -1, throttle_idx].item())
            gt_deg.append(next_states[0, -1, deg_idx].item())
            pred_deg.append(deg_pred[0, -1, 0].item())
            gt_speed.append(next_states[0, -1, speed_idx].item())
            pred_speed.append(speed_pred[0, -1, 0].item())
    
    # Smoothing function
    def _smooth(x, k):
        if k <= 1: return x
        x = np.asarray(x)
        kernel = np.ones(k)/k
        return np.convolve(x, kernel, mode="same")

    # Create time axis
    t = np.arange(len(gt_deg))
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.suptitle(f"Scenario {scenario_num} - Policy Inference Results", y=1.02)

    ax1.set_title(f"Battery Degradation - Scenario {scenario_num}")
    ax2.set_title(f"Vehicle Speed - Scenario {scenario_num}")
    ax3.set_title(f"Vehicle Throttle - Scenario {scenario_num}")
    
    # Plot degradation
    ax1.plot(t, gt_deg, label="GT Degradation", alpha=0.7, color='tab:blue')
    ax1.plot(t, pred_deg, label="Predicted Degradation", alpha=0.7, color='tab:orange')
    if smooth_win > 1:
        ax1.plot(t, _smooth(pred_deg, smooth_win), '--', label="Smoothed Pred", color='tab:red')
    ax1.set_ylabel("Degradation")
    ax1.legend()
    
    # Plot speed
    ax2.plot(t, gt_speed, label="GT Speed", alpha=0.7, color='tab:blue')
    ax2.plot(t, pred_speed, label="Predicted Speed", alpha=0.7, color='tab:orange')
    if smooth_win > 1:
        ax2.plot(t, _smooth(pred_speed, smooth_win), '--', label="Smoothed Speed", color='tab:red')
    ax2.set_ylabel("Speed")
    ax2.legend()
    
    # Plot throttle actions
    ax3.plot(t, throttle_actions, label="Throttle Actions", alpha=1.0, color='tab:green')
    ax3.plot(t, gt_throttle, label="GT Throttle Actions", alpha=0.7, color='tab:orange')
    if smooth_win > 1:
        ax3.plot(t, _smooth(throttle_actions, smooth_win), '--', label="Smoothed Throttle", color='tab:red')
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Throttle")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"scenario_{scenario_num}_policy_inference_{run_id}.png")

    return {
        'gt_degradation': np.array(gt_deg),
        'pred_degradation': np.array(pred_deg),
        'throttle_actions': np.array(throttle_actions),
        'gt_speed': np.array(gt_speed),
        'pred_speed': np.array(pred_speed)
    }

def evaluate_all_scenarios(agent, dynamics_model, scenarios, feature_cols, action_cols, transformer, run_id):
    """Evaluate agent on all scenarios"""
    scenario_results = []
    
    for i, scenario_path in enumerate(scenarios, 1):
        print(f"Evaluating scenario {i}/{len(scenarios)}")
        
        results = sac_policy_windowed_inference(
            test_csv_path=scenario_path,
            state_cols=feature_cols,  # Should be your original state columns
            action_cols=action_cols,
            actor=agent.actor,  # Pass the trained actor directly
            dynamics=dynamics_model,
            transformer=transformer,
            scenario_num=i,
            run_id=run_id,  # Use filename as run_id
            device=DEVICE
        )
        scenario_results.append(results)
    
    return scenario_results


if __name__ == "__main__":
    
    print(f"Device: {DEVICE}")    
    unnormalised_cols = ['sim_simulation_time', 'sim_dt_physics', 'sim_battery_time_accel']
    state_cols = ['sim_speed', 'sim_front_rpm', 'sim_rear_rpm', 'sim_range', 
                  'sim_battery_degradation', 'sim_battery_soh', 'sim_battery_temp', 
                  #'sim_battery_temp_std', 
                  'sim_battery_soc', 'sim_battery_curr',
                  'sim_battery_cooling_temp', 'carla_roll', 'carla_pitch', 'carla_yaw', 
                  'carla_env_temp'
                  ]
    categorical_cols = ['sim_battery_age_factor', 'carla_traffic_light']
    action_cols = ['sim_throttle', 'carla_steering']
    numeric_cols = list(set(state_cols) - set(categorical_cols))

    print("Preparing data loader...")
    training_path = "F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/train"
    test_path = "F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/logs_augmented/test"

    #Dataset+Loader of sliding windows
    train_loader, val_loader, feature_cols, ct = make_loaders(
        csv_dir             = training_path,
        unnormalised_cols   = unnormalised_cols,
        numeric_cols        = state_cols,
        categorical_cols    = categorical_cols,
        action_cols         = action_cols,
        batch_size          = BATCH_SIZE,
        step                = 1,
        seq_len             = SEQ_LENGTH,
        num_workers         = 0,
        epoch_windows       = EPOCH_WINDOWS   #pr epoch cap
    )
    
    pprint(f"Training data loader created with {len(train_loader.dataset)} samples.")
    pprint(f"Features:{feature_cols}")

    deg_idx = list(feature_cols).index('num__sim_battery_degradation') #num__
    rng_idx = list(feature_cols).index('num__sim_range')
    speed_idx = list(feature_cols).index('num__sim_speed')

    state_dim  = len(feature_cols)
    action_dim = len(action_cols)

    print(f"Feature dimenstions:{state_dim}")
    print(f"Action dimenstions:{action_dim}")

    print("Training Dynamics Model...")
    dynamics_model, all_val_losses, all_train_losses = pretrain_dynamics_model(train_loader, state_dim, action_dim, val_loader, deg_idx, rng_idx, speed_idx, ct)
    
    #dynamics_model = DynamicsModel(
    #    state_dim   = state_dim,
    #    action_dim  = action_dim,
    #    num_layers  = 3,            # match trained model
    #    hidden_size = 128
    #).to(DEVICE)
    #
    #dynamics_model.load_state_dict(torch.load("F:/Onedrive/Uni/MSc_uddannelse/4_semester/KandidatThesis/Thesis_Implementation/Scripts/RL_network/checkpoints/Model_a91c/dynamic_model_pretrain_a91c.pth"))
    #dynamics_model.eval()
    #
    ## Get list of evaluation scenarios
    #eval_scenarios = sorted([os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.csv')])
    #
    #seeds = [42, 123, 456, 789, 1011, 1213]
    #all_results = []
    #pprint("Training SAC agents...")
    #for seed in seeds:
    #    # Set random seeds
    #    torch.manual_seed(seed)
    #    np.random.seed(seed)
    #    
    #    # Train new agent
    #    agent = train_sac_with_experience_buffer(dynamics_model, feature_cols, action_cols)
    #    
    #    # Evaluate on all scenarios
    #    seed_results = evaluate_all_scenarios(
    #        agent=agent,
    #        dynamics_model=dynamics_model,
    #        scenarios=eval_scenarios,
    #        feature_cols=state_cols,  # Your original state columns
    #        action_cols=action_cols,
    #        transformer=ct,
    #        run_id=f"seed_{seed}"
    #    )
    #    all_results.append(seed_results)
#
    ## Statistical analysis across all runs
    #degradation_diffs = []
    #for seed_results in all_results:
    #    for scenario in seed_results:
    #        # Calculate percentage difference between predicted and ground truth degradation
    #        diff = np.mean(scenario['pred_degradation'] - np.mean(scenario['gt_degradation']))
    #        degradation_diffs.append(diff)
#
    #mean_diff = np.mean(degradation_diffs)
    #std_diff = np.std(degradation_diffs)
    #
    ## Calculate 95% confidence interval
    #from scipy import stats
    #ci = stats.t.interval(0.95, len(degradation_diffs)-1, 
    #                    loc=mean_diff, 
    #                    scale=stats.sem(degradation_diffs))
#
    #print("\nFinal Results:")
    #print(f"Average degradation difference: {mean_diff:.4f} ± {std_diff:.4f}")
    #print(f"95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")
#
    ## Save final results
    #results_save_path = MODEL_DIR / "sac_evaluation_results.json"
    #with open(results_save_path, 'w') as f:
    #    json.dump({
    #        "mean_degradation_diff": mean_diff,
    #        "std_degradation_diff": std_diff,
    #        "confidence_interval": ci.tolist(),
    #        "seeds": seeds,
    #        "results": all_results
    #    }, f, indent=4)
    #print(f"Results saved to {results_save_path}")