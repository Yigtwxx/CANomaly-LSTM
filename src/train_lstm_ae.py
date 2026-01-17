# train_lstm_ae.py
# ==============================================================
# 📌 FILE PURPOSE: CAN-Bus Anomaly Detection with LSTM Autoencoder
# - Feature extraction from can_data.csv (payload, iat, id-onehot)
# - Sequence creation with sliding window
# - Training with ONLY normal windows (unsupervised/semi-supervised)
# - Reconstruction error calculation and threshold determination
# - Classification report and error logging
# ==============================================================

# Required libraries
import numpy as np
import pandas as pd
import torch                               # PyTorch main module
from torch import nn                       # Neural network layers
from torch.utils.data import Dataset, DataLoader  # Data management
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Normalization
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.metrics import precision_recall_fscore_support, classification_report

# -------------------- HYPERPARAMETERS --------------------
DATA_PATH = "can_data.csv"   # Input CSV file
WINDOW_SIZE = 50             # Window length (how many messages form a sequence)
STRIDE = 5                   # Sliding step (1=dense, 5=faster processing)
BATCH_SIZE = 64              # Mini-batch size
EPOCHS = 40                  # Number of training epochs
LR = 1e-3                    # Learning rate
HIDDEN_SIZE = 64             # LSTM hidden layer size
LATENT_SIZE = 32             # Compressed representation size (bottleneck)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
OUT_ERRORS = "recon_errors.csv"  # Error log file
print("Device:", DEVICE)

# -------------------- 1) LOAD DATA --------------------
df = pd.read_csv(DATA_PATH)

# Fill b0-b7 columns with 0 if missing (safety check)
for k in range(8):
    if f"b{k}" not in df.columns:
        df[f"b{k}"] = 0

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# IAT (Inter-Arrival Time): Time difference between consecutive messages
# Important feature for anomaly detection (catches burst attacks)
df["iat"] = df["timestamp"].diff().fillna(0.0)

# Numericalize CAN IDs (indexing)
unique_ids = sorted(df["can_id"].unique())
id_map = {v:i for i,v in enumerate(unique_ids)}  # ID -> index mapping
df["id_idx"] = df["can_id"].map(id_map)

# One-Hot Encoding: Convert categorical IDs to binary vectors
# E.g., 3 IDs -> [1,0,0], [0,1,0], [0,0,1]
onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
id_onehot = onehot.fit_transform(df[["id_idx"]])  # (N, n_ids)

# Convert payload and IAT features to numpy array
payload_cols = [f"b{k}" for k in range(8)]
X_payload = df[payload_cols].values.astype(np.float32)   # (N, 8)
X_iat = df[["iat"]].values.astype(np.float32)            # (N, 1)

# Concatenate feature vector: [ID-onehot | payload(8) | iat(1)]
X = np.concatenate([id_onehot.astype(np.float32), X_payload, X_iat], axis=1)
print("Feature shape:", X.shape)

# Normalize continuous features (excluding one-hot)
# StandardScaler: makes mean=0, std=1 (facilitates model training)
num_id_cols = id_onehot.shape[1]
scaler = StandardScaler()
X[:, num_id_cols:] = scaler.fit_transform(X[:, num_id_cols:])

# Get labels (if available), otherwise assume all normal
labels = df.get("label", pd.Series(0, index=df.index)).values

# -------------------- 2) CREATE SLIDING WINDOWS --------------------
def make_windows(arr, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Split time series data into overlapping windows.
    arr: input (N, F) -> output (num_windows, window_size, F)
    Sliding window: Required format for sequence models like LSTM
    """
    windows = []
    for start in range(0, arr.shape[0] - window_size + 1, stride):
        windows.append(arr[start:start+window_size])
    return np.stack(windows)

windows = make_windows(X, WINDOW_SIZE, STRIDE)  # (W, T, F)

# Determine label for each window
# If AT LEAST 1 anomaly exists in window -> window is anomaly (label=1)
win_labels = []
for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
    seg = labels[start:start+WINDOW_SIZE]
    win_labels.append(int(seg.sum() > 0))  # True if any 1 exists
win_labels = np.array(win_labels)

print("windows shape:", windows.shape, "num anomalies:", win_labels.sum())

# -------------------- 3) TRAIN WITH ONLY NORMAL WINDOWS --------------------
# Anomaly detection principle: Train model only with normal data
# Model learns normal behavior, cannot reconstruct anomalies -> high error

normal_idx = np.where(win_labels==0)[0]  # Indices of normal windows
train_idx, val_idx = train_test_split(normal_idx, test_size=0.2, random_state=42)
X_train = windows[train_idx]
X_val = windows[val_idx]

# Test: ALL windows (normal + anomaly) - for evaluation
X_test = windows
y_test = win_labels

# PyTorch Dataset class (for data loader)
class WinDataset(Dataset):
    def __init__(self, arr):
        self.x = torch.tensor(arr, dtype=torch.float32)
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, idx):
        return self.x[idx]

# DataLoader: Provides data in mini-batches
train_loader = DataLoader(WinDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(WinDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(WinDataset(X_test),  batch_size=BATCH_SIZE, shuffle=False)

# -------------------- 4) MODEL: LSTM AUTOENCODER --------------------
class LSTMAE(nn.Module):
    """
    LSTM Autoencoder architecture:
    
    ENCODER: LSTM -> last timestep -> FC -> latent (compressed representation)
    DECODER: latent -> FC -> LSTM -> reconstructed sequence
    
    Working Principle:
    - Trained with normal data, learns normal patterns
    - Cannot reconstruct anomalies -> high reconstruction error
    - If error > threshold, mark as anomaly
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE):
        super().__init__()
        # Encoder: Convert input sequence to hidden state
        self.enc = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Bottleneck: Compress hidden state to latent dimension
        self.fc1 = nn.Linear(hidden_size, latent_size)
        # Decoder preparation: Expand latent to hidden dimension
        self.fc2 = nn.Linear(latent_size, hidden_size)
        # Decoder: Reconstruct sequence from hidden state
        self.dec = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        # Encoder: (B, T, F) -> (B, T, H) -> last step (B, H)
        enc_out, _ = self.enc(x)
        last = enc_out[:, -1, :]           # Output of last timestep
        
        # Bottleneck: (B, H) -> (B, Z)
        z = torch.tanh(self.fc1(last))     # tanh: normalize between -1 and 1
        
        # Decoder preparation: (B, Z) -> (B, H) -> (B, T, H)
        dec_in = torch.relu(self.fc2(z)).unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decoder: (B, T, H) -> convert to (B, T, F)
        dec_out, _ = self.dec(dec_in)
        return dec_out  # Reconstructed sequence

# Create model, optimizer and loss function
model = LSTMAE(input_size=windows.shape[2]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)  # Adam optimizer
crit = nn.MSELoss()  # Mean Squared Error - reconstruction loss

# -------------------- 5) TRAINING LOOP --------------------
for epoch in range(1, EPOCHS+1):
    # ---- Train ----
    model.train()  # Set layers like Dropout, BatchNorm to training mode
    tloss = 0.0
    for b in train_loader:
        b = b.to(DEVICE)
        opt.zero_grad()          # Reset gradients
        rec = model(b)           # Forward pass - reconstruction
        loss = crit(rec, b)      # Calculate loss (input vs output difference)
        loss.backward()          # Backpropagation
        opt.step()               # Update weights
        tloss += loss.item()*b.size(0)
    tloss /= len(train_loader.dataset)

    # ---- Validation ----
    model.eval()  # Set to evaluation mode
    vloss = 0.0
    with torch.no_grad():  # Disable gradient computation (for speed)
        for b in val_loader:
            b = b.to(DEVICE)
            rec = model(b)
            loss = crit(rec, b)
            vloss += loss.item()*b.size(0)
    vloss /= len(val_loader.dataset)

    print(f"Epoch {epoch}/{EPOCHS} train_loss={tloss:.6f} val_loss={vloss:.6f}")

# -------------------- 6) CALCULATE ERRORS ON ALL WINDOWS --------------------
def compute_errors(dataloader):
    """
    Calculate average MSE (reconstruction error) for each window.
    High error = model couldn't reconstruct this window well = anomaly suspicion
    """
    model.eval()
    errs = []
    with torch.no_grad():
        for b in dataloader:
            b = b.to(DEVICE)
            rec = model(b)
            # MSE for each window: average across (B, T, F) dimensions
            batch_err = torch.mean((rec - b)**2, dim=(1,2)).cpu().numpy()
            errs.append(batch_err)
    return np.concatenate(errs)

all_errors = compute_errors(test_loader)

# -------------------- 7) THRESHOLD SELECTION (MAXIMIZE F1) --------------------
# Find best threshold by maximizing F1 score
best_thr, best_f1 = None, -1
for thr in np.linspace(all_errors.min(), all_errors.max(), 200):
    preds = (all_errors > thr).astype(int)  # Error > threshold means anomaly
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    if f > best_f1:
        best_f1 = f
        best_thr = thr
print("best thr:", best_thr, "best F1:", best_f1)

# Final predictions and report with selected threshold
preds = (all_errors > best_thr).astype(int)
print(classification_report(y_test, preds, zero_division=0))

# -------------------- 8) SAVE ERROR + LABEL CSV --------------------
# Save errors for visualization and analysis
out_df = pd.DataFrame({"error": all_errors, "label": y_test.astype(int)})
out_df.to_csv(OUT_ERRORS, index=False)
print("Saved errors to", OUT_ERRORS)
