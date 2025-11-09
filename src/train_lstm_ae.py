# train_lstm_ae.py
# --------------------------------------------------------------
# Amaç:
# - can_data.csv dosyasını okuyup özellik çıkarma (payload, iat, id-onehot)
# - Sliding window ile sekans oluşturma
# - LSTM Autoencoder ile SADECE normal pencereler üzerinde eğitim
# - Tüm pencerelerde yeniden-üretim hatası (reconstruction) hesaplama
# - Eşiği (threshold) F1'i maksimize edecek şekilde tarayıp seçme
# - Classification report yazdırma
# - recon_errors.csv dosyasına (error, label) kaydetme
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report

# -------------------- HİPERPARAMETRELER --------------------
DATA_PATH = "can_data.csv"   # Girdi CSV
WINDOW_SIZE = 50             # Pencere uzunluğu (mesaj sayısı)
STRIDE = 5                   # Kaydırma adımı (1 → yoğun, 5 → hızlı)
BATCH_SIZE = 64
EPOCHS = 40                  
LR = 1e-3
HIDDEN_SIZE = 64
LATENT_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ERRORS = "recon_errors.csv"  # Pencere başına hata+etiket kaydı
print("Device:", DEVICE)

# -------------------- 1) VERİYİ YÜKLE --------------------
df = pd.read_csv(DATA_PATH)

# Güvenlik: b0..b7 yoksa ekle (0 doldur)
for k in range(8):
    if f"b{k}" not in df.columns:
        df[f"b{k}"] = 0

# Zaman sırasına koy ve iat (inter-arrival time) çıkar
df = df.sort_values("timestamp").reset_index(drop=True)
df["iat"] = df["timestamp"].diff().fillna(0.0)

# CAN ID'leri indeksle (sayısallaştır)
unique_ids = sorted(df["can_id"].unique())
id_map = {v:i for i,v in enumerate(unique_ids)}
df["id_idx"] = df["can_id"].map(id_map)

# One-hot encode (ID'leri kategori olarak genişletme)
onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
id_onehot = onehot.fit_transform(df[["id_idx"]])  # shape: (N, n_ids)

# Ham payload ve iat sütunlarını numpy'a al
payload_cols = [f"b{k}" for k in range(8)]
X_payload = df[payload_cols].values.astype(np.float32)   # (N, 8)
X_iat = df[["iat"]].values.astype(np.float32)            # (N, 1)

# Özellik vektörü: [ID-onehot | payload(8) | iat(1)]
X = np.concatenate([id_onehot.astype(np.float32), X_payload, X_iat], axis=1)
print("Feature shape:", X.shape)

# Sürekli özellikleri ölçekle (one-hot hariç)
num_id_cols = id_onehot.shape[1]
scaler = StandardScaler()
X[:, num_id_cols:] = scaler.fit_transform(X[:, num_id_cols:])

# Eğer etiket sütunu varsa kullan, yoksa hepsini normal varsay
labels = df.get("label", pd.Series(0, index=df.index)).values

# -------------------- 2) SLIDING WINDOW OLUŞTUR --------------------
def make_windows(arr, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    arr: (N, F) -> (num_windows, window_size, F)
    """
    windows = []
    for start in range(0, arr.shape[0] - window_size + 1, stride):
        windows.append(arr[start:start+window_size])
    return np.stack(windows)

windows = make_windows(X, WINDOW_SIZE, STRIDE)  # (W, T, F)

# Her pencere için etiket: pencere içinde en az 1 anomali varsa -> 1
win_labels = []
for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
    seg = labels[start:start+WINDOW_SIZE]
    win_labels.append(int(seg.sum() > 0))
win_labels = np.array(win_labels)

print("windows shape:", windows.shape, "num anomalies:", win_labels.sum())

# -------------------- 3) SADECE NORMAL PENCERELERLE EĞİT --------------------
normal_idx = np.where(win_labels==0)[0]
# Eğitim/validasyon ayır (sadece normalde)
train_idx, val_idx = train_test_split(normal_idx, test_size=0.2, random_state=42)
X_train = windows[train_idx]
X_val = windows[val_idx]
# Test: tüm pencereler (normal + anomali)
X_test = windows
y_test = win_labels

# PyTorch Dataset/DataLoader
class WinDataset(Dataset):
    def __init__(self, arr): self.x = torch.tensor(arr, dtype=torch.float32)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, idx): return self.x[idx]

train_loader = DataLoader(WinDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(WinDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(WinDataset(X_test),  batch_size=BATCH_SIZE, shuffle=False)

# -------------------- 4) MODEL: LSTM AUTOENCODER --------------------
class LSTMAE(nn.Module):
    """
    Encoder: LSTM -> son zaman adımının gizli hali -> latent (fc)
    Decoder: latent -> fc ile gizliye -> her zaman adımı için LSTM üzerinden çıkış (input_size)
    Loss: MSE (reconstruction)
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE):
        super().__init__()
        self.enc = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, latent_size)
        self.fc2 = nn.Linear(latent_size, hidden_size)
        self.dec = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.enc(x)          # (B, T, H)
        last = enc_out[:, -1, :]          # son zaman adımı (B, H)
        z = torch.tanh(self.fc1(last))    # latent (B, Z)
        dec_in = torch.relu(self.fc2(z)).unsqueeze(1).repeat(1, x.size(1), 1)  # (B,T,H)
        dec_out, _ = self.dec(dec_in)     # (B, T, F)  -> reconstruct
        return dec_out

model = LSTMAE(input_size=windows.shape[2]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.MSELoss()  # Reconstruction kaybı

# -------------------- 5) EĞİTİM DÖNGÜSÜ --------------------
for epoch in range(1, EPOCHS+1):
    # ---- Train ----
    model.train()
    tloss = 0.0
    for b in train_loader:
        b = b.to(DEVICE)
        opt.zero_grad()
        rec = model(b)              # yeniden üretim
        loss = crit(rec, b)         # MSE
        loss.backward()
        opt.step()
        tloss += loss.item()*b.size(0)
    tloss /= len(train_loader.dataset)

    # ---- Validation (normal pencerelerde) ----
    model.eval()
    vloss = 0.0
    with torch.no_grad():
        for b in val_loader:
            b = b.to(DEVICE)
            rec = model(b)
            loss = crit(rec, b)
            vloss += loss.item()*b.size(0)
    vloss /= len(val_loader.dataset)

    print(f"Epoch {epoch}/{EPOCHS} train_loss={tloss:.6f} val_loss={vloss:.6f}")

# -------------------- 6) TÜM PENCERELERDE HATA HESABI --------------------
def compute_errors(dataloader):
    """
    Her pencere için ortalama MSE (B,T,F boyutlarındaki farkın kare ortalaması).
    """
    model.eval()
    errs = []
    with torch.no_grad():
        for b in dataloader:
            b = b.to(DEVICE)
            rec = model(b)
            batch_err = torch.mean((rec - b)**2, dim=(1,2)).cpu().numpy()
            errs.append(batch_err)
    return np.concatenate(errs)

all_errors = compute_errors(test_loader)

# -------------------- 7) EŞİK SEÇİMİ (F1 MAKSİMİZE) --------------------
best_thr, best_f1 = None, -1
for thr in np.linspace(all_errors.min(), all_errors.max(), 200):
    preds = (all_errors > thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    if f > best_f1:
        best_f1 = f; best_thr = thr
print("best thr:", best_thr, "best F1:", best_f1)

# Seçilen eşikle son rapor
preds = (all_errors > best_thr).astype(int)
print(classification_report(y_test, preds, zero_division=0))

# -------------------- 8) HATA + ETİKET CSV KAYDI --------------------
out_df = pd.DataFrame({"error": all_errors, "label": y_test.astype(int)})
out_df.to_csv(OUT_ERRORS, index=False)
print("Saved errors to", OUT_ERRORS)
