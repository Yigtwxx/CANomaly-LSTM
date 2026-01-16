# train_lstm_ae.py
# ==============================================================
# 📌 DOSYA AMACI: LSTM Autoencoder ile CAN-Bus Anomali Tespiti
# - can_data.csv'den özellik çıkarma (payload, iat, id-onehot)
# - Sliding window ile sekans oluşturma
# - SADECE normal pencerelerle eğitim (unsupervised/semi-supervised)
# - Yeniden üretim hatası hesaplama ve eşik belirleme
# - Classification report ve hata kaydı
# ==============================================================

# Gerekli kütüphaneler
import numpy as np
import pandas as pd
import torch                               # PyTorch ana modül
from torch import nn                       # Sinir ağı katmanları
from torch.utils.data import Dataset, DataLoader  # Veri yönetimi
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Normalizasyon
from sklearn.model_selection import train_test_split  # Veri bölme
from sklearn.metrics import precision_recall_fscore_support, classification_report

# -------------------- HİPERPARAMETRELER --------------------
DATA_PATH = "can_data.csv"   # Girdi CSV dosyası
WINDOW_SIZE = 50             # Pencere uzunluğu (kaç mesaj bir sekans oluşturur)
STRIDE = 5                   # Kaydırma adımı (1=yoğun, 5=hızlı işlem)
BATCH_SIZE = 64              # Mini-batch boyutu
EPOCHS = 40                  # Eğitim epoch sayısı
LR = 1e-3                    # Öğrenme oranı (learning rate)
HIDDEN_SIZE = 64             # LSTM gizli katman boyutu
LATENT_SIZE = 32             # Sıkıştırılmış temsil boyutu (bottleneck)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU varsa kullan
OUT_ERRORS = "recon_errors.csv"  # Hata kayıt dosyası
print("Device:", DEVICE)

# -------------------- 1) VERİYİ YÜKLE --------------------
df = pd.read_csv(DATA_PATH)

# b0-b7 sütunları yoksa 0 ile doldur (güvenlik kontrolü)
for k in range(8):
    if f"b{k}" not in df.columns:
        df[f"b{k}"] = 0

# Zaman sırasına göre sırala
df = df.sort_values("timestamp").reset_index(drop=True)

# IAT (Inter-Arrival Time): Ardışık mesajlar arası zaman farkı
# Anomali tespitinde önemli bir özellik (burst saldırılarını yakalar)
df["iat"] = df["timestamp"].diff().fillna(0.0)

# CAN ID'leri sayısallaştır (indeksleme)
unique_ids = sorted(df["can_id"].unique())
id_map = {v:i for i,v in enumerate(unique_ids)}  # ID -> indeks eşlemesi
df["id_idx"] = df["can_id"].map(id_map)

# One-Hot Encoding: Kategorik ID'leri binary vektörlere çevir
# Örn: 3 ID varsa -> [1,0,0], [0,1,0], [0,0,1]
onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
id_onehot = onehot.fit_transform(df[["id_idx"]])  # (N, n_ids)

# Payload ve IAT özelliklerini numpy array'e çevir
payload_cols = [f"b{k}" for k in range(8)]
X_payload = df[payload_cols].values.astype(np.float32)   # (N, 8)
X_iat = df[["iat"]].values.astype(np.float32)            # (N, 1)

# Özellik vektörünü birleştir: [ID-onehot | payload(8) | iat(1)]
X = np.concatenate([id_onehot.astype(np.float32), X_payload, X_iat], axis=1)
print("Feature shape:", X.shape)

# Sürekli özellikleri normalize et (one-hot hariç)
# StandardScaler: ortalama=0, std=1 yapar (model eğitimini kolaylaştırır)
num_id_cols = id_onehot.shape[1]
scaler = StandardScaler()
X[:, num_id_cols:] = scaler.fit_transform(X[:, num_id_cols:])

# Etiketleri al (varsa), yoksa hepsini normal varsay
labels = df.get("label", pd.Series(0, index=df.index)).values

# -------------------- 2) SLIDING WINDOW OLUŞTUR --------------------
def make_windows(arr, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Zaman serisi verisini üst üste binen pencerelere böl.
    arr: (N, F) şeklinde girdi -> (num_windows, window_size, F) çıktı
    Sliding window: LSTM gibi sekans modelleri için gerekli format
    """
    windows = []
    for start in range(0, arr.shape[0] - window_size + 1, stride):
        windows.append(arr[start:start+window_size])
    return np.stack(windows)

windows = make_windows(X, WINDOW_SIZE, STRIDE)  # (W, T, F)

# Her pencere için etiket belirle
# Pencere içinde EN AZ 1 anomali varsa -> pencere anomali (label=1)
win_labels = []
for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
    seg = labels[start:start+WINDOW_SIZE]
    win_labels.append(int(seg.sum() > 0))  # Herhangi bir 1 varsa True
win_labels = np.array(win_labels)

print("windows shape:", windows.shape, "num anomalies:", win_labels.sum())

# -------------------- 3) SADECE NORMAL PENCERELERLE EĞİT --------------------
# Anomali tespiti prensibi: Modeli sadece normal veriyle eğit
# Model normal davranışı öğrenir, anomalileri yeniden üretemez -> yüksek hata

normal_idx = np.where(win_labels==0)[0]  # Normal pencerelerin indeksleri
train_idx, val_idx = train_test_split(normal_idx, test_size=0.2, random_state=42)
X_train = windows[train_idx]
X_val = windows[val_idx]

# Test: TÜM pencereler (normal + anomali) - değerlendirme için
X_test = windows
y_test = win_labels

# PyTorch Dataset sınıfı (veri yükleyici için)
class WinDataset(Dataset):
    def __init__(self, arr):
        self.x = torch.tensor(arr, dtype=torch.float32)
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, idx):
        return self.x[idx]

# DataLoader: Mini-batch halinde veri sağlar
train_loader = DataLoader(WinDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(WinDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(WinDataset(X_test),  batch_size=BATCH_SIZE, shuffle=False)

# -------------------- 4) MODEL: LSTM AUTOENCODER --------------------
class LSTMAE(nn.Module):
    """
    LSTM Autoencoder mimarisi:
    
    ENCODER: LSTM -> son zaman adımı -> FC -> latent (sıkıştırılmış temsil)
    DECODER: latent -> FC -> LSTM -> yeniden oluşturulmuş sekans
    
    Çalışma Prensibi:
    - Normal veriyle eğitilir, normal paternleri öğrenir
    - Anomali gelince yeniden üretemez -> yüksek reconstruction error
    - Hata > eşik ise anomali olarak işaretle
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE):
        super().__init__()
        # Encoder: Girdi sekansını gizli duruma çevir
        self.enc = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Bottleneck: Gizli durumu latent boyuta sıkıştır
        self.fc1 = nn.Linear(hidden_size, latent_size)
        # Decoder hazırlık: Latent'i gizli boyuta genişlet
        self.fc2 = nn.Linear(latent_size, hidden_size)
        # Decoder: Gizli durumdan sekansı yeniden oluştur
        self.dec = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        # Encoder: (B, T, F) -> (B, T, H) -> son adım (B, H)
        enc_out, _ = self.enc(x)
        last = enc_out[:, -1, :]           # Son zaman adımının çıktısı
        
        # Bottleneck: (B, H) -> (B, Z)
        z = torch.tanh(self.fc1(last))     # tanh: -1 ile 1 arası normalize
        
        # Decoder hazırlık: (B, Z) -> (B, H) -> (B, T, H)
        dec_in = torch.relu(self.fc2(z)).unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decoder: (B, T, H) -> (B, T, F)'e çevir
        dec_out, _ = self.dec(dec_in)
        return dec_out  # Yeniden oluşturulmuş sekans

# Model, optimizer ve loss fonksiyonu oluştur
model = LSTMAE(input_size=windows.shape[2]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)  # Adam optimizer
crit = nn.MSELoss()  # Mean Squared Error - reconstruction loss

# -------------------- 5) EĞİTİM DÖNGÜSÜ --------------------
for epoch in range(1, EPOCHS+1):
    # ---- Train (Eğitim) ----
    model.train()  # Dropout, BatchNorm gibi katmanları eğitim moduna al
    tloss = 0.0
    for b in train_loader:
        b = b.to(DEVICE)
        opt.zero_grad()          # Gradyanları sıfırla
        rec = model(b)           # Forward pass - yeniden üretim
        loss = crit(rec, b)      # Kayıp hesapla (girdi vs çıktı farkı)
        loss.backward()          # Backpropagation
        opt.step()               # Ağırlıkları güncelle
        tloss += loss.item()*b.size(0)
    tloss /= len(train_loader.dataset)

    # ---- Validation (Doğrulama) ----
    model.eval()  # Değerlendirme moduna al
    vloss = 0.0
    with torch.no_grad():  # Gradyan hesaplama kapalı (hız için)
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
    Her pencere için ortalama MSE (reconstruction error) hesapla.
    Yüksek hata = model bu pencereyi iyi yeniden üretemedi = anomali şüphesi
    """
    model.eval()
    errs = []
    with torch.no_grad():
        for b in dataloader:
            b = b.to(DEVICE)
            rec = model(b)
            # Her pencere için MSE: (B, T, F) boyutlarında ortalama
            batch_err = torch.mean((rec - b)**2, dim=(1,2)).cpu().numpy()
            errs.append(batch_err)
    return np.concatenate(errs)

all_errors = compute_errors(test_loader)

# -------------------- 7) EŞİK SEÇİMİ (F1 MAKSİMİZE) --------------------
# En iyi threshold'u F1 skorunu maksimize ederek bul
best_thr, best_f1 = None, -1
for thr in np.linspace(all_errors.min(), all_errors.max(), 200):
    preds = (all_errors > thr).astype(int)  # Hata > eşik ise anomali
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    if f > best_f1:
        best_f1 = f
        best_thr = thr
print("best thr:", best_thr, "best F1:", best_f1)

# Seçilen eşikle final tahminler ve rapor
preds = (all_errors > best_thr).astype(int)
print(classification_report(y_test, preds, zero_division=0))

# -------------------- 8) HATA + ETİKET CSV KAYDI --------------------
# Görselleştirme ve analiz için hataları kaydet
out_df = pd.DataFrame({"error": all_errors, "label": y_test.astype(int)})
out_df.to_csv(OUT_ERRORS, index=False)
print("Saved errors to", OUT_ERRORS)
