# generate_can_dataset.py
# --------------------------------------------------------------
# Amaç:
# - Sentetik (yapay) CAN-Bus verisi üretmek.
# - Normal trafiğe ek olarak 4 farklı saldırı bloğu enjekte etmek:
#   1) Spoofed Message (sahte mesaj enjeksiyonu)
#   2) Replay (önceki mesajları tekrar hızlıca yollama)
#   3) Unauthorized Device (yetkisiz ID'lerden mesaj)
#   4) Corruption (payload bozulması)
# Çıktı:
# - can_data.csv  -> timestamp, can_id, dlc, b0..b7, label(0=normal,1=anomali)
# ---------------------------------------------------------

import csv, random, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -------------------- PARAMETRELER -----------------
OUT_CSV = "can_data.csv"     # Üretilecek dosya adı
NUM_NORMAL = 20000           # Üretilecek normal mesaj sayısı
NUM_SPIKE_EVENTS = 4         # Kaç ayrı saldırı BLOĞU yerleştirilecek
RANDOM_SEED = 42             # Tekrarlanabilirlik için sabit tohum
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Araç içinde sık görülebilecek örnek CAN ID'leri (tamamen örnek amaçlı)
normal_ids = [0x100, 0x101, 0x200, 0x300, 0x350, 0x7FF]

# Yetkisiz cihaz/saldırganın kullanacağı ID'ler (normal setin dışında)
unauth_ids = [0x9AA, 0x9AB]

def random_payload():
    """
    1–8 byte arası rastgele uzunlukta payload üret, geri kalanı 0 ile doldur.
    CAN çerçevesinde DLC 0–8 byte olabilir; burada 8'e pad ediyoruz.
    """
    length = random.randint(1, 8)
    b = [random.randint(0,255) for _ in range(length)]
    b += [0]*(8-length)  # 8 byte'a tamamla
    return b

# Başlangıç zaman damgası – milisaniye cinsinden ilerleteceğiz.

start = datetime.now()

# Her mesaj arasındaki temel aralık ~5 ms. (sentetik)
ts = lambda idx: (start + timedelta(milliseconds=idx*5)).timestamp() * 1000.0

rows = []

# -------------------- 1) NORMAL TRAFİK --------------------
# NUM_NORMAL adet normal mesaj üret ve listeye ekle.
for i in range(NUM_NORMAL):
    t = ts(i)
    cid = random.choice(normal_ids)     # normal ID'lerden biri
    b = random_payload()                # rastgele payload (byte)
    dlc = 8
    rows.append({
        "timestamp": t,
        "can_id": cid,
        "dlc": dlc,
        **{f"b{k}": b[k] for k in range(8)},
        "label": 0                       # 0 = normal
    })

# -------------------- 2) SALDIRI BLOKLARI EKLE --------------------
# Normal akış içine 4 farklı noktaya saldırı blokları yerleştir.
total = len(rows)
# Saldırı bloklarının başlangıç pozisyonları (normal trafiğin ortalarına yerleştiriyoruz)
insert_positions = sorted(random.sample(range(100, total-500), NUM_SPIKE_EVENTS))

# ---- Saldırı türleri için yardımcı fonksiyonlar ----
def insert_spoof(pos):
    """ Sahte mesaj: normal ID'lerle ama anormal payload paterni """
    for j in range(50):  # 50 anomali mesajı
        t = rows[pos + j]["timestamp"] + j*1.0
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": random.choice(normal_ids),      # normal görünen ID
            "dlc": 8,
            # b0'ı kasıtlı olarak 0xFF yapıp "anormal paterne" örnek veriyoruz
            **{f"b{k}": (0xFF if k==0 else random.randint(0,255)) for k in range(8)},
            "label": 1
        })

def insert_replay(pos):
    """ Replay: daha önceki 50 mesajı kopyalayıp hızlıca yeniden gönder """
    seg_start = max(0, pos - 200)       # bir miktar geriden bir segment çek
    segment = rows[seg_start:seg_start+50]
    base_time = rows[pos]["timestamp"]
    for idx, s in enumerate(segment):
        t = base_time + idx*1.0         # sık aralıklarla yeniden zamanla
        new = s.copy()
        new["timestamp"] = t
        new["label"] = 1                # bu blok anomali
        rows.insert(pos+idx, new)

def insert_unauthorized(pos):
    """ Yetkisiz cihaz: normal set dışında ID'lerle mesaj yağmuru """
    for j in range(80):
        t = rows[pos + j]["timestamp"] + j*2.0
        cid = random.choice(unauth_ids) # saldırgan ID
        b = [random.randint(0,255) for _ in range(8)]
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": cid,
            "dlc": 8,
            **{f"b{k}": b[k] for k in range(8)},
            "label": 1
        })

def insert_corruption(pos):
    """ Payload bozulması: bazı byte'ları tersle/boz """
    for j in range(60):
        t = rows[pos + j]["timestamp"] + j*3.0
        orig = rows[pos + j]
        # %50 ihtimalle byte'ı ~x (bitwise not) ile boz
        b = [((~orig.get(f"b{k}",0)) & 0xFF) if random.random()<0.5 else orig.get(f"b{k}",0)
             for k in range(8)]
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": orig["can_id"],   # aynı ID, bozuk payload
            "dlc": 8,
            **{f"b{k}": b[k] for k in range(8)},
            "label": 1
        })

# Seçilen pozisyonlara sırasıyla saldırı bloklarını uygula.
for i, pos in enumerate(insert_positions):
    # Listeye insert ettikçe uzunluk arttığı için pozisyonu az kaydırıyoruz
    current_pos = min(len(rows)-200, pos + i*100)
    if i % 4 == 0:
        insert_spoof(current_pos)
    elif i % 4 == 1:
        insert_replay(current_pos)
    elif i % 4 == 2:
        insert_unauthorized(current_pos)
    else:
        insert_corruption(current_pos)

# Insert işlemleri zaman sırasını bozduğu için tekrar sırala
rows = sorted(rows, key=lambda r: r["timestamp"])

# Zaman damgalarını tekrar düzgün artan sabit aralıkla normalize et (okunabilirlik için)
first_ts = rows[0]["timestamp"]
for i,r in enumerate(rows):
    r["timestamp"] = first_ts + i*5.0

# DataFrame'e çevir ve diske yaz
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUT_CSV}")
