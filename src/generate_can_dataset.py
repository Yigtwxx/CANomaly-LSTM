# generate_can_dataset.py
# ==============================================================
# 📌 DOSYA AMACI: Sentetik CAN-Bus verisi üretmek
# - Normal trafik + 4 saldırı türü (Spoof, Replay, Unauthorized, Corruption)
# - Çıktı: can_data.csv (timestamp, can_id, dlc, b0-b7, label)
# ==============================================================

# Gerekli kütüphaneler
import csv, random, time      # random: rastgele veri üretimi
import numpy as np            # numpy: bilimsel hesaplamalar
import pandas as pd           # pandas: veri işleme ve CSV yazma
from datetime import datetime, timedelta  # zaman damgası oluşturma

# -------------------- HİPERPARAMETRELER --------------------
OUT_CSV = "can_data.csv"     # Çıktı dosya adı
NUM_NORMAL = 20000           # Normal mesaj sayısı
NUM_SPIKE_EVENTS = 4         # Saldırı bloğu sayısı
RANDOM_SEED = 42             # Tekrarlanabilirlik için tohum değeri

# Rastgele sayı üretecini başlat (aynı tohum = aynı sonuç)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# CAN ID tanımları
# normal_ids: Araçta meşru olarak görülen ECU ID'leri (motor, fren vb.)
normal_ids = [0x100, 0x101, 0x200, 0x300, 0x350, 0x7FF]
# unauth_ids: Saldırgan cihazların kullanacağı ID'ler (normal dışı)
unauth_ids = [0x9AA, 0x9AB]

def random_payload():
    """
    1-8 byte arası rastgele CAN payload üret, 8 byte'a doldur.
    CAN mesajları max 8 byte veri taşır, DLC uzunluğu belirtir.
    """
    length = random.randint(1, 8)                    # Gerçek veri uzunluğu
    b = [random.randint(0,255) for _ in range(length)]  # Rastgele byte'lar
    b += [0]*(8-length)                              # Kalanı sıfırla doldur
    return b

# Zaman damgası oluşturma
start = datetime.now()  # Başlangıç zamanı
# ts: Her mesaj için 5ms aralıklı zaman damgası üret (milisaniye cinsinden)
ts = lambda idx: (start + timedelta(milliseconds=idx*5)).timestamp() * 1000.0

rows = []  # Tüm mesajların tutulacağı liste

# -------------------- 1) NORMAL TRAFİK OLUŞTUR --------------------
# NUM_NORMAL adet temiz (anomali olmayan) mesaj üret
for i in range(NUM_NORMAL):
    t = ts(i)                            # Zaman damgası
    cid = random.choice(normal_ids)      # Normal ID seç
    b = random_payload()                 # Rastgele payload
    dlc = 8                              # Veri uzunluğu (Data Length Code)
    rows.append({
        "timestamp": t,
        "can_id": cid,
        "dlc": dlc,
        **{f"b{k}": b[k] for k in range(8)},  # b0-b7 sütunları
        "label": 0                            # 0 = normal (anomali değil)
    })

# -------------------- 2) SALDIRI BLOKLARINI EKLE --------------------
total = len(rows)
# Saldırı pozisyonları: İlk 100 ve son 500 mesaj hariç rastgele seç
insert_positions = sorted(random.sample(range(100, total-500), NUM_SPIKE_EVENTS))

def insert_spoof(pos):
    """
    SPOOFING SALDIRISI: Normal ID ile anormal payload gönder.
    Saldırgan meşru ECU gibi davranır ama payload'da imza bırakır (b0=0xFF).
    """
    for j in range(50):  # 50 sahte mesaj
        t = rows[pos + j]["timestamp"] + j*1.0
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": random.choice(normal_ids),  # Normal görünen ID
            "dlc": 8,
            # b0=0xFF: Anomali imzası (model bunu öğrenir)
            **{f"b{k}": (0xFF if k==0 else random.randint(0,255)) for k in range(8)},
            "label": 1  # Anomali
        })

def insert_replay(pos):
    """
    REPLAY SALDIRISI: Geçmiş mesajları kopyalayıp hızlıca tekrar gönder.
    Saldırgan önceki trafiği kaydedip yeniden oynatır.
    """
    seg_start = max(0, pos - 200)        # Kopyalanacak segment başlangıcı
    segment = rows[seg_start:seg_start+50]
    base_time = rows[pos]["timestamp"]
    for idx, s in enumerate(segment):
        t = base_time + idx*1.0          # Hızlı tekrar (1ms aralık)
        new = s.copy()
        new["timestamp"] = t
        new["label"] = 1                 # Tekrarlanan mesaj = anomali
        rows.insert(pos+idx, new)

def insert_unauthorized(pos):
    """
    YETKİSİZ CİHAZ SALDIRISI: Normal set dışı ID'lerle mesaj yağmuru.
    Bilinmeyen bir cihaz CAN bus'a bağlanmış durumu simüle eder.
    """
    for j in range(80):  # 80 yetkisiz mesaj
        t = rows[pos + j]["timestamp"] + j*2.0
        cid = random.choice(unauth_ids)  # Şüpheli ID (0x9AA veya 0x9AB)
        b = [random.randint(0,255) for _ in range(8)]
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": cid,
            "dlc": 8,
            **{f"b{k}": b[k] for k in range(8)},
            "label": 1
        })

def insert_corruption(pos):
    """
    BOZULMA SALDIRISI: Payload byte'larını bitwise NOT ile boz.
    Fiziksel parazit veya kötü amaçlı veri manipülasyonunu simüle eder.
    """
    for j in range(60):  # 60 bozuk mesaj
        t = rows[pos + j]["timestamp"] + j*3.0
        orig = rows[pos + j]
        # %50 ihtimalle byte'ı tersle (~x), diğer yarısı orijinal kalır
        b = [((~orig.get(f"b{k}",0)) & 0xFF) if random.random()<0.5 else orig.get(f"b{k}",0)
             for k in range(8)]
        rows.insert(pos + j, {
            "timestamp": t,
            "can_id": orig["can_id"],
            "dlc": 8,
            **{f"b{k}": b[k] for k in range(8)},
            "label": 1
        })

# Her pozisyona sırasıyla farklı saldırı türü uygula
for i, pos in enumerate(insert_positions):
    # Insert işlemleri listeyi uzattığı için pozisyonu ayarla
    current_pos = min(len(rows)-200, pos + i*100)
    if i % 4 == 0:
        insert_spoof(current_pos)
    elif i % 4 == 1:
        insert_replay(current_pos)
    elif i % 4 == 2:
        insert_unauthorized(current_pos)
    else:
        insert_corruption(current_pos)

# Zaman sırasına göre sırala (insert işlemleri sıralamayı bozabilir)
rows = sorted(rows, key=lambda r: r["timestamp"])

# Zaman damgalarını düzgün artan formata normalize et
first_ts = rows[0]["timestamp"]
for i,r in enumerate(rows):
    r["timestamp"] = first_ts + i*5.0  # 5ms aralıklı düzgün seri

# DataFrame'e çevir ve CSV olarak kaydet
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUT_CSV}")
