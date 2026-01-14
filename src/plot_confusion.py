# plot_confusion.py (Seaborn versiyonu)
# ==============================================================
# 📌 DOSYA AMACI: Confusion Matrix ve Sınıflandırma Raporu
# - recon_errors.csv'den hata ve etiket değerlerini oku
# - F1 skorunu maksimize eden eşiği otomatik bul
# - Seaborn ile şık ısı haritası görselleştirmesi
# - Çıktılar: confusion_matrix.png, confusion_report.txt
# ==============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                    # Görselleştirme kütüphanesi
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# -------------------- AYARLAR --------------------
CSV_PATH = "recon_errors.csv"   # Girdi dosyası (train_lstm_ae.py çıktısı)
USER_THRESHOLD = None           # Elle eşik belirtmek için (örn: 0.7), None ise otomatik

# -------------------- YARDIMCI FONKSİYONLAR --------------------
def find_column(df, candidates):
    """
    DataFrame'de olası sütun isimlerinden uygun olanı bul.
    Farklı veri formatlarıyla uyumluluk sağlar.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------- DOSYA KONTROLÜ --------------------
if not os.path.exists(CSV_PATH):
    print(f"❌ Hata: '{CSV_PATH}' bulunamadı. train_lstm_ae.py çalıştırılmış mı?")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print("📄 CSV sütunları:", list(df.columns))

# Hata ve etiket sütunlarını tespit et
# Farklı isimlendirmeler için alternatif isimler kontrol edilir
err_col = find_column(df, ['error', 'recon_error', 'reconstruction_error'])
lab_col = find_column(df, ['label', 'y_true', 'anomaly', 'target'])

if err_col is None or lab_col is None:
    print("⚠️ Beklenen sütunlar bulunamadı. Lütfen dosyayı kontrol et.")
    sys.exit(1)

errors = df[err_col].values           # Reconstruction hataları
y_true = df[lab_col].astype(int).values  # Gerçek etiketler (0=normal, 1=anomali)

# -------------------- EŞİK BELİRLEME --------------------
if USER_THRESHOLD is not None:
    thr = float(USER_THRESHOLD)
    auto_info = "(manuel eşik)"
else:
    # Otomatik eşik arama: F1 skorunu maksimize et
    # Arama aralığı: %50 ile %99.9 persentil arası
    cand_thrs = np.linspace(np.percentile(errors, 50), np.percentile(errors, 99.9), 200)
    best_f1, best_thr = -1, cand_thrs[0]
    for t in cand_thrs:
        preds = (errors > t).astype(int)  # Hata > eşik ise anomali
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    thr = best_thr
    auto_info = f"(otomatik, en iyi F1={best_f1:.4f})"

print(f"\n🔹 Kullanılan eşik: {thr:.6f} {auto_info}")

# -------------------- TAHMİN VE METRİKLER --------------------
y_pred = (errors > thr).astype(int)  # Binary tahminler
cm = confusion_matrix(y_true, y_pred)  # 2x2 confusion matrix

# Confusion matrix değerlerini ayrıştır
# TN (True Negative): Doğru tahmin edilen normal
# FP (False Positive): Yanlışlıkla anomali denen normal
# FN (False Negative): Kaçırılan anomali
# TP (True Positive): Doğru tahmin edilen anomali
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

print("\n📊 Confusion Matrix:")
print(cm)
print("\n📈 Classification Report:\n", classification_report(y_true, y_pred, digits=4))

# -------------------- SEABORN GÖRSELLEŞTİRME --------------------
plt.figure(figsize=(6,5))
sns.set_theme(style="darkgrid")  # Seaborn tema ayarı

# Isı haritası (heatmap) olarak confusion matrix çiz
ax = sns.heatmap(cm,
                 annot=True, fmt="d",      # Hücrelerde sayı göster
                 cmap="coolwarm",          # Renk paleti
                 cbar=True,                # Renk skalası
                 linewidths=0.6,           # Hücre çizgi kalınlığı
                 linecolor='gray',
                 annot_kws={"size":14, "weight":"bold"})

# Başlık ve etiketler
plt.title("Confusion Matrix (Seaborn Heatmap)", fontsize=15, weight="bold")
plt.xlabel(f"Predicted Label\nThreshold={thr:.6f}", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'])
plt.yticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'], rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("🖼️ confusion_matrix.png dosyası kaydedildi.")

# -------------------- METİN RAPORU --------------------
with open("confusion_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Eşik değeri: {thr:.6f} {auto_info}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\n")
    f.write(classification_report(y_true, y_pred, digits=4))
print("🗒️ confusion_report.txt kaydedildi.")

# -------------------- ÖZET --------------------
acc = (tp+tn)/(tp+tn+fp+fn)  # Accuracy hesapla
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print("Tamamlandı.")
