# plot_confusion.py (Seaborn versiyonu)
# --------------------------------------------------------------
# Amaç:
# - recon_errors.csv içindeki hata (error) ve etiket (label) değerleriyle
#   confusion matrix ve sınıflandırma raporu üretmek.
# - Görselleştirmeyi Seaborn ile daha şık (ısı haritası).
# --------------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# -------------------- AYARLAR --------------------
CSV_PATH = "recon_errors.csv"
USER_THRESHOLD = None  # elle eşik (örn: 0.7) vermezsen otomatik bulur

# -------------------- YARDIMCI --------------------
def find_column(df, candidates):
    """Kullanılabilir sütun isimlerinden uygun olanı döndürür."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------- DOSYA KONTROL --------------------
if not os.path.exists(CSV_PATH):
    print(f"❌ Hata: '{CSV_PATH}' bulunamadı. train_lstm_ae.py çalıştırılmış mı?")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print("📄 CSV sütunları:", list(df.columns))

# error ve label sütunlarını tespit et
err_col = find_column(df, ['error', 'recon_error', 'reconstruction_error'])
lab_col = find_column(df, ['label', 'y_true', 'anomaly', 'target'])

if err_col is None or lab_col is None:
    print("⚠️ Beklenen sütunlar bulunamadı. Lütfen dosyayı kontrol et.")
    sys.exit(1)

errors = df[err_col].values
y_true = df[lab_col].astype(int).values

# -------------------- EŞİK BELİRLEME --------------------
if USER_THRESHOLD is not None:
    thr = float(USER_THRESHOLD)
    auto_info = "(manuel eşik)"
else:
    # F1 skorunu maksimize eden eşiği otomatik bul
    cand_thrs = np.linspace(np.percentile(errors, 50), np.percentile(errors, 99.9), 200)
    best_f1, best_thr = -1, cand_thrs[0]
    for t in cand_thrs:
        preds = (errors > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    thr = best_thr
    auto_info = f"(otomatik, en iyi F1={best_f1:.4f})"

print(f"\n🔹 Kullanılan eşik: {thr:.6f} {auto_info}")

# -------------------- PREDİKSİYON VE METRİKLER --------------------
y_pred = (errors > thr).astype(int)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

print("\n📊 Confusion Matrix:")
print(cm)
print("\n📈 Classification Report:\n", classification_report(y_true, y_pred, digits=4))

# -------------------- SEABORN GÖRSEL --------------------
plt.figure(figsize=(6,5))
sns.set_theme(style="darkgrid")

# Confusion matrix'i normalize etmeden ısı haritası olarak çiz
ax = sns.heatmap(cm,
                 annot=True, fmt="d",
                 cmap="coolwarm",
                 cbar=True, linewidths=0.6,
                 linecolor='gray',
                 annot_kws={"size":14, "weight":"bold"})

# Başlıklar ve etiketler
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
acc = (tp+tn)/(tp+tn+fp+fn)
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print("Tamamlandı.")
