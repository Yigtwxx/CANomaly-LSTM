# plot_regression.py
# ==============================================================
# 📌 DOSYA AMACI: Doğrusal Regresyon Analizi ve Görselleştirme
# - Girdi CSV'den sayısal sütunlar seç veya otomatik tespit
# - Linear Regression modeli uygula
# - R², RMSE, Pearson korelasyonu hesapla
# - Çıktılar: regression_plot.png, regression_report.txt, regression_predictions.csv
# ==============================================================
# Kullanım örnekleri:
#   python src/plot_regression.py
#   python src/plot_regression.py --csv data/can_data.csv --x rpm --y speed
#   python src/plot_regression.py --csv data/recon_errors.csv --auto
# ==============================================================

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression   # Doğrusal regresyon modeli
from sklearn.metrics import r2_score, mean_squared_error  # Değerlendirme metrikleri
from scipy.stats import pearsonr  # Korelasyon hesabı

# Otomatik sütun tespiti için bilinen sayısal sütun isimleri
COMMON_NUMERIC = [
    "speed","rpm","load","throttle","voltage","current","temp","torque",
    "pressure","time","timestamp","recon_error","recon_err","error"
]

def pick_numeric_columns(df: pd.DataFrame, prefer_list=COMMON_NUMERIC):
    """
    DataFrame'den regresyon için uygun iki sayısal sütun seç.
    
    Öncelik sırası:
    1. prefer_list'teki bilinen sütun isimleri
    2. İlk iki sayısal sütun
    
    Sabit sütunlar (tüm değerler aynı) hariç tutulur.
    """
    # Sadece sayısal sütunları al
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Tek değerli (varyansı 0) sütunları at
    num_df = num_df.loc[:, num_df.nunique(dropna=True) > 1]
    
    if num_df.shape[1] < 2:
        raise ValueError("Regresyon için en az iki değişken içeren sayısal sütun bulunamadı.")

    # Bilinen sütun isimlerinden eşleşen varsa kullan
    candidates = [c for c in prefer_list if c in num_df.columns]
    if len(candidates) >= 2:
        return candidates[0], candidates[1]

    # Aksi halde ilk iki sayısal sütunu kullan
    cols = list(num_df.columns[:2])
    return cols[0], cols[1]


def load_data(csv_path: Path):
    """CSV dosyasını yükle ve doğrula."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV boş görünüyor.")
    return df


def fit_and_plot(df: pd.DataFrame, x_col: str, y_col: str, outdir: Path):
    """
    Doğrusal regresyon modeli oluştur, metrikler hesapla ve görselleştir.
    
    Parametreler:
        df: Veri çerçevesi
        x_col: Bağımsız değişken (X ekseni)
        y_col: Bağımlı değişken (Y ekseni)
        outdir: Çıktı klasörü
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # NaN değerleri temizle
    data = df[[x_col, y_col]].dropna()
    if data.shape[0] < 2:
        raise ValueError("Yeterli sayıda satır yok (NaN temizliği sonrası).")

    X = data[[x_col]].values  # (n, 1) şeklinde olmalı
    y = data[y_col].values    # (n,) şeklinde hedef

    # Model oluştur ve eğit
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)  # Tahminler

    # -------------------- METRİKLER --------------------
    # R² (Determination Coefficient): Modelin açıklayıcılık gücü (1'e ne kadar yakınsa o kadar iyi)
    r2 = r2_score(y, y_pred)
    
    # MSE (Mean Squared Error): Ortalama kare hata
    mse = mean_squared_error(y, y_pred)
    
    # RMSE (Root MSE): Karekök ortalama hata (orijinal ölçeğe geri döner)
    rmse = np.sqrt(mse)
    
    # Pearson korelasyon katsayısı ve p-değeri
    try:
        corr, pval = pearsonr(data[x_col].values, data[y_col].values)
    except Exception:
        corr, pval = np.nan, np.nan

    # -------------------- GRAFİK ÇİZ --------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.6, label="Veri (scatter)")  # Veri noktaları
    
    # Regresyon çizgisi için pürüzsüz x değerleri
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, linewidth=2, label="Doğrusal regresyon")

    # Model denklemi: y = m*x + b
    coef = float(model.coef_[0])       # Eğim (katsayı)
    intercept = float(model.intercept_) # Y-kesişim
    eq = f"y = {coef:.4f} * x + {intercept:.4f}"
    subtitle = f"R² = {r2:.4f} | RMSE = {rmse:.4f} | Corr = {corr:.4f} (p={pval:.2g})"

    plt.title(f"Regression: {y_col} ~ {x_col}\n{eq}\n{subtitle}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()

    fig_path = outdir / "regression_plot.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # -------------------- ÇIKTILARI KAYDET --------------------
    pred_path = outdir / "regression_predictions.csv"
    rep_path = outdir / "regression_report.txt"

    # Tahmin CSV
    pd.DataFrame({
        x_col: data[x_col].values,
        y_col: y,
        "y_pred": y_pred
    }).to_csv(pred_path, index=False)

    # Metin raporu
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("Linear Regression Report\n")
        f.write("-" * 28 + "\n")
        f.write(f"X (bağımsız): {x_col}\n")
        f.write(f"Y (bağımlı):  {y_col}\n")
        f.write(f"Eşitlik:      {eq}\n")
        f.write(f"R²:           {r2:.6f}\n")
        f.write(f"RMSE:         {rmse:.6f}\n")
        f.write(f"Pearson r:    {corr:.6f}\n")
        f.write(f"p-değeri:     {pval:.6g}\n")
        f.write(f"Satır sayısı: {len(data)}\n")
        f.write(f"Görsel:       {fig_path}\n")
        f.write(f"Tahmin CSV:   {pred_path}\n")

    print(f"[OK] Grafik: {fig_path}")
    print(f"[OK] Rapor:  {rep_path}")
    print(f"[OK] Tahmin: {pred_path}")


def main():
    """Komut satırı argümanlarını işle ve regresyon analizi yap."""
    parser = argparse.ArgumentParser(
        description="Basit doğrusal regresyon grafiği üretir ve çıktıları kaydeder."
    )
    parser.add_argument("--csv", type=str, default="data/can_data.csv",
                        help="Girdi CSV yolu (varsayılan: data/can_data.csv)")
    parser.add_argument("--x", type=str, default=None, help="X ekseni sütun adı")
    parser.add_argument("--y", type=str, default=None, help="Y ekseni sütun adı")
    parser.add_argument("--outdir", type=str, default="outputs", 
                        help="Çıkış klasörü (varsayılan: outputs)")
    parser.add_argument("--auto", action="store_true",
                        help="Sütun isimleri verilmemişse otomatik en iyi iki sütunu seç")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    df = load_data(csv_path)

    # Sütun seçimi
    x_col, y_col = args.x, args.y
    if (x_col is None or y_col is None):
        x_col, y_col = pick_numeric_columns(df)

    # --auto bayrağı aktifse otomatik seç
    if args.auto:
        x_col, y_col = pick_numeric_columns(df)

    print(f"Kullanılan CSV: {csv_path}")
    print(f"Sütunlar: X='{x_col}', Y='{y_col}'")
    fit_and_plot(df, x_col, y_col, outdir)


if __name__ == "__main__":
    main()
