# plot_regression.py
# ==============================================================
# 📌 FILE PURPOSE: Linear Regression Analysis and Visualization
# - Select numeric columns from input CSV or auto-detect
# - Apply Linear Regression model
# - Calculate R², RMSE, Pearson correlation
# - Outputs: regression_plot.png, regression_report.txt, regression_predictions.csv
# ==============================================================
# Usage examples:
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

from sklearn.linear_model import LinearRegression   # Linear regression model
from sklearn.metrics import r2_score, mean_squared_error  # Evaluation metrics
from scipy.stats import pearsonr  # Correlation calculation

# Known numeric column names for automatic column detection
COMMON_NUMERIC = [
    "speed","rpm","load","throttle","voltage","current","temp","torque",
    "pressure","time","timestamp","recon_error","recon_err","error"
]

def pick_numeric_columns(df: pd.DataFrame, prefer_list=COMMON_NUMERIC):
    """
    Select two suitable numeric columns from DataFrame for regression.
    
    Priority order:
    1. Known column names from prefer_list
    2. First two numeric columns
    
    Constant columns (all values the same) are excluded.
    """
    # Get only numeric columns
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Remove single-value (zero variance) columns
    num_df = num_df.loc[:, num_df.nunique(dropna=True) > 1]
    
    if num_df.shape[1] < 2:
        raise ValueError("Could not find at least two numeric columns with variable values for regression.")

    # Use matching columns from known column names if available
    candidates = [c for c in prefer_list if c in num_df.columns]
    if len(candidates) >= 2:
        return candidates[0], candidates[1]

    # Otherwise use first two numeric columns
    cols = list(num_df.columns[:2])
    return cols[0], cols[1]


def load_data(csv_path: Path):
    """Load and validate CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV appears to be empty.")
    return df


def fit_and_plot(df: pd.DataFrame, x_col: str, y_col: str, outdir: Path):
    """
    Create linear regression model, calculate metrics and visualize.
    
    Parameters:
        df: Data frame
        x_col: Independent variable (X axis)
        y_col: Dependent variable (Y axis)
        outdir: Output folder
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Clean NaN values
    data = df[[x_col, y_col]].dropna()
    if data.shape[0] < 2:
        raise ValueError("Not enough rows (after NaN cleanup).")

    X = data[[x_col]].values  # Must be (n, 1) shape
    y = data[y_col].values    # (n,) shape target

    # Create and train model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)  # Predictions

    # -------------------- METRICS --------------------
    # R² (Determination Coefficient): Model's explanatory power (closer to 1 is better)
    r2 = r2_score(y, y_pred)
    
    # MSE (Mean Squared Error): Average squared error
    mse = mean_squared_error(y, y_pred)
    
    # RMSE (Root MSE): Square root of mean error (returns to original scale)
    rmse = np.sqrt(mse)
    
    # Pearson correlation coefficient and p-value
    try:
        corr, pval = pearsonr(data[x_col].values, data[y_col].values)
    except Exception:
        corr, pval = np.nan, np.nan

    # -------------------- PLOT GRAPH --------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.6, label="Data (scatter)")  # Data points
    
    # Smooth x values for regression line
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, linewidth=2, label="Linear regression")

    # Model equation: y = m*x + b
    coef = float(model.coef_[0])       # Slope (coefficient)
    intercept = float(model.intercept_) # Y-intercept
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

    # -------------------- SAVE OUTPUTS --------------------
    pred_path = outdir / "regression_predictions.csv"
    rep_path = outdir / "regression_report.txt"

    # Prediction CSV
    pd.DataFrame({
        x_col: data[x_col].values,
        y_col: y,
        "y_pred": y_pred
    }).to_csv(pred_path, index=False)

    # Text report
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("Linear Regression Report\n")
        f.write("-" * 28 + "\n")
        f.write(f"X (independent): {x_col}\n")
        f.write(f"Y (dependent):   {y_col}\n")
        f.write(f"Equation:        {eq}\n")
        f.write(f"R²:              {r2:.6f}\n")
        f.write(f"RMSE:            {rmse:.6f}\n")
        f.write(f"Pearson r:       {corr:.6f}\n")
        f.write(f"p-value:         {pval:.6g}\n")
        f.write(f"Row count:       {len(data)}\n")
        f.write(f"Plot:            {fig_path}\n")
        f.write(f"Prediction CSV:  {pred_path}\n")

    print(f"[OK] Plot: {fig_path}")
    print(f"[OK] Report: {rep_path}")
    print(f"[OK] Predictions: {pred_path}")


def main():
    """Process command line arguments and perform regression analysis."""
    parser = argparse.ArgumentParser(
        description="Generates simple linear regression plot and saves outputs."
    )
    parser.add_argument("--csv", type=str, default="data/can_data.csv",
                        help="Input CSV path (default: data/can_data.csv)")
    parser.add_argument("--x", type=str, default=None, help="X axis column name")
    parser.add_argument("--y", type=str, default=None, help="Y axis column name")
    parser.add_argument("--outdir", type=str, default="outputs", 
                        help="Output folder (default: outputs)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select best two columns if column names not provided")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    df = load_data(csv_path)

    # Column selection
    x_col, y_col = args.x, args.y
    if (x_col is None or y_col is None):
        x_col, y_col = pick_numeric_columns(df)

    # Auto-select if --auto flag is active
    if args.auto:
        x_col, y_col = pick_numeric_columns(df)

    print(f"CSV used: {csv_path}")
    print(f"Columns: X='{x_col}', Y='{y_col}'")
    fit_and_plot(df, x_col, y_col, outdir)


if __name__ == "__main__":
    main()
