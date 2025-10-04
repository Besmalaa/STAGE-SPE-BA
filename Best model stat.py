import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- CONFIG ----
DATA_PATH = "dataset_ml_enriched.csv"
TARGET_COL = "COEFF_SOL"
ID_COL = "ID_DECLARATION"
ALPHA_OPT = 2.357143  # meilleur alpha trouvé
BASE_FEATURES = [
    "TOTAL_ETAT_RECAP_4000C",
    "TOTAL_ETAT_RECAP_2000E",
    "TOTAL_ENGAGEMENTS_NET",
    "TOTAL_EQUIVALENT_RISQUE",
    "TOTAL_POS_LONGUE_BILAN"
]

# ---- charger données ----
df = pd.read_csv(DATA_PATH, sep=";")
if ID_COL in df.columns:
    df = df.drop(columns=[ID_COL])
y = df[TARGET_COL].values.copy()
n_samples = df.shape[0]

# ---- features de base + interactions paires ----
X_df = df[BASE_FEATURES].copy()
for i in range(len(BASE_FEATURES)):
    for j in range(i+1, len(BASE_FEATURES)):
        a, b = BASE_FEATURES[i], BASE_FEATURES[j]
        X_df[f"{a}_x_{b}"] = df[a] * df[b]
        denom = df[b].replace(0, np.nan).fillna(1e-6)
        X_df[f"{a}_div_{b}"] = df[a] / denom
        X_df[f"{a}_minus_{b}"] = df[a] - df[b]

# winsorisation 5%-95% pour petit dataset
p_low, p_high = 0.05, 0.95
for c in X_df.columns:
    lo, hi = X_df[c].quantile(p_low), X_df[c].quantile(p_high)
    X_df[c] = X_df[c].clip(lo, hi).fillna(X_df[c].median())

X_arr = X_df.values.astype(float)

# ---- LOOCV Ridge ----
loo = LeaveOneOut()
preds = np.zeros_like(y, dtype=float)

for train_idx, test_idx in loo.split(X_arr):
    X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
    y_tr = y[train_idx]
    try:
        model = Ridge(alpha=ALPHA_OPT)
        model.fit(X_tr, y_tr)
        preds[test_idx] = model.predict(X_te)
    except Exception:
        preds[test_idx] = np.median(y_tr)

# ---- métriques ----
r2 = r2_score(y, preds)
mae = mean_absolute_error(y, preds)
rmse = math.sqrt(mean_squared_error(y, preds))

print("\n=== LOOCV Ridge avec features affinées ===")
print(f"R2   = {r2:.6f}")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
