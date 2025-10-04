import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

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

# Supprimer colonnes inutiles
drop_cols = [ID_COL, "DATE_ARRETE", "CODE_PARTICIPANT"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Features + cible
y = df[TARGET_COL].values.copy()
X_df = df[BASE_FEATURES].copy()

# ---- interactions paires ----
for i in range(len(BASE_FEATURES)):
    for j in range(i+1, len(BASE_FEATURES)):
        a, b = BASE_FEATURES[i], BASE_FEATURES[j]
        X_df[f"{a}_x_{b}"] = df[a] * df[b]
        denom = df[b].replace(0, np.nan).fillna(1e-6)
        X_df[f"{a}_div_{b}"] = df[a] / denom
        X_df[f"{a}_minus_{b}"] = df[a] - df[b]

# ---- winsorisation ----
p_low, p_high = 0.05, 0.95
for c in X_df.columns:
    lo, hi = X_df[c].quantile(p_low), X_df[c].quantile(p_high)
    X_df[c] = X_df[c].clip(lo, hi).fillna(X_df[c].median())

X_arr = X_df.values.astype(float)

# ---- Bootstrapping ----
n_bootstrap = 500  # nombre d’échantillons générés
rng = np.random.default_rng(42)

X_boot, y_boot = [], []
for _ in range(n_bootstrap):
    idx = rng.choice(len(X_arr), size=len(X_arr), replace=True)
    X_boot.append(X_arr[idx])
    y_boot.append(y[idx])

X_boot = np.vstack(X_boot)
y_boot = np.hstack(y_boot)

# ---- Entraînement + prédictions ----
model = Ridge(alpha=ALPHA_OPT)
y_pred = cross_val_predict(model, X_boot, y_boot, cv=5)

# ---- métriques ----
r2 = r2_score(y_boot, y_pred)
mae = mean_absolute_error(y_boot, y_pred)
rmse = math.sqrt(mean_squared_error(y_boot, y_pred))

print("\n=== Résultats Bootstrapping Ridge ===")
print(f"R2   = {r2:.6f}")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")

# ---- Entraînement final et sauvegarde ----
model.fit(X_boot, y_boot)
joblib.dump(model, "ridge_bootstrap_model.pkl")
print("\n✅ Modèle sauvegardé sous 'ridge_bootstrap_model.pkl'")
