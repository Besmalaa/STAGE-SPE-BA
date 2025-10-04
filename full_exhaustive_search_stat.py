"""
full_exhaustive_search.py
Usage: place in folder with dataset_ml_clean.csv, activate venv, then run.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import HuberRegressor, Ridge, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
import traceback, time, csv, os, math
from sklearn.base import clone

# ---------------- CONFIG ----------------
DATA_PATH = "dataset_ml_clean.csv"
TARGET_COL = "COEFF_SOL"
ID_COL = "ID_DECLARATION"   # dropped if present

# Ressources / runtime tuning
N_JOBS = -1                 # -1 = all cores for joblib chunks (outer)
CHUNK_SIZE = 200            # sauvegarde par bloc
MAX_FEATURES_TO_KEEP = 10   # top-k correlated features to consider
MIN_COMBO = 1               # min combo size
MAX_COMBO = 5               # max combo size (reduce if too many)
RUN_HEAVY_MODELS = True     # include XGBoost and CatBoost
N_ESTIMATORS = 500          # reduce to 200 for faster runs
USE_LOOCV = True            # True -> LeaveOneOut (stable for small datasets), False -> 5-fold
EARLY_STOP = 50             # early stopping rounds for boosters (if used)
OUT_CSV = "results_full_search.csv"
ERROR_LOG = "errors_full_search.log"
RANDOM_STATE = 42
TOP_FOR_STACKING = 4        # number of top models to use for stacking
STACKING_META = "Ridge"     # meta-model for stacking ("Ridge" or "ElasticNet")
# ----------------------------------------

# ---- Load data ----
df = pd.read_csv(DATA_PATH, sep=";")
if ID_COL in df.columns:
    df = df.drop(columns=[ID_COL])
y = df[TARGET_COL].values
n_samples = df.shape[0]
print(f"[INFO] Loaded {DATA_PATH} -> {df.shape[0]} rows, {df.shape[1]} cols")

# ---- Candidate features by correlation ----
cor = df.corr(numeric_only=True)[TARGET_COL].abs().sort_values(ascending=False)
candidate_feats = cor.index[1:MAX_FEATURES_TO_KEEP+1].tolist()
print(f"[INFO] Candidate features ({len(candidate_feats)}): {candidate_feats}")

# ---- Scalers / transforms ----
def safe_log1p(X):
    Xc = np.array(X, dtype=float)
    # clip negatives to 0 before log1p
    Xc = np.clip(Xc, a_min=0, a_max=None)
    return np.log1p(Xc)

def safe_sqrt(X):
    Xc = np.array(X, dtype=float)
    Xc = np.clip(Xc, a_min=0, a_max=None)
    return np.sqrt(Xc)

scalers = {
    "none": lambda X: X,
    "standard": lambda X: StandardScaler().fit_transform(X),
    "robust": lambda X: RobustScaler().fit_transform(X),
    "log1p": lambda X: safe_log1p(X),
    "sqrt": lambda X: safe_sqrt(X),
}

# ---- Model factory (create fresh instances inside evaluate) ----
def get_model_instance(name):
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=0.05,
                             num_leaves=31, objective="huber", n_jobs=1, random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=N_ESTIMATORS, learning_rate=0.05, max_depth=5,
                            n_jobs=1, objective="reg:squarederror", random_state=RANDOM_STATE, verbosity=0)
    if name == "CatBoost":
        # CatBoost uses its own Pool for early stopping; verbosity=0 for silent
        return CatBoostRegressor(iterations=N_ESTIMATORS, learning_rate=0.05, depth=6, verbose=0, random_state=RANDOM_STATE)
    if name == "Huber":
        return HuberRegressor(max_iter=2000, alpha=1e-4)
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "ElasticNet":
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE)
    raise ValueError("Unknown model name")

model_names = ["LightGBM", "Ridge", "ElasticNet", "Huber"]
if RUN_HEAVY_MODELS:
    model_names += ["XGBoost", "CatBoost"]

print(f"[INFO] Models: {model_names}")

# ---- CV strategy ----
if USE_LOOCV:
    cv = LeaveOneOut()
    print("[INFO] Using LOOCV (Leave-One-Out CV) - stable for small datasets")
else:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("[INFO] Using 5-fold KFold CV")

# ---- Generate tasks ----
tasks = []
for k in range(MIN_COMBO, min(MAX_COMBO, len(candidate_feats)) + 1):
    for combo in combinations(candidate_feats, k):
        for scaler_name in scalers.keys():
            for model_name in model_names:
                tasks.append((combo, scaler_name, model_name))
print(f"[INFO] Total tasks: {len(tasks)}")

# ---- resume support: existing results skip ----
def task_key(combo, scaler, model):
    return f"{model}||{scaler}||{','.join(combo)}"

existing_keys = set()
if os.path.exists(OUT_CSV):
    try:
        df_exist = pd.read_csv(OUT_CSV)
        for _, r in df_exist.iterrows():
            k = task_key(tuple(r['features_serial'].split(",")) if 'features_serial' in r else tuple(eval(r['features'])),
                         r['scaler'], r['model'])
            existing_keys.add(k)
        print(f"[INFO] Found existing results: {len(existing_keys)} rows. Will skip these tasks.")
    except Exception:
        print("[WARN] Could not read existing CSV for resume. Starting fresh.")

# ---- safe evaluation function ----
def evaluate_task(combo, scaler_name, model_name):
    key = task_key(combo, scaler_name, model_name)
    try:
        # Safety: build fresh X dataframe
        X_combo = df[list(combo)].copy()

        # Feature engineering: interactions, ratios, differences (careful safe ops)
        if len(combo) > 1:
            for i in range(len(combo)):
                for j in range(i+1, len(combo)):
                    a, b = combo[i], combo[j]
                    X_combo[f"{a}_x_{b}"] = df[a] * df[b]
                    # safe divide: replace zeros with nan then add eps
                    denom = df[b].replace(0, np.nan).fillna(1e-6)
                    X_combo[f"{a}_div_{b}"] = df[a] / denom
                    X_combo[f"{a}_minus_{b}"] = df[a] - df[b]

        # Winsorisation (1%-99% or 5%-95% depending on sample size)
        p_low = 0.01 if n_samples >= 100 else 0.05
        p_high = 0.99 if n_samples >= 100 else 0.95
        for c in X_combo.columns:
            lower, upper = X_combo[c].quantile(p_low), X_combo[c].quantile(p_high)
            X_combo[c] = X_combo[c].clip(lower, upper).fillna(X_combo[c].median())

        # Transformation
        scaler_fn = scalers[scaler_name]
        X_trans = scaler_fn(X_combo.values if hasattr(X_combo, "values") else X_combo)

        # ensure ndarray float
        X_arr = np.array(X_trans, dtype=float)

        # model instance
        model = get_model_instance(model_name)

        # LOOCV or KFold
        y_pred = np.zeros_like(y, dtype=float)
        # For boosters supporting early stopping we could implement eval_set; but with CV we fit on folds
        for train_idx, test_idx in cv.split(X_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Handle CatBoost early stopping using Pool if used and EARLY_STOP>0
            if model_name == "CatBoost" and EARLY_STOP > 0:
                # CatBoost can accept X_train,y_train directly; early stopping via eval_set
                try:
                    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=EARLY_STOP)
                except Exception:
                    # fallback without early stopping
                    model.fit(X_train, y_train)
            elif model_name == "XGBoost" and EARLY_STOP > 0:
                # xgboost scikit-learn API supports early_stopping_rounds with eval_set
                try:
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=EARLY_STOP, verbose=False)
                except Exception:
                    model.fit(X_train, y_train)
            else:
                # normal fit (Huber, Ridge, ElasticNet, LightGBM)
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            y_pred[test_idx] = np.array(preds, dtype=float).reshape(-1,)

        # metrics
        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y, y_pred)))

        return {
            "features": str(combo),
            "features_serial": ",".join(combo),
            "scaler": scaler_name,
            "model": model_name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "error": ""
        }

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()
        with open(ERROR_LOG, "a", encoding="utf-8") as ef:
            ef.write(f"\n--- {time.asctime()} ERROR for {key} ---\n{err}\n{tb}\n")
        return {
            "features": str(combo),
            "features_serial": ",".join(combo),
            "scaler": scaler_name,
            "model": model_name,
            "R2": -999.0,
            "MAE": None,
            "RMSE": None,
            "error": err
        }

# ---- Prepare tasks to run (skip existing) ----
all_tasks = [(c, s, m) for (c, s, m) in tasks if task_key(c, s, m) not in existing_keys]
print(f"[INFO] Tasks to run after skip: {len(all_tasks)}")

# ensure output csv with header
def write_header_if_needed(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["features", "features_serial", "scaler", "model", "R2", "MAE", "RMSE", "error"])
            writer.writeheader()
write_header_if_needed(OUT_CSV)

# ---- Run in chunks ----
for i in range(0, len(all_tasks), CHUNK_SIZE):
    chunk = all_tasks[i:i+CHUNK_SIZE]
    print(f"[INFO] Running chunk {i//CHUNK_SIZE + 1} / {((len(all_tasks)-1)//CHUNK_SIZE)+1}  size={len(chunk)}")
    # compute results in parallel for this chunk
    results_chunk = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(evaluate_task)(combo, scaler_name, model_name) for (combo, scaler_name, model_name) in chunk
    )
    # append to CSV
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["features", "features_serial", "scaler", "model", "R2", "MAE", "RMSE", "error"])
        for r in results_chunk:
            writer.writerow(r)
    # quick summary of chunk
    valid = [r for r in results_chunk if r["R2"] != -999.0]
    if valid:
        best = sorted(valid, key=lambda x: -x["R2"])[0]
        print(f"[CHUNK RESULT] Best in chunk: model={best['model']} scaler={best['scaler']} feats={best['features']} R2={best['R2']:.4f}")
    else:
        print("[CHUNK RESULT] No valid results in this chunk")
print("[DONE] All chunks processed. Results saved to:", OUT_CSV)

# ---- Optional stacking stage ----
# After full results saved, build stacking using top models (if enough valid rows)
try:
    df_all = pd.read_csv(OUT_CSV)
    df_valid = df_all[df_all['R2'] != -999.0].copy()
    df_valid = df_valid.sort_values(by='R2', ascending=False)
    top_rows = df_valid.head(50)  # take top 50 diverse candidates
    # choose top distinct models to stack
    stack_candidates = []
    seen = set()
    for _, r in top_rows.iterrows():
        key = (r['model'], r['scaler'], r['features_serial'])
        if r['model'] not in seen:
            stack_candidates.append(r)
            seen.add(r['model'])
        if len(stack_candidates) >= TOP_FOR_STACKING:
            break

    if len(stack_candidates) >= 2:
        print(f"[STACKING] Building stacking from top {len(stack_candidates)} models")
        from sklearn.model_selection import cross_val_predict
        meta_X = np.zeros((n_samples, len(stack_candidates)))
        model_objs = []
        for idx, r in enumerate(stack_candidates):
            combo = tuple(r['features_serial'].split(","))
            scaler_name = r['scaler']
            model_name = r['model']
            # build features and instantiate model
            X_combo = df[list(combo)].copy()
            if len(combo) > 1:
                for i in range(len(combo)):
                    for j in range(i+1, len(combo)):
                        a, b = combo[i], combo[j]
                        X_combo[f"{a}_x_{b}"] = df[a] * df[b]
                        denom = df[b].replace(0, np.nan).fillna(1e-6)
                        X_combo[f"{a}_div_{b}"] = df[a] / denom
                        X_combo[f"{a}_minus_{b}"] = df[a] - df[b]
            # winsor
            p_low = 0.01 if n_samples >= 100 else 0.05
            p_high = 0.99 if n_samples >= 100 else 0.95
            for c in X_combo.columns:
                lower, upper = X_combo[c].quantile(p_low), X_combo[c].quantile(p_high)
                X_combo[c] = X_combo[c].clip(lower, upper).fillna(X_combo[c].median())
            X_trans = scalers[scaler_name](X_combo.values)
            X_arr = np.array(X_trans, dtype=float)
            model_obj = get_model_instance(model_name)
            # out-of-fold predictions as meta feature
            preds = cross_val_predict(model_obj, X_arr, y, cv=LeaveOneOut() if USE_LOOCV else KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), n_jobs=1)
            meta_X[:, idx] = preds
            model_objs.append((model_name, model_obj, combo, scaler_name))
        # meta model fit on meta_X
        if STACKING_META == "Ridge":
            meta_model = Ridge(alpha=1.0)
        else:
            meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        try:
            meta_preds = cross_val_predict(meta_model, meta_X, y, cv=LeaveOneOut() if USE_LOOCV else KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), n_jobs=1)
            r2_meta = r2_score(y, meta_preds)
            mae_meta = mean_absolute_error(y, meta_preds)
            rmse_meta = math.sqrt(mean_squared_error(y, meta_preds))
            print(f"[STACKING RESULT] Meta-model {STACKING_META} R2={r2_meta:.4f} MAE={mae_meta:.2f} RMSE={rmse_meta:.2f}")
            # append stacking summary to CSV
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["features", "features_serial", "scaler", "model", "R2", "MAE", "RMSE", "error"])
                writer.writerow({"features": str([r['features'] for r in stack_candidates]),
                                 "features_serial": ";".join([r['features_serial'] for r in stack_candidates]),
                                 "scaler": "meta",
                                 "model": f"STACK_{STACKING_META}",
                                 "R2": r2_meta, "MAE": mae_meta, "RMSE": rmse_meta, "error": ""})
        except Exception as e:
            print("[STACKING] Stacking failed:", e)
    else:
        print("[STACKING] Not enough distinct candidate models for stacking")
except Exception as e:
    print("[STACKING] Could not run stacking stage:", e)

print("[FINISHED] Full exhaustive search script finished.")
