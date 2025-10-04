import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
from scipy.stats import zscore

# ================= CONFIGURATION =================
DATA_PATH = "dataset_ml_enriched.csv"
TARGET_COL = "COEFF_SOL"
BANK_COL = "CODE_PARTICIPANT"
DATE_COL = "DATE_ARRETE"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_FEATURES = [
    "TOTAL_FONDS_PROPRES",
    "TOTAL_ENGAGEMENTS_NET",
    "TOTAL_EQUIVALENT_RISQUE",
    "TOTAL_POS_LONGUE_BILAN",
    "TOTAL_POS_COURTE_BILAN",
    "TOTAL_ETAT_RECAP_4000C",
    "TOTAL_ETAT_RECAP_2000E"
]

# ================= IMPORTATION DU DATASET =================
df = pd.read_csv(DATA_PATH, sep=";")
df = df.dropna(subset=[TARGET_COL] + BASE_FEATURES)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values([BANK_COL, DATE_COL]).reset_index(drop=True)
df["days_since_start"] = (df[DATE_COL] - df[DATE_COL].min()).dt.days

# ================= FEATURE ENGINEERING =================
df["RATIO_ENG_NET_FONDS"] = df["TOTAL_ENGAGEMENTS_NET"] / (df["TOTAL_FONDS_PROPRES"] + 1e-9)
df["RATIO_EQUIV_FONDS"] = df["TOTAL_EQUIVALENT_RISQUE"] / (df["TOTAL_FONDS_PROPRES"] + 1e-9)
df["RATIO_POS_LONGUE"] = df["TOTAL_POS_LONGUE_BILAN"] / (df["TOTAL_POS_COURTE_BILAN"] + 1e-9)
df["DIFF_ETAT_RECAP"] = df["TOTAL_ETAT_RECAP_4000C"] - df["TOTAL_ETAT_RECAP_2000E"]
df["SUM_ENG_EQUIV"] = df["TOTAL_ENGAGEMENTS_NET"] + df["TOTAL_EQUIVALENT_RISQUE"]

lag_cols = BASE_FEATURES + [TARGET_COL]
for col in lag_cols:
    df[f"{col}_lag1"] = df.groupby(BANK_COL)[col].shift(1)
    df[f"{col}_lag2"] = df.groupby(BANK_COL)[col].shift(2)
    df[f"{col}_pct1"] = df.groupby(BANK_COL)[col].pct_change(1)
    df[f"{col}_roll2"] = df.groupby(BANK_COL)[col].shift(1).rolling(2, min_periods=1).mean()
    df[f"{col}_roll3"] = df.groupby(BANK_COL)[col].shift(1).rolling(3, min_periods=1).mean()
df.fillna(0, inplace=True)

df["ENG_PER_FONDS"] = df["TOTAL_ENGAGEMENTS_NET_lag1"] / (df["TOTAL_FONDS_PROPRES_lag1"] + 1e-9)
df["EQV_PER_FONDS"] = df["TOTAL_EQUIVALENT_RISQUE_lag1"] / (df["TOTAL_FONDS_PROPRES_lag1"] + 1e-9)
df["POS_LONGUE_OVER_SHORT"] = df["TOTAL_POS_LONGUE_BILAN_lag1"] / (df["TOTAL_POS_COURTE_BILAN_lag1"] + 1e-9)

FEATURES = BASE_FEATURES + [
    "days_since_start",
    "RATIO_ENG_NET_FONDS", "RATIO_EQUIV_FONDS", "RATIO_POS_LONGUE",
    "DIFF_ETAT_RECAP", "SUM_ENG_EQUIV"
]
lag_features = []
for col in lag_cols:
    lag_features += [f"{col}_lag1", f"{col}_lag2", f"{col}_pct1", f"{col}_roll2", f"{col}_roll3"]
FEATURES += lag_features + ["ENG_PER_FONDS", "EQV_PER_FONDS", "POS_LONGUE_OVER_SHORT"]
FEATURES = [f for f in FEATURES if f in df.columns]

# ================= NORMALISATION =================
X = df[FEATURES].values
y = df[TARGET_COL].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= MODEL =================
def make_lgb():
    return lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.02, num_leaves=64,
        max_depth=-1, min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_STATE, verbose=-1
    )
def make_ridge():
    return Ridge(alpha=1.0)

train_ratio = 0.8
split_idx = int(len(df) * train_ratio)
X_train, y_train = X_scaled[:split_idx], y[:split_idx]
X_test, y_test = X_scaled[split_idx:], y[split_idx:]

def augment_X(X, y, n_copies=5, scale=0.05):
    aug_X, aug_y = [X], [y]
    for _ in range(n_copies):
        factors = np.random.uniform(1-scale, 1+scale, size=X.shape)
        aug_X.append(X * factors)
        aug_y.append(y)
    return np.vstack(aug_X), np.concatenate(aug_y)

X_train_aug, y_train_aug = augment_X(X_train, y_train, n_copies=5)
ridge = make_ridge()
lgbm = make_lgb()
ridge.fit(X_train_aug, y_train_aug)
lgbm.fit(X_train_aug, y_train_aug)

# ================= PREDICTIONS FUTURES =================
def predict_future(df, periods=3):
    results = []
    df_sorted = df.sort_values([BANK_COL, DATE_COL]).copy()
    for bank, group in df_sorted.groupby(BANK_COL):
        last_row = group.iloc[-1].copy()
        future_row = last_row.copy()
        last_date = last_row[DATE_COL]
        for p in range(1, periods+1):
            future_date = last_date + pd.DateOffset(months=3*p)
            future_row[DATE_COL] = future_date
            similar_banks = df_sorted[df_sorted[BANK_COL] != bank].copy()
            for f in BASE_FEATURES:
                ratio_last = last_row["TOTAL_FONDS_PROPRES"] / (last_row["TOTAL_ENGAGEMENTS_NET"] + 1e-9)
                ratio_sim = similar_banks["TOTAL_FONDS_PROPRES"] / (similar_banks["TOTAL_ENGAGEMENTS_NET"] + 1e-9)
                sim_scores = 1 / (np.abs(ratio_sim - ratio_last) + 1e-6)
                pct_changes = similar_banks[f].pct_change().fillna(0)
                weighted_change = np.sum(sim_scores.values * pct_changes.values) / np.sum(sim_scores.values)
                future_row[f] = future_row[f] * (1 + weighted_change)
            # recalcul features dérivées
            future_row["RATIO_ENG_NET_FONDS"] = future_row["TOTAL_ENGAGEMENTS_NET"] / (future_row["TOTAL_FONDS_PROPRES"] + 1e-9)
            future_row["RATIO_EQUIV_FONDS"] = future_row["TOTAL_EQUIVALENT_RISQUE"] / (future_row["TOTAL_FONDS_PROPRES"] + 1e-9)
            future_row["RATIO_POS_LONGUE"] = future_row["TOTAL_POS_LONGUE_BILAN"] / (future_row["TOTAL_POS_COURTE_BILAN"] + 1e-9)
            future_row["DIFF_ETAT_RECAP"] = future_row["TOTAL_ETAT_RECAP_4000C"] - future_row["TOTAL_ETAT_RECAP_2000E"]
            future_row["SUM_ENG_EQUIV"] = future_row["TOTAL_ENGAGEMENTS_NET"] + future_row["TOTAL_EQUIVALENT_RISQUE"]
            future_row["ENG_PER_FONDS"] = future_row["TOTAL_ENGAGEMENTS_NET"] / (future_row["TOTAL_FONDS_PROPRES"] + 1e-9)
            future_row["EQV_PER_FONDS"] = future_row["TOTAL_EQUIVALENT_RISQUE"] / (future_row["TOTAL_FONDS_PROPRES"] + 1e-9)
            future_row["POS_LONGUE_OVER_SHORT"] = future_row["TOTAL_POS_LONGUE_BILAN"] / (future_row["TOTAL_POS_COURTE_BILAN"] + 1e-9)
            X_future = scaler.transform(future_row[FEATURES].values.reshape(1, -1))
            pred_r = ridge.predict(X_future)[0]
            pred_l = lgbm.predict(X_future)[0]
            pred_ens = 0.7 * pred_r + 0.3 * pred_l
            results.append({
                BANK_COL: bank,
                "DATE_PRED": future_date,
                "COEFF_SOL_PRED": pred_ens
            })
    return pd.DataFrame(results)

df_future = predict_future(df, periods=3)

# ================= ALERTES =================
RELATIVE_FACTOR = 0.80
PCT_DROP = 25.0
SECTOR_QUANTILE = 0.10
ANOMALY_STD = 2.0

bank_means = df.groupby(BANK_COL)[TARGET_COL].mean()
df_future_sorted = df_future.sort_values([BANK_COL, "DATE_PRED"]).copy()
df_future_sorted["VAR_PCT"] = df_future_sorted.groupby(BANK_COL)["COEFF_SOL_PRED"].pct_change() * 100
sector_threshold = df_future_sorted["COEFF_SOL_PRED"].quantile(SECTOR_QUANTILE)
df_future_sorted["Z_SCORE"] = df_future_sorted.groupby(BANK_COL)["COEFF_SOL_PRED"].transform(lambda x: zscore(x, nan_policy='omit'))

def detect_alert(row):
    alerts = []
    bank = row[BANK_COL]
    pred = row["COEFF_SOL_PRED"]
    mean_hist = bank_means.get(bank, np.nan)
    if not np.isnan(mean_hist) and pred < RELATIVE_FACTOR * mean_hist:
        alerts.append("Relatif faible")
    if not np.isnan(row["VAR_PCT"]) and row["VAR_PCT"] < -PCT_DROP:
        alerts.append(f"Chute >{PCT_DROP}%")
    if pred < sector_threshold:
        alerts.append("Sous quantile secteur")
    if not np.isnan(row["Z_SCORE"]) and abs(row["Z_SCORE"]) > ANOMALY_STD:
        alerts.append("Anomalie z-score")
    return "OK" if len(alerts)==0 else ";".join(alerts)

df_future_sorted["ALERTE_MULTI"] = df_future_sorted.apply(detect_alert, axis=1)
# ================= STREAMLIT APP =================
st.set_page_config(layout="wide")
st.title("Exportation temporelle de la solvabilité des banques")

# ================= NAVIGATION =================
pages = ["Vue globale", "Prédictions futures", "Graphiques", "Alertes"]

# Initialisation du state
if "page" not in st.session_state:
    st.session_state.page = pages[0]

st.sidebar.markdown("### Navigation")
# Un seul bouton cliquable par page
for p in pages:
    if st.sidebar.button(p, key=p):
        st.session_state.page = p

page = st.session_state.page

# ================= SIDEBAR BANQUES =================
all_banks = df_future_sorted[BANK_COL].unique().tolist()
select_all = st.sidebar.checkbox("Sélectionner toutes les banques", value=True)
if select_all:
    selected_banks = all_banks
else:
    selected_banks = st.sidebar.multiselect("Banques", options=all_banks, default=[])

# --------- Filtrage ----------
df_display = df_future_sorted.copy()
if page == "Alertes" and not selected_banks:
    df_display = df_display[df_display["ALERTE_MULTI"] != "OK"]
elif selected_banks:
    df_display = df_display[df_display[BANK_COL].isin(selected_banks)]

# ================= PAGE: Vue globale =================
if page == "Vue globale":
    st.subheader("Résumé global des banques")
    st.markdown(f"- Nombre de banques total: **{len(all_banks)}**")
    st.markdown(f"- Moyenne du coefficient de solvabilité: **{df_future_sorted['COEFF_SOL_PRED'].mean():.2f}**")
    st.markdown(f"- Min / Max du coefficient de solvabilité: **{df_future_sorted['COEFF_SOL_PRED'].min():.2f} / {df_future_sorted['COEFF_SOL_PRED'].max():.2f}**")
    st.markdown(f"- Total des alertes : **{(df_future_sorted['ALERTE_MULTI']!='OK').sum()}**")

    fig = px.line(df_future_sorted, x="DATE_PRED", y="COEFF_SOL_PRED", color=BANK_COL,
                  markers=True, title="Évolution globale des coefficients")
    fig.update_layout(xaxis_title="Trimestre", yaxis_title="Coefficient de solvabilité", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ================= PAGE: Prédictions futures =================
elif page == "Prédictions futures":
    st.subheader("Timeline des coefficients prédits")
    for bank in df_display[BANK_COL].unique():
        st.markdown(f"**Banque : {bank}**")
        df_bank = df_display[df_display[BANK_COL]==bank][["DATE_PRED", "COEFF_SOL_PRED"]]
        df_bank = df_bank.set_index(pd.Index([bank]*len(df_bank)))
        st.dataframe(df_bank)

# ================= PAGE: Graphiques =================
elif page == "Graphiques":
    st.subheader("Comparaison graphique des banques")
    fig = px.line(df_display, x="DATE_PRED", y="COEFF_SOL_PRED", color=BANK_COL,
                  markers=True, title="Évolution du coefficient de solvabilité")
    fig.update_layout(xaxis_title="Trimestre", yaxis_title="Coefficient de solvabilité",
                      legend_title="Banque", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Valeurs détaillées")
    df_display_indexed = df_display[["DATE_PRED", "COEFF_SOL_PRED", BANK_COL]].set_index(BANK_COL)
    st.dataframe(df_display_indexed)

# ================= PAGE: Alertes =================
elif page == "Alertes":
    st.subheader("Alertes multi-critères")
    if df_display.empty:
        st.info("Aucune alerte à afficher pour la sélection actuelle.")
    else:
        for bank in df_display[BANK_COL].unique():
            st.markdown(f"### Banque : {bank}")
            df_bank_alerts = df_display[df_display[BANK_COL]==bank][["DATE_PRED", "COEFF_SOL_PRED", "ALERTE_MULTI"]]
            df_bank_alerts = df_bank_alerts.set_index(pd.Index([bank]*len(df_bank_alerts)))
            st.dataframe(df_bank_alerts)
            
            # Graphique par banque avec alertes en rouge
            fig = px.line(df_bank_alerts.reset_index(), x="DATE_PRED", y="COEFF_SOL_PRED", markers=True)
            alerts_df = df_bank_alerts[df_bank_alerts["ALERTE_MULTI"]!="OK"].reset_index()
            if not alerts_df.empty:
                fig.add_scatter(x=alerts_df["DATE_PRED"], y=alerts_df["COEFF_SOL_PRED"], mode="markers",
                                marker=dict(color="red", size=10), name="Alertes")
            fig.update_layout(title=f"Évolution du coefficient pour {bank}", xaxis_title="Trimestre",
                              yaxis_title="Coefficient de solvabilité", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
