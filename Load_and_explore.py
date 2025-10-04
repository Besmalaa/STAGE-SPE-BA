# load_and_explore.py
import pandas as pd


CSV_FILE = "Table ML.csv"

# Si ton CSV est séparé par ";" change sep=";"
df = pd.read_csv(CSV_FILE, sep=",", encoding="utf-8")

# Vue d'ensemble
print("Shape:", df.shape)
print("\nColonnes:", df.columns.tolist())
print("\nTypes:")
print(df.dtypes)

# Aperçu
print("\nHead:")
print(df.head(5))

# Comptage des valeurs manquantes par colonne
print("\nMissing (count per col):")
print(df.isnull().sum())

# Statistiques descriptives pour les colonnes numériques
print("\nDescribe (num):")
print(df.describe().T)
