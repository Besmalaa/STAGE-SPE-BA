import pandas as pd

print("HELLo ")
# Charger le fichier résultats
df = pd.read_csv("results_full_search.csv")

# Trier par R² décroissant
df_sorted = df.sort_values(by="R2", ascending=False)

# Sauvegarder les résultats triés dans un nouveau fichier
df_sorted.to_csv("resultats_full_search_sorted.csv", index=False)

# Afficher les 10 meilleurs résultats
print(df_sorted.head(10))
