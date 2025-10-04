# STAGE-SPE-BA


# Projet: Prédiction du Coefficient de Solvabilité des Banques

## 1. Description

Ce projet utilise des modèles de machine learning (Ridge, LightGBM) pour prédire le **coefficient de solvabilité** des banques sur plusieurs périodes futures. L'application propose une interface **Streamlit** permettant de visualiser les prévisions, les alertes, et d’explorer les données par banque.

Fonctionnalités principales :

* Prévisions trimestrielles des coefficients de solvabilité.
* Comparaison des banques sur des graphiques interactifs.
* Détection d’alertes multi-critères (variation importante, sous quantile secteur, anomalie z-score).
* Sélection flexible des banques (une ou plusieurs, toutes, ou aucune).
* Tableaux interactifs triés par banque et période.

---

## 2. Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/Besmalaa/STAGE-SPE-BA
cd STAGE-SPE-BA
```

2. Créer un environnement virtuel (recommandé) :

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 3. Structure du projet

```
.
├── app.py               
├── dataset_ml_enriched.csv 
├── pipeline.ipynb         
├── models/               
├── requirements.txt       
└── README.md
```

---

## 4. Usage

Pour lancer l'application :

```bash
streamlit run app.py
```

### Navigation dans l'application :

* **Vue globale** : résumé des banques et graphiques globaux.
* **Prédictions futures** : timeline des coefficients par banque.
* **Graphiques** : comparaison graphique des banques sélectionnées.
* **Alertes** : affichage des alertes séparées par banque.

### Sélection des banques :

* Checkbox pour **sélectionner toutes les banques** ou aucune.
* Multiselect pour choisir **une ou plusieurs banques**.

---

## 5. Préparation des données

* Le dataset `dataset_ml_enriched.csv` doit contenir au minimum les colonnes :

  * `CODE_PARTICIPANT`
  * `DATE_ARRETE`
  * `COEFF_SOL`
  * Features financières : `TOTAL_FONDS_PROPRES`, `TOTAL_ENGAGEMENTS_NET`, etc.

* Le notebook `pipeline.ipynb` permet :
  
  * le nettoyage et la transformation des données
  * la création de features dérivées et lag
  * l’entraînement et la sauvegarde des modèles (statique du coeff de sol et celui qui permet l extrapolation temporelle)

---

## 6. Modèles utilisés

* **Ridge Regression**
* **LightGBM Regressor**
* **Ensemble** pondéré des deux modèles pour améliorer la précision

---

## 7. Alertes

Les alertes sont générées selon plusieurs critères :

* Coefficient faible par rapport à l’historique de la banque.
* Chute du coefficient supérieure à un certain pourcentage.
* Coefficient inférieur au quantile du secteur.
* Anomalie détectée via z-score.

---

## 8. Résultats attendus

* Tableaux triés par banque et période.
* Graphiques interactifs pour chaque banque ou pour toutes.
* Alertes visibles séparément par banque dans l’onglet **Alertes**.

---

## 9. Dépendances principales

* Python >= 3.9
* pandas, numpy, scikit-learn
* lightgbm
* streamlit
* plotly
* scipy

---

## 10. Contact / Auteur

Ton Nom – [mb_bouameur@esi.dz](mb_bouameur@esi.dz)
Lien GitHub : [https://github.com/Besmalaa/STAGE-SPE-BA](https://github.com/Besmalaa/STAGE-SPE-BA)

