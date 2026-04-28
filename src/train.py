"""
train.py

Carga los datos procesados, entrena LightGBM con GridSearchCV
y guarda el modelo final en models/.

Uso:
    python src/train.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Carga los datos procesados."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def find_optimal_threshold(y_test, y_prob) -> float:
    """Encuentra el umbral que maximiza el F1-Score."""
    from sklearn.metrics import f1_score

    umbrales = np.arange(0.1, 0.9, 0.01)
    mejor_f1 = 0
    mejor_umbral = 0.5

    for umbral in umbrales:
        y_pred = (y_prob >= umbral).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

    return round(mejor_umbral, 2)


def train():
    # Carga de datos
    X_train, y_train, X_test, y_test = load_data()

    # Validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hiperparámetros
    params = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 6],
        "learning_rate": [0.05, 0.1]
    }

    # Modelo
    lgb_model = lgb.LGBMClassifier(random_state=42, verbosity=-1)

    # GridSearchCV
    print("Entrenando LightGBM con GridSearchCV...")
    grid = GridSearchCV(
        lgb_model, params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"Mejores parámetros: {grid.best_params_}")
    print(f"Mejor ROC-AUC en CV: {grid.best_score_:.4f}")

    # Threshold tuning
    y_prob = grid.predict_proba(X_test)[:, 1]
    umbral = find_optimal_threshold(y_test, y_prob)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\nROC-AUC en test: {auc:.4f}")
    print(f"Umbral óptimo:   {umbral}")

    # Guardar modelo
    os.makedirs("models", exist_ok=True)

    modelo_final = {
        "modelo":  grid.best_estimator_,
        "umbral":  umbral,
        "params":  grid.best_params_,
        "roc_auc": round(auc, 4)
    }

    with open("models/lightgbm_fraud.pkl", "wb") as f:
        pickle.dump(modelo_final, f)

    print("\nModelo guardado correctamente en models/lightgbm_fraud.pkl")


if __name__ == "__main__":
    train()