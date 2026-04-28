"""
evaluate.py

Carga el modelo final y evalúa su rendimiento sobre el test.
Genera las métricas y visualizaciones principales.

Uso:
    python src/evaluate.py
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
import seaborn as sns
import os

sns.set_theme(style="darkgrid")


def load_model():
    """Carga el modelo y el umbral guardados."""
    with open("models/lightgbm_fraud.pkl", "rb") as f:
        modelo_final = pickle.load(f)
    print(f"Modelo cargado | Umbral: {modelo_final['umbral']}")
    return modelo_final["modelo"], modelo_final["umbral"]


def load_data():
    """Carga los datos de test."""
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    print(f"X_test: {X_test.shape} | Fraudes: {y_test.sum()}")
    return X_test, y_test


def print_metrics(y_test, y_pred, y_prob):
    """Imprime las métricas principales."""
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    print("\n=== MÉTRICAS FINALES ===")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Average Precision:  {ap:.4f}")
    print(f"\nFraudes reales:     {y_test.sum()}")
    print(f"Fraudes detectados: {y_pred.sum()}")
    print(f"Fraudes perdidos:   {(y_test - y_pred).clip(0).sum()}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude'])}")


def plot_confusion_matrix(y_test, y_pred, out_dir):
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legítima", "Fraude"],
                yticklabels=["Legítima", "Fraude"], ax=ax)
    ax.set_title("Matriz de confusión — LightGBM")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ confusion_matrix.png")


def plot_roc_curve(y_test, y_prob, out_dir):
    """Genera y guarda la curva ROC."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"LightGBM (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1, label="Modelo aleatorio (AUC = 0.50)")
    ax.set_title("Curva ROC — LightGBM")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ roc_curve.png")


def plot_precision_recall(y_test, y_prob, out_dir):
    """Genera y guarda la curva Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="crimson", linewidth=2,
            label=f"LightGBM (AP = {ap:.4f})")
    ax.axhline(y=y_test.mean(), color="gray", linestyle="--",
               linewidth=1, label=f"Modelo aleatorio (AP = {y_test.mean():.4f})")
    ax.set_title("Curva Precision-Recall — LightGBM")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ precision_recall.png")


def plot_feature_importance(modelo, X_test, out_dir):
    """Genera y guarda la importancia de variables."""
    importancias = pd.DataFrame({
        "variable":    X_test.columns,
        "importancia": modelo.feature_importances_
    }).sort_values("importancia", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importancias["variable"][::-1],
            importancias["importancia"][::-1], color="steelblue")
    ax.set_title("Importancia de variables — LightGBM")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ feature_importance.png")


def evaluate():
    # Carga
    modelo, umbral = load_model()
    X_test, y_test = load_data()

    # Predicciones
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)

    # Métricas
    print_metrics(y_test, y_pred, y_prob)

    # Visualizaciones
    out_dir = "reports/images"
    os.makedirs(out_dir, exist_ok=True)

    print("\nGenerando visualizaciones...")
    plot_confusion_matrix(y_test, y_pred, out_dir)
    plot_roc_curve(y_test, y_prob, out_dir)
    plot_precision_recall(y_test, y_prob, out_dir)
    plot_feature_importance(modelo, X_test, out_dir)

    print(f"\nVisualizaciones guardadas en {out_dir}/")


if __name__ == "__main__":
    evaluate()