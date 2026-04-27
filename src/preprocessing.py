"""
preprocessing.py

Carga el dataset original, aplica el preprocessing completo
y guarda los datos procesados en data/processed/.

Uso:
    python src/preprocessing.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset original."""
    df = pd.read_csv(path)
    print(f"Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    return df


def split_features_target(df: pd.DataFrame):
    """Separa variables de entrada y variable objetivo."""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def get_pipeline() -> Pipeline:
    """
    Construye el Pipeline completo:
        1. RobustScaler — escala Amount y Time
        2. SMOTE        — balancea las clases
        3. (El modelo se añade en train.py)
    """
    return Pipeline([
        ("scaler", RobustScaler()),
        ("smote",  SMOTE(random_state=42)),
    ])


def run():
    # Rutas
    raw_path  = "data/creditcard.csv"
    out_dir   = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    # Carga
    df = load_data(raw_path)

    # Separar X e y
    X, y = split_features_target(df)

    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")

    # Aplicar Pipeline solo en train
    pipeline = get_pipeline()
    X_train_proc, y_train_proc = pipeline.fit_resample(X_train, y_train)

    # Escalar X_test con lo aprendido en train
    X_test_proc = pipeline.named_steps["scaler"].transform(X_test)
    X_test_proc = pd.DataFrame(X_test_proc, columns=X_test.columns)

    print(f"Después de SMOTE — Train: {X_train_proc.shape[0]:,} filas")

    # Guardar
    pd.DataFrame(X_train_proc, columns=X_train.columns).to_csv(
        f"{out_dir}/X_train.csv", index=False)
    pd.Series(y_train_proc, name="Class").to_csv(
        f"{out_dir}/y_train.csv", index=False)
    X_test_proc.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)

    print("Datos guardados correctamente en data/processed/")


if __name__ == "__main__":
    run()