# 🔍 Detección de Fraude en Tarjetas de Crédito

Modelo de Machine Learning para detectar transacciones fraudulentas en
tarjetas de crédito, aplicando técnicas avanzadas para el tratamiento
de datasets extremadamente desbalanceados.

## 📊 Dataset

- **Fuente:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transacciones totales:** 284.807
- **Fraudes:** 492 (0.17%)
- **Variables:** 30 (V1-V28 anonimizadas por PCA + Amount + Time)

## 🛠️ Tecnologías

Python · Pandas · NumPy · Scikit-learn · LightGBM · XGBoost · SMOTE · Matplotlib · Seaborn

## 🗺️ Estructura del proyecto

```
Proyecto-ML-fraude/
├── data/
│   ├── creditcard.csv
│   └── processed/          # Datos preprocesados
├── models/
│   └── lightgbm_fraud.pkl  # Modelo final guardado
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── images/             # Visualizaciones generadas
├── src/                    # Scripts .py (pipeline final)
├── requirements.txt
└── README.md
```

## ⚙️ Metodología

### 1. EDA
- Análisis del desbalance extremo de clases (0.17% fraudes)
- Distribución de importes: fraudes concentrados en importes bajos (mediana 9.25€)
- Análisis temporal: fraudes ocurren también de madrugada
- Correlación de variables V1-V28 con la clase objetivo

### 2. Preprocessing
- Split 80/20 estratificado antes de cualquier transformación
- Escalado de Amount y Time con RobustScaler (resistente a outliers)
- SMOTE aplicado **solo en train** para evitar data leakage

### 3. Entrenamiento y comparativa de modelos

| Modelo | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Regresión Logística | 0.9712 | 0.0588 | 0.9184 | 0.1106 |
| Random Forest | 0.9615 | 0.8617 | 0.8265 | 0.8438 |
| XGBoost | 0.9811 | 0.6204 | 0.8673 | 0.7234 |
| **LightGBM** | **0.9790** | **0.6364** | **0.8571** | **0.7304** |

### 4. Threshold Tuning

| Modelo | Umbral | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost optimizado | 0.85 | 0.8367 | 0.8367 | 0.8367 |
| **LightGBM optimizado** | **0.80** | **0.8384** | **0.8469** | **0.8426** |

### 5. Modelo final — LightGBM (umbral 0.80)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.9790 |
| Average Precision | 0.8288 |
| Precision | 0.8384 |
| Recall | 0.8469 |
| F1-Score | 0.8426 |
| Fraudes detectados | 83 / 98 (84.7%) |
| Falsas alarmas | 16 / 56.864 (0.03%) |

## 🚀 Reproducir el proyecto

**1. Clona el repositorio**
```bash
git clone https://github.com/LucasBalaguer/Proyecto-ML-fraude.git
cd Proyecto-ML-fraude
```

**2. Crea el entorno virtual e instala las dependencias**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**3. Descarga el dataset**

Descarga `creditcard.csv` desde [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
y colócalo en la carpeta `data/`.

**4. Ejecuta los notebooks en orden**
```
01_eda.ipynb
02_preprocessing.ipynb
03_training.ipynb
04_evaluation.ipynb
```

## 📈 Resultados principales

- El modelo detecta el **84.7% de los fraudes reales**
- Solo **16 falsas alarmas** de 56.864 transacciones legítimas
- **487x mejor** que un modelo aleatorio en Average Precision
- Variables más importantes: **V14, V4, Amount, V12**

## 👤 Autor

**Lucas Balaguer**
[LinkedIn](https://www.linkedin.com/in/lucasbalaguer/) · [GitHub](https://github.com/LucasBalaguer)