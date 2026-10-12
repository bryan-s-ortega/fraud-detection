# Proyecto MLOps: Detección de Fraude en Tarjetas de Crédito con MLflow

Este repositorio implementa un flujo completo de Machine Learning y MLOps para la detección de transacciones fraudulentas en tarjetas de crédito, utilizando datos sintéticos basados en el estándar metodológico de la industria:
[*Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook*](https://fraud-detection-handbook.github.io/fraud-detection-handbook/).

---

## Estructura del Proyecto

```text
mlops/
├── data/
│   └── transactions_simulated.csv          # Dataset generado (~58k transacciones con 6 escenarios de fraude)
├── notebooks/
│   └── modeling.ipynb                      # Notebook enfocado exclusivamente en Exploración de Datos (EDA)
├── src/
│   ├── __init__.py
│   ├── data_generator.py                   # Simulador de perfiles, terminales y escenarios de fraude
│   └── features.py                         # Limpieza de datos y extracción de características
├── tests/
│   ├── __init__.py
│   ├── test_data_generator.py              # Pruebas unitarias del generador
│   ├── test_features.py                    # Pruebas unitarias de limpieza y features
│   └── test_train.py                       # Pruebas unitarias del script de entrenamiento y reproducibilidad
├── ANALISIS_EXPERIMENTOS.md                # Análisis detallado de las 5 corridas y comparativa en MLflow
├── mlflow.db                               # Base de datos SQLite local de MLflow Tracking
├── mlruns/                                 # Directorio local de artefactos y experimentos de MLflow
├── pyproject.toml                          # Configuración del proyecto, dependencias y pytest
├── requirements.txt                        # Lock de versiones exactas de librerías para reproducibilidad
└── README.md                               # Documentación del proyecto
```

---

## 1. Exploración de Datos (EDA)

El notebook [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb) se conserva **únicamente para la exploración de datos**:
- **Paso 1:** Carga e inspección de estructura, tipos y diccionario de datos de transacciones financieras.
- **Taxonomía:** Definición de la problemática técnica y operativa de los 6 escenarios de fraude sintéticos.
- **Paso 2:** Análisis Exploratorio de Datos (EDA) cuantitativo y visual:
  - Estadísticas descriptivas de variables numéricas.
  - Evaluación del desbalance severo de clases (~$4.8\%$ fraudes vs ~$95.2\%$ legítimas).
  - Distribuciones de montos en escala logarítmica y perfiles por escenario de fraude.
  - Análisis temporal a 30 días y conclusiones del comportamiento de los patrones de fraude.

---

## 2. Pipeline de Entrenamiento Parametrizado (`src/train.py`)

Toda la lógica de ingeniería de características, entrenamiento, evaluación y versionado de artefactos reside en el script ejecutable [`src/train.py`](src/train.py).

### Características Principales:
- **Modularidad:** Funciones con responsabilidades únicas (`load_data`, `preprocess_data`, `build_pipeline`, `train_model`, `evaluate_model`, `evaluate_scenarios`, `generate_evaluation_plot`, `run_training_pipeline`).
- **Parametrización CLI:** Control de hiperparámetros mediante `argparse`.
- **Valores por defecto:** Se ejecuta sin requerir argumentos manuales.
- **Instrumentación con MLflow Tracking:**
  - Hiperparámetros con `mlflow.log_param`
  - Métricas de evaluación con `mlflow.log_metric` (PR-AUC, ROC-AUC, Recall, Precision, F1, Accuracy, etc.)
  - Artefactos con `mlflow.log_artifact` (Matriz de confusión y curvas en PNG, resumen en JSON, desglose por escenario en CSV)
  - Modelo serializado con `mlflow.sklearn.log_model`
- **Fijación de Semilla:** Parámetro `--random-state` aplicado a generadores pseudo-aleatorios y estimadores para asegurar reproducibilidad determinista.

### Opciones de Línea de Comandos:

```bash
uv run python train.py --help
```

| Argumento | Tipo | Por Defecto | Descripción |
| :--- | :---: | :---: | :--- |
| `--C` | `float` | `1.0` | Inverso de la fuerza de regularización. |
| `--penalty` | `str` | `l2` | Norma de regularización (`l1` o `l2`). |
| `--solver` | `str` | `liblinear` | Algoritmo de optimización del clasificador. |
| `--class-weight` | `str` | `balanced` | Ponderación de clases (`balanced` o `none`). |
| `--max-iter` | `int` | `100` | Iteraciones máximas para convergencia del solver. |
| `--random-state` | `int` | `42` | Semilla aleatoria para reproducibilidad determinista. |
| `--split-day` | `int` | `21` | Día de partición temporal train/test. |
| `--data-path` | `str` | `data/transactions_simulated.csv` | Ruta al CSV de transacciones. |
| `--tracking-uri` | `str` | `sqlite:///mlflow.db` | URI de MLflow Tracking. |
| `--experiment-name` | `str` | `fraud_detection_credit_card` | Nombre del experimento de MLflow. |
| `--run-name` | `str` | `None` | Nombre descriptivo del run en MLflow. |

---

## 3. Ejecución de Corridas y Comparación en MLflow

### Ejecución básica (valores por defecto)
```bash
uv run python src/train.py
```

### Ejecución de corridas experimentales
```bash
# Corrida 1: Baseline L2 C=1.0 balanced
uv run python src/train.py --C 1.0 --penalty l2 --solver liblinear --class-weight balanced --random-state 42 --run-name "exp_1_c1.0_l2_balanced"

# Corrida 2: Fuerte regularización L2 C=0.01 balanced
uv run python src/train.py --C 0.01 --penalty l2 --solver liblinear --class-weight balanced --random-state 42 --run-name "exp_2_c0.01_l2_balanced"

# Corrida 3: Baja regularización L2 C=10.0 balanced
uv run python src/train.py --C 10.0 --penalty l2 --solver liblinear --class-weight balanced --random-state 42 --run-name "exp_3_c10.0_l2_balanced"

# Corrida 4: Regularización L1 Sparsity C=0.1 balanced
uv run python src/train.py --C 0.1 --penalty l1 --solver liblinear --class-weight balanced --random-state 42 --run-name "exp_4_c0.1_l1_balanced"

# Corrida 5: Regularización L1 C=1.0 balanced
uv run python src/train.py --C 1.0 --penalty l1 --solver liblinear --class-weight balanced --random-state 42 --run-name "exp_5_c1.0_l1_balanced"

# Corrida 6: Sin balanceo de clases (class_weight none)
uv run python src/train.py --C 1.0 --penalty l2 --solver liblinear --class-weight none --random-state 42 --run-name "exp_6_c1.0_l2_unweighted"
```

### Interfaz Gráfica de MLflow
Para contrastar y comparar visualmente todas las corridas en la interfaz web de MLflow:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Luego abrir [http://localhost:5000](http://localhost:5000) en el navegador:
1. Seleccionar el experimento `fraud_detection_credit_card`.
2. Marcar las casillas de las corridas a comparar.
3. Hacer clic en **Compare** para ver las tablas y gráficos interactivos de hiperparámetros vs. métricas (PR-AUC, ROC-AUC, Recall, Precision).

---

## 4. Reproducibilidad y Gestión de Dependencias

1. **Gestión de paquetes con `uv`:** El entorno y dependencias exactas se encuentran fijados en [`requirements.txt`](requirements.txt) y gestionados a través de `pyproject.toml` y `uv.lock`.
2. **Reproducibilidad Numérica:** Al fijar `--random-state 42` y mantener fijos los argumentos, cualquier ejecución reproduce bit a bit las predicciones, matrices de confusión y métricas registradas en MLflow.
3. **Pruebas Automatizadas con `pytest`:**
   ```bash
   uv run pytest
   ```
   Valida el generador sintético, la ingeniería de características y el comportamiento determinista de `train.py`.
