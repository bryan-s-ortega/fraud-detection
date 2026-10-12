"""Script de entrenamiento y evaluación para detección de fraude en tarjetas de crédito.

Este script migra y automatiza la lógica del pipeline de modelado:
- Carga de datos
- Limpieza y feature engineering
- División temporal
- Entrenamiento con scikit-learn
- Optimización de umbral de decisión basado en la curva Precision-Recall (AUC-PR)
- Evaluación exhaustiva de métricas por escenario de fraude
- Instrumentación y versionado de experimentos con MLflow Tracking
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silenciar mensajes informativos y advertencias de deprecación
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

from src.data_generator import generate_full_dataset
from src.features import clean_transactions_data, extract_features

SCENARIO_NAMES = {
    0: "0: Legítima",
    1: "1: Monto > 220",
    2: "2: Terminal Comprometida",
    3: "3: Tarjeta Comprometida (Monto x2)",
    4: "4: Card Testing (Monto < 3)",
    5: "5: Ráfaga en 24 Horas",
    6: "6: Fraude Geográfico",
}

DEFAULT_FEATURE_COLS = [
    "TX_AMOUNT",
    "TX_AMOUNT_LOG",
    "TX_DURING_WEEKEND",
    "TX_DURING_NIGHT",
    "TX_HOUR",
    "TX_IS_MICRO_AMOUNT",
]


def load_data(data_path: str = "data/transactions_simulated.csv") -> pd.DataFrame:
    """Carga el dataset de transacciones o lo genera sintéticamente si no existe."""
    if not os.path.exists(data_path):
        print(f"Dataset no encontrado en '{data_path}'. Generando con el simulador...")
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        df = generate_full_dataset(
            n_customers=1000,
            n_terminals=1500,
            nb_days=30,
            inject_issues=True,
        )
        df.to_csv(data_path, index=False)
        print(f"Dataset generado exitosamente en: {data_path}")
    else:
        df = pd.read_csv(data_path)
        print(f"Dataset cargado desde: {data_path} ({len(df):,} filas)")
    return df


def preprocess_data(
    df: pd.DataFrame,
    split_day: int = 21,
    feature_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Ejecuta la limpieza, feature engineering y partición temporal de datos."""
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    cleaned_df = clean_transactions_data(df)
    enriched_df = extract_features(cleaned_df)

    train_mask = enriched_df["TX_TIME_DAYS"] < split_day
    test_mask = enriched_df["TX_TIME_DAYS"] >= split_day

    X_train = enriched_df.loc[train_mask, feature_cols].copy()
    y_train = enriched_df.loc[train_mask, "TX_FRAUD"].copy()

    X_test = enriched_df.loc[test_mask, feature_cols].copy()
    y_test = enriched_df.loc[test_mask, "TX_FRAUD"].copy()

    return X_train, y_train, X_test, y_test, enriched_df


def build_pipeline(
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "liblinear",
    class_weight: Optional[str] = "balanced",
    max_iter: int = 100,
    random_state: int = 42,
) -> Pipeline:
    """Construye el pipeline con StandardScaler y LogisticRegression."""
    weight = None if class_weight in (None, "none", "None") else class_weight

    classifier = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Entrena el pipeline sobre el conjunto de entrenamiento."""
    pipeline.fit(X_train, y_train)
    return pipeline


def find_best_threshold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Encuentra el umbral óptimo sobre la curva Precision-Recall que maximiza el F1-Score.

    Retorna:
        Tuple de (best_threshold, best_f1, precision_at_best, recall_at_best)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Evitar división por cero en cálculo de F1
    sum_pr = precisions[:-1] + recalls[:-1]
    f1_scores = np.where(
        sum_pr > 0,
        2 * (precisions[:-1] * recalls[:-1]) / np.maximum(sum_pr, 1e-12),
        0.0,
    )

    if len(thresholds) == 0 or np.all(f1_scores == 0):
        return 0.5, 0.0, 0.0, 0.0

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    precision_at_best = float(precisions[best_idx])
    recall_at_best = float(recalls[best_idx])

    return best_threshold, best_f1, precision_at_best, recall_at_best


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    """Calcula métricas clave (AUC-PR, ROC-AUC), busca el umbral óptimo y evalúa predicciones."""
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # 1. Métricas independientes del umbral (prioridad de optimización)
    pr_auc = float(average_precision_score(y_test, y_pred_proba))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    # 2. Selección del umbral óptimo maximizando F1 en la curva Precision-Recall
    best_threshold, best_f1, prec_opt, rec_opt = find_best_threshold(y_test, y_pred_proba)

    # 3. Predicciones con el umbral óptimo
    y_pred_opt = (y_pred_proba >= best_threshold).astype(int)

    # Métricas con umbral óptimo
    acc_opt = float(accuracy_score(y_test, y_pred_opt))
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = [int(v) for v in cm_opt.ravel()]

    # Métricas de referencia con umbral estándar 0.5
    y_pred_default = pipeline.predict(X_test)
    acc_def = float(accuracy_score(y_test, y_pred_default))
    prec_def = float(precision_score(y_test, y_pred_default, pos_label=1, zero_division=0))
    rec_def = float(recall_score(y_test, y_pred_default, pos_label=1, zero_division=0))
    f1_def = float(f1_score(y_test, y_pred_default, pos_label=1, zero_division=0))

    metrics = {
        # Métricas principales enfocadas
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        # Métricas al umbral óptimo seleccionado
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "optimal_precision": prec_opt,
        "optimal_recall": rec_opt,
        "optimal_accuracy": acc_opt,
        "optimal_tp": tp_opt,
        "optimal_fp": fp_opt,
        "optimal_tn": tn_opt,
        "optimal_fn": fn_opt,
        # Métricas estándar (umbral 0.5)
        "default_precision": prec_def,
        "default_recall": rec_def,
        "default_f1": f1_def,
        "default_accuracy": acc_def,
    }

    return metrics, y_pred_opt, y_pred_proba, best_threshold


def evaluate_scenarios(
    y_test: pd.Series,
    y_pred: np.ndarray,
    test_scenarios: pd.Series,
) -> pd.DataFrame:
    """Calcula Recall y Precision desglosados para cada tipo de fraude con el umbral dado."""
    records = []
    # Total de alertas generadas por el modelo (todas las transacciones predichas como fraude)
    total_alerts = int((y_pred == 1).sum())
    # Falsos positivos globales (transacciones predichas como fraude pero que eran legítimas: escenario 0)
    legitimate_mask = (test_scenarios == 0).values
    total_fps = int((y_pred[legitimate_mask] == 1).sum())

    for sc in sorted(test_scenarios.unique()):
        if sc == 0:
            continue
        mask = (test_scenarios == sc).values
        total_cases = int(mask.sum())
        detected_cases = int((y_pred[mask] == 1).sum())

        # 1. Recall por escenario: ¿Qué porcentaje de este tipo de fraude se detectó?
        recall_sc = (detected_cases / total_cases) if total_cases > 0 else 0.0

        # 2. Precision específica frente a falsos positivos legítimos:
        # TP_sc / (TP_sc + FP_legit)
        denom_prec = detected_cases + total_fps
        prec_sc = (detected_cases / denom_prec) if denom_prec > 0 else 0.0

        # 3. Share of alerts: ¿Qué porcentaje de todas las alertas corresponde a este escenario?
        share_alerts = (detected_cases / total_alerts) if total_alerts > 0 else 0.0

        records.append(
            {
                "scenario_id": int(sc),
                "scenario_name": SCENARIO_NAMES.get(sc, f"Escenario {sc}"),
                "total_cases": total_cases,
                "detected_cases": detected_cases,
                "recall": round(recall_sc, 4),
                "precision": round(prec_sc, 4),
                "share_of_alerts": round(share_alerts, 4),
            }
        )
    return pd.DataFrame(records)


def generate_evaluation_plot(
    y_test: pd.Series,
    y_pred_opt: np.ndarray,
    y_pred_proba: np.ndarray,
    metrics: Dict[str, float],
    output_path: str,
) -> str:
    """Genera y guarda la figura de evaluación destacando el umbral óptimo y curvas PR y ROC."""
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Matriz de confusión con umbral óptimo
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        annot_kws={"size": 13},
    )
    axes[0].set_title(
        f"Matriz de Confusión (Umbral Óptimo = {metrics['best_threshold']:.3f})\n"
        f"TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}"
    )
    axes[0].set_xlabel("Predicción del Modelo")
    axes[0].set_ylabel("Valor Real")
    axes[0].set_xticklabels(["Legítima (0)", "Fraude (1)"])
    axes[0].set_yticklabels(["Legítima (0)", "Fraude (1)"])

    # 2. Curva Precision-Recall
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(
        rec_curve,
        prec_curve,
        color="#1f77b4",
        lw=2.5,
        label=f"PR-AUC = {metrics['pr_auc']:.4f}",
    )
    # Marcar el punto del umbral óptimo
    axes[1].scatter(
        [metrics["optimal_recall"]],
        [metrics["optimal_precision"]],
        color="#d62728",
        s=90,
        zorder=5,
        label=f"Umbral Óptimo ({metrics['best_threshold']:.3f})\nF1 = {metrics['best_f1']:.3f}",
    )
    baseline_hazard = float(y_test.mean())
    axes[1].axhline(
        y=baseline_hazard,
        color="grey",
        linestyle="--",
        label=f"Azar ({baseline_hazard * 100:.1f}%)",
    )
    axes[1].set_title("Curva Precision-Recall (AUC-PR)")
    axes[1].set_xlabel("Recall (Sensibilidad)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    # 3. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[2].plot(
        fpr,
        tpr,
        color="#2ca02c",
        lw=2.5,
        label=f"ROC-AUC = {metrics['roc_auc']:.4f}",
    )
    axes[2].plot([0, 1], [0, 1], color="grey", linestyle="--", label="Azar (0.50)")
    axes[2].set_title("Curva ROC (ROC-AUC)")
    axes[2].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[2].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Configura y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento parametrizado enfocado en ROC-AUC y AUC-PR con optimización de umbral.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hiperparámetros del modelo
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización.",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help="Norma de regularización ('l1' o 'l2').",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="liblinear",
        choices=["liblinear", "saga", "lbfgs"],
        help="Algoritmo de optimización del clasificador.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Estrategia de balanceo de clases ('balanced' o 'none').",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Número máximo de iteraciones para la convergencia.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )

    # Flujo de datos y tracking
    parser.add_argument(
        "--split-day",
        type=int,
        default=21,
        help="Día de corte temporal train/test.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/transactions_simulated.csv",
        help="Ruta al dataset CSV.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="URI de MLflow Tracking.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="fraud_detection_credit_card",
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre descriptivo para la corrida.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts_temp",
        help="Directorio temporal local para artefactos.",
    )

    return parser.parse_args(args)


def run_training_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Ejecuta el ciclo de entrenamiento, optimización de umbral, evaluación y logging a MLflow."""
    # 1. Asegurar reproducibilidad fijando semillas
    np.random.seed(args.random_state)

    # 2. Cargar y preparar datos
    df = load_data(args.data_path)
    X_train, y_train, X_test, y_test, enriched_df = preprocess_data(
        df=df,
        split_day=args.split_day,
        feature_cols=DEFAULT_FEATURE_COLS,
    )

    # 3. Construir pipeline
    pipeline = build_pipeline(
        penalty=args.penalty,
        C=args.C,
        solver=args.solver,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    # 4. Configurar MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"exp_C{args.C}_{args.penalty}_{args.class_weight}_s{args.random_state}"

    os.makedirs(args.artifacts_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n[MLflow] Iniciando Run: '{run_name}' (ID: {run_id})")

        # Registro de hiperparámetros
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", args.C)
        mlflow.log_param("penalty", args.penalty)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("split_day", args.split_day)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("train_records", int(len(X_train)))
        mlflow.log_param("test_records", int(len(X_test)))
        mlflow.log_param("features", ",".join(DEFAULT_FEATURE_COLS))

        # Entrenamiento
        train_model(pipeline, X_train, y_train)

        # Evaluación con optimización de umbral
        metrics, y_pred_opt, y_pred_proba, best_thresh = evaluate_model(pipeline, X_test, y_test)

        # Registro de métricas clave en MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Desglose de Recall y Precision por cada escenario con el umbral óptimo
        test_mask = enriched_df["TX_TIME_DAYS"] >= args.split_day
        test_scenarios = enriched_df.loc[test_mask, "TX_FRAUD_SCENARIO"]
        scenario_df = evaluate_scenarios(y_test, y_pred_opt, test_scenarios)

        # Registro de métricas por escenario en MLflow
        for _, row in scenario_df.iterrows():
            sc_id = int(row["scenario_id"])
            mlflow.log_metric(f"recall_scenario_{sc_id}", float(row["recall"]))
            mlflow.log_metric(f"precision_scenario_{sc_id}", float(row["precision"]))
            mlflow.log_metric(f"share_alerts_scenario_{sc_id}", float(row["share_of_alerts"]))

        # Generación y guardado de artefactos
        # Artefacto 1: Gráficos de evaluación (Matriz con umbral óptimo + Curvas PR y ROC)
        plot_path = os.path.join(args.artifacts_dir, f"evaluation_plots_{run_id}.png")
        generate_evaluation_plot(
            y_test=y_test,
            y_pred_opt=y_pred_opt,
            y_pred_proba=y_pred_proba,
            metrics=metrics,
            output_path=plot_path,
        )
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # Artefacto 2: Métricas estructuradas en JSON
        metrics_json_path = os.path.join(args.artifacts_dir, f"metrics_{run_id}.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_json_path, artifact_path="metrics")

        # Artefacto 3: Reporte detallado de escenarios en CSV
        scenario_csv_path = os.path.join(args.artifacts_dir, f"scenario_eval_{run_id}.csv")
        scenario_df.to_csv(scenario_csv_path, index=False)
        mlflow.log_artifact(scenario_csv_path, artifact_path="reports")

        # Registro del modelo serializado
        input_example = X_test.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
        )

        # Limpieza de archivos temporales
        for fpath in [plot_path, metrics_json_path, scenario_csv_path]:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

        print("[MLflow] Métricas clave registradas:")
        print(f"  AUC-PR (Foco Principal): {metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC:                 {metrics['roc_auc']:.4f}")
        print(f"  Umbral Óptimo Selecc.:   {metrics['best_threshold']:.4f}")
        print(f"  F1 al Umbral Óptimo:     {metrics['best_f1']:.4f}")
        print(f"  Recall al Umbral Óptimo: {metrics['optimal_recall']:.4f}")
        print(f"  Prec. al Umbral Óptimo:  {metrics['optimal_precision']:.4f}")
        print(f"[MLflow] Run '{run_name}' finalizado con éxito.")

        return {
            "run_id": run_id,
            "metrics": metrics,
            "scenario_eval": scenario_df,
            "best_threshold": best_thresh,
        }


def main() -> None:
    """Punto de entrada principal CLI."""
    args = parse_args(sys.argv[1:])
    run_training_pipeline(args)


if __name__ == "__main__":
    main()
