"""
Módulo de transformación y limpieza de características para detección de fraude.
"""

import numpy as np
import pandas as pd


def clean_transactions_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame de transacciones:
    1. Elimina duplicados de transacciones idénticas (idempotencia).
    2. Imputa o descarta registros con valores nulos críticos (CUSTOMER_ID, TX_AMOUNT).
    3. Trata tipos de datos incorrectos (convierte TX_DATETIME a datetime).
    4. Trata outliers inválidos (ej. montos negativos o valores sin sentido físico).
    """
    cleaned_df = df.copy()

    # 1. Tratar tipos incorrectos: TX_DATETIME a tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(cleaned_df['TX_DATETIME']):
        cleaned_df['TX_DATETIME'] = pd.to_datetime(cleaned_df['TX_DATETIME'])

    # 2. Eliminar duplicados
    # Una transacción con mismo tiempo, cliente, terminal y monto es un duplicado por reintento
    initial_len = len(cleaned_df)
    subset_cols = ['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT']
    cleaned_df = cleaned_df.drop_duplicates(subset=subset_cols, keep='first')

    # 3. Tratar valores faltantes
    # En pasarelas de pago, no se puede evaluar riesgo sin identificador de cliente ni monto
    cleaned_df = cleaned_df.dropna(subset=['CUSTOMER_ID', 'TX_AMOUNT'])
    cleaned_df['CUSTOMER_ID'] = cleaned_df['CUSTOMER_ID'].astype(int)
    cleaned_df['TERMINAL_ID'] = cleaned_df['TERMINAL_ID'].astype(int)

    # 4. Tratar outliers no válidos
    # Transacciones con monto <= 0 son inconsistencias operativas (reversos o errores técnicos)
    cleaned_df = cleaned_df[cleaned_df['TX_AMOUNT'] > 0]

    # Reordenar y reindexar
    cleaned_df = cleaned_df.sort_values('TX_DATETIME').reset_index(drop=True)
    return cleaned_df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características de contexto temporal y de comportamiento transaccional
    siguiendo las recomendaciones del Fraud Detection Handbook:
    - TX_DURING_WEEKEND: Indica si la transacción ocurrió en fin de semana.
    - TX_DURING_NIGHT: Indica si la transacción ocurrió en horario nocturno (00:00 a 06:00).
    - TX_HOUR: Hora del día (0-23).
    - TX_DAY_OF_WEEK: Día de la semana (0=Lunes, 6=Domingo).
    - TX_AMOUNT_LOG: Logaritmo natural del monto para normalizar asimetría positiva.
    """
    data = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(data['TX_DATETIME']):
        data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])

    hour = data['TX_DATETIME'].dt.hour
    weekday = data['TX_DATETIME'].dt.weekday

    data['TX_HOUR'] = hour
    data['TX_DAY_OF_WEEK'] = weekday
    data['TX_DURING_WEEKEND'] = weekday.isin([5, 6]).astype(int)
    data['TX_DURING_NIGHT'] = ((hour >= 0) & (hour < 6)).astype(int)
    data['TX_AMOUNT_LOG'] = np.log1p(data['TX_AMOUNT'])
    data['TX_IS_MICRO_AMOUNT'] = (data['TX_AMOUNT'] < 3.0).astype(int)

    return data
