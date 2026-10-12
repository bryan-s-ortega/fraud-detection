"""
Pruebas unitarias para limpieza de datos y extracción de características.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import clean_transactions_data, extract_features


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'TRANSACTION_ID': [0, 1, 2, 3, 4],
        'TX_DATETIME': [
            '2024-01-01 02:30:00',
            '2024-01-01 02:30:00',  # Duplicado de la fila 0
            '2024-01-01 14:15:00',
            '2024-01-06 23:00:00',  # Sabado (fin de semana)
            '2024-01-02 10:00:00',  # Con monto negativo
        ],
        'CUSTOMER_ID': [101, 101, np.nan, 102, 103],
        'TERMINAL_ID': [500, 500, 501, 502, 503],
        'TX_AMOUNT': [50.0, 50.0, 75.0, 120.0, -10.0],
        'TX_FRAUD': [0, 0, 0, 1, 0],
    })


def test_clean_transactions_data(sample_raw_data):
    cleaned = clean_transactions_data(sample_raw_data)

    # 1. Duplicado eliminado
    # 2. NaN en CUSTOMER_ID eliminado
    # 3. Monto negativo eliminado
    # De 5 filas originales, deben quedar 2 (fila 0 y fila 3)
    assert len(cleaned) == 2
    assert (cleaned['TX_AMOUNT'] > 0).all()
    assert cleaned['CUSTOMER_ID'].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(cleaned['TX_DATETIME'])


def test_extract_features(sample_raw_data):
    cleaned = clean_transactions_data(sample_raw_data)
    featured = extract_features(cleaned)

    assert 'TX_HOUR' in featured.columns
    assert 'TX_DURING_WEEKEND' in featured.columns
    assert 'TX_DURING_NIGHT' in featured.columns
    assert 'TX_AMOUNT_LOG' in featured.columns

    # Fila con hora 02:30 debe tener TX_DURING_NIGHT == 1
    night_row = featured[featured['TX_HOUR'] == 2]
    assert (night_row['TX_DURING_NIGHT'] == 1).all()

    # Fila de sabado debe tener TX_DURING_WEEKEND == 1
    sat_row = featured[featured['TX_DAY_OF_WEEK'] == 5]
    assert (sat_row['TX_DURING_WEEKEND'] == 1).all()
