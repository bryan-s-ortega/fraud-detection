"""
Pruebas unitarias para el generador de datos sintéticos de detección de fraude.
"""

import os
import pandas as pd
import pytest
from src.data_generator import (
    generate_customer_profiles_table,
    generate_terminal_profiles_table,
    get_list_terminals_within_radius,
    generate_full_dataset,
)


def test_generate_customer_profiles():
    n_cust = 10
    cust_df = generate_customer_profiles_table(n_customers=n_cust, random_state=42)
    assert len(cust_df) == n_cust
    assert 'CUSTOMER_ID' in cust_df.columns
    assert 'mean_amount' in cust_df.columns
    assert (cust_df['mean_amount'] > 0).all()


def test_generate_terminal_profiles():
    n_term = 15
    term_df = generate_terminal_profiles_table(n_terminals=n_term, random_state=42)
    assert len(term_df) == n_term
    assert 'TERMINAL_ID' in term_df.columns
    assert 'x_terminal_id' in term_df.columns


def test_terminals_within_radius():
    cust_df = generate_customer_profiles_table(n_customers=1, random_state=42)
    term_df = generate_terminal_profiles_table(n_terminals=20, random_state=42)
    x_y_term = term_df[['x_terminal_id', 'y_terminal_id']].values.astype(float)

    terminals = get_list_terminals_within_radius(cust_df.iloc[0], x_y_term, r=100.0)
    # Con r=100 en un cuadrante 100x100, al menos un terminal o casi todos deben caer dentro
    assert len(terminals) > 0


def test_generate_full_dataset_smoke():
    df = generate_full_dataset(n_customers=20, n_terminals=30, nb_days=2, inject_issues=True)
    assert isinstance(df, pd.DataFrame)
    assert 'TX_FRAUD' in df.columns
    assert len(df) > 0
