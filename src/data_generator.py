"""
Simulador de transacciones de tarjetas de crédito basado en el libro:
'Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook'
(https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
con extensión avanzada a 6 escenarios de fraude de la industria financiera real.
"""

import os
import random
import numpy as np
import pandas as pd


def generate_customer_profiles_table(n_customers: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Genera perfiles de clientes con sus propiedades de gasto y ubicación geográfica."""
    np.random.seed(random_state)
    customer_id_properties = []

    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)
        mean_amount = np.random.uniform(5, 100)
        std_amount = mean_amount / 2
        mean_nb_tx_per_day = np.random.uniform(0, 4)

        customer_id_properties.append([
            customer_id,
            x_customer_id,
            y_customer_id,
            mean_amount,
            std_amount,
            mean_nb_tx_per_day,
        ])

    columns = [
        'CUSTOMER_ID',
        'x_customer_id',
        'y_customer_id',
        'mean_amount',
        'std_amount',
        'mean_nb_tx_per_day',
    ]
    return pd.DataFrame(customer_id_properties, columns=columns)


def generate_terminal_profiles_table(n_terminals: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Genera perfiles de terminales de pago con su ubicación geográfica en una cuadrícula 100x100."""
    np.random.seed(random_state)
    terminal_id_properties = []

    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)
        terminal_id_properties.append([terminal_id, x_terminal_id, y_terminal_id])

    columns = ['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id']
    return pd.DataFrame(terminal_id_properties, columns=columns)


def get_list_terminals_within_radius(
    customer_profile: pd.Series,
    x_y_terminals: np.ndarray,
    r: float = 5.0
) -> list:
    """Obtiene la lista de terminales disponibles dentro del radio de cercanía r del cliente."""
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)
    squared_diff = np.square(x_y_customer - x_y_terminals)
    dist = np.sqrt(np.sum(squared_diff, axis=1))
    return list(np.where(dist < r)[0])


def generate_transactions_table(
    customer_profile: pd.Series,
    start_date: str = '2024-01-01',
    nb_days: int = 30
) -> pd.DataFrame:
    """Genera las transacciones legítimas para un perfil de cliente específico."""
    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))

    for day in range(nb_days):
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        if nb_tx > 0:
            for _ in range(nb_tx):
                time_tx = int(np.random.normal(86400 / 2, 20000))
                if 0 < time_tx < 86400:
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)
                    amount = np.round(amount, 2)

                    if len(customer_profile.available_terminals) > 0:
                        terminal_id = random.choice(customer_profile.available_terminals)
                        customer_transactions.append([
                            time_tx + day * 86400,
                            day,
                            int(customer_profile.CUSTOMER_ID),
                            int(terminal_id),
                            amount,
                        ])

    columns = ['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT']
    df = pd.DataFrame(customer_transactions, columns=columns)

    if len(df) > 0:
        df['TX_DATETIME'] = pd.to_datetime(df['TX_TIME_SECONDS'], unit='s', origin=start_date)
        df = df[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']]

    return df


def add_frauds(
    customer_profiles_table: pd.DataFrame,
    terminal_profiles_table: pd.DataFrame,
    transactions_df: pd.DataFrame,
    start_date: str = '2024-01-01',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Añade transacciones fraudulentas cubriendo 6 escenarios clave de la industria financiera:
    - Escenario 1: Montos anormalmente altos (> 220).
    - Escenario 2: Terminales comprometidas (phishing/skimming) durante 28 días.
    - Escenario 3: Clientes comprometidos (monto x5) durante 14 días.
    - Escenario 4: Card Testing (Ráfaga de micro-transacciones automatizadas de $0.50 a $2.50).
    - Escenario 5: Velocidad Imposible (Impossible Travel / Clonación con gran distancia y poco tiempo).
    - Escenario 6: Fraude Nocturno atípico (Out-of-Hours compras de madrugada de alto monto).
    """
    df = transactions_df.copy()
    df['TX_FRAUD'] = 0
    df['TX_FRAUD_SCENARIO'] = 0

    max_days = int(df.TX_TIME_DAYS.max())

    # --- Escenario 1: Montos altos (> 220) ---
    s1_mask = df.TX_AMOUNT > 220
    df.loc[s1_mask, 'TX_FRAUD'] = 1
    df.loc[s1_mask, 'TX_FRAUD_SCENARIO'] = 1

    # --- Escenario 2: Terminales comprometidas durante 28 días ---
    for day in range(max_days + 1):
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
        comp_mask = (
            (df.TX_TIME_DAYS >= day) &
            (df.TX_TIME_DAYS < day + 28) &
            (df.TERMINAL_ID.isin(compromised_terminals)) &
            (df['TX_FRAUD'] == 0)
        )
        df.loc[comp_mask, 'TX_FRAUD'] = 1
        df.loc[comp_mask, 'TX_FRAUD_SCENARIO'] = 2

    # --- Escenario 3: Clientes comprometidos durante 14 días (monto x5) ---
    for day in range(max_days + 1):
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
        comp_mask = (
            (df.TX_TIME_DAYS >= day) &
            (df.TX_TIME_DAYS < day + 14) &
            (df.CUSTOMER_ID.isin(compromised_customers)) &
            (df['TX_FRAUD'] == 0)
        )
        comp_indices = df[comp_mask].index.tolist()
        if len(comp_indices) > 0:
            random.seed(day)
            k = max(1, int(len(comp_indices) / 3))
            chosen = random.sample(comp_indices, k=min(k, len(comp_indices)))
            df.loc[chosen, 'TX_AMOUNT'] = np.round(df.loc[chosen, 'TX_AMOUNT'] * 5, 2)
            df.loc[chosen, 'TX_FRAUD'] = 1
            df.loc[chosen, 'TX_FRAUD_SCENARIO'] = 3

    # --- Escenario 4: Card Testing (Micro-transacciones en ráfaga de bots) ---
    # Los atacantes validan números probando montos de $0.50 a $2.50 en ráfaga
    burst_rows = []
    random.seed(random_state + 10)
    np.random.seed(random_state + 10)
    for day in range(0, max_days + 1, 2):
        target_cust = int(customer_profiles_table.CUSTOMER_ID.sample(n=1, random_state=day + 100).iloc[0])
        target_term = int(terminal_profiles_table.TERMINAL_ID.sample(n=1, random_state=day + 200).iloc[0])
        base_time = int(np.random.uniform(3600, 80000)) + day * 86400
        n_burst = random.randint(4, 7)
        for b in range(n_burst):
            burst_rows.append({
                'TX_TIME_SECONDS': base_time + b * random.randint(15, 60),
                'TX_TIME_DAYS': day,
                'CUSTOMER_ID': target_cust,
                'TERMINAL_ID': target_term,
                'TX_AMOUNT': round(random.uniform(0.50, 2.50), 2),
                'TX_FRAUD': 1,
                'TX_FRAUD_SCENARIO': 4
            })
    if burst_rows:
        df_burst = pd.DataFrame(burst_rows)
        df_burst['TX_DATETIME'] = pd.to_datetime(df_burst['TX_TIME_SECONDS'], unit='s', origin=start_date)
        df = pd.concat([df, df_burst], ignore_index=True)

    # --- Escenario 5: Velocidad Imposible (Impossible Travel / Clonación) ---
    # Misma tarjeta transacciona en dos terminales con distancia > 60 en < 30 minutos
    terminals_xy = terminal_profiles_table.set_index('TERMINAL_ID')[['x_terminal_id', 'y_terminal_id']]
    cand_subset = df[(df['TX_FRAUD'] == 0) & (df['TX_AMOUNT'] > 20)]
    n_sample_travel = min(80, len(cand_subset))
    travel_rows = []
    if n_sample_travel > 0:
        candidates = cand_subset.sample(n=n_sample_travel, random_state=random_state + 50).index
        for idx in candidates:
            orig_r = df.loc[idx]
            orig_tid = int(orig_r['TERMINAL_ID'])
            if orig_tid in terminals_xy.index:
                ox, oy = terminals_xy.loc[orig_tid, ['x_terminal_id', 'y_terminal_id']]
                dists = np.sqrt((terminals_xy['x_terminal_id'] - ox)**2 + (terminals_xy['y_terminal_id'] - oy)**2)
                far_terminals = dists[dists > 60].index
                if len(far_terminals) > 0:
                    ftid = int(np.random.choice(far_terminals))
                    travel_rows.append({
                        'TX_TIME_SECONDS': int(orig_r['TX_TIME_SECONDS']) + random.randint(180, 1800),
                        'TX_TIME_DAYS': int(orig_r['TX_TIME_DAYS']),
                        'CUSTOMER_ID': int(orig_r['CUSTOMER_ID']),
                        'TERMINAL_ID': ftid,
                        'TX_AMOUNT': round(float(orig_r['TX_AMOUNT']) * random.uniform(1.2, 2.5), 2),
                        'TX_FRAUD': 1,
                        'TX_FRAUD_SCENARIO': 5
                    })
    if travel_rows:
        df_travel = pd.DataFrame(travel_rows)
        df_travel['TX_DATETIME'] = pd.to_datetime(df_travel['TX_TIME_SECONDS'], unit='s', origin=start_date)
        df = pd.concat([df, df_travel], ignore_index=True)

    # --- Escenario 6: Fraude Nocturno de Alto Monto (Out-of-Hours Sleeper Fraud) ---
    # Compras entre 01:00 y 04:30 AM con monto superior al triple del gasto medio del cliente
    cust_mean_map = customer_profiles_table.set_index('CUSTOMER_ID')['mean_amount'].to_dict()
    df_dt = pd.to_datetime(df['TX_DATETIME'])
    hours = df_dt.dt.hour
    night_subset = df[(hours >= 1) & (hours <= 4) & (df['TX_FRAUD'] == 0)]
    n_sample_night = min(90, len(night_subset))
    if n_sample_night > 0:
        night_candidates = night_subset.sample(n=n_sample_night, random_state=random_state + 80).index
        for idx in night_candidates:
            cid = df.loc[idx, 'CUSTOMER_ID']
            if pd.notna(cid) and int(cid) in cust_mean_map:
                c_mean = cust_mean_map[int(cid)]
                df.loc[idx, 'TX_AMOUNT'] = round(c_mean * random.uniform(3.5, 6.0), 2)
                df.loc[idx, 'TX_FRAUD'] = 1
                df.loc[idx, 'TX_FRAUD_SCENARIO'] = 6

    # Reordenar cronológicamente y asignar IDs
    df = df.sort_values('TX_TIME_SECONDS').reset_index(drop=True)
    df['TRANSACTION_ID'] = df.index
    return df


def inject_realistic_data_issues(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Introduce anomalías realistas de calidad de datos para la fase de limpieza:
    1. Duplicados accidentales (reintentos de pasarela de pago).
    2. Valores nulos en variables clave.
    3. Outliers no válidos (montos negativos por errores operativos de reversa).
    """
    np.random.seed(random_state)
    df_dirty = df.copy()

    # 1. Duplicados: duplicar un ~0.3% de filas
    n_dupes = int(len(df_dirty) * 0.003)
    dupe_rows = df_dirty.sample(n=n_dupes, random_state=random_state)
    df_dirty = pd.concat([df_dirty, dupe_rows], ignore_index=True)

    # 2. Valores nulos: introducir NaN en ~0.2% de CUSTOMER_ID y ~0.2% en TX_AMOUNT
    null_cust_idx = df_dirty.sample(frac=0.002, random_state=random_state).index
    df_dirty.loc[null_cust_idx, 'CUSTOMER_ID'] = np.nan

    null_amount_idx = df_dirty.sample(frac=0.002, random_state=random_state + 1).index
    df_dirty.loc[null_amount_idx, 'TX_AMOUNT'] = np.nan

    # 3. Outliers inválidos (montos negativos o error de captura)
    invalid_amount_idx = df_dirty.sample(n=5, random_state=random_state + 2).index
    df_dirty.loc[invalid_amount_idx, 'TX_AMOUNT'] = -99.99

    # Reordenar por tiempo y resetear índice
    df_dirty = df_dirty.sort_values('TX_TIME_SECONDS').reset_index(drop=True)
    df_dirty['TRANSACTION_ID'] = df_dirty.index
    return df_dirty


def generate_full_dataset(
    n_customers: int = 1000,
    n_terminals: int = 1500,
    nb_days: int = 30,
    start_date: str = '2024-01-01',
    r: float = 5.0,
    inject_issues: bool = True
) -> pd.DataFrame:
    """Genera el dataset completo de transacciones simuladas con 6 escenarios de fraude."""
    customer_profiles = generate_customer_profiles_table(n_customers=n_customers, random_state=42)
    terminal_profiles = generate_terminal_profiles_table(n_terminals=n_terminals, random_state=42)

    x_y_terminals = terminal_profiles[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles['available_terminals'] = customer_profiles.apply(
        lambda row: get_list_terminals_within_radius(row, x_y_terminals, r=r),
        axis=1
    )

    transactions_list = [
        generate_transactions_table(customer_profiles.iloc[i], start_date=start_date, nb_days=nb_days)
        for i in range(len(customer_profiles))
    ]
    transactions_df = pd.concat(transactions_list, ignore_index=True)
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    transactions_df['TRANSACTION_ID'] = transactions_df.index

    col_order = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID',
                 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']
    transactions_df = transactions_df[col_order]

    # Agregar los 6 escenarios de fraude
    labeled_df = add_frauds(customer_profiles, terminal_profiles, transactions_df, start_date=start_date)

    if inject_issues:
        labeled_df = inject_realistic_data_issues(labeled_df, random_state=42)

    return labeled_df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    output_path = 'data/transactions_simulated.csv'
    print("Generando dataset simulado con 6 escenarios de fraude...")
    df = generate_full_dataset(n_customers=1000, n_terminals=1500, nb_days=30, inject_issues=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generado exitosamente en '{output_path}' con {len(df):,} registros.")
    print("Desglose de escenarios de fraude:")
    print(df['TX_FRAUD_SCENARIO'].value_counts().sort_index())
