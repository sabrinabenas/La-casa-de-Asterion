### prediccion de precios
import psutil
import os
import polars as pl
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# ==========================================
# FUNCIONES DE FILTRADO DE OUTLIERS para total sales
# ==========================================
def filter_outliers_from_field(
    df, field = "total_sales", iqr_multiplier = 1.5
):
    """
    Elimina outliers de un campo específico usando el IQR (Interquartile Range) por tienda.
    """
    # Calcular Q1 y Q3 por tienda
    quantiles_by_store = (
        df.group_by("store_id")
        .agg(
            [
                pl.col(field).quantile(0.25).alias("Q1"),
                pl.col(field).quantile(0.75).alias("Q3"),
            ]
        )
        .with_columns(
            (pl.col("Q3") - pl.col("Q1")).alias("IQR"),
        )
        .with_columns(
            (pl.col("Q1") - iqr_multiplier * pl.col("IQR")).alias("lower_bound"),
            (pl.col("Q3") + iqr_multiplier * pl.col("IQR")).alias("upper_bound"),
        )
    )

    # Unir Q1, Q3 e IQR al DataFrame original
    enriched = df.join(quantiles_by_store, on="store_id", how="left")

    filtered_df = enriched.filter(
        (pl.col(field) >= pl.col("lower_bound"))
        & (pl.col(field) <= pl.col("upper_bound"))
    ).drop(["Q1", "Q3", "IQR", "lower_bound", "upper_bound"])

    return filtered_df


# ==========================================
# FUNCIONES DE MODELO SINUSOIDAL
# ==========================================

def sinusoidal_func(x, A, B, C, D, S):
    """
    Función sinusoidal: A * sin(B * x + C) + D
    A: Amplitud
    B: Frecuencia
    C: Fase
    D: Offset vertical
    S: Pendiente
    """
    return A * np.sin(B * x + C) + D + S * x


def get_price_sinusoidal(test_row, sinusoidal_params, reference_date):
    """
    Estima el precio usando el ajuste sinusoidal para una fila de test

    Args:
        test_row: Fila del DataFrame de test
        sinusoidal_params: parámetros del ajuste sinusoidal [A, B, C, D, S]
        reference_date: fecha de referencia para calcular días transcurridos

    Returns:
        float: Precio estimado
    """
    # Convertir fecha a numérico
    days_from_start = (test_row["date"] - reference_date).days

    A, B, C, D, S = sinusoidal_params
    predicted_price = sinusoidal_func(days_from_start, A, B, C, D, S)

    return max(0, predicted_price)  # Asegurar precio no negativo


def fit_sinusoidal_price_model_polars(demanda_lazy: pl.LazyFrame):
    """
    Versión optimizada que usa Polars para el ajuste sinusoidal
    """
    try:
        # Agrupar por día usando Polars (más eficiente)
        demanda_df_daily = (
            demanda_lazy
            .group_by("date")
            .agg([
                pl.col("price").mean(),
                pl.col("quantity").sum(),
                pl.col("total_sales").sum(),
            ])
            .sort("date")
            .with_columns([
                ((pl.col("date") - pl.col("date").min()).dt.total_days()).alias("date_numeric")
            ])
            .collect()
            .to_pandas()
        )

        x_data = demanda_df_daily["date_numeric"].values
        y_data = demanda_df_daily["price"].values

        # Valores iniciales para el ajuste
        A_init = (y_data.max() - y_data.min()) / 2
        D_init = y_data.mean()
        B_init = 2 * np.pi / 365
        C_init = 0
        S_init = (
            (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
            if len(x_data) > 1
            else 0
        )

        # Ajustar la sinusoidal
        popt, pcov = curve_fit(
            sinusoidal_func,
            x_data,
            y_data,
            p0=[A_init, B_init, C_init, D_init, S_init],
            maxfev=10000,
        )

        # Calcular R² sin crear columnas adicionales en memoria
        A_fit, B_fit, C_fit, D_fit, S_fit = popt
        predicted_prices = sinusoidal_func(x_data, A_fit, B_fit, C_fit, D_fit, S_fit)
        
        ss_res = np.sum((y_data - predicted_prices) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"   R² ajuste sinusoidal: {r_squared:.3f}")
        
        reference_date = demanda_df_daily["date"].min()
        sinusoidal_params = [A_fit, B_fit, C_fit, D_fit, S_fit]
        
        return sinusoidal_params, reference_date, r_squared
        
    except Exception as e:
        print(f"   ❌ Error en ajuste sinusoidal: {e}")
        return None, None, None

# ==========================================
# FUNCIONES DE CARGA DE DATOS 
# ==========================================

def load_selected_cols_region(lazy_df, subgrupo: str, region: str = None):
    """Carga datos filtrados por subgrupo y opcionalmente por región"""
    query = lazy_df.select(
        pl.col("store_subgroup_date_id"),
        pl.col("store_id"),
        pl.col("subgroup"),
        pl.col("date"),
        # Joined Features
        pl.col("cluster"),
        pl.col("base_price"),
        pl.col("costos"),
        pl.col("region"),
        pl.col("store_type"),
        # Engineered Features
        pl.col("estacion"),
        pl.col("day"),
        pl.col("day_sin"),
        pl.col("day_cos"),
        pl.col("weekday"),
        pl.col("weekday_sin"),
        pl.col("weekday_cos"),
        pl.col("month"),
        pl.col("month_sin"),
        pl.col("month_cos"),
        pl.col("year"),
        pl.col("loyalty_count"),
        pl.col("non_loyalty_count"),
        pl.col("brand_house"),
        pl.col("category"),
        # Target
        pl.col("price"),
        pl.col("quantity"),
        pl.col("total_sales"),
    ).filter(pl.col("subgroup") == subgrupo)

    # Filtrar por región si se especifica
    if region is not None:
        query = query.filter(pl.col("region") == region)

    return query.collect().to_pandas()


def load_selected_cols_test_region(
    lazy_df: pl.LazyFrame, subgrupo: str, region: str = None
) -> pd.DataFrame:
    """Carga datos de test filtrados por subgrupo y opcionalmente por región"""
    query = lazy_df.select(
        pl.col("store_subgroup_date_id"),
        pl.col("store_id"),
        pl.col("subgroup"),
        pl.col("brand_house"),
        pl.col("date"),
        # Joined Features
        pl.col("cluster"),
        pl.col("region"),
        pl.col("store_type"),
        # Engineered Features
        pl.col("estacion"),
        pl.col("day"),
        pl.col("day_sin"),
        pl.col("day_cos"),
        pl.col("weekday"),
        pl.col("weekday_sin"),
        pl.col("weekday_cos"),
        pl.col("month"),
        pl.col("month_sin"),
        pl.col("month_cos"),
        pl.col("year"),
        pl.col("loyalty_count"),
        pl.col("non_loyalty_count"),
        pl.col("days_since_start"),
    ).filter(pl.col("subgroup") == subgrupo)

    # Filtrar por región si se especifica
    if region is not None:
        query = query.filter(pl.col("region") == region)

    return query.collect().to_pandas()


def load_selected_cols_test(lazy_df: pl.LazyFrame, subgrupo: str) -> pd.DataFrame:
    return (
        lazy_df.select(
            pl.col("store_subgroup_date_id"),
            pl.col("store_id"),
            pl.col("subgroup"),
            pl.col("brand_house"),
            pl.col("date"),
            # Joined Features
            pl.col("cluster"),
            pl.col("region"),
            pl.col("store_type"),
            # Engineered Features
            pl.col("estacion"),
            pl.col("day"),
            pl.col("day_sin"),
            pl.col("day_cos"),
            pl.col("weekday"),
            pl.col("weekday_sin"),
            pl.col("weekday_cos"),
            pl.col("month"),
            pl.col("month_sin"),
            pl.col("month_cos"),
            pl.col("year"),
            pl.col("loyalty_count"),
            pl.col("non_loyalty_count"),
            # Nuevos features
            pl.col("days_since_start")
        )
        .filter(subgroup=subgrupo)
        .collect()
        .to_pandas()
    )

# ==========================================
# FUNCIONES DE PREPARACIÓN DE FEATURES (VERSIÓN POLARS OPTIMIZADA)
# ==========================================

def prepare_test_features_polars(test_df: pl.LazyFrame, demanda_lazy: pl.LazyFrame, 
                               sinusoidal_params=None, reference_date=None):
    """
    Versión optimizada que prepara features usando solo Polars hasta el final
    """
    # Vectorizar estimación de precios si tenemos parámetros sinusoidales
    if sinusoidal_params is not None and reference_date is not None:
        A, B, C, D, S = sinusoidal_params
        test_df = test_df.with_columns([
            pl.max_horizontal(
                0,
                A * (B * ((pl.col("date") - reference_date).dt.total_days()) + C).sin() + 
                D + S * ((pl.col("date") - reference_date).dt.total_days())
            ).alias("price")
        ])
    else:
        # Fallback: usar promedio por día de la semana (más eficiente que apply)
        price_by_weekday = (
            demanda_lazy
            .group_by(pl.col("date").dt.weekday())
            .agg(pl.col("price").mean().alias("avg_price"))
            .rename({"date": "weekday_ref"})
        )
        
        test_df = test_df.join(
            price_by_weekday,
            left_on=pl.col("date").dt.weekday(),
            right_on="weekday_ref",
            how="left"
        ).with_columns(
            pl.col("avg_price").fill_null(
                demanda_lazy.select(pl.col("price").mean()).collect().item()
            ).alias("price")
        ).drop("avg_price")
    
    # Vectorizar estimación de otras variables usando joins eficientes
    for variable in ["quantity", "base_price", "costos", "price"]:  # CUIDADO
        if variable in demanda_lazy.collect_schema().names():
            # Crear lookup table por día de la semana y día del mes
            lookup_table = (
                demanda_lazy
                .with_columns([
                    pl.col("date").dt.weekday().alias("weekday_ref"),
                    pl.col("date").dt.day().alias("day_ref")
                ])
                .group_by(["weekday_ref", "day_ref"])
                .agg(pl.col(variable).mean().alias(f"avg_{variable}"))
            )
            
            # Join con fallback
            test_df = test_df.with_columns([
                pl.col("date").dt.weekday().alias("weekday_ref"),
                pl.col("date").dt.day().alias("day_ref")
            ]).join(
                lookup_table,
                on=["weekday_ref", "day_ref"],
                how="left"
            )
            
            # Fallback para valores nulos
            fallback_value = demanda_lazy.select(pl.col(variable).mean()).collect().item()
            test_df = test_df.with_columns(
                pl.max_horizontal(0, pl.col(f"avg_{variable}").fill_null(fallback_value)).alias(variable)
            ).drop([f"avg_{variable}", "weekday_ref", "day_ref"])
    
    # Calcular ratios de forma vectorizada
    test_df = test_df.with_columns([
        (pl.col("price") / pl.col("costos")).alias("avg_profit_ratio"),
        (pl.col("price") / pl.col("base_price")).alias("avg_price_ratio"),
    ])
    
    return test_df

# ==========================================
# FUNCIONES DE ENTRENAMIENTO Y PREDICCIÓN
# ==========================================

def train_and_predict(demand_grouped_df, test_df_subgroup, pipeline_region):
    """
    Entrena el modelo y genera predicciones
    
    Returns:
        array: predicciones de total_sales
    """
    # Preparar datos de entrenamiento
    demand_grouped_df = demand_grouped_df.dropna().reset_index(drop=True)
    
    y_train = demand_grouped_df["total_sales"]
    X_train = demand_grouped_df.drop(
        columns=[
            "date",
            "index",
            "total_sales",
            "region",
            "subgroup", 
            "day",
            "weekday",
            "month",
            "costos",
            "base_price",
            "price",  
        ],
        axis=1,
        errors="ignore"
    )
    
    # Entrenar modelo
    pipeline_region.fit(X_train, y_train)
    
    # Preparar datos de test
    X_test_no_enc = test_df_subgroup.loc[:, X_train.columns.tolist()]
    
    # Generar predicciones
    predictions = pipeline_region.predict(X_test_no_enc)
    
    return predictions


# ==========================================
# FUNCIONES DE PROCESAMIENTO POR PRODUCTO
# ==========================================
def process_basketball_special_case(lazy_test_df, subgrupo):
    """
    Maneja el caso especial del Basketball (ventas = 0)
    """
    test_df_subgroup = load_selected_cols_test(lazy_test_df, subgrupo=subgrupo)
    test_df_subgroup["total_sales"] = 0
    
    return {
        store_subgroup_date_id: total_sales
        for store_subgroup_date_id, total_sales in zip(
            test_df_subgroup["store_subgroup_date_id"],
            test_df_subgroup["total_sales"],
        )
    }


def process_product_region_combination_polars(prod, region, lazy_grouped_region, 
                                            lazy_wide_df, lazy_test_df, pipeline_region):
    """
    Versión completamente optimizada usando Polars hasta el último momento
    """
    print(f"   Procesando: {prod} - {region}")
    
    # 1. Filtrar datos de entrenamiento (mantener como LazyFrame)
    demand_grouped_lazy = lazy_grouped_region.filter(pl.col("subgroup") == prod)
    if region is not None:
        demand_grouped_lazy = demand_grouped_lazy.filter(pl.col("region") == region)
    # 2. Contar filas antes del filtrado
    rows_before = demand_grouped_lazy.select(pl.len()).collect().item()
    print(f"   📊 Filas antes filtrado: {rows_before:,}")
    
    # 3. Filtrar outliers (mantener lazy)
    demand_grouped_lazy = filter_outliers_from_field(
        demand_grouped_lazy, field="total_sales", iqr_multiplier=3.5
    )
    
    rows_after = demand_grouped_lazy.select(pl.len()).collect().item()
    removed_rows = rows_before - rows_after
    removal_percentage = (removed_rows / rows_before) * 100 if rows_before > 0 else 0
    print(f"   🗑️ Outliers removidos: {removed_rows:,} ({removal_percentage:.1f}%)")
    
    # 4. Datos históricos para modelo sinusoidal (mantener lazy)
    demanda_lazy = lazy_wide_df.filter(pl.col("subgroup") == prod)
    
    if region is not None:
        demanda_lazy = demanda_lazy.filter(pl.col("region") == region)
        
    demanda_lazy = demanda_lazy.sort("date")
    
    # 5. Ajustar modelo sinusoidal usando versión optimizada
    sinusoidal_params, reference_date, r_squared = fit_sinusoidal_price_model_polars(demanda_lazy)
    
    # 6. Preparar datos de test (mantener lazy hasta lo último)
    test_lazy = lazy_test_df.filter(pl.col("subgroup") == prod)
    
    if region is not None:
        test_lazy = test_lazy.filter(pl.col("region") == region)

    
    # 7. Preparar features del test con versión optimizada
    test_lazy = prepare_test_features_polars(
        test_lazy, demanda_lazy, sinusoidal_params, reference_date
    )
    
    # 8. SOLO AHORA convertir a pandas para el modelo (momento crítico de memoria)
    demand_grouped_df = demand_grouped_lazy.collect().to_pandas().dropna().reset_index(drop=True)
    test_df_subgroup = test_lazy.collect().to_pandas()
    
    demand_grouped_df['residuos_precio'] = demand_grouped_df['price'] - sinusoidal_func(
        demand_grouped_df["days_since_start"],
        *sinusoidal_params
    )
    # print('mal')
    if demand_grouped_df['residuos_precio'].isnull().any():
        raise ValueError("Error en el rellenado de residuos")

    test_df_subgroup["residuos_precio"] = test_df_subgroup["price"] - sinusoidal_func(
    test_df_subgroup["days_since_start"],
    *sinusoidal_params
    )

    # 9. Entrenar y predecir (parte que requiere pandas)
    predictions = train_and_predict(demand_grouped_df, test_df_subgroup, pipeline_region)
    
    # 10. Crear resultados como diccionario
    results = {
        row["store_subgroup_date_id"]: pred 
        for row, pred in zip(test_df_subgroup.to_dict('records'), predictions)
    }
    
    # 11. Limpiar memoria inmediatamente
    del demand_grouped_df, test_df_subgroup
    
    print(f"   ✅ Generadas {len(results):,} predicciones")
    return results

# ==========================================
# FUNCIONES DE MONITOREO
# ==========================================

def check_memory():
    """Función para monitorear uso de memoria"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

# ==========================================
# FUNCIONES PARA GANANCIAS
# ==========================================

def train_and_predict_ganancias(demand_grouped_df, test_df_subgroup, pipeline_region):
    """
    Entrena el modelo y genera predicciones
    
    Returns:
        array: predicciones de total_sales
    """
    # Preparar datos de entrenamiento
    demand_grouped_df = demand_grouped_df.dropna().reset_index(drop=True)
    
    y_train = demand_grouped_df["total_sales"]
    y_test = test_df_subgroup["total_sales"]
    X_train = demand_grouped_df.drop(
        columns=[
            "date",
            "index",
            "total_sales",
            "region",
            "subgroup", 
            "day",
            "weekday",
            "month",
            "costos",
            "base_price",
            "price", 
            "store_subgroup_date_id"
        ],
        axis=1,
        errors="ignore"
    )
    
    # Entrenar modelo
    pipeline_region.fit(X_train, y_train)
    
    # Preparar datos de test
    X_test_no_enc = test_df_subgroup.loc[:, X_train.columns.tolist()]
    
    # Generar predicciones
    predictions = pipeline_region.predict(X_test_no_enc)
    print(f'score: {pipeline_region.score(X_test_no_enc, y_test)}')
    return predictions, pipeline_region


def process_product_region_combination_polars_ganancias(prod, region, lazy_grouped_region, 
                                            lazy_wide_df, lazy_test_df, pipeline_region):
    """
    Versión completamente optimizada usando Polars hasta el último momento
    """
    print(f"   Procesando: {prod} - {region}")
    
    # 1. Filtrar datos de entrenamiento (mantener como LazyFrame)
    demand_grouped_lazy = lazy_grouped_region.filter(pl.col("subgroup") == prod)
    if region is not None:
        demand_grouped_lazy = demand_grouped_lazy.filter(pl.col("region") == region)
    # 2. Contar filas antes del filtrado
    rows_before = demand_grouped_lazy.select(pl.len()).collect().item()
    print(f"   📊 Filas antes filtrado: {rows_before:,}")
    
    # 3. Filtrar outliers (mantener lazy)
    demand_grouped_lazy = filter_outliers_from_field(
        demand_grouped_lazy, field="total_sales", iqr_multiplier=3.5
    )
    
    rows_after = demand_grouped_lazy.select(pl.len()).collect().item()
    removed_rows = rows_before - rows_after
    removal_percentage = (removed_rows / rows_before) * 100 if rows_before > 0 else 0
    print(f"   🗑️ Outliers removidos: {removed_rows:,} ({removal_percentage:.1f}%)")
    
    # 4. Datos históricos para modelo sinusoidal (mantener lazy)
    demanda_lazy = lazy_wide_df.filter(pl.col("subgroup") == prod)
    
    if region is not None:
        demanda_lazy = demanda_lazy.filter(pl.col("region") == region)
        
    demanda_lazy = demanda_lazy.sort("date")
    
    # 5. Ajustar modelo sinusoidal usando versión optimizada
    sinusoidal_params, reference_date, r_squared = fit_sinusoidal_price_model_polars(demanda_lazy)
    
    # 6. Preparar datos de test (mantener lazy hasta lo último)
    test_lazy = lazy_test_df.filter(pl.col("subgroup") == prod)
    
    if region is not None:
        test_lazy = test_lazy.filter(pl.col("region") == region)

    
    # 7. Preparar features del test con versión optimizada
    test_lazy = prepare_test_features_polars(
        test_lazy, demanda_lazy, sinusoidal_params, reference_date
    )
    
    # 8. SOLO AHORA convertir a pandas para el modelo (momento crítico de memoria)
    demand_grouped_df = demand_grouped_lazy.collect().to_pandas().dropna().reset_index(drop=True)
    test_df_subgroup = test_lazy.collect().to_pandas()
    
    demand_grouped_df['residuos_precio'] = demand_grouped_df['price'] - sinusoidal_func(
        demand_grouped_df["days_since_start"],
        *sinusoidal_params
    )
    if demand_grouped_df['residuos_precio'].isnull().any():
        raise ValueError("Error en el rellenado de residuos")

    test_df_subgroup["residuos_precio"] = test_df_subgroup["price"] - sinusoidal_func(
    test_df_subgroup["days_since_start"],
    *sinusoidal_params
    )

    predictions, pipeline = train_and_predict_ganancias(demand_grouped_df, test_df_subgroup, pipeline_region)
    
    # 10. Crear resultados como diccionario
    results = {
        row["store_subgroup_date_id"]: pred 
        for row, pred in zip(test_df_subgroup.to_dict('records'), predictions)
    }
    
    # 11. Limpiar memoria inmediatamente
    #del demand_grouped_df, test_df_subgroup
    
    print(f"   ✅ Generadas {len(results):,} predicciones")
    return results, pipeline, test_df_subgroup, sinusoidal_params