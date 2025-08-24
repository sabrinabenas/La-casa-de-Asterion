import math
from datetime import date
from pathlib import Path

import polars as pl

pl.enable_string_cache()


GLOBAL_MIN_DATE = date(2021, 1, 1) # Fixed global origin date to mirror training script


def obtener_estacion_expr():
    """
    Retorna una expresión de Polars para determinar la estación del año basada en una columna de fecha.
    """
    return (
        pl.when(
            # Invierno: Diciembre 21+ o Enero-Febrero o Marzo 1-19
            ((pl.col("DATE").dt.month() == 12) & (pl.col("DATE").dt.day() >= 21))
            | (pl.col("DATE").dt.month().is_in([1, 2]))
            | ((pl.col("DATE").dt.month() == 3) & (pl.col("DATE").dt.day() < 20))
        )
        .then(pl.lit("Invierno"))
        .when(
            # Primavera: Marzo 20+ o Abril-Mayo o Junio 1-20
            ((pl.col("DATE").dt.month() == 3) & (pl.col("DATE").dt.day() >= 20))
            | (pl.col("DATE").dt.month().is_in([4, 5]))
            | ((pl.col("DATE").dt.month() == 6) & (pl.col("DATE").dt.day() < 21))
        )
        .then(pl.lit("Primavera"))
        .when(
            # Verano: Junio 21+ o Julio-Agosto o Septiembre 1-21
            ((pl.col("DATE").dt.month() == 6) & (pl.col("DATE").dt.day() >= 21))
            | (pl.col("DATE").dt.month().is_in([7, 8]))
            | ((pl.col("DATE").dt.month() == 9) & (pl.col("DATE").dt.day() < 22))
        )
        .then(pl.lit("Verano"))
        .otherwise(
            pl.lit("Otoño")
        )  # Septiembre 22+ o Octubre-Noviembre o Diciembre 1-20
    )


files = {
    p.stem.removeprefix("eci_"): pl.scan_parquet(p)
    for p in Path("data").glob("*.parquet")
}


def parse_flexible_date(col_name: str) -> pl.Expr:
    """Parsea fechas intentando múltiples formatos tras normalizar.

    Formatos soportados (en orden de prioridad):
    - YYYY-MM-DD
    - DD-MM-YYYY
    - MM-DD-YYYY

    Normalización previa: quita sufijo horario " 00:00:00" y reemplaza "/" por "-".
    """
    base = (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_suffix(" 00:00:00")
        .str.replace_all("/", "-")
    )
    return pl.coalesce(
        [
            base.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            base.str.strptime(pl.Date, "%d-%m-%Y", strict=False),
            base.str.strptime(pl.Date, "%m-%d-%Y", strict=False),
        ]
    ).alias(col_name)



customer_features = (
    files["customer_data"]
    .group_by(["city", "state"])
    .agg(
        pl.col("loyalty_number")
        .count()
        .alias(
            "loyalty_count"
        ),  # Using non-nan count of loyalty_number as proxy of loyalty customers count
        pl.col("loyalty_number")
        .is_null()
        .sum()
        .alias("non_loyalty_count"),  # Idem above, but counting non-loyalty customers
    )
    .with_columns(
        [
            pl.col("city").cast(pl.Categorical),
            pl.col("state").cast(pl.Categorical),
        ]
    )
)

df: pl.LazyFrame = (
    files["ids_test"]
    .with_columns(pl.col("STORE_SUBGROUP_DATE_ID").str.split("_").alias("id_parts"))
    .with_columns(
        pl.col("id_parts").list.get(0).alias("STORE_ID"),
        pl.col("id_parts").list.get(1).alias("subgroup"),
        pl.col("id_parts").list.get(2).str.to_date().alias("DATE"),
    )
    .drop("id_parts")
    .join(
        files["stores"].join(files["stores_clusters"], on="STORE_ID", how="left"),
        on="STORE_ID",
        how="left",
    )
    .with_columns(
        pl.col("STORE_ID").cast(pl.Categorical),
        pl.col("subgroup").cast(pl.Categorical),
        pl.col("BRAND").cast(
            pl.Categorical
        ),  
        pl.col("STORE_NAME").cast(
            pl.Categorical
        ),  
        pl.col("CITY").cast(pl.Categorical),
        pl.col("STATE").cast(pl.Categorical),
        pl.col("ZIP").cast(pl.Categorical),
        pl.col("STORE_TYPE").cast(pl.Categorical),
        pl.col("REGION").cast(pl.Categorical),
        pl.col("CLUSTER").cast(pl.Categorical),
        # ---- Robust OPENDATE / CLOSEDATE parsing (multi-format) ----
        parse_flexible_date("OPENDATE"),
        parse_flexible_date("CLOSEDATE"),
    )
    .drop(
        # "BRAND",
        "STORE_NAME",  
        "ADDRESS1",
        "ADDRESS2",
        "ZIP",
        "BRAND_right",
        "STORE_NAME_right",
    )
    .with_columns(
        pl.col("DATE").dt.day().alias("day"),
        pl.col("DATE").dt.weekday().alias("weekday"),
        pl.col("DATE").dt.month().alias("month"),
        pl.col("DATE").dt.year().alias("year"),
        obtener_estacion_expr().alias("estacion"),
    )
    .with_columns(
        [
            (pl.col("day") / 31 * (2 * math.pi)).sin().alias("day_sin"),
            (pl.col("day") / 31 * (2 * math.pi)).cos().alias("day_cos"),
            (pl.col("weekday") / 7 * (2 * math.pi)).sin().alias("weekday_sin"),
            (pl.col("weekday") / 7 * (2 * math.pi)).cos().alias("weekday_cos"),
            (pl.col("month") / 12 * (2 * math.pi)).sin().alias("month_sin"),
            (pl.col("month") / 12 * (2 * math.pi)).cos().alias("month_cos"),
            (
                (pl.col("DATE") - pl.lit(GLOBAL_MIN_DATE))
                .dt.total_days()
                .alias("days_since_start")
            ),
            (
                ((pl.col("DATE") - pl.col("OPENDATE")).dt.total_days() // 365).alias(
                    "years_since_open"
                )
            ),
        ]
    )
    .rename(lambda column_name: column_name.lower())
    .rename({"brand": "brand_house"})
    .join(customer_features, on=["city", "state"], how="left")
)

if __name__ == "__main__":
    df.sink_parquet("data/ids_test_expanded.parquet")
