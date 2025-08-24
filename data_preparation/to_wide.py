import math
from datetime import date
from pathlib import Path

import polars as pl

pl.enable_string_cache()

# Fixed global origin date (dataset known start) for consistent days_since_start
GLOBAL_MIN_DATE = date(2021, 1, 1)


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
    if p.stem.startswith("eci_")
}


def parse_flexible_date(col_name: str) -> pl.Expr:
    """Intento robusto de parseo de fechas en múltiples formatos comunes.

    Soporta:
    - YYYY-MM-DD
    - DD-MM-YYYY
    - MM-DD-YYYY

    Normaliza primero:
    - Elimina sufijo horario " 00:00:00" si aparece.
    - Reemplaza "/" por "-".

    Nota sobre ambigüedad (ej. 03-04-2021): el orden de los strptime determina
    la preferencia. Actualmente priorizamos formato ISO, luego día-mes-año (más común
    en datos locales), y por último mes-día-año. Cambiar el orden si se prefiere
    priorizar formato estadounidense.
    """
    base = (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_suffix(" 00:00:00")
        .str.replace_all("/", "-")
    )
    return pl.coalesce([
        base.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        base.str.strptime(pl.Date, "%d-%m-%Y", strict=False),
        base.str.strptime(pl.Date, "%m-%d-%Y", strict=False),
    ]).alias(col_name)

# subgroup_features = (
#     files["product_master"]
#     .group_by("subgroup")
#     .agg(
#         pl.sum("costos").alias("subgroup_costos_sum"),
#         pl.sum("base_price").alias("subgroup_base_price_sum"),
#     )
#     .with_columns(pl.col("subgroup").cast(pl.Categorical))
#     .rename({"subgroup": "SUBGROUP"})  # Rename to match the main dataframe column
#     # Note that this group by is naive and does not consider that some items in a subgroup may
#     # represent a much larger portion of the sales than others. We could consider a weighted average
#     # or a more complex aggregation if needed.
# )

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
    files["transactions"]
    .join(
        files["stores"].join(files["stores_clusters"], on="STORE_ID", how="left"),
        on="STORE_ID",
        how="left",
    )
    .join(
        files["product_master"].join(files["product_groups"], on="sku", how="full"),
        left_on=["SUBGROUP", "SKU"],
        right_on=["subgroup", "sku"],
        how="left",
    )
    .drop(
        "STORE_NAME_right",
        "BRAND_right",
        "sku_right",
        "product_name_right",
    )
    .drop(
        "TRANSACTION_ID",
        "SKU",
        "ADDRESS1",
        "ADDRESS2",
    )
    .with_columns(
        pl.col("DATE").cast(pl.Date),
        pl.col("STORE_ID").cast(pl.Categorical),
        pl.col("SUBGROUP").cast(pl.Categorical),
        pl.col("BRAND").cast(
            pl.Categorical
        ),  # este es distinta a brand, cual es la diferencia? El vendedor?
        pl.col("STORE_NAME").cast(
            pl.Categorical
        ),  # esta asoociado a una brand_house + city
        pl.col("CITY").cast(pl.Categorical),
        pl.col("STATE").cast(pl.Categorical),
        pl.col("ZIP").cast(pl.Categorical),
        pl.col("STORE_TYPE").cast(pl.Categorical),
        pl.col("REGION").cast(pl.Categorical),
        pl.col("CLUSTER").cast(pl.Categorical),
        # ---- Robust OPENDATE / CLOSEDATE parsing (multi-format) ----
        # Ahora soporta también MM-DD-YYYY usando helper reutilizable.
        parse_flexible_date("OPENDATE"),
        parse_flexible_date("CLOSEDATE"),
        pl.col("category").cast(pl.Categorical),
        pl.col("group").cast(pl.Categorical),
        pl.col("brand").cast(pl.Categorical),
        pl.col("price_group_id").cast(pl.Categorical),
        pl.col("price_group_name").cast(pl.Categorical),
        pl.col("group_type").cast(pl.Categorical),
        # pl.col("STORE_SUBGROUP_DATE_ID").cast(pl.Categorical),
    )
    .with_columns(
        pl.col("DATE").dt.day().alias("day"),
        pl.col("DATE").dt.weekday().alias("weekday"),
        pl.col("DATE").dt.month().alias("month"),
        pl.col("DATE").dt.year().alias("year"),
        estacion=obtener_estacion_expr(),
        # competencia=
    )
    .with_columns(
        [
            (pl.col("day") / 31 * (2 * math.pi)).sin().alias("day_sin"),
            (pl.col("day") / 31 * (2 * math.pi)).cos().alias("day_cos"),
            (pl.col("weekday") / 7 * (2 * math.pi)).sin().alias("weekday_sin"),
            (pl.col("weekday") / 7 * (2 * math.pi)).cos().alias("weekday_cos"),
            (pl.col("month") / 12 * (2 * math.pi)).sin().alias("month_sin"),
            (pl.col("month") / 12 * (2 * math.pi)).cos().alias("month_cos"),
            # Use the globally cached minimum date so train/test align.
            (
                (pl.col("DATE") - pl.lit(GLOBAL_MIN_DATE))
                .dt
                .total_days()
                .alias("days_since_start")
            ),
            (
                ((pl.col("DATE") - pl.col("OPENDATE")).dt.total_days() // 365)
                .alias("years_since_open")
            )
        ]
    )
    .with_columns(
        [
            (pl.col("PRICE") - pl.col("base_price")).alias("price_diff"),
            (pl.col("PRICE") / pl.col("base_price")).alias("price_ratio"),
            (pl.col("PRICE") - pl.col("costos")).alias("profit_diff"),
            (pl.col("PRICE") / pl.col("costos")).alias("profit_ratio"),
        ]
    )
    # .join(subgroup_features, on="SUBGROUP")  # Join on uppercase SUBGROUP before rename
    .rename(lambda col_name: col_name.lower() if col_name != "BRAND" else "brand_house")
    .join(customer_features, on=["city", "state"], how="left")
)

if __name__ == "__main__":
    df.sink_parquet("data/wide.parquet")


# product_name : Brand + algo + subgroup, creo que la podemos sacar
# tenemos que sacar los productos que nunca tuvo alguna venta Asterion? Creemos que no
# Tratar a los libros como un solo producto!
# Cómo encontramos la distribucion de precios de cada producto para tirar outliers?

### en Kaggle solo subimos la demanda que queremos predecir (enero 2024)
### en el informe tenemos que agregar la optimizacion de precios
