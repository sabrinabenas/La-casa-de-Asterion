# La casa de Asterión  
**Competencia de datos: Estimación de demanda y optimización de precios para artículos para el hogar**

---

## Descripción
Contamos con más de **20 millones de registros de transacciones** de productos, correspondientes al período **2021–2023** (dataset *train*).  
El objetivo es **predecir la demanda en la primera semana de 2024** (dataset *test*). Además queremos **optimizar los precios y maximizar la ganancia** para una semana dada.

---

## Preparación de los datos

Para trabajar de manera eficiente, transformamos los datos originales a formato **Parquet**, lo que permite un acceso más rápido y escalable.

```bash
pixi run to_parquet
```

Luego generamos dos datasets en formato wide (uno para train y otro para test), asegurando que contengan los mismos features y sean totalmente compatibles:

```bash
pixi run to_wide
pixi run expand_test_features
```

Estos archivos se encuentran en la carpeta data_preparation. Para correrlos hay que subir los .csv a data. 

---
## Modelos implementados

1. **Estimación de precios con HistGradientBoostingRegressor (HGBR)**  
   Entrenamos un modelo de boosting para estimar la demanda en función de:
   - Variables categóricas (producto, tienda, subgrupo, etc.)
   - Variables numéricas (precio, cantidad, estacionalidad, etc.)

   El modelo se ajusta sobre los datos de entrenamiento (2021–2023) y genera predicciones de demanda para la primera semana de 2024.

2. **Optimización de ganancias con scipy.optimize**  
   Dado el modelo de demanda, planteamos una función objetivo de ganancia:

   $\textit{ganancia} = (p - c) \cdot q(p)$

   donde:  
   - $p$: precio de venta  
   - $c$: costo  
   - $q(p)$: demanda estimada por el modelo.

   Con `scipy.optimize` buscamos el precio $p$ que maximiza la ganancia esperada para cada subgrupo de productos.

---

## Entorno de trabajo

Toda la gestión de dependencias y ejecución de comandos se realizó con pixi, lo que asegura entornos reproducibles.

Para inicializar el entorno ejecutar 
```bash
pixi install
pixi shell
```
y para cada comando:

```bash
pixi run <comando>
```

Ref: https://pixi.sh/latest/