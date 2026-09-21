"""
Lógica de limpieza compartida por train.py y app.py.

Todo lo que está aquí es copia fiel del notebook ProyectoFinal.ipynb.
Si algo cambia en el notebook, se cambia SOLO aquí, para que el modelo
entrenado y el dashboard nunca se desincronicen.
"""
import re

import pandas as pd

# Notebook celda 8: columnas que se conservan (el resto se descarta)
USECOLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
]

# Notebook celda 15: unificación de etiquetas históricas de Product.
# (Money transfers / Virtual currency NO se unifican a propósito.)
MAPEO_PRODUCT_DUPLICADO = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit card or prepaid card": "Credit card",
    "Checking or savings account": "Bank account or service",
}

# Notebook celda 17: se conservan las top 8 categorías, el resto pasa a "Other"
TOP_N_PRODUCTS = 8

# Notebook celda 24: muestreo estratificado para NLP
SAMPLE_N = 100_000


def clean_text(text):
    """Notebook celda 21."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b[x]{2,}\b", " redacted ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_product_unified(df):
    """
    Notebook celdas 15 y 17: crea 'Product_unified' (Product original se conserva).
    Las top 8 categorías se calculan sobre TODAS las filas del df, no solo las
    que tienen narrativa, igual que en el notebook.
    """
    df["Product_unified"] = df["Product"].replace(MAPEO_PRODUCT_DUPLICADO)
    top_prods = df["Product_unified"].value_counts().head(TOP_N_PRODUCTS).index.tolist()
    df["Product_unified"] = df["Product_unified"].where(
        df["Product_unified"].isin(top_prods), "Other"
    )
    return df


def stratified_nlp_sample(nlp_df, sample_n=SAMPLE_N):
    """Notebook celda 24: muestra estratificada por Product_unified."""
    if len(nlp_df) <= sample_n:
        return nlp_df
    parts = []
    for _, group in nlp_df.groupby("Product_unified"):
        n = max(500, int(sample_n * len(group) / len(nlp_df)))
        parts.append(group.sample(min(len(group), n), random_state=42))
    return pd.concat(parts).reset_index(drop=True)
