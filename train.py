import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from preprocessing import (
    USECOLS,
    add_product_unified,
    clean_text,
    stratified_nlp_sample,
)

NARRATIVE = "Consumer complaint narrative"


def load_raw():
    """Lee rows.parquet (preferido) o rows.csv, solo con las columnas del notebook."""
    if os.path.exists("rows.parquet"):
        df = pd.read_parquet("rows.parquet")
        return df[[c for c in USECOLS if c in df.columns]]
    if os.path.exists("rows.csv"):
        return pd.read_csv("rows.csv", low_memory=False)[USECOLS]
    raise FileNotFoundError(
        "No se encontró 'rows.parquet' ni 'rows.csv' en la raíz del proyecto."
    )


def main():
    os.makedirs("models", exist_ok=True)

    print("📥 Cargando dataset...")
    df = load_raw()
    print(f"Dimensiones iniciales: {df.shape}")

    # Notebook celdas 15-17: unificar Product y agrupar el resto en 'Other'
    df = add_product_unified(df)

    # Notebook celdas 23-24: solo filas con narrativa + muestreo estratificado
    nlp_df = df[df[NARRATIVE].notna()].copy()
    print(f"Filas con narrativa: {len(nlp_df):,} de {len(df):,}")
    nlp_df = stratified_nlp_sample(nlp_df)
    print(f"Filas para NLP tras muestreo: {len(nlp_df):,}")

    # Notebook celda 45 filtra `!= 'Otros'`, pero esa etiqueta no existe (la clase
    # se llama 'Other'), así que en el notebook el filtro no hace nada y 'Other'
    # SÍ participa en el modelo. Aquí se replica eso: no se filtra nada.
    nlp_model = nlp_df.copy()
    nlp_model["text_clean"] = nlp_model[NARRATIVE].map(clean_text)
    print(nlp_model["Product_unified"].value_counts())

    # Notebook celda 48: TF-IDF ajustado sobre todo el texto, luego split 80/20
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X = tfidf.fit_transform(nlp_model["text_clean"])
    y = nlp_model["Product_unified"]

    idx = np.arange(len(nlp_model))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    print("🤖 Entrenando LogisticRegression...")
    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # Notebook celda 52: baseline para comparar
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)

    print("\n" + "=" * 50)
    print(f"Logistic Regression - Accuracy: {acc:.4f} | F1 macro: {f1_macro:.4f}")
    print(
        f"Baseline            - Accuracy: {accuracy_score(y_test, y_dummy):.4f} | "
        f"F1 macro: {f1_score(y_test, y_dummy, average='macro'):.4f}"
    )
    print("=" * 50 + "\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(log_reg, "models/model.joblib")
    joblib.dump(tfidf, "models/tfidf.joblib")
    print("✅ Modelo guardado en models/model.joblib")
    print("✅ Vectorizador guardado en models/tfidf.joblib")


if __name__ == "__main__":
    main()
