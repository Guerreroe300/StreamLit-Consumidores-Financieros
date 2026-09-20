import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean_narrative(text):
    """
    Limpieza básica coincidente con el preprocesamiento de textos de la CFPB:
    Remueve patrones de anonimización (ej. XXXX, XX/XX/XXXX) y caracteres especiales.
    """
    if not isinstance(text, str):
        return ""
    # Remover patrones de anonimización como XXXX o xx/xx/xxxx
    text = re.sub(r'X{2,}', '', text)
    text = re.sub(r'x{2,}', '', text)
    # Remover espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # Asegurar existencia del directorio de destino para los modelos
    os.makedirs("models", exist_ok=True)
    
    data_path = "rows.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No se encontró el archivo '{data_path}'. "
            "Asegúrate de colocar el CSV original de la CFPB en la raíz del proyecto."
        )
    
    print("📥 Cargando dataset original...")
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Dimensiones iniciales del dataset: {df.shape}")
    
    # 1. Limpieza de datos (Seguir lógica estricta del notebook)
    # Conservar únicamente registros con narrativa y producto válidos
    df_clean = df.dropna(subset=["Consumer complaint narrative", "Product"]).copy()
    print(f"Registros con narrativa válida: {len(df_clean):,}")

    # --- NUEVO: Mapeo de categorías duplicadas ---
    print("🔄 Consolidando categorías de productos...")
    mapeo_product_duplicado = {
        "Credit card": "Credit card or prepaid card",
        "Prepaid card": "Credit card or prepaid card",
        "Payday loan": "Payday loan, title loan, or personal loan",
        "Virtual currency": "Money transfer, virtual currency, or money service",
        "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
        "Money transfers": "Money transfer, virtual currency, or money service"
    }
    df_clean["Product"] = df_clean["Product"].replace(mapeo_product_duplicado)
    # ----------------------------------------------
    
    # 2. Aplicar limpieza de texto sobre la narrativa
    print("🧹 Aplicando limpieza de texto y remoción de tokens anonimizados...")
    df_clean["narrative_clean"] = df_clean["Consumer complaint narrative"].apply(clean_narrative)
    
    # Filtrar textos que hayan quedado vacíos tras la limpieza
    df_clean = df_clean[df_clean["narrative_clean"].str.len() > 10]
    
    X = df_clean["narrative_clean"]
    y = df_clean["Product"]
    
    # 3. Divisón de Train / Test
    print("✂️ Dividiendo datos en entrenamiento y prueba (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Vectorización TF-IDF
    print("🔠 Transformando textos mediante TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 5. Entrenamiento del Modelo (Regresión Logística según notebook)
    print("🤖 Entrenando LogisticRegression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train_tfidf, y_train)
    
    # 6. Evaluación de Métricas
    print("📊 Evaluando modelo...")
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    
    print("\n" + "="*50)
    print(f"Accuracy:         {acc:.4f}")
    print(f"F1-Score Weighted: {f1_w:.4f}")
    print("="*50 + "\n")
    print("Reporte de Clasificación Detallado:")
    print(classification_report(y_test, y_pred))
    
    # 7. Guardar artefactos en la ruta consumida por Streamlit (models/)
    model_path = "models/model.joblib"
    tfidf_path = "models/tfidf.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, tfidf_path)
    
    print(f"✅ Modelo guardado exitosamente en: {model_path}")
    print(f"✅ Vectorizador guardado exitosamente en: {tfidf_path}")

if __name__ == "__main__":
    main()
