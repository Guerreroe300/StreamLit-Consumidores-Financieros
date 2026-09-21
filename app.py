import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import re

# ---------------------------------------------------------
# Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="CFPB Financial Complaint Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Funciones de Soporte
# ---------------------------------------------------------
def clean_narrative(text):
    """Limpia el texto del usuario aplicando las mismas reglas del modelo."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'X{2,}', '', text)
    text = re.sub(r'x{2,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# Carga y Generación de Datos
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """
    Carga el dataset real desde data/complaints_sample.csv.
    Si no existe el archivo, genera un dataset sintético realista de 1,000 registros.
    """
    data_path = "rows.parquet"
    
    # Mapeo de categorías duplicadas según el descubrimiento del notebook
    mapeo_product_duplicado = {
        "Credit card": "Credit card or prepaid card",
        "Prepaid card": "Credit card or prepaid card",
        "Payday loan": "Payday loan, title loan, or personal loan",
        "Virtual currency": "Money transfer, virtual currency, or money service",
        "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
        "Money transfers": "Money transfer, virtual currency, or money service"
    }

    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        # Asegurar formato de fecha
        for date_col in ["Date received", "Date", "date_received"]:
            if date_col in df.columns:
                df["Date received"] = pd.to_datetime(df[date_col], errors="coerce")
                break
        
        # --- NUEVO: Aplicar mapeo al dataset real ---
        if "Product" in df.columns:
            df["Product"] = df["Product"].replace(mapeo_product_duplicado)
            
        return df
    else:
        # DATASET SINTÉTICO DE RESPALDO (1,000 quejas simuladas)
        np.random.seed(42)
        n_samples = 1000
        
        products = [
            "Credit reporting, credit repair services, or other personal consumer reports",
            "Debt collection",
            "Mortgage",
            "Credit card or prepaid card",
            "Checking or savings account",
            "Student loan",
            "Vehicle loan or lease"
        ]
        prod_p = [0.42, 0.20, 0.15, 0.12, 0.06, 0.03, 0.02]
        
        companies = [
            "EQUIFAX INC.", "EXPERIAN INFORMATION SOLUTIONS INC.", "TRANSUNION INTERMEDIATE HOLDINGS",
            "BANK OF AMERICA, NATIONAL ASSOCIATION", "WELLS FARGO & COMPANY", "JPMORGAN CHASE & CO.",
            "CITIBANK, N.A.", "CAPITAL ONE FINANCIAL CORPORATION", "NAVIENT CORPORATION", "SYNCHRONY FINANCIAL"
        ]
        comp_p = [0.22, 0.20, 0.18, 0.10, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02]
        
        states = ["CA", "FL", "TX", "NY", "GA", "IL", "PA", "OH", "NC", "MI", "VA", "AZ", "NJ", "TN"]
        
        issues = [
            "Incorrect information on your report",
            "Attempts to collect debt not owed",
            "Trouble during payment process",
            "Managing an account",
            "Problem with a credit reporting company's investigation",
            "Improper use of your report",
            "Fees or interest charged improperly"
        ]
        
        responses = [
            "Closed with explanation",
            "Closed with non-monetary relief",
            "Closed with monetary relief",
            "Untimely response",
            "In progress"
        ]
        
        dates = pd.date_range(start="2021-01-01", end="2023-12-31", periods=n_samples)
        
        df_mock = pd.DataFrame({
            "Date received": np.random.choice(dates, size=n_samples),
            "Product": np.random.choice(products, p=prod_p, size=n_samples),
            "Issue": np.random.choice(issues, size=n_samples),
            "Company": np.random.choice(companies, p=comp_p, size=n_samples),
            "State": np.random.choice(states, size=n_samples),
            "Timely response?": np.random.choice(["Yes", "No"], p=[0.96, 0.04], size=n_samples),
            "Company response to consumer": np.random.choice(responses, p=[0.74, 0.15, 0.07, 0.02, 0.02], size=n_samples),
            "Consumer complaint narrative": [
                "There are incorrect transactions reported on my account that need immediate correction."
            ] * n_samples
        })
        
        # Aplicar mapeo también al sintético por seguridad
        if "Product" in df_mock.columns:
            df_mock["Product"] = df_mock["Product"].replace(mapeo_product_duplicado)
            
        return df_mock

@st.cache_resource
def load_ml_artifacts():
    model_path = "models/model.joblib"
    vectorizer_path = "models/tfidf.joblib"
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    vectorizer = joblib.load(vectorizer_path) if os.path.exists(vectorizer_path) else None
    return model, vectorizer

df_raw = load_data()
model, vectorizer = load_ml_artifacts()

# ---------------------------------------------------------
# Barra Lateral (Sidebar) & Filtros
# ---------------------------------------------------------
st.sidebar.title("🏦 CFPB Analytics")
st.sidebar.markdown("---")

st.sidebar.subheader("👥 Equipo de Desarrollo")
st.sidebar.markdown("""
- **Carlos Flores**
- **Mauro Galindo**
- **Alejandro Guerrero**
- **Nicolás Castro**
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtros Exploratorios")

# Filtro por Producto
all_products = list(df_raw["Product"].dropna().unique()) if "Product" in df_raw.columns else []
selected_products = st.sidebar.multiselect("Filtrar por Producto:", options=all_products, default=all_products)

# Filtro por Estado
all_states = list(df_raw["State"].dropna().unique()) if "State" in df_raw.columns else []
selected_states = st.sidebar.multiselect("Filtrar por Estado (EE.UU.):", options=all_states, default=all_states[:5] if len(all_states) > 5 else all_states)

# Aplicar filtros
df_filtered = df_raw.copy()
if selected_products and "Product" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Product"].isin(selected_products)]
if selected_states and "State" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["State"].isin(selected_states)]

# ---------------------------------------------------------
# Interfaz Principal
# ---------------------------------------------------------
st.title("🏦 Dashboard de Quejas Financieras (CFPB)")
st.caption("Plataforma interactiva de análisis exploratorio, clasificación NLP en tiempo real y clustering.")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard / EDA", "🔮 Clasificador NLP", "🔍 Clustering y Subtemas"])

# =========================================================
# PESTAÑA 1: DASHBOARD / EDA AMPLIADO
# =========================================================
with tab1:
    st.header("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Visión detallada de las tendencias, volúmenes y comportamiento de las quejas registradas.")
    
    if df_filtered.empty:
        st.warning("No hay datos que coincidan con los filtros seleccionados en la barra lateral.")
    else:
        # --- TARJETAS DE KPIS ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric("Total de Quejas", f"{len(df_filtered):,}")
            
        with kpi2:
            top_prod = df_filtered["Product"].mode()[0] if "Product" in df_filtered.columns and not df_filtered.empty else "N/A"
            # Acortar nombre largo si es necesario
            short_prod = (top_prod[:28] + "...") if len(top_prod) > 30 else top_prod
            st.metric("Producto Más Reportado", short_prod)
            
        with kpi3:
            top_comp = df_filtered["Company"].mode()[0] if "Company" in df_filtered.columns and not df_filtered.empty else "N/A"
            short_comp = (top_comp[:25] + "...") if len(top_comp) > 27 else top_comp
            st.metric("Compañía con Más Quejas", short_comp)
            
        with kpi4:
            if "Timely response?" in df_filtered.columns:
                pct_timely = (df_filtered["Timely response?"].value_counts(normalize=True).get("Yes", 0)) * 100
                st.metric("Respuesta a Tiempo", f"{pct_timely:.1f}%")
            else:
                st.metric("Respuesta a Tiempo", "N/A")

        st.markdown("---")

        # --- FILA 1 DE GRÁFICOS ---
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("1. Quejas por Categoría de Producto")
            if "Product" in df_filtered.columns:
                prod_counts = df_filtered["Product"].value_counts().reset_index()
                prod_counts.columns = ["Producto", "Cantidad"]
                fig_prod = px.bar(
                    prod_counts,
                    x="Cantidad",
                    y="Producto",
                    orientation="h",
                    color="Cantidad",
                    color_continuous_scale="Blues",
                    title="Volumen Total por Producto"
                )
                fig_prod.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=400)
                st.plotly_chart(fig_prod, use_container_width=True)

        with col_g2:
            st.subheader("2. Top 10 Compañías Más Reportadas")
            if "Company" in df_filtered.columns:
                comp_counts = df_filtered["Company"].value_counts().head(10).reset_index()
                comp_counts.columns = ["Compañía", "Cantidad"]
                fig_comp = px.bar(
                    comp_counts,
                    x="Cantidad",
                    y="Compañía",
                    orientation="h",
                    color="Cantidad",
                    color_continuous_scale="Reds",
                    title="Top 10 Compañías con Mayor Número de Quejas"
                )
                fig_comp.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=400)
                st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # --- FILA 2 DE GRÁFICOS ---
        col_g3, col_g4 = st.columns(2)

        with col_g3:
            st.subheader("3. Principales Problemas Reportados (Issues)")
            if "Issue" in df_filtered.columns:
                issue_counts = df_filtered["Issue"].value_counts().head(7).reset_index()
                issue_counts.columns = ["Problema", "Cantidad"]
                fig_issue = px.bar(
                    issue_counts,
                    x="Cantidad",
                    y="Problema",
                    orientation="h",
                    color="Cantidad",
                    color_continuous_scale="Viridis",
                    title="Tipos de Problemas Más Frecuentes"
                )
                fig_issue.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=400)
                st.plotly_chart(fig_issue, use_container_width=True)

        with col_g4:
            st.subheader("4. Tipo de Respuesta de las Compañías")
            if "Company response to consumer" in df_filtered.columns:
                resp_counts = df_filtered["Company response to consumer"].value_counts().reset_index()
                resp_counts.columns = ["Respuesta", "Cantidad"]
                fig_resp = px.pie(
                    resp_counts,
                    values="Cantidad",
                    names="Respuesta",
                    hole=0.4,
                    title="Distribución de Respuestas Entregadas al Consumidor"
                )
                fig_resp.update_layout(height=400)
                st.plotly_chart(fig_resp, use_container_width=True)

        st.markdown("---")

        # --- FILA 3 DE GRÁFICOS ---
        col_g5, col_g6 = st.columns(2)

        with col_g5:
            st.subheader("5. Evolución Temporal de Quejas")
            if "Date received" in df_filtered.columns and not df_filtered["Date received"].isna().all():
                # Se cambia "M" por "ME" para compatibilidad con Pandas >= 2.2.0
                df_time = df_filtered.set_index("Date received").resample("ME").size().reset_index(name="Cantidad")
                fig_time = px.line(
                    df_time,
                    x="Date received",
                    y="Cantidad",
                    markers=True,
                    title="Tendencia Mensual de Quejas Ingresadas"
                )
                fig_time.update_layout(height=380)
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("Columna de fecha no disponible o sin formato válido.")

        with col_g6:
            st.subheader("6. Distribución Por Estado")
            if "State" in df_filtered.columns:
                state_counts = df_filtered["State"].value_counts().head(10).reset_index()
                state_counts.columns = ["Estado", "Cantidad"]
                fig_state = px.bar(
                    state_counts,
                    x="Estado",
                    y="Cantidad",
                    color="Cantidad",
                    color_continuous_scale="Purples",
                    title="Top 10 Estados con Mayor Volumen"
                )
                fig_state.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_state, use_container_width=True)

# =========================================================
# PESTAÑA 2: CLASIFICADOR NLP
# =========================================================
with tab2:
    st.header("Predicción de Categoría de Queja en Tiempo Real")
    st.markdown("Ingresa el texto de una queja en inglés o selecciona un ejemplo para probar el modelo de clasificación.")
    
    ejemplos = {
        "Seleccionar un ejemplo...": "",
        "Ejemplo 1: Tarjeta de Crédito": "I noticed unauthorized charges on my credit card statement last month. I called customer service immediately to dispute the transactions, but they refused to remove the fees and credited wrong interest amounts.",
        "Ejemplo 2: Hipoteca / Mortgage": "The bank increased my monthly mortgage escrow payment significantly without providing a detailed explanation or escrow shortage statement. I have requested documentation multiple times.",
        "Ejemplo 3: Cobro de Deuda / Debt Collection": "A debt collector has been calling my workplace repeatedly despite being informed that I am not allowed to receive personal calls. They are threatening legal action for a debt I do not owe.",
        "Ejemplo 4: Reporte de Crédito": "Equifax is reporting a 90-day late payment on my credit profile for an account that was closed two years ago in good standing. I sent dispute letters with proof but no changes were made."
    }
    
    opcion_ejemplo = st.selectbox("📌 Cargar un ejemplo rápido:", list(ejemplos.keys()))
    texto_defecto = ejemplos[opcion_ejemplo] if opcion_ejemplo != "Seleccionar un ejemplo..." else ""
    
    user_narrative = st.text_area(
        "Narrativa de la Queja (Consumer Complaint Narrative):",
        value=texto_defecto,
        height=150,
        placeholder="Escribe o pega aquí el texto de la queja..."
    )
    
    btn_predict = st.button("⚡ Clasificar Queja", type="primary")
    
    if btn_predict:
        if not user_narrative.strip():
            st.error("Por favor, ingresa un texto para analizar.")
        else:
            if model is None or vectorizer is None:
                st.warning("⚠️ No se encontraron los modelos pre-entrenados en `models/`. Mostrando simulación de inferencia:")
                
                clases_mock = [
                    "Credit reporting, credit repair services, or other personal consumer reports",
                    "Debt collection",
                    "Mortgage",
                    "Credit card or prepaid card",
                    "Checking or savings account"
                ]
                pred_mock = np.random.choice(clases_mock)
                probs_mock = np.random.dirichlet(np.ones(len(clases_mock)))
                
                st.success(f"### Categoría Predicha (Simulación): **{pred_mock}**")
                
                df_prob = pd.DataFrame({
                    "Categoría": clases_mock,
                    "Probabilidad": probs_mock
                }).sort_values("Probabilidad", ascending=True)
                
                fig_prob = px.bar(df_prob, x="Probabilidad", y="Categoría", orientation="h", title="Probabilidades Estimadas")
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                # --- NUEVO: Limpiar el texto ingresado por el usuario ---
                clean_text = clean_narrative(user_narrative)
                
                # Transformar el texto limpio en lugar del texto crudo
                X_tfidf = vectorizer.transform([clean_text])
                prediction = model.predict(X_tfidf)[0]
                
                st.success(f"### Categoría Predicha: **{prediction}**")
                
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_tfidf)[0]
                    classes = model.classes_
                    
                    df_prob = pd.DataFrame({
                        "Categoría": classes,
                        "Probabilidad": probabilities
                    }).sort_values("Probabilidad", ascending=True)
                    
                    fig_prob = px.bar(
                        df_prob, 
                        x="Probabilidad", 
                        y="Categoría", 
                        orientation="h",
                        color="Probabilidad",
                        color_continuous_scale="Blues",
                        title="Distribución de Probabilidades por Categoría"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

# =========================================================
# PESTAÑA 3: CLUSTERING / SUBTEMAS
# =========================================================
with tab3:
    st.header("Análisis de Subtemas y Agrupamiento (Clustering)")
    st.markdown("Agrupamiento no supervisado de las narrativas para descubrir patrones y problemáticas recurrentes.")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Clusters de Temas Identificados")
        cluster_info = pd.DataFrame({
            "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
            "Tema Principal": ["Cargos No Autorizados", "Disputas en Buró de Crédito", "Pagos Hipotecarios / Escrow", "Acoso por Cobranza de Deuda"],
            "Palabras Clave": ["card, charge, fee, bank, account", "report, credit, dispute, bureau, equifax", "mortgage, loan, escrow, payment, interest", "debt, call, collection, phone, owe"]
        })
        st.dataframe(cluster_info, use_container_width=True, hide_index=True)

    with col_c2:
        st.subheader("Distribución Porcentual de Clusters")
        df_cluster_dist = pd.DataFrame({
            "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
            "Porcentaje": [32, 38, 18, 12]
        })
        fig_donut = px.pie(
            df_cluster_dist, 
            values="Porcentaje", 
            names="Cluster", 
            hole=0.4,
            title="Proporción de Narrativas por Grupo"
        )
        st.plotly_chart(fig_donut, use_container_width=True)
