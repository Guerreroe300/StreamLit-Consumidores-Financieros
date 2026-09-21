import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

from preprocessing import USECOLS, add_product_unified, clean_text

# ---------------------------------------------------------
# Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="CFPB Financial Complaint Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Carga y Generación de Datos
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """
    Carga rows.parquet y aplica la misma preparación del notebook:
      - solo las 10 columnas de DATA_DICT (celda 8)
      - 'Date received' a datetime (celda 9)
      - Product_unified: unificación de etiquetas + top 8 y "Other" (celdas 15 y 17)
      - has_narrative (celda 23)
    """
    df = pd.read_parquet("rows.parquet")
    df = df[[c for c in USECOLS if c in df.columns]]
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    df = add_product_unified(df)
    df["has_narrative"] = df["Consumer complaint narrative"].notna()
    return df

@st.cache_resource
def load_ml_artifacts():
    model_path = "models/model.joblib"
    vectorizer_path = "models/tfidf.joblib"
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    vectorizer = joblib.load(vectorizer_path) if os.path.exists(vectorizer_path) else None
    return model, vectorizer

if not os.path.exists("rows.parquet"):
    st.error("No se encontró `rows.parquet` en la raíz del proyecto.")
    st.stop()

df_raw = load_data()
model, vectorizer = load_ml_artifacts()

# ---------------------------------------------------------
# Barra Lateral (Sidebar) & Filtros
# ---------------------------------------------------------
st.sidebar.title("CFPB Analytics")
st.sidebar.markdown("---")

st.sidebar.subheader("Equipo de Desarrollo")
st.sidebar.markdown("""
- **Carlos Flores**
- **Mauro Galindo**
- **Alejandro Guerrero**
- **Nicolás Castro**
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros Exploratorios")

# Filtro por Producto
all_products = list(df_raw["Product_unified"].dropna().unique())
selected_products = st.sidebar.multiselect("Filtrar por Producto:", options=all_products, default=all_products)

# Filtro por Estado
all_states = sorted(df_raw["State"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Filtrar por Estado (EE.UU.):",
    options=all_states,
    default=[],
    help="Vacío = todos los estados.",
)

# Aplicar filtros
df_filtered = df_raw.copy()
if selected_products:
    df_filtered = df_filtered[df_filtered["Product_unified"].isin(selected_products)]
if selected_states:
    df_filtered = df_filtered[df_filtered["State"].isin(selected_states)]

# ---------------------------------------------------------
# Interfaz Principal
# ---------------------------------------------------------
st.title("Dashboard de Quejas Financieras (CFPB)")
st.caption("Plataforma interactiva de análisis exploratorio y clasificación NLP en tiempo real.")

tab1, tab2 = st.tabs(["Dashboard / EDA", "Clasificador NLP"])

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
            top_prod = df_filtered["Product_unified"].mode()[0]
            # Acortar nombre largo si es necesario
            short_prod = (top_prod[:28] + "...") if len(top_prod) > 30 else top_prod
            st.metric("Producto Más Reportado", short_prod)
            
        with kpi3:
            top_comp = df_filtered["Company"].mode()[0] if "Company" in df_filtered.columns and not df_filtered.empty else "N/A"
            short_comp = (top_comp[:25] + "...") if len(top_comp) > 27 else top_comp
            st.metric("Compañía con Más Quejas", short_comp)
            
        with kpi4:
            pct_narr = df_filtered["has_narrative"].mean() * 100
            st.metric("Quejas con Narrativa", f"{pct_narr:.1f}%")

        st.markdown("---")

        # --- FILA 1 DE GRÁFICOS ---
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("1. Quejas por Categoría de Producto")
            prod_counts = df_filtered["Product_unified"].value_counts().reset_index()
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
            st.subheader("4. % de Quejas con Narrativa por Producto")
            narr_by_prod = (
                df_filtered.groupby("Product_unified")["has_narrative"].mean()
                .mul(100).sort_values().reset_index()
            )
            narr_by_prod.columns = ["Producto", "% con narrativa"]
            fig_narr = px.bar(
                narr_by_prod,
                x="% con narrativa",
                y="Producto",
                orientation="h",
                color="% con narrativa",
                color_continuous_scale="Purples",
                title="Porcentaje de Quejas que Incluyen Narrativa"
            )
            fig_narr.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_narr, use_container_width=True)

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
    
    opcion_ejemplo = st.selectbox("Cargar un ejemplo rápido:", list(ejemplos.keys()))
    texto_defecto = ejemplos[opcion_ejemplo] if opcion_ejemplo != "Seleccionar un ejemplo..." else ""
    
    user_narrative = st.text_area(
        "Narrativa de la Queja (Consumer Complaint Narrative):",
        value=texto_defecto,
        height=150,
        placeholder="Escribe o pega aquí el texto de la queja..."
    )
    
    btn_predict = st.button("Clasificar Queja", type="primary")
    
    if btn_predict:
        if not user_narrative.strip():
            st.error("Por favor, ingresa un texto para analizar.")
        else:
            if model is None or vectorizer is None:
                st.error("No se encontraron los modelos en `models/`. Ejecuta `python train.py` primero.")
            else:
                # Misma limpieza que el notebook (clean_text) antes de vectorizar
                X_tfidf = vectorizer.transform([clean_text(user_narrative)])
                prediction = model.predict(X_tfidf)[0]
                
                st.success(f"### Categoría Predicha: **{prediction}**")
                
                probabilities = model.predict_proba(X_tfidf)[0]
                df_prob = pd.DataFrame({
                    "Categoría": model.classes_,
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
