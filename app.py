import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
st.set_page_config(page_title="Visor de Precipitaciones - Antioquia", layout="wide")

# Cargar datos
@st.cache_data
def cargar_datos():
    # Datos de precipitaciones
    pptn = pd.read_csv("Transp_Est_Pptn.csv")
    pptn["Estacion"] = pptn["Estacion"].astype(str)

    # Datos de metadatos
    meta = pd.read_csv("EstHM_CV.csv")
    meta["Estacion"] = meta["Estacion"].astype(str)

    # Unión por columna 'Estacion'
    df = pd.merge(meta, pptn, on="Estacion", how="inner")
    return df, pptn, meta

df, pptn, meta = cargar_datos()

# ----------------------------
# SIDEBAR - FILTROS
# ----------------------------
st.sidebar.header("Filtros")

# Selector de estaciones
estaciones_unicas = sorted(df["Estacion"].unique())
col1, col2 = st.sidebar.columns([1, 1])

if "estaciones_sel" not in st.session_state:
    st.session_state.estaciones_sel = estaciones_unicas

def seleccionar_todas():
    st.session_state.estaciones_sel = estaciones_unicas

def limpiar_seleccion():
    st.session_state.estaciones_sel = []

with col1:
    if st.button("Seleccionar todas"):
        seleccionar_todas()
with col2:
    if st.button("Limpiar"):
        limpiar_seleccion()

estaciones_sel = st.sidebar.multiselect(
    "Selecciona estaciones",
    options=estaciones_unicas,
    default=st.session_state.estaciones_sel
)
st.session_state.estaciones_sel = estaciones_sel

# Selector de rango de años
años = [col for col in pptn.columns if col.isdigit()]
años = sorted(map(int, años))
min_año, max_año = min(años), max(años)
rango_años = st.sidebar.slider("Rango de años", min_año, max_año, (min_año, max_año))

# ----------------------------
# FILTRAR DATOS
# ----------------------------
columnas_datos = ["Estacion"] + [str(a) for a in range(rango_años[0], rango_años[1] + 1)]
df_filtrado = df[df["Estacion"].isin(estaciones_sel)][columnas_datos + list(meta.columns[1:])]

# ----------------------------
# PESTAÑAS PRINCIPALES
# ------
