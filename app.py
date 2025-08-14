import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd

st.set_page_config(page_title="Visualizador de Lluvia", layout="wide")

# -------------------------------
# Función para cargar datos
# -------------------------------
@st.cache_data
def cargar_csv(nombre_archivo, nombre_session):
    ruta = Path("data") / nombre_archivo
    if ruta.exists():
        return pd.read_csv(ruta)
    elif nombre_session in st.session_state:
        return st.session_state[nombre_session]
    else:
        return None

# -------------------------------
# Cargar datos
# -------------------------------
with st.sidebar:
    st.header("📂 Carga de datos")
    uploaded_meta = st.file_uploader("Sube EstHM_CV.csv", type="csv")
    if uploaded_meta:
        st.session_state["meta"] = pd.read_csv(uploaded_meta)

    uploaded_pptn = st.file_uploader("Sube Transp_Est_Pptn.csv", type="csv")
    if uploaded_pptn:
        st.session_state["pptn_raw"] = pd.read_csv(uploaded_pptn)

    uploaded_df = st.file_uploader("Sube DatosProcesados.csv", type="csv")
    if uploaded_df:
        st.session_state["df"] = pd.read_csv(uploaded_df)

meta = cargar_csv("EstHM_CV.csv", "meta")
pptn_raw = cargar_csv("Transp_Est_Pptn.csv", "pptn_raw")
df = cargar_csv("DatosProcesados.csv", "df")

if meta is None or pptn_raw is None or df is None:
    st.error("⚠️ Faltan datos. Sube los 3 archivos CSV en la barra lateral o colócalos en la carpeta `data/`.")
    st.stop()

# -------------------------------
# Sidebar configuración
# -------------------------------
st.sidebar.header("🎨 Configuración de Colores")
invertir_colores = st.sidebar.checkbox("Invertir colores (Azul = Mucha lluvia)", value=True)
paleta_manual = st.sidebar.selectbox(
    "Paleta de colores",
    ["viridis", "plasma", "inferno", "coolwarm", "Spectral", "turbo"],
    index=0
)

# -------------------------------
# Slider de años
# -------------------------------
if "Año" in pptn_raw.columns:
    años_disponibles = sorted(pptn_raw["Año"].dropna().unique())
    año_seleccionado = st.sidebar.select_slider(
        "📅 Año de análisis",
        options=["Promedio"] + list(años_disponibles),
        value="Promedio"
    )
else:
    año_seleccionado = "Promedio"

# -------------------------------
# Procesamiento
# -------------------------------
st.title("🌧 Visualizador de Lluvia")

if año_seleccionado == "Promedio":
    datos = pptn_raw.groupby("Estacion")["Precipitacion"].mean().reset_index()
else:
    datos = pptn_raw[pptn_raw["Año"] == año_seleccionado].groupby("Estacion")["Precipitacion"].mean().reset_index()

# Merge con coordenadas
if "lat" in meta.columns and "lon" in meta.columns:
    datos = datos.merge(meta[["Estacion", "lat", "lon"]], on="Estacion", how="left")

# -------------------------------
# Mapa
# -------------------------------
shapefile_path = Path("data") / "cuencas.shp"
cmap = sns.color_palette(paleta_manual, as_cmap=True)
if invertir_colores:
    cmap = cmap.reversed()

titulo_mapa = f"Precipitación {'Promedio' if año_seleccionado == 'Promedio' else 'en ' + str(año_seleccionado)}"

if shapefile_path.exists():
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.merge(datos, left_on="nombre_estacion", right_on="Estacion", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf.plot(column="Precipitacion", cmap=cmap, legend=True, ax=ax)
    plt.title(titulo_mapa, fontsize=16)
    st.pyplot(fig)

else:
    st.warning("No se encontró shapefile de cuencas en `data/`. Mostrando mapa de puntos.")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = plt.scatter(
        datos["lon"], datos["lat"],
        c=datos["Precipitacion"],
        cmap=cmap,
        s=80,
        edgecolor="black"
    )
    plt.colorbar(sc, label="Precipitación")
    plt.title(titulo_mapa, fontsize=16)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    st.pyplot(fig)

# -------------------------------
# Tabla de datos
# -------------------------------
st.subheader("📊 Datos de Precipitación")
st.dataframe(datos)
