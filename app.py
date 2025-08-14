import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================
st.set_page_config(page_title="Visualizador de Lluvias", layout="wide")

# ==============================
# FUNCIÓN PARA CARGAR DATOS
# ==============================
@st.cache_data
def cargar_datos():
    meta = pd.read_csv(Path("data/EstHM_CV.csv"))
    pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn_.csv"))

    # Convertir primera columna a string para asegurar el merge
    meta["Estacion"] = meta["Estacion"].astype(str)
    pptn_raw["Estacion"] = pptn_raw["Estacion"].astype(str)

    # Unir metadatos con precipitaciones
    df = pd.merge(meta, pptn_raw, on="Estacion", how="inner")

    return df, pptn_raw, meta

# ==============================
# FUNCIÓN PARA CARGAR SHAPEFILE
# ==============================
@st.cache_data
def cargar_shapefile():
    gdf = gpd.read_file(Path("data/Estaciones.shp"))
    gdf["Estacion"] = gdf["Estacion"].astype(str)
    return gdf

# ==============================
# FUNCIÓN PARA GRAFICAR MAPA
# ==============================
def graficar_mapa(gdf, datos, anio):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Sin fondo de calles, solo puntos
    gdf.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.3)

    # Graficar precipitaciones con colores invertidos
    datos.plot(
        column=str(anio),
        cmap="Blues_r",  # azul = más lluvia
        legend=True,
        ax=ax,
        markersize=50
    )

    ax.set_title(f"Precipitación en {anio}", fontsize=14)
    ax.axis("off")
    st.pyplot(fig)

# ==============================
# CARGA DE DATOS
# ==============================
df, pptn_raw, meta = cargar_datos()
gdf = cargar_shapefile()

# ==============================
# INTERFAZ
# ==============================
st.title("Visualizador de Precipitaciones")

opcion = st.radio("Modo de visualización", ["Manual", "Animación automática"])

anios = [col for col in pptn_raw.columns if col != "Estacion"]

if opcion == "Manual":
    anio_sel = st.selectbox("Seleccione un año", anios)
    gdf_merged = gdf.merge(pptn_raw[["Estacion", anio_sel]], on="Estacion", how="left")
    graficar_mapa(gdf, gdf_merged, anio_sel)

else:
    velocidad = 3  # segundos entre imágenes
    placeholder = st.empty()

    for anio in anios:
        gdf_merged = gdf.merge(pptn_raw[["Estacion", anio]], on="Estacion", how="left")
        with placeholder.container():
            graficar_mapa(gdf, gdf_merged, anio)
        time.sleep(velocidad)
