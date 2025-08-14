import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import os

# ========== CONFIG STREAMLIT ==========
st.set_page_config(page_title="Visualizador de Lluvia", layout="wide")

st.title("🌧️ Visualizador de Precipitación - Animación y Control Manual")

# ========== CONTROLES ==========
modo = st.radio("Selecciona el modo", ["Animación automática", "Control manual"])
invertir_colores = st.checkbox("Invertir colores (azul = más lluvia)", value=True)
tiempo_entre_frames = st.slider("Segundos entre imágenes (solo para animación)", 1, 10, 3)

# Estado de animación
if "animando" not in st.session_state:
    st.session_state.animando = False

if modo == "Animación automática":
    col1, col2 = st.columns(2)
    if col1.button("▶️ Iniciar animación"):
        st.session_state.animando = True
    if col2.button("⏸️ Detener animación"):
        st.session_state.animando = False

# ========== CARGA DE DATOS ==========
@st.cache_data
def cargar_datos():
    meta = pd.read_csv("data/EstHM_CV.csv")
    pptn = pd.read_csv("data/Transp_Est_Pptn.csv")
    meta["Estacion"] = meta["Estacion"].astype(str)
    pptn["Estacion"] = pptn["Estacion"].astype(str)
    df = pd.merge(pptn, meta, on="Estacion", how="inner")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitud, df.Latitud), crs="EPSG:4326")
    anios = [col for col in pptn.columns if col != "Estacion"]
    return gdf, anios

gdf, anios = cargar_datos()

# ========== FUNCIÓN PARA PLOTEAR UN AÑO ==========
def plot_precipitacion(anio):
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(
        ax=ax,
        column=anio,
        cmap="Blues_r" if invertir_colores else "Blues",
        legend=True,
        markersize=80
    )
    ax.set_title(f"Precipitación en {anio}", fontsize=16)
    ax.axis("off")
    st.pyplot(fig)

# ========== MODO MANUAL ==========
if modo == "Control manual":
    if "indice_anio" not in st.session_state:
        st.session_state.indice_anio = 0

    col_prev, col_next = st.columns([1, 1])
    if col_prev.button("⬅️ Anterior"):
        st.session_state.indice_anio = (st.session_state.indice_anio - 1) % len(anios)
    if col_next.button("➡️ Siguiente"):
        st.session_state.indice_anio = (st.session_state.indice_anio + 1) % len(anios)

    plot_precipitacion(anios[st.session_state.indice_anio])

# ========== FUNCIÓN PARA GENERAR GIF ==========
def generar_gif():
    frames = []
    for anio in anios:
        fig, ax = plt.subplots(figsize=(8, 6))
        gdf.plot(
            ax=ax,
            column=anio,
            cmap="Blues_r" if invertir_colores else "Blues",
            legend=True,
            markersize=80
        )
        ax.set_title(f"Precipitación en {anio}", fontsize=16)
        ax.axis("off")
        ruta_temp = f"frame_{anio}.png"
        plt.savefig(ruta_temp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frames.append(imageio.imread(ruta_temp))
        os.remove(ruta_temp)
    imageio.mimsave("lluvias.gif", frames, duration=tiempo_entre_frames, loop=0)

# ========== MODO AUTOMÁTICO ==========
if modo == "Animación automática":
    if st.button("💾 Generar GIF"):
        generar_gif()
        st.success("GIF generado!")
        st.image("lluvias.gif")

    if st.session_state.animando:
        generar_gif()
        st.image("lluvias.gif")
