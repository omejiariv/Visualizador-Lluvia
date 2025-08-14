import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import time
from pathlib import Path

st.set_page_config(page_title="Visualizador de Lluvia", layout="wide")

@st.cache_data
def cargar_datos():
    meta = pd.read_csv(Path("data/EstHM_CV.csv"))
    pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))
    pptn = pptn_raw.melt(id_vars=["Estacion"], var_name="Año", value_name="Precipitacion")
    df = pd.merge(meta, pptn, on="Estacion", how="inner")
    return df, pptn_raw, meta

df, pptn_raw, meta = cargar_datos()

# Selección de animación
estaciones = df["Estacion"].unique()
est_sel = st.selectbox("Selecciona la estación", estaciones)
df_est = df[df["Estacion"] == est_sel]

# Opción de colores
invertir = st.checkbox("Invertir colores (Azul = más lluvia)")
if invertir:
    cmap = "Blues_r"
else:
    cmap = "Blues"

# Animación en bucle
iniciar = st.button("Iniciar animación")
detener = st.button("Detener animación")

fig, ax = plt.subplots(figsize=(6, 6))
gdf = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta.Longitud, meta.Latitud), crs="EPSG:4326")

if iniciar:
    st.session_state["animando"] = True
if detener:
    st.session_state["animando"] = False

if "animando" not in st.session_state:
    st.session_state["animando"] = False

placeholder = st.empty()

while st.session_state["animando"]:
    for año in sorted(df_est["Año"].unique()):
        df_año = df[df["Año"] == año]
        gdf_merge = gdf.merge(df_año, on="Estacion", how="left")

        fig, ax = plt.subplots(figsize=(6, 6))
        gdf_merge.plot(column="Precipitacion", cmap=cmap, legend=True, ax=ax, markersize=50)
        ax.set_title(f"Precipitación {año}", fontsize=16)
        ax.axis("off")
        
        placeholder.pyplot(fig)
        time.sleep(3)

        if not st.session_state["animando"]:
            break
