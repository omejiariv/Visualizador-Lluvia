import streamlit as st
import pandas as pd
import shapefile  # PyShp para leer shapefiles
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path
import time

# ======================
# Cargar datos
# ======================
@st.cache_data
def cargar_datos():
    df = pd.read_csv(Path("data/EstHM_CV.csv"))
    pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))
    return df, pptn_raw

# ======================
# Cargar shapefile sin geopandas
# ======================
@st.cache_data
def cargar_shapefile(ruta):
    sf = shapefile.Reader(ruta)
    shapes = sf.shapes()
    return shapes

# ======================
# Dibujar mapa
# ======================
def dibujar_mapa(shapes, df, fecha, invertir=False):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Dibujar polígonos del shapefile
    for shape in shapes:
        puntos = np.array(shape.points)
        partes = list(shape.parts) + [len(puntos)]
        for i in range(len(partes) - 1):
            seg = puntos[partes[i]:partes[i+1]]
            ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=0.5)

    # Colorear estaciones
    data_fecha = df[df["fecha"] == fecha]
    norm = Normalize(vmin=df["pptn"].min(), vmax=df["pptn"].max())

    cmap = plt.cm.Blues_r if invertir else plt.cm.Blues
    sc = ax.scatter(data_fecha["lon"], data_fecha["lat"],
                    c=data_fecha["pptn"], cmap=cmap, norm=norm,
                    s=50, edgecolor="black")

    plt.colorbar(sc, ax=ax, label="Precipitación (mm)")
    ax.set_title(f"Lluvia - {fecha}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect('equal')
    return fig

# ======================
# MAIN APP
# ======================
st.title("Visualizador de Lluvia - Sin GeoPandas")

df, pptn_raw = cargar_datos()
shapes = cargar_shapefile("data/limite_municipal.shp")

fechas = sorted(df["fecha"].unique())
modo = st.radio("Modo de visualización", ["Manual", "Automático"])
invertir_colores = st.checkbox("Invertir colores (Azul = más lluvia)")

if modo == "Manual":
    fecha_sel = st.selectbox("Seleccione fecha", fechas)
    fig = dibujar_mapa(shapes, df, fecha_sel, invertir=invertir_colores)
    st.pyplot(fig)

else:
    boton_inicio = st.button("Iniciar animación")
    if boton_inicio:
        for fecha in fechas:
            fig = dibujar_mapa(shapes, df, fecha, invertir=invertir_colores)
            st.pyplot(fig)
            time.sleep(3)
