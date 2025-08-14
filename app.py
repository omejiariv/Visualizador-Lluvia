import pandas as pd
import shapefile  # PyShp
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import time

# ========================================
# Función para cargar datos
# ========================================
@st.cache_data
def cargar_datos():
    # Carga CSV de metadatos
    meta = pd.read_csv(Path("data/EstHM_CV.csv"))

    # Carga CSV transpuesto de precipitaciones
    pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))

    # Unión de datos
    df = pd.merge(pptn_raw, meta, on="Estacion", how="inner")

    return df, pptn_raw, meta

# ========================================
# Función para leer shapefile con PyShp
# ========================================
def leer_shapefile(ruta_shp):
    sf = shapefile.Reader(ruta_shp)
    shapes = sf.shapes()
    records = sf.records()
    fields = [f[0] for f in sf.fields[1:]]  # Ignorar el primer campo 'DeletionFlag'

    return shapes, records, fields

# ========================================
# Función para graficar mapa
# ========================================
def graficar_mapa(shapes, records, fields, df, year, invert_colors=False):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Escoger colores
    cmap = plt.cm.Blues_r if invert_colors else plt.cm.Blues

    # Obtener valores de precipitación
    pptn_dict = df.set_index("Estacion")[str(year)].to_dict()
    min_val = min(pptn_dict.values())
    max_val = max(pptn_dict.values())

    for shape_rec, rec in zip(shapes, records):
        est_id = rec[0]  # Asumiendo que el primer campo es la estación
        if est_id in pptn_dict:
            color_val = (pptn_dict[est_id] - min_val) / (max_val - min_val + 1e-6)
            color = cmap(color_val)
        else:
            color = (0.8, 0.8, 0.8)

        # Dibujar polígono
        x, y = zip(*shape_rec.points)
        ax.fill(x, y, color=color, edgecolor="black", linewidth=0.5)

    ax.set_title(f"Mapa de precipitación - {year}", fontsize=14)
    ax.axis("equal")
    ax.axis("off")

    return fig

# ========================================
# Interfaz Streamlit
# ========================================
st.title("Visualizador de Lluvia por Año")

df, pptn_raw, meta = cargar_datos()

# Leer shapefile
ruta_shp = Path("data/estaciones.shp")  # Cambiar al nombre correcto
shapes, records, fields = leer_shapefile(ruta_shp)

# Opciones de interfaz
invert_colors = st.checkbox("Invertir colores (Azul = más lluvia)", value=False)
modo_animacion = st.checkbox("Animar todos los años", value=False)

if not modo_animacion:
    year = st.selectbox("Selecciona un año", sorted([c for c in pptn_raw.columns if c != "Estacion"]))
    fig = graficar_mapa(shapes, records, fields, df, year, invert_colors=invert_colors)
    st.pyplot(fig)
else:
    # Animación
    years = sorted([c for c in pptn_raw.columns if c != "Estacion"])
    stop = st.button("Detener animación")
    for y in years:
        if stop:
            break
        fig = graficar_mapa(shapes, records, fields, df, y, invert_colors=invert_colors)
        st.pyplot(fig)
        time.sleep(3)
