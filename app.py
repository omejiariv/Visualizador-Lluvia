import pandas as pd
import shapefile  # PyShp
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import time

# ========================================
# Cargar datos CSV
# ========================================
@st.cache_data
def cargar_datos():
    meta = pd.read_csv(Path("data/EstHM_CV.csv"))
    pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))
    df = pd.merge(pptn_raw, meta, on="Estacion", how="inner")
    return df, pptn_raw, meta

# ========================================
# Leer shapefile con PyShp y detectar campo ID estación
# ========================================
def leer_shapefile(ruta_shp):
    sf = shapefile.Reader(ruta_shp)
    shapes = sf.shapes()
    records = sf.records()
    fields = [f[0] for f in sf.fields[1:]]

    # Intentar detectar el campo que contiene el ID de la estación
    posibles_ids = ["estacion", "id", "codigo", "cod", "id_estacion"]
    campo_id = None
    for candidato in posibles_ids:
        for f in fields:
            if f.strip().lower() == candidato:
                campo_id = f
                break
        if campo_id:
            break

    if campo_id:
        id_idx = fields.index(campo_id)
    else:
        st.warning("No se detectó automáticamente el campo de ID de estación. Usando la primera columna.")
        id_idx = 0

    return shapes, records, fields, id_idx

# ========================================
# Validar que el shapefile tenga estaciones en los datos
# ========================================
def validar_shapefile(records, id_idx, estaciones_csv):
    ids_shp = {rec[id_idx] for rec in records}
    ids_csv = set(estaciones_csv)
    comunes = ids_shp & ids_csv
    if not comunes:
        return False, ids_shp, ids_csv
    return True, ids_shp, ids_csv

# ========================================
# Graficar mapa
# ========================================
def graficar_mapa(shapes, records, df, year, id_idx, invert_colors=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.Blues_r if invert_colors else plt.cm.Blues

    pptn_dict = df.set_index("Estacion")[str(year)].to_dict()
    min_val, max_val = min(pptn_dict.values()), max(pptn_dict.values())

    for shape_rec, rec in zip(shapes, records):
        est_id = rec[id_idx]
        if est_id in pptn_dict:
            color_val = (pptn_dict[est_id] - min_val) / (max_val - min_val + 1e-6)
            color = cmap(color_val)
        else:
            color = (0.8, 0.8, 0.8)

        x, y = zip(*shape_rec.points)
        ax.fill(x, y, color=color, edgecolor="black", linewidth=0.5)

    ax.set_title(f"Mapa de precipitación - {year}", fontsize=14)
    ax.axis("equal")
    ax.axis("off")
    return fig

# ========================================
# Interfaz
# ========================================
st.title("Visualizador de Lluvia por Año")

# Cargar datos de lluvia
df, pptn_raw, meta = cargar_datos()

# Detectar años automáticamente
years = sorted([c for c in pptn_raw.columns if c != "Estacion"])

# Rutas por defecto
ruta_shp = Path("data/estaciones.shp")
ruta_shx = Path("data/estaciones.shx")
ruta_dbf = Path("data/estaciones.dbf")

# Verificar si los archivos existen
if not (ruta_shp.exists() and ruta_shx.exists() and ruta_dbf.exists()):
    st.warning("No se encontró el shapefile completo en la carpeta `data/`. "
               "Debe contener `.shp`, `.shx` y `.dbf` con el mismo nombre.")

    st.info("Puede subir manualmente los tres archivos del shapefile:")
    uploaded_shp = st.file_uploader("Suba el archivo .shp", type="shp")
    uploaded_shx = st.file_uploader("Suba el archivo .shx", type="shx")
    uploaded_dbf = st.file_uploader("Suba el archivo .dbf", type="dbf")

    if uploaded_shp and uploaded_shx and uploaded_dbf:
        # Guardar en carpeta data/ para uso futuro
        Path("data").mkdir(exist_ok=True)
        with open(ruta_shp, "wb") as f:
            f.write(uploaded_shp.read())
        with open(ruta_shx, "wb") as f:
            f.write(uploaded_shx.read())
        with open(ruta_dbf, "wb") as f:
            f.write(uploaded_dbf.read())

        shapes, records, fields, id_idx = leer_shapefile(ruta_shp)
        valido, ids_shp, ids_csv = validar_shapefile(records, id_idx, df["Estacion"])
        if not valido:
            st.error("El shapefile subido no contiene ninguna estación que coincida con los datos de lluvia.")
            st.stop()
    else:
        st.stop()
else:
    shapes, records, fields, id_idx = leer_shapefile(ruta_shp)
    valido, ids_shp, ids_csv = validar_shapefile(records, id_idx, df["Estacion"])
    if not valido:
        st.error("El shapefile en `data/` no contiene estaciones presentes en los datos de lluvia.")
        st.stop()

# Controles de la interfaz
invert_colors = st.checkbox("Invertir colores (Azul = más lluvia)", value=False)
modo_animacion = st.checkbox("Animar todos los años", value=False)

if not modo_animacion:
    year = st.selectbox("Selecciona un año", years)
    fig = graficar_mapa(shapes, records, df, year, id_idx, invert_colors=invert_colors)
    st.pyplot(fig)
else:
    stop = st.button("Detener animación")
    for y in years:
        if stop:
            break
        fig = graficar_mapa(shapes, records, df, y, id_idx, invert_colors=invert_colors)
        st.pyplot(fig)
        time.sleep(3)
