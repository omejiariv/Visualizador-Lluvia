import os
import pandas as pd
import shapefile
import streamlit as st
import matplotlib.pyplot as plt
import time

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================
# Función para cargar CSV
# ==========================
@st.cache_data
def cargar_datos():
    pptn_path = os.path.join(DATA_DIR, "lluvia.csv")
    estaciones_path = os.path.join(DATA_DIR, "estaciones.csv")

    if not os.path.exists(pptn_path) or not os.path.exists(estaciones_path):
        st.warning("No se encontraron los archivos CSV. Por favor súbelos.")
        lluvia_file = st.file_uploader("Sube lluvia.csv", type=["csv"], key="lluvia_csv")
        estaciones_file = st.file_uploader("Sube estaciones.csv", type=["csv"], key="estaciones_csv")

        if lluvia_file and estaciones_file:
            with open(pptn_path, "wb") as f:
                f.write(lluvia_file.getbuffer())
            with open(estaciones_path, "wb") as f:
                f.write(estaciones_file.getbuffer())
            st.success("Archivos CSV guardados en data/. Recarga la aplicación.")
            st.stop()
        else:
            st.stop()

    try:
        pptn_raw = pd.read_csv(pptn_path, encoding="utf-8")
        meta = pd.read_csv(estaciones_path, encoding="utf-8")
    except UnicodeDecodeError:
        pptn_raw = pd.read_csv(pptn_path, encoding="latin-1")
        meta = pd.read_csv(estaciones_path, encoding="latin-1")

    if "Estacion" not in pptn_raw.columns or "Estacion" not in meta.columns:
        st.error("Los CSV deben contener la columna 'Estacion'.")
        st.stop()

    df = pd.merge(pptn_raw, meta, on="Estacion", how="inner")
    return df, pptn_raw, meta

# ==========================
# Función para cargar shapefiles
# ==========================
def cargar_shapefiles():
    shp_path = os.path.join(DATA_DIR, "mapa.shp")
    shx_path = os.path.join(DATA_DIR, "mapa.shx")
    dbf_path = os.path.join(DATA_DIR, "mapa.dbf")

    if not (os.path.exists(shp_path) and os.path.exists(shx_path) and os.path.exists(dbf_path)):
        st.warning("No se encontraron shapefiles. Por favor súbelos (.shp, .shx, .dbf).")
        uploaded_files = st.file_uploader(
            "Sube shapefiles", type=["shp", "shx", "dbf"], accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                with open(os.path.join(DATA_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success("Shapefiles guardados en data/. Recarga la aplicación.")
            st.stop()
        else:
            st.stop()

    sf = shapefile.Reader(shp_path)
    return sf

# ==========================
# App principal
# ==========================
st.title("Visualizador de Lluvia con Mapas y Animación de Imágenes")

# Cargar datos
df, pptn_raw, meta = cargar_datos()
sf = cargar_shapefiles()

st.write("Datos cargados correctamente:")
st.dataframe(df.head())

# ==========================
# Bucle de imágenes optimizado
# ==========================
image_files = [f for f in os.listdir("imagenes") if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if image_files:
    if "slideshow_running" not in st.session_state:
        st.session_state.slideshow_running = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Iniciar animación"):
            st.session_state.slideshow_running = True
    with col2:
        if st.button("⏹ Detener animación"):
            st.session_state.slideshow_running = False

    img_container = st.empty()

    while st.session_state.slideshow_running:
        for img in image_files:
            if not st.session_state.slideshow_running:
                break
            img_container.image(os.path.join("imagenes", img), use_container_width=True)
            time.sleep(3)
else:
    st.warning("No se encontraron imágenes en la carpeta 'imagenes'.")
