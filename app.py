import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shapefile
from pathlib import Path
import os
import time

# ============================
# Cargar datos con validación
# ============================
@st.cache_data
def cargar_datos():
    try:
        meta = pd.read_csv(Path("data/EstHM_CV.csv"))
        pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))
    except FileNotFoundError:
        st.error("❌ No se encontraron 'EstHM_CV.csv' y/o 'Transp_Est_Pptn.csv' en la carpeta data/.")
        st.stop()

    # Normalizar nombres de columnas
    meta.columns = meta.columns.str.strip().str.lower()
    pptn_raw.columns = pptn_raw.columns.str.strip().str.lower()

    # Depuración
    st.write("📄 Columnas en meta:", list(meta.columns))
    st.write("📄 Columnas en pptn_raw:", list(pptn_raw.columns))

    # Validar existencia
    if "estacion" not in meta.columns:
        st.error(f"El archivo 'EstHM_CV.csv' no contiene la columna 'Estacion'. Columnas: {list(meta.columns)}")
        st.stop()
    if "estacion" not in pptn_raw.columns:
        st.error(f"El archivo 'Transp_Est_Pptn.csv' no contiene la columna 'Estacion'. Columnas: {list(pptn_raw.columns)}")
        st.stop()

    # Unir
    df = pd.merge(pptn_raw, meta, on="estacion", how="inner")

    return df, pptn_raw, meta

# ============================
# Cargar shapefile con control de errores
# ============================
def cargar_shapefile(path):
    base = Path(path)
    if not base.with_suffix(".shp").exists():
        st.error(f"❌ No se encontró el archivo {base}.shp")
        return None
    for ext in [".shx", ".dbf"]:
        if not base.with_suffix(ext).exists():
            st.error(f"❌ Falta {base}{ext}. El shapefile necesita .shp, .shx y .dbf en la misma carpeta.")
            return None
    try:
        return shapefile.Reader(str(base.with_suffix(".shp")))
    except Exception as e:
        st.error(f"Error cargando shapefile: {e}")
        return None

# ============================
# Guardar shapefiles subidos
# ============================
def guardar_archivos_subidos(archivos):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    for archivo in archivos:
        with open(data_dir / archivo.name, "wb") as f:
            f.write(archivo.read())
    st.success("📂 Archivos guardados en /data para usos futuros.")

# ============================
# Dibujar mapa
# ============================
def dibujar_mapa(sf, color="Blues"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for shape_rec in sf.shapeRecords():
        x = [p[0] for p in shape_rec.shape.points]
        y = [p[1] for p in shape_rec.shape.points]
        ax.plot(x, y, color="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig)

# ============================
# Streamlit App
# ============================
st.title("🌧️ Visualizador de Lluvia")

# Subida manual
archivos = st.file_uploader(
    "Sube shapefiles (.shp, .shx, .dbf)",
    type=["shp", "shx", "dbf"],
    accept_multiple_files=True
)
if archivos:
    guardar_archivos_subidos(archivos)

# Cargar shapefile
sf = cargar_shapefile("data/mapa")
if sf:
    dibujar_mapa(sf, color="Blues")

# Cargar CSVs
df, pptn_raw, meta = cargar_datos()
st.write(df.head())

# Animación de mapas
if sf and st.button("▶ Iniciar animación"):
    while True:
        for i in range(3):
            dibujar_mapa(sf, color="Blues_r")  # Azul = más lluvia
            time.sleep(3)
        if st.button("⏹ Detener animación"):
            break
