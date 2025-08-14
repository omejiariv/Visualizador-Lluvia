import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapefile  # PyShp
from pathlib import Path
from scipy.interpolate import griddata
import time

# ============================
# FUNCIONES AUXILIARES
# ============================
def shapefiles_existen(basepath):
    return all(Path(f"{basepath}{ext}").exists() for ext in [".shp", ".shx", ".dbf"])

def guardar_shapefiles(archivos):
    Path("data").mkdir(exist_ok=True)
    nombres_permitidos = {".shp": "mapa.shp", ".shx": "mapa.shx", ".dbf": "mapa.dbf"}
    for archivo in archivos:
        ext = Path(archivo.name).suffix.lower()
        if ext in nombres_permitidos:
            with open(Path("data") / nombres_permitidos[ext], "wb") as f:
                f.write(archivo.read())

def cargar_shapefile(path):
    try:
        return shapefile.Reader(path)
    except Exception as e:
        st.error(f"Error al leer shapefile: {e}")
        return None

@st.cache_data
def cargar_datos():
    try:
        pptn_raw = pd.read_csv("data/lluvia.csv")
        meta = pd.read_csv("data/estaciones.csv")
    except FileNotFoundError:
        st.error("Faltan archivos de datos: 'lluvia.csv' y/o 'estaciones.csv'.")
        return None, None, None

    # Validación de columnas
    if "Estacion" not in pptn_raw.columns or "Estacion" not in meta.columns:
        st.error("Archivos CSV no contienen la columna 'Estacion'.")
        return None, None, None

    df = pd.merge(pptn_raw, meta, on="Estacion", how="inner")
    return df, pptn_raw, meta

def generar_mapa(df, fecha, shapefile_obj, invertir_colores=False):
    datos_fecha = df[df["Fecha"] == fecha]
    if datos_fecha.empty:
        st.warning(f"No hay datos para {fecha}")
        return

    x = datos_fecha["Longitud"].values
    y = datos_fecha["Latitud"].values
    z = datos_fecha["Precipitacion"].values

    xi = np.linspace(min(x), max(x), 200)
    yi = np.linspace(min(y), max(y), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="cubic")

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = "Blues_r" if invertir_colores else "Blues"
    cs = ax.contourf(xi, yi, zi, cmap=cmap, levels=15)

    for shape_rec in shapefile_obj.shapeRecords():
        x_coords = [i[0] for i in shape_rec.shape.points]
        y_coords = [i[1] for i in shape_rec.shape.points]
        ax.plot(x_coords, y_coords, "k-", linewidth=0.5)

    plt.colorbar(cs, ax=ax, label="Precipitación (mm)")
    ax.set_title(f"Mapa de precipitación - {fecha}")
    st.pyplot(fig)

# ============================
# INTERFAZ PRINCIPAL
# ============================
st.title("🌧️ Visualizador de Lluvia")

# Paso 1: Verificar shapefiles
if not shapefiles_existen("data/mapa"):
    st.info("📂 No se han encontrado shapefiles completos. Sube los tres archivos: .shp, .shx, .dbf")
    archivos = st.file_uploader(
        "Sube shapefiles (.shp, .shx, .dbf)",
        type=["shp", "shx", "dbf"],
        accept_multiple_files=True
    )
    if archivos:
        guardar_shapefiles(archivos)
        if shapefiles_existen("data/mapa"):
            st.success("✅ Archivos shapefile guardados correctamente en /data.")
        else:
            st.error("⚠️ Faltan uno o más archivos. Asegúrate de subir los tres: .shp, .shx, .dbf")
        st.stop()
else:
    st.success("✅ Shapefiles encontrados en /data.")

# Paso 2: Cargar shapefile
sf = cargar_shapefile("data/mapa.shp")
if not sf:
    st.stop()

# Paso 3: Cargar datos de lluvia
df, pptn_raw, meta = cargar_datos()
if df is None:
    st.stop()

# Paso 4: Controles
invertir_colores = st.checkbox("Invertir colores (Azul = Mucha lluvia)", value=False)
modo_animacion = st.radio("Modo", ["Manual", "Animación en bucle"])

if modo_animacion == "Manual":
    fecha_sel = st.selectbox("Selecciona una fecha", sorted(df["Fecha"].unique()))
    generar_mapa(df, fecha_sel, sf, invertir_colores)
else:
    fechas = sorted(df["Fecha"].unique())
    if st.button("Iniciar animación"):
        placeholder = st.empty()
        while True:
            for fecha in fechas:
                with placeholder:
                    generar_mapa(df, fecha, sf, invertir_colores)
                time.sleep(3)
                if st.button("Detener"):
                    st.stop()
