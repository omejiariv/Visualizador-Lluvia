import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from io import BytesIO
import shapefile

# Optional libs (no fatales)
try:
    import plotly.express as px
except Exception:
    px = None
    st.warning("plotly no instalado: algunos gráficos interactivos no estarán disponibles. Añade 'plotly' a requirements.txt para habilitarlos.")

try:
    import pydeck as pdk
except Exception:
    pdk = None
    st.warning("pydeck no instalado: el mapa interactivo no estará disponible. Añade 'pydeck' a requirements.txt para habilitarlo.")

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except Exception:
    sns = None
    import matplotlib.pyplot as plt

# ==========================
# Configuración inicial
# ==========================
st.set_page_config(page_title="Visualizador de Lluvia - robusto", layout="wide")
st.title("🌧️ Visualizador de Lluvia - Antioquia (robusto)")

DATA_DIR = "data"
IMAGES_DIR = "imagenes"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
BASE_PATHS = [Path("data"), Path(".")]

# ==========================
# Funciones auxiliares
# ==========================
def buscar_archivo_parcial(patron):
    """Busca un CSV cuyo nombre contenga 'patron' en data/ o en la raíz."""
    for base in BASE_PATHS:
        if not base.exists():
            continue
        for file in base.glob("*.csv"):
            if patron.lower() in file.name.lower():
                return file
    return None

def leer_csv_flexible(f):
    """Lee un pd.read_csv aceptando Path o UploadedFile, con fallback de engine."""
    try:
        return pd.read_csv(f)
    except Exception as e:
        return pd.read_csv(f, engine="python")

def estandarizar_nombre_columna(df):
    """Estandariza el nombre de la columna 'Estacion' en un DataFrame."""
    nombres_validos = ['Estacion', 'estacion', 'ID_Estacion', 'estacion_id', 'nombre_est']
    for col in df.columns:
        if col.strip().lower() in [n.lower() for n in nombres_validos]:
            df.rename(columns={col: 'Estacion'}, inplace=True)
            return df
    return df

# ==========================
# Cargar datos (robusto)
# ==========================
@st.cache_data
def cargar_datos(prec_file_obj=None, meta_file_obj=None):
    """
    Carga los archivos CSV de precipitación y metadatos.
    Prioriza el uploader, luego busca en el disco.
    """
    prec_path = None
    meta_path = None

    if prec_file_obj is None:
        prec_path = buscar_archivo_parcial("Transp_Est_Pptn") or buscar_archivo_parcial("lluvia")
    if meta_file_obj is None:
        meta_path = buscar_archivo_parcial("EstHM_CV") or buscar_archivo_parcial("estaciones")

    if prec_file_obj is not None:
        pptn_raw = leer_csv_flexible(prec_file_obj)
        prec_source = "uploader"
    elif prec_path is not None:
        pptn_raw = leer_csv_flexible(prec_path)
        prec_source = str(prec_path)
    else:
        pptn_raw = None
        prec_source = None

    if meta_file_obj is not None:
        meta_df = leer_csv_flexible(meta_file_obj)
        meta_source = "uploader"
    elif meta_path is not None:
        meta_df = leer_csv_flexible(meta_path)
        meta_source = str(meta_path)
    else:
        meta_df = None
        meta_source = None

    if pptn_raw is not None:
        pptn_raw = estandarizar_nombre_columna(pptn_raw)
    if meta_df is not None:
        meta_df = estandarizar_nombre_columna(meta_df)

    if pptn_raw is None or meta_df is None:
        return None, None, prec_source, meta_source

    if "Estacion" not in pptn_raw.columns or "Estacion" not in meta_df.columns:
        st.error(
            "Los CSV deben contener una columna de estación con un nombre como: 'Estacion', 'estacion', 'ID_Estacion', etc."
        )
        st.stop()

    return pptn_raw, meta_df, prec_source, meta_source

@st.cache_data
def cargar_shapefiles():
    """Carga los archivos de shapefile desde la carpeta de datos."""
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
            st.success("Shapefiles guardados en data/. Por favor, recarga la aplicación.")
            st.stop()
        else:
            st.stop()

    sf = shapefile.Reader(shp_path)
    return sf

# -----------------------
# Procesar precipitaciones
# -----------------------
def procesar_precipitaciones(pptn_raw):
    """
    Convierte el DataFrame de precipitaciones a formato largo.
    Maneja varios formatos de entrada.
    """
    if pptn_raw is None:
        return None

    first_col = pptn_raw.columns[0]
    first_val = str(pptn_raw.iloc[0, 0]).strip()

    def es_anio(v):
        try:
            n = int(v)
            return 1900 <= n <= 2100
        except Exception:
            return False

    if es_anio(first_val) or first_col.lower() in ['año', 'ano', 'year', 'fecha']:
        df_wide = pptn_raw.rename(columns={first_col: "Año"})
        df_long = df_wide.melt(id_vars=["Año"], var_name="Estacion", value_name="Precipitacion")
    else:
        try:
            df_trans = pptn_raw.set_index(first_col).transpose().reset_index()
            df_trans = df_trans.rename(columns={"index": "Año"})
            df_long = df_trans.melt(id_vars=["Año"], var_name="Estacion", value_name="Precipitacion")
        except Exception:
            df_long = pptn_raw.melt(id_vars=[first_col], var_name="Año", value_name="Precipitacion")
            df_long = df_long.rename(columns={first_col: "Estacion", "Año": "Año"})

    df_long['Año'] = pd.to_numeric(df_long['Año'], errors='coerce')
    df_long['Estacion'] = df_long['Estacion'].astype(str)
    df_long['Precipitacion'] = pd.to_numeric(df_long['Precipitacion'], errors='coerce')
    df_long = df_long.dropna(subset=['Año', 'Estacion']).reset_index(drop=True)
    df_long['Año'] = df_long['Año'].astype(int)

    return df_long

# -----------------------
# UI: Cargar archivos y mostrar fuentes
# -----------------------
st.sidebar.header("Archivos")
prec_file_u = st.sidebar.file_uploader("CSV precipitaciones (ej. Transp_Est_Pptn...)", type=["csv"])
meta_file_u = st.sidebar.file_uploader("CSV metadatos (ej. EstHM_CV...)", type=["csv"])

try:
    pptn_raw, meta_df, prec_source, meta_source = cargar_datos(prec_file_obj=prec_file_u, meta_file_obj=meta_file_u)
    sf = cargar_shapefiles()
except Exception as e:
    st.error(f"Error al cargar datos o shapefiles. Detalles: {e}")
    st.stop()

if pptn_raw is None or meta_df is None:
    st.error("No se encontraron los archivos necesarios. Súbelos o colócalos en la carpeta 'data/'.")
    st.stop()

st.sidebar.markdown("**Fuentes detectadas**")
st.sidebar.write(f"- Precipitaciones: `{prec_source}`")
st.sidebar.write(f"- Metadatos: `{meta_source}`" if meta_source else "- Metadatos: (no cargados)")

# -----------------------
# Procesar precip y unir con meta
# -----------------------
prec_long = procesar_precipitaciones(pptn_raw)
if prec_long is None or prec_long.empty:
    st.error("No se pudo procesar el CSV de precipitaciones. Revisa el formato.")
    st.stop()

merged = pd.merge(prec_long, meta_df, on='Estacion', how='left')

# -----------------------
# Controles: estaciones y años
# -----------------------
st.sidebar.header("Filtros")
estaciones_all = sorted(prec_long['Estacion'].unique().astype(str).tolist())

if 'est_selected' not in st.session_state:
    st.session_state['est_selected'] = estaciones_all.copy()

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Seleccionar todo"):
        st.session_state['est_selected'] = estaciones_all.copy()
with col_b:
    if st.button("Limpiar selección"):
        st.session_state['est_selected'] = []

estaciones_sel = st.sidebar.multiselect("Estaciones", estaciones_all, default=st.session_state['est_selected'])
st.session_state['est_selected'] = estaciones_sel

anio_min_data = int(prec_long['Año'].min())
anio_max_data = int(prec_long['Año'].max())
YEAR_MIN, YEAR_MAX = 1970, 2021
slider_min = max(YEAR_MIN, anio_min_data)
slider_max = min(YEAR_MAX, anio_max_data)
if slider_min > slider_max:
    slider_min, slider_max = anio_min_data, anio_max_data

rango = st.sidebar.slider("Rango de años (eje X)",
                          min_value=int(YEAR_MIN), max_value=int(YEAR_MAX),
                          value=(int(slider_min), int(slider_max)))

df_filtrado = merged[(merged['Año'] >= rango[0]) & (merged['Año'] <= rango[1])]
if estaciones_sel:
    df_filtrado = df_filtrado[df_filtrado['Estacion'].isin(estaciones_sel)]
else:
    st.warning("No hay estaciones seleccionadas. Usa 'Seleccionar todo' o marca algunas en la lista.")
    st.stop()

# -----------------------
# Pestañas: tabla, gráficos, stats, info, mapa, animación
# -----------------------
tabs = st.tabs(["Tabla", "Gráficos", "Estadísticas", "Info estaciones", "Mapa", "Animación de Imágenes"])

# TAB 1: Tabla
with tabs[0]:
    st.subheader("Tabla de observaciones (filtrada)")
    st.dataframe(df_filtrado[['Estacion','Año','Precipitacion'] + [c for c in merged.columns if c not in ['Estacion','Año','Precipitacion']]], use_container_width=True)

# TAB 2: Gráficos
with tabs[1]:
    st.subheader("Gráficos")
    tipo = st.radio("Tipo", ['Línea', 'Barras', 'Boxplot'], horizontal=True)
    df_plot = df_filtrado[['Estacion', 'Año', 'Precipitacion']].copy()

    if tipo == 'Línea' and px is not None:
        years_full = list(range(YEAR_MIN, YEAR_MAX + 1))
        df_plot = df_plot[(df_plot['Año'] >= YEAR_MIN) & (df_plot['Año'] <= YEAR_MAX)]
        fig = px.line(df_plot, x='Año', y='Precipitacion', color='Estacion', markers=True)
        tickvals = list(range(YEAR_MIN, YEAR_MAX + 1, 5))
        fig.update_xaxes(tickmode='array', tickvals=tickvals)
        st.plotly_chart(fig, use_container_width=True)
    elif tipo == 'Barras' and px is not None:
        df_plot = df_plot[(df_plot['Año'] >= rango[0]) & (df_plot['Año'] <= rango[1])]
        fig = px.bar(df_plot, x='Año', y='Precipitacion', color='Estacion', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        if sns is not None:
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=df_plot, x='Estacion', y='Precipitacion')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.write("Instala seaborn/plotly para ver gráficos más atractivos.")

# TAB 3: Estadísticas
with tabs[2]:
    st.subheader("Estadísticas por estación")
    rows = []
    for est in estaciones_sel:
        serie = df_filtrado[df_filtrado['Estacion'] == est]['Precipitacion'].dropna()
        if serie.empty:
            rows.append({'Estacion': est, 'Promedio': np.nan, 'Mínimo': np.nan, 'Año Mín': None, 'Máximo': np.nan, 'Año Máx': None})
            continue
        minv = serie.min()
        maxv = serie.max()
        meanv = serie.mean()
        anio_min = int(df_filtrado[(df_filtrado['Estacion'] == est) & (df_filtrado['Precipitacion'] == minv)]['Año'].values[0])
        anio_max = int(df_filtrado[(df_filtrado['Estacion'] == est) & (df_filtrado['Precipitacion'] == maxv)]['Año'].values[0])
        rows.append({'Estacion': est, 'Promedio': round(meanv, 2), 'Mínimo': minv, 'Año Mín': anio_min, 'Máximo': maxv, 'Año Máx': anio_max})
    st.dataframe(pd.DataFrame(rows).set_index('Estacion'), use_container_width=True)

# TAB 4: Info estaciones (metadatos)
with tabs[3]:
    st.subheader("Metadatos de estaciones")
    if meta_df is None:
        st.info("No se han cargado metadatos. Sube EstHM_CV.csv para ver la información completa.")
    else:
        st.dataframe(meta_df, use_container_width=True)

# TAB 5: Mapa
with tabs[4]:
    st.subheader("Mapa de estaciones - puntos tamaño/color por precipitación")
    if meta_df is None or pdk is None:
        st.info("Para mostrar el mapa, sube el archivo de metadatos con coordenadas (ej. 'x' y 'y') y asegúrate de tener instalado pydeck.")
    else:
        colnames = [c.lower() for c in meta_df.columns]
        lon_candidates = [c for c in meta_df.columns if c.lower() in ['x', 'lon', 'long', 'longitud', 'longitude']]
        lat_candidates = [c for c in meta_df.columns if c.lower() in ['y', 'lat', 'latitude', 'latitud']]

        if not lon_candidates or not lat_candidates:
            st.warning("No se encontraron columnas de coordenadas en metadatos.")
        else:
            lon_col = lon_candidates[0]
            lat_col = lat_candidates[0]

            ppt_prom = df_filtrado.groupby('Estacion')['Precipitacion'].mean().reset_index().rename(columns={'Precipitacion': 'ppt_media'})
            meta_map = meta_df.copy()
            meta_map[lon_col] = pd.to_numeric(meta_map[lon_col], errors='coerce')
            meta_map[lat_col] = pd.to_numeric(meta_map[lat_col], errors='coerce')
            meta_map = meta_map.dropna(subset=[lon_col, lat_col])
            meta_map = meta_map.merge(ppt_prom, on='Estacion', how='left')

            min_p = float(meta_map['ppt_media'].min(skipna=True) if not meta_map['ppt_media'].isna().all() else 0.0)
            max_p = float(meta_map['ppt_media'].max(skipna=True) if not meta_map['ppt_media'].isna().all() else 1.0)
            MIN_R, MAX_R = 2000, 8000

            def color_from_p(p):
                if pd.isna(p):
                    return [150, 150, 150, 160]
                ratio = (p - min_p) / (max_p - min_p + 1e-9)
                r = int(255 * (1 - ratio))
                g = int(122 * ratio)
                b = int(255 * ratio)
                return [r, g, b, 180]

            meta_map['color'] = meta_map['ppt_media'].map(lambda v: color_from_p(v))
            meta_map['radius'] = meta_map['ppt_media'].map(lambda v: (((v - min_p) / (max_p - min_p + 1e-9)) * (MAX_R - MIN_R) + MIN_R) if not pd.isna(v) else MIN_R)

            midpoint = (meta_map[lat_col].mean(), meta_map[lon_col].mean())
            layer = pdk.Layer("ScatterplotLayer", data=meta_map, get_position=[lon_col, lat_col], get_color="color", get_radius="radius", pickable=True)
            view = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=7, pitch=30)
            tooltip = {"html": "<b>Estación:</b> {Estacion}<br/><b>Precipación (media):</b> {ppt_media:.2f} mm", "style": {"color": "white"}}
            deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip)
            st.pydeck_chart(deck, use_container_width=True)

            # Animación por año (manual + automático)
            st.markdown("---")
            st.markdown("**Animación temporal (año por año)**")
            years = sorted(df_filtrado['Año'].unique())
            if not years:
                st.info("No hay años en el conjunto de datos filtrado.")
            else:
                colA, colB, colC = st.columns([2, 1, 1])
                with colA:
                    year_slider = st.slider("Año (manual)", min_value=int(min(years)), max_value=int(max(years)), value=int(min(years)))
                with colB:
                    auto = st.checkbox("Modo automático (time-lapse)", value=False)
                with colC:
                    speed = st.selectbox("Velocidad", options=['Lento (2s)', 'Medio (1s)', 'Rápido (0.5s)'], index=0)
                
                delay = 2.0 if speed.startswith('Lento') else (1.0 if speed.startswith('Medio') else 0.5)
                mapa_placeholder = st.empty()

                def render_year(y):
                    df_y = df_filtrado[df_filtrado['Año'] == int(y)]
                    ppt_y = df_y.groupby('Estacion')['Precipitacion'].mean().reset_index().rename(columns={'Precipitacion': 'ppt_media'})
                    mm = meta_map.copy()
                    mm = mm.merge(ppt_y, on='Estacion', how='left')
                    mm['color'] = mm['ppt_media'].map(lambda v: color_from_p(v))
                    mm['radius'] = mm['ppt_media'].map(lambda v: (((v - min_p) / (max_p - min_p + 1e-9)) * (MAX_R - MIN_R) + MIN_R) if not pd.isna(v) else MIN_R)
                    
                    if mm.empty:
                        mapa_placeholder.write(f"No hay datos para el año {y}")
                        return

                    layer_y = pdk.Layer("ScatterplotLayer", data=mm, get_position=[lon_col, lat_col], get_color="color", get_radius="radius", pickable=True)
                    deck_y = pdk.Deck(layers=[layer_y], initial_view_state=view, tooltip=tooltip)
                    mapa_placeholder.pydeck_chart(deck_y, use_container_width=True)

                if auto:
                    for y in years:
                        render_year(y)
                        time.sleep(delay)
                else:
                    render_year(year_slider)

# TAB 6: Animación de imágenes
with tabs[5]:
    st.subheader("Animación de Imágenes")
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()

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

        if st.session_state.slideshow_running:
            for img in image_files:
                if not st.session_state.slideshow_running:
                    break
                img_container.image(os.path.join(IMAGES_DIR, img), use_container_width=True)
                time.sleep(3)
            
            if st.session_state.slideshow_running:
                st.session_state.slideshow_running = False
    else:
        st.warning(f"No se encontraron imágenes en la carpeta '{IMAGES_DIR}'.")

# -----------------------
# Descarga de datos filtrados
# -----------------------
def to_excel_bytes(df_):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        df_.to_excel(writer, index=False, sheet_name='Datos')
    return out.getvalue()

st.sidebar.markdown("---")
st.sidebar.header("Exportar")
st.sidebar.download_button(
    "Descargar CSV (filtrado)",
    data=df_filtrado.to_csv(index=False).encode('utf-8'),
    file_name="precipitacion_filtrada.csv",
    mime="text/csv"
)
st.sidebar.download_button(
    "Descargar Excel (filtrado)",
    data=to_excel_bytes(df_filtrado),
    file_name="precipitacion_filtrada.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.info("Si sigues viendo errores en lectura de archivos: comprueba que los CSV y shapefiles existen en la carpeta `data/` del repo (o súbelos con los uploaders).")

