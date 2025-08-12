# Estructura de proyecto recomendada para GitHub:
#
# 📂 proyecto-lluvia/
# ├── app.py                  # Este archivo con el código principal
# ├── requirements.txt        # Lista de dependencias para instalar en Streamlit Cloud
# ├── README.md                # Descripción del proyecto
#
# --- Contenido de requirements.txt ---
# streamlit
# pandas
# numpy
# plotly==5.22.0
# pydeck
# folium (opcional)
#
# --- Contenido sugerido de README.md ---
# # 🌧 Visualizador de Precipitación Anual
# Aplicación web interactiva desarrollada en Streamlit para visualizar y analizar datos de lluvias anuales por estación.
#
# ## 📋 Características
# - Carga de archivo CSV directamente desde la interfaz (datos y metadatos).
# - Filtros por estación y rango de años (1970-2021 por defecto).
# - Botones para "Seleccionar todo" / "Limpiar" estaciones.
# - Gráficos interactivos (línea, barras y boxplot) con eje X con años 1970-2021.
# - Pestaña de información de estaciones con metadatos y coordenadas.
# - Mapa interactivo mostrando la ubicación de las estaciones (pydeck).
#
# ## 🚀 Cómo ejecutar localmente
# ```bash
# pip install -r requirements.txt
# streamlit run app.py
# ```

import streamlit as st
import pandas as pd
import numpy as np
import os

# import plotly y pydeck con manejo de error claro
try:
    import plotly.express as px
except Exception:
    st.error("La librería 'plotly' no está disponible. Asegúrate de incluir 'plotly' en requirements.txt.")
    st.stop()

try:
    import pydeck as pdk
except Exception:
    st.warning("pydeck no está disponible. El mapa interactivo no funcionará sin pydeck. Agrega 'pydeck' a requirements.txt para habilitar mapas.")

st.set_page_config(page_title="Visualizador de Precipitación", layout='wide')
st.title("🌧 Visualizador de Precipitación Anual")

# ---------- CARGA DE ARCHIVOS ----------
st.sidebar.header("Datos y metadatos")
archivo = st.sidebar.file_uploader("Sube archivo CSV de precipitaciones (formato: columnas: Año, <est1>, <est2>, ...)", type=["csv"], key="data_csv")
meta_file = st.sidebar.file_uploader("(Opcional) Sube EstHM_CV.csv con metadatos de estaciones", type=["csv"], key="meta_csv")

# También permitir usar archivos en el repositorio si existen (útil en despliegue desde GitHub)
if archivo is None and os.path.exists("Estaciones_Pptn_Todas.csv"):
    archivo = "Estaciones_Pptn_Todas.csv"
if meta_file is None and os.path.exists("EstHM_CV.csv"):
    meta_file = "EstHM_CV.csv"

# ---------- LECTURA DE DATOS PRINCIPALES ----------
if archivo is None:
    st.info("Por favor sube el archivo CSV con los datos de precipitación desde la barra lateral.")
    st.stop()

# cargar dataframe
try:
    if isinstance(archivo, str):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_csv(archivo)
except Exception as e:
    st.error(f"Error leyendo el CSV de datos: {e}")
    st.stop()

# Normalizar nombre primera columna a 'Año'
df.rename(columns={df.columns[0]: "Año"}, inplace=True)
# intentar convertir a entero (si falla, dejarlo como está y mostrar advertencia)
try:
    df['Año'] = df['Año'].astype(int)
except Exception:
    st.warning("La columna de años no pudo convertirse a entero. Asegúrate que contenga años como 1970, 1971, ...")

# columnas de estaciones (asumimos que las columnas desde la 2da en adelante son estaciones)
estaciones_all = list(df.columns[1:])

# ---------- SIDEBAR: Selección de estaciones y años ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros principales")

# Inicializar estado para selección
if 'est_selected' not in st.session_state:
    st.session_state['est_selected'] = estaciones_all.copy()

# botones de seleccionar todo / limpiar
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Seleccionar todo"):
        st.session_state['est_selected'] = estaciones_all.copy()
with col2:
    if st.button("Limpiar selección"):
        st.session_state['est_selected'] = []

# multiselect (muestra las que están en session_state)
estaciones_seleccionadas = st.sidebar.multiselect("Selecciona estaciones", estaciones_all, default=st.session_state['est_selected'], key='multi_est')
# sincronizar session_state con el multiselect result
st.session_state['est_selected'] = estaciones_seleccionadas

# Rango de años forzado a 1970-2021 como pidió el usuario
YEAR_MIN, YEAR_MAX = 1970, 2021
# ajustar si el dataset está fuera de ese rango
data_year_min = int(df['Año'].min())
data_year_max = int(df['Año'].max())

slider_min = max(YEAR_MIN, data_year_min)
slider_max = min(YEAR_MAX, data_year_max)
if slider_min > slider_max:
    # no hay intersección; usar data years
    slider_min, slider_max = data_year_min, data_year_max

rango_años = st.sidebar.slider("Selecciona rango de años", min_value=int(YEAR_MIN), max_value=int(YEAR_MAX), value=(int(slider_min), int(slider_max)))

# ---------- FILTRAR DATOS ----------
selected_cols = ['Año'] + estaciones_seleccionadas
if len(estaciones_seleccionadas) == 0:
    st.warning("No has seleccionado estaciones. Usa 'Seleccionar todo' o marca algunas en la lista.")
    st.stop()

# Filtrar por rango de años en los datos existentes
df_filtrado = df[(df['Año'] >= rango_años[0]) & (df['Año'] <= rango_años[1])][selected_cols]

# ---------- CARGA DE METADATOS (EstHM_CV.csv) ----------
meta_df = None
if meta_file is not None:
    try:
        if isinstance(meta_file, str):
            meta_df = pd.read_csv(meta_file)
        else:
            meta_df = pd.read_csv(meta_file)
        st.sidebar.success("Metadatos cargados")
    except Exception as e:
        st.sidebar.error(f"Error leyendo EstHM_CV.csv: {e}")

# ---------- PESTAÑAS PRINCIPALES ----------
tabs = st.tabs(["Tabla de Datos", "Gráficos", "Estadísticas", "Info Estaciones", "Mapa"]) 

# --- TAB 0: Tabla de Datos ---
with tabs[0]:
    st.subheader("Tabla de precipitaciones (filtrada)")
    st.dataframe(df_filtrado, use_container_width=True)

# --- TAB 1: Gráficos ---
with tabs[1]:
    st.subheader("Visualización de Datos")
    tipo_grafico = st.radio("Tipo de gráfico", ["Línea", "Barras", "Boxplot"], horizontal=True)

    # preparacion para plotly
    df_melt = df_filtrado.melt(id_vars=['Año'], var_name='Estacion', value_name='Precipitacion')

    if tipo_grafico == 'Línea':
        # forzar eje x desde 1970 hasta 2021 con ticks por año (siempre mostrar esos años)
        years = list(range(YEAR_MIN, YEAR_MAX + 1))
        # Restringir datos para plot a ese rango
        df_plot = df_melt[(df_melt['Año'] >= YEAR_MIN) & (df_melt['Año'] <= YEAR_MAX)]
        fig = px.line(df_plot, x='Año', y='Precipitacion', color='Estacion', markers=True)
        # ajustar ticks: mostrar cada 5 años para no saturar (pero eje contiene todos los años)
        tickvals = list(range(YEAR_MIN, YEAR_MAX + 1, 5))
        fig.update_xaxes(tickmode='array', tickvals=tickvals, tickformat='d')

    elif tipo_grafico == 'Barras':
        df_plot = df_melt[(df_melt['Año'] >= rango_años[0]) & (df_melt['Año'] <= rango_años[1])]
        fig = px.bar(df_plot, x='Año', y='Precipitacion', color='Estacion', barmode='group')
        tickvals = list(range(max(YEAR_MIN, rango_años[0]), min(YEAR_MAX, rango_años[1]) + 1, 5))
        fig.update_xaxes(tickmode='array', tickvals=tickvals, tickformat='d')

    else:  # Boxplot
        fig = px.box(df_melt, x='Estacion', y='Precipitacion')

    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Estadísticas ---
with tabs[2]:
    st.subheader("Estadísticas Básicas")
    estaciones_stats = []
    for estacion in estaciones_seleccionadas:
        # proteger contra columnas vacías
        if estacion not in df_filtrado.columns:
            continue
        ser = df_filtrado[estacion].dropna()
        if ser.empty:
            estaciones_stats.append({
                'Estacion': estacion,
                'Promedio': np.nan,
                'Mínimo': np.nan,
                'Año Mín': None,
                'Máximo': np.nan,
                'Año Máx': None
            })
            continue
        min_val = ser.min()
        max_val = ser.max()
        anio_min = int(df_filtrado[df_filtrado[estacion] == min_val]['Año'].values[0]) if (df_filtrado[df_filtrado[estacion] == min_val].shape[0] > 0) else None
        anio_max = int(df_filtrado[df_filtrado[estacion] == max_val]['Año'].values[0]) if (df_filtrado[df_filtrado[estacion] == max_val].shape[0] > 0) else None
        estaciones_stats.append({
            'Estacion': estacion,
            'Promedio': round(float(ser.mean()), 2),
            'Mínimo': min_val,
            'Año Mín': anio_min,
            'Máximo': max_val,
            'Año Máx': anio_max
        })

    stats_df = pd.DataFrame(estaciones_stats).set_index('Estacion')
    st.dataframe(stats_df, use_container_width=True)

# --- TAB 3: Info Estaciones ---
with tabs[3]:
    st.subheader("Información de Estaciones y Metadatos")
    if meta_df is None:
        st.info("No se han cargado metadatos. Sube EstHM_CV.csv en la barra lateral para ver información adicional de estaciones.")
    else:
        # mostrar columnas relevantes si existen
        requested_fields = ['Porc_datos','Celda_XY','Conteo_celda','Nombre_Est','Depto','Mpio','AH','Z_SZH','x','y']
        existing_fields = [c for c in requested_fields if c in meta_df.columns]
        if len(existing_fields) == 0:
            st.warning("El archivo de metadatos no contiene las columnas esperadas. Muestra las columnas disponibles a continuación.")
            st.dataframe(meta_df.head(), use_container_width=True)
        else:
            st.write("Campos disponibles en metadatos:", existing_fields)
            # Necesitamos un campo para hacer join entre columnas (nombres de estaciones o códigos)
            st.write("A continuación se muestra la tabla de metadatos filtrada por las estaciones seleccionadas (si es posible hacer la unión).")

            # Intentar detectar columna en meta_df que empata con los nombres de las columnas de datos
            possible_join_cols = []
            for col in meta_df.columns:
                # convertir ambos a strings y valorar si hay intersección
                meta_vals = meta_df[col].astype(str).unique()
                inter = set(meta_vals).intersection(set([str(s) for s in estaciones_all]))
                if len(inter) > 0:
                    possible_join_cols.append(col)

            join_col = None
            if len(possible_join_cols) == 1:
                join_col = possible_join_cols[0]
            elif len(possible_join_cols) > 1:
                join_col = st.selectbox("Selecciona la columna de metadatos que coincide con los IDs de estación", possible_join_cols)
            else:
                st.info("No se detectó automáticamente una columna para unir metadatos con las columnas de datos. Si conoces el campo que contiene el ID de estación, selecciona manualmente.")
                join_col = st.selectbox("Selecciona columna para unir (opcional)", ['---'] + list(meta_df.columns))
                if join_col == '---':
                    join_col = None

            if join_col is None:
                # solo mostrar la tabla con las columnas solicitadas
                st.dataframe(meta_df[existing_fields].set_index('Nombre_Est') if 'Nombre_Est' in meta_df.columns else meta_df[existing_fields], use_container_width=True)
            else:
                # realizar join
                meta_df_copy = meta_df.copy()
                meta_df_copy[join_col] = meta_df_copy[join_col].astype(str)
                df_meta_filtered = meta_df_copy[meta_df_copy[join_col].isin([str(s) for s in estaciones_seleccionadas])]
                if df_meta_filtered.empty:
                    st.warning("No se encontraron coincidencias entre las estaciones seleccionadas y el metadato elegido. Muestra tabla de metadatos completa en su lugar.")
                    st.dataframe(meta_df[existing_fields].head(200), use_container_width=True)
                else:
                    show_cols = [c for c in existing_fields if c in df_meta_filtered.columns]
                    if 'Nombre_Est' in df_meta_filtered.columns:
                        df_meta_filtered = df_meta_filtered.set_index('Nombre_Est')
                    st.dataframe(df_meta_filtered[show_cols], use_container_width=True)

# --- TAB 4: Mapa ---
with tabs[4]:
    st.subheader("Mapa de estaciones (Antioquia)")
    if meta_df is None:
        st.info("Para ver el mapa, sube EstHM_CV.csv que contenga campos 'x' (longitud) y 'y' (latitud). Si tus coordenadas están en otra proyección o con otros nombres, renómbralas en el CSV antes de subir.")
    else:
        if not ('x' in meta_df.columns and 'y' in meta_df.columns):
            st.warning("El archivo de metadatos no contiene columnas 'x' y 'y'. No se puede trazar el mapa sin coordenadas.")
            if 'x' in meta_df.columns or 'y' in meta_df.columns:
                st.write("Columnas disponibles:", [c for c in ['x','y'] if c in meta_df.columns])
        else:
            # convertir a float, filtrar por estaciones seleccionadas si hay join
            coords = meta_df.copy()
            coords = coords.dropna(subset=['x','y'])
            try:
                coords['lon'] = coords['x'].astype(float)
                coords['lat'] = coords['y'].astype(float)
            except Exception as e:
                st.error(f"Error convirtiendo coordenadas: {e}")

            # si detectamos columna join, filtrar por seleccionadas
            if 'Nombre_Est' in coords.columns and any([str(s) in coords['Nombre_Est'].astype(str).values for s in estaciones_seleccionadas]):
                coords_plot = coords[coords['Nombre_Est'].astype(str).isin([str(s) for s in estaciones_seleccionadas])]
            else:
                # intentar filtrar por cualquier columna que coincida
                coords_plot = coords.copy()
                filtered = False
                for col in coords_plot.columns:
                    if coords_plot[col].astype(str).isin([str(s) for s in estaciones_seleccionadas]).any():
                        coords_plot = coords_plot[coords_plot[col].astype(str).isin([str(s) for s in estaciones_seleccionadas])]
                        filtered = True
                        break
                if not filtered:
                    # usar todas las estaciones del archivo de metadatos
                    coords_plot = coords_plot

            if coords_plot.empty:
                st.warning("No hay coordenadas para las estaciones seleccionadas en los metadatos.")
            else:
                # mostrar mapa con pydeck
                midpoint = (coords_plot['lat'].mean(), coords_plot['lon'].mean())
                st.write(f"Mostrando {len(coords_plot)} estaciones en el mapa. Centrado en: {midpoint}")
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=coords_plot,
                    get_position='[lon, lat]',
                    get_radius=5000,
                    pickable=True,
                    auto_highlight=True,
                )
                view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=7, pitch=0)
                r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Nombre_Est}
{Depto} - {Mpio}"})
                st.pydeck_chart(r)

st.caption("Desarrollado con Streamlit - Visualizador de precipitación anual. Mejoras: seleccionar todo/limpiar, rango 1970-2021, pestaña metadatos y mapa.")

    st.caption("Desarrollado con Streamlit - Visualizador de precipitación anual")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")

