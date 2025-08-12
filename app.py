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
#
# --- Contenido sugerido de README.md ---
# # 🌧 Visualizador de Precipitación Anual
# Aplicación web interactiva desarrollada en Streamlit para visualizar y analizar datos de lluvias anuales por estación.
#
# ## 📋 Características
# - Carga de archivo CSV directamente desde la interfaz.
# - Filtros por estación y rango de años.
# - Gráficos interactivos (línea, barras y boxplot).
# - Estadísticas básicas (promedio, mínimo y máximo con años asociados).
#
# ## 🚀 Cómo ejecutar localmente
# ```bash
# pip install -r requirements.txt
# streamlit run app.py
# ```
#
# ## ☁️ Despliegue en Streamlit Cloud
# 1. Sube este repositorio a GitHub.
# 2. Ve a [Streamlit Cloud](https://streamlit.io/cloud) y crea una nueva app.
# 3. Selecciona el repositorio y apunta `app.py` como archivo principal.
# 4. Haz clic en **Deploy**.
#
# ## 📷 Capturas de Pantalla
# *(Agrega imágenes de ejemplo de tu app aquí)*

import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except ModuleNotFoundError:
    st.error("La librería 'plotly' no está instalada. Asegúrate de incluirla en requirements.txt con la versión recomendada.")
    st.stop()

st.title("🌧 Visualizador de Precipitación Anual")

# Subida de archivo CSV
archivo = st.file_uploader("Sube un archivo CSV con datos de lluvia anual", type=["csv"])

if archivo is not None:
    try:
        # Cargar datos
        df = pd.read_csv(archivo)
        df.rename(columns={df.columns[0]: "Año"}, inplace=True)
        df["Año"] = df["Año"].astype(int)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        st.stop()

    # Sidebar - Selección de estaciones y rango de años
    estaciones = df.columns[1:]
    año_min, año_max = df["Año"].min(), df["Año"].max()

    st.sidebar.title("Opciones de filtro")
    estaciones_seleccionadas = st.sidebar.multiselect("Selecciona estaciones", estaciones, default=estaciones)
    rango_años = st.sidebar.slider("Selecciona rango de años", min_value=int(año_min), max_value=int(año_max), value=(int(año_min), int(año_max)))

    # Filtrar datos
    df_filtrado = df[(df["Año"] >= rango_años[0]) & (df["Año"] <= rango_años[1])][["Año"] + list(estaciones_seleccionadas)]

    # Tabs de contenido
    tabs = st.tabs(["Tabla de Datos", "Gráficos", "Estadísticas"])

    # --- TABLA ---
    with tabs[0]:
        st.subheader("Tabla de precipitaciones")
        st.dataframe(df_filtrado, use_container_width=True)

    # --- GRÁFICOS ---
    with tabs[1]:
        st.subheader("Visualización de Datos")
        tipo_grafico = st.radio("Tipo de gráfico", ["Línea", "Barras", "Boxplot"], horizontal=True)

        df_melt = df_filtrado.melt(id_vars=["Año"], var_name="Estacion", value_name="Precipitacion")

        if tipo_grafico == "Línea":
            fig = px.line(df_melt, x="Año", y="Precipitacion", color="Estacion", markers=True)
        elif tipo_grafico == "Barras":
            fig = px.bar(df_melt, x="Año", y="Precipitacion", color="Estacion", barmode="group")
        elif tipo_grafico == "Boxplot":
            fig = px.box(df_melt, x="Estacion", y="Precipitacion")

        st.plotly_chart(fig, use_container_width=True)

    # --- ESTADÍSTICAS ---
    with tabs[2]:
        st.subheader("Estadísticas Básicas")

        estaciones_stats = []
        for estacion in estaciones_seleccionadas:
            min_val = df_filtrado[estacion].min()
            max_val = df_filtrado[estacion].max()
            anio_min = df_filtrado[df_filtrado[estacion] == min_val]["Año"].values[0]
            anio_max = df_filtrado[df_filtrado[estacion] == max_val]["Año"].values[0]
            estaciones_stats.append({
                "Estacion": estacion,
                "Promedio": round(df_filtrado[estacion].mean(), 2),
                "Mínimo": f"{min_val} ({anio_min})",
                "Máximo": f"{max_val} ({anio_max})"
            })

        st.dataframe(pd.DataFrame(estaciones_stats).set_index("Estacion"), use_container_width=True)

    st.caption("Desarrollado con Streamlit - Visualizador de precipitación anual")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
