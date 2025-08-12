import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from pathlib import Path
from io import BytesIO

# =========================
# CONFIGURACIÓN DE LA PÁGINA
# =========================
st.set_page_config(
    page_title="Visualizador de Lluvia - Antioquia",
    page_icon="🌧️",
    layout="wide"
)

# =========================
# FUNCIÓN PARA CARGAR DATOS
# =========================
@st.cache_data
def cargar_datos():
    pptn = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))
    meta = pd.read_csv(Path("data/EstHM_CV.csv"))

    # Convertir a formato largo
    pptn_long = pptn.melt(
        id_vars=["Estacion"],
        var_name="Año",
        value_name="Precipitación"
    )
    pptn_long["Año"] = pptn_long["Año"].astype(int)

    # Unir metadatos
    df = pptn_long.merge(meta, on="Estacion", how="left")

    return df, pptn, meta

# =========================
# FUNCIÓN PARA DESCARGAR EN EXCEL
# =========================
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
    processed_data = output.getvalue()
    return processed_data

df, pptn_raw, meta = cargar_datos()

# =========================
# INTERFAZ
# =========================
st.title("🌧️ Visualizador de Lluvia - Antioquia")

# ---- SIDEBAR ----
st.sidebar.header("Filtros")

# Filtro estaciones
estaciones_unicas = sorted(df["Estacion"].unique())
col1, col2 = st.sidebar.columns(2)
if col1.button("Seleccionar todo"):
    estaciones_sel = estaciones_unicas
elif col2.button("Limpiar"):
    estaciones_sel = []
else:
    estaciones_sel = st.sidebar.multiselect(
        "Seleccionar estaciones",
        estaciones_unicas,
        default=estaciones_unicas
    )

# Filtro años
años_unicos = sorted(df["Año"].unique())
col3, col4 = st.sidebar.columns(2)
if col3.button("Todos los años"):
    años_sel = años_unicos
elif col4.button("Limpiar años"):
    años_sel = []
else:
    años_sel = st.sidebar.multiselect(
        "Seleccionar años",
        años_unicos,
        default=años_unicos
    )

# =========================
# FILTRADO DE DATOS
# =========================
df_filtrado = df[df["Estacion"].isin(estaciones_sel) & df["Año"].isin(años_sel)]

# =========================
# BOTONES DE DESCARGA
# =========================
if not df_filtrado.empty:
    st.download_button(
        label="📥 Descargar CSV",
        data=df_filtrado.to_csv(index=False).encode("utf-8"),
        file_name="precipitacion_filtrada.csv",
        mime="text/csv"
    )

    st.download_button(
        label="📥 Descargar Excel",
        data=to_excel(df_filtrado),
        file_name="precipitacion_filtrada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================
# PESTAÑAS PRINCIPALES
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualización",
    "📈 Estadísticas",
    "📋 Metadatos",
    "🗺️ Mapa"
])

# ---- TABLA Y GRÁFICA ----
with tab1:
    st.subheader("Datos filtrados")
    st.dataframe(df_filtrado)

    if not df_filtrado.empty:
        fig = px.line(
            df_filtrado,
            x="Año",
            y="Precipitación",
            color="Estacion",
            markers=True,
            title="Precipitación anual por estación"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- ESTADÍSTICAS ----
with tab2:
    if not df_filtrado.empty:
        st.subheader("Resumen estadístico")
        promedio = df_filtrado.groupby("Estacion")["Precipitación"].mean()
        max_val = df_filtrado.loc[df_filtrado["Precipitación"].idxmax()]
        min_val = df_filtrado.loc[df_filtrado["Precipitación"].idxmin()]

        st.write("### Promedio por estación")
        st.dataframe(promedio)

        st.write("### Máximo")
        st.write(max_val)

        st.write("### Mínimo")
        st.write(min_val)
    else:
        st.warning("No hay datos para calcular estadísticas.")

# ---- METADATOS ----
with tab3:
    st.subheader("Metadatos de estaciones")
    meta_cols = [
        "Estacion", "Nombre_Est", "Porc_datos", "Celda_XY",
        "Conteo_celda", "Depto", "Mpio", "AH", "Z_SZH", "x", "y"
    ]
    st.dataframe(meta[meta_cols])

# ---- MAPA ----
with tab4:
    st.subheader("Mapa de estaciones - Antioquia")
    if not meta.empty:
        meta_map = meta.dropna(subset=["x", "y"])
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=meta_map["y"].mean(),
                longitude=meta_map["x"].mean(),
                zoom=7,
                pitch=0
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=meta_map,
                    get_position='[x, y]',
                    get_fill_color='[0, 100, 200, 160]',
                    get_radius=5000,
                    pickable=True
                )
            ],
            tooltip={"text": "{Nombre_Est}\nEstación: {Estacion}"}
        ))
