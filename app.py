import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="Visualizador de Lluvia", layout="wide")

# =====================
# FUNCIONES
# =====================
@st.cache_data
def cargar_datos():
    meta = pd.read_csv(Path("data/EstHM_CV.csv"))
    pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))

    pptn_raw.rename(columns={pptn_raw.columns[0]: "Estacion"}, inplace=True)
    pptn_raw["Estacion"] = pd.to_numeric(pptn_raw["Estacion"], errors="coerce")

    df = pptn_raw.melt(id_vars=["Estacion"], var_name="Anio", value_name="Precipitacion")
    df["Anio"] = pd.to_numeric(df["Anio"], errors="coerce")
    df["Precipitacion"] = pd.to_numeric(df["Precipitacion"], errors="coerce")

    return df, pptn_raw, meta


# =====================
# CARGA DE DATOS
# =====================
df, pptn_raw, meta = cargar_datos()

# =====================
# FILTROS
# =====================
st.sidebar.header("Filtros")
anios_disponibles = sorted(df["Anio"].dropna().unique())
anio_sel = st.sidebar.multiselect("Seleccionar año(s)", anios_disponibles, default=anios_disponibles)

df_filtrado = df[df["Anio"].isin(anio_sel)]

# =====================
# MAPA 1 - PROMEDIO
# =====================
st.subheader("🗺 Mapa de promedio de lluvia (Azul = mucha lluvia, Rojo = poca lluvia)")

estaciones_filtradas = df_filtrado["Estacion"].unique()
meta_filtrada = meta[meta["Estacion"].isin(estaciones_filtradas)]

if {"Latitud", "Longitud"}.issubset(meta_filtrada.columns):
    ppt_prom = df_filtrado.groupby("Estacion")["Precipitacion"].mean().reset_index()
    ppt_prom.rename(columns={"Precipitacion": "ppt_media"}, inplace=True)

    meta_filtrada = meta_filtrada.merge(ppt_prom, on="Estacion", how="left")
    meta_filtrada = meta_filtrada.rename(columns={"Latitud": "lat", "Longitud": "lon"})

    max_radio = 8000
    min_radio = 2000
    meta_filtrada["radio"] = ((meta_filtrada["ppt_media"] - meta_filtrada["ppt_media"].min()) /
                              (meta_filtrada["ppt_media"].max() - meta_filtrada["ppt_media"].min() + 1e-6)) \
                              * (max_radio - min_radio) + min_radio

    min_ppt = meta_filtrada["ppt_media"].min()
    max_ppt = meta_filtrada["ppt_media"].max()

    def color_gradiente_invertido(ppt):
        ratio = (ppt - min_ppt) / (max_ppt - min_ppt + 1e-6)
        r = int(255 * (1 - ratio))
        g = int(122 * ratio)
        b = int(255 * ratio)
        return [r, g, b, 200]

    meta_filtrada["color"] = meta_filtrada["ppt_media"].apply(color_gradiente_invertido)

    capa_puntos = pdk.Layer(
        "ScatterplotLayer",
        data=meta_filtrada,
        get_position=["lon", "lat"],
        get_radius="radio",
        get_color="color",
        pickable=True
    )

    vista_inicial = pdk.ViewState(
        latitude=meta_filtrada["lat"].mean(),
        longitude=meta_filtrada["lon"].mean(),
        zoom=6,
        pitch=40
    )

    tooltip = {
        "html": "<b>Estación:</b> {Estacion}<br/><b>Altura:</b> {Altura} m<br/><b>Precipitación media:</b> {ppt_media:.2f} mm",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    mapa = pdk.Deck(
        layers=[capa_puntos],
        initial_view_state=vista_inicial,
        tooltip=tooltip
    )

    st.pydeck_chart(mapa)

    st.markdown(f"""
        <div style="
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 12px;
            box-shadow: 0px 0px 8px rgba(0,0,0,0.2);
            width: 150px;
        ">
            <b>Leyenda</b><br>
            <div style="width: 100%; height: 15px; 
                        background: linear-gradient(to right, rgb(255,0,0), rgb(0,0,255)); 
                        border: 1px solid #555; margin: 5px 0;"></div>
            <div style="display: flex; justify-content: space-between;">
                <span>{min_ppt:.0f} mm</span>
                <span>{max_ppt:.0f} mm</span>
            </div>
            <small>Tamaño ∝ precipitación media</small>
        </div>
    """, unsafe_allow_html=True)

else:
    st.warning("No se encontraron columnas 'Latitud' y 'Longitud' en los metadatos.")

# =====================
# MAPA 2 - ANIMACIÓN TEMPORAL
# =====================
st.subheader("⏳ Animación temporal por año")

anio_anim = st.slider("Seleccionar año", int(min(anios_disponibles)), int(max(anios_disponibles)), int(min(anios_disponibles)))
df_anio = df[df["Anio"] == anio_anim]
meta_anio = meta.merge(df_anio.groupby("Estacion")["Precipitacion"].mean().reset_index(), on="Estacion", how="left")
meta_anio = meta_anio.rename(columns={"Latitud": "lat", "Longitud": "lon"})

min_ppt_a = meta_anio["Precipitacion"].min()
max_ppt_a = meta_anio["Precipitacion"].max()

meta_anio["radio"] = ((meta_anio["Precipitacion"] - min_ppt_a) / (max_ppt_a - min_ppt_a + 1e-6)) * (max_radio - min_radio) + min_radio

meta_anio["color"] = meta_anio["Precipitacion"].apply(lambda ppt: [
    int(255 * (1 - (ppt - min_ppt_a) / (max_ppt_a - min_ppt_a + 1e-6))),
    int(122 * ((ppt - min_ppt_a) / (max_ppt_a - min_ppt_a + 1e-6))),
    int(255 * ((ppt - min_ppt_a) / (max_ppt_a - min_ppt_a + 1e-6))),
    200
])

capa_anim = pdk.Layer(
    "ScatterplotLayer",
    data=meta_anio,
    get_position=["lon", "lat"],
    get_radius="radio",
    get_color="color",
    pickable=True
)

vista_anim = pdk.ViewState(
    latitude=meta_anio["lat"].mean(),
    longitude=meta_anio["lon"].mean(),
    zoom=6,
    pitch=40
)

tooltip_anim = {
    "html": "<b>Estación:</b> {Estacion}<br/><b>Precipitación:</b> {Precipitacion:.2f} mm",
    "style": {"backgroundColor": "darkgreen", "color": "white"}
}

mapa_anim = pdk.Deck(
    layers=[capa_anim],
    initial_view_state=vista_anim,
    tooltip=tooltip_anim
)

st.pydeck_chart(mapa_anim)
