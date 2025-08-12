import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

# =====================
# CARGA Y PROCESAMIENTO
# =====================
@st.cache_data
def cargar_datos():
    # Leer datos base
    df = pd.read_csv(Path("data/EstHM_CV.csv"))
    pptn_raw = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))

    # Ajustar formato de pptn_raw
    pptn_t = pptn_raw.set_index("Estacion").T.reset_index()
    pptn_t.rename(columns={"index": "Fecha"}, inplace=True)

    # Convertir fechas
    pptn_t["Fecha"] = pd.to_datetime(pptn_t["Fecha"], errors="coerce")

    # Aplanar datos de precipitaciones
    pptn_melt = pptn_t.melt(id_vars="Fecha", var_name="Estacion", value_name="Precipitacion")
    pptn_melt.dropna(subset=["Precipitacion"], inplace=True)

    # Unir con metadatos
    df = pptn_melt.merge(df, on="Estacion", how="left")

    return df, pptn_raw

# =====================
# CARGAR DATOS
# =====================
df, pptn_raw = cargar_datos()

# =====================
# INTERFAZ DE STREAMLIT
# =====================
st.set_page_config(page_title="Visualizador de Lluvia", layout="wide")
st.title("🌧 Visualizador de Lluvia")

# Mostrar datos
st.subheader("Datos Combinados")
st.dataframe(df.head())

# =====================
# FILTROS
# =====================
estaciones = st.multiselect("Selecciona estaciones:", df["Estacion"].unique())
if estaciones:
    df = df[df["Estacion"].isin(estaciones)]

# Rango de fechas
if not df.empty:
    fecha_min = df["Fecha"].min()
    fecha_max = df["Fecha"].max()

    rango_fechas = st.date_input(
        "Selecciona rango de fechas:",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max
    )

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        df = df[(df["Fecha"] >= pd.to_datetime(rango_fechas[0])) &
                (df["Fecha"] <= pd.to_datetime(rango_fechas[1]))]

# =====================
# BOTÓN DE DESCARGA
# =====================
def to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Datos Filtrados")
    return output.getvalue()

if not df.empty:
    excel_bytes = to_excel(df)
    st.download_button(
        label="⬇ Descargar datos filtrados en Excel",
        data=excel_bytes,
        file_name="datos_filtrados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================
# GRÁFICOS
# =====================
if not df.empty:
    st.subheader("Precipitación a lo largo del tiempo")
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x="Fecha", y="Precipitacion", hue="Estacion")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Histograma
    st.subheader("Distribución de Precipitaciones")
    plt.figure(figsize=(10,5))
    sns.histplot(df["Precipitacion"], kde=True)
    st.pyplot(plt)
else:
    st.warning("No hay datos para mostrar con los filtros seleccionados.")
