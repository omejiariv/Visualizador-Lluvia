import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ===== Función para cargar datos =====
@st.cache_data
def cargar_datos():
    df = pd.read_csv(Path("data/EstHM_CV.csv"))
    pptn = pd.read_csv(Path("data/Transp_Est_Pptn.csv"))
    meta = df.merge(pptn, on="Estacion", how="inner")
    return df, pptn, meta

# ===== Cargar datos =====
df, pptn_raw, meta = cargar_datos()

st.title("📊 Visualizador de Cobertura de Precipitaciones")
st.markdown("Explora la cobertura de datos de estaciones meteorológicas a lo largo del tiempo.")

# ===== Selección de parámetros =====
mostrar_porcentaje = st.checkbox("Mostrar porcentaje de cobertura", value=True)
paletas = ["viridis", "plasma", "coolwarm", "YlGnBu", "magma"]
paleta_sel = st.selectbox("Selecciona paleta de colores:", paletas, index=0)
invertir_colores = st.checkbox("Invertir colores", value=False)

# ===== Calcular cobertura =====
cobertura = pptn_raw.set_index("Estacion").notna().groupby(level=0).mean()
if mostrar_porcentaje:
    cobertura *= 100

cbar_label = "% de cobertura" if mostrar_porcentaje else "Cobertura (proporción)"

# ===== Ajuste de intensidad =====
vmin = st.slider(
    "Valor mínimo de la escala de colores:",
    0.0 if mostrar_porcentaje else 0,
    100.0 if mostrar_porcentaje else 12,
    0.0 if mostrar_porcentaje else 0
)
vmax = st.slider(
    "Valor máximo de la escala de colores:",
    0.0 if mostrar_porcentaje else 0,
    100.0 if mostrar_porcentaje else 12,
    100.0 if mostrar_porcentaje else 12
)

# ===== Mapa de calor =====
cmap_final = paleta_sel + ("_r" if invertir_colores else "")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    cobertura,
    cmap=cmap_final,
    cbar_kws={'label': cbar_label},
    ax=ax,
    vmin=vmin,
    vmax=vmax
)
ax.set_title("Cobertura de datos por estación y año", fontsize=14)
ax.set_xlabel("Año")
ax.set_ylabel("Estación")
st.pyplot(fig)
