import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt

# 1. Configuración general de la página web
st.set_page_config(page_title="Dashboard IIoT, SPC, KPIs & Sostenibilidad - La Artesanal", layout="wide")
st.title("Panel de Control IIoT, SPC, KPIs y Sostenibilidad - La Artesanal")
st.markdown("Monitoreo en tiempo real, Capacidad de Proceso, KPIs y Huella Ambiental del **Horno Principal (Centro de Trabajo 2)**")

ARCHIVO_CSV_MQTT = "historial_mqtt_horno.csv"

# Parámetros extraídos de Odoo ERP y Simulación (Fases 2 y 3)
COSTO_HORA_HORNO_CS = 90.00       # Costo por hora del CT Horno en Odoo (C$)
COSTO_ESTANDAR_TORTA_CS = 310.23  # Costo estándar BOM Torta de Chocolate Artesanal (PT-001) en Odoo (C$)
META_OEE_ODOO_PCT = 90.0          # Objetivo de eficiencia configurado en Odoo (%)
TIEMPO_CICLO_HORAS = 0.5          # 30 minutos (0.5 h) por lote de horneado
POTENCIA_NOMINAL_HORNO_KW = 5.0   # Consumo nominal de 5 kW definido en Fase 3
POTENCIA_MEZCLADO_KW = 1.5        # Potencia estimada Área de Mezclado (37 min = 0.616 h por lote)
TARIFA_KWH_CS = 8.50              # Tarifa eléctrica industrial referencial (C$/kWh)
FACTOR_EMISION_RED_CO2 = 0.42     # Factor de emisión red eléctrica (kg CO2eq / kWh)
HUELLA_BOM_INSUMOS_CO2 = 3.85     # Huella materias primas BOM por Torta PT-001 (kg CO2eq / lote)

# 2. Cargar artefactos de Machine Learning exportados en la Fase 3
@st.cache_resource
def load_models():
    modelo = joblib.load("modelo_mantenimiento_horno.pkl")
    escalador = joblib.load("escalador_horno.pkl")
    return modelo, escalador

modelo_rf, scaler = load_models()

# 3. Panel lateral: Selección de fuente de datos y Parámetros SPC / Sostenibilidad
st.sidebar.header("Configuración de Telemetría")
modo_fuente = st.sidebar.radio(
    "Seleccione el origen de los datos:",
    ["Flujo MQTT Local (Tiempo Real)", "Simulador Manual (Sliders Fase 3)"]
)

st.sidebar.divider()
st.sidebar.header("Parámetros SPC y Sostenibilidad (Fase 5)")
lei_temp = st.sidebar.number_input("LEI - Límite Esp. Inferior (°C)", value=165.0, step=1.0)
les_temp = st.sidebar.number_input("LES - Límite Esp. Superior (°C)", value=195.0, step=1.0)
ventana_lotes = st.sidebar.slider("Ventana de lotes a evaluar", min_value=20, max_value=200, value=60, step=10)
filtrar_anomalias = st.sidebar.checkbox("Filtrar picos de anomalías simuladas (>200°C) en SPC", value=False)
aplicar_eficiencia = st.sidebar.checkbox("Simular Estrategias de Eficiencia Energética y Circularidad", value=True)

# =========================================================================
# MÓDULO 1: APLICACIÓN DE SPC (GRÁFICOS DE CONTROL, Cp, Cpk)
# =========================================================================
def renderizar_modulo_spc(df_datos, col_variable="Temperatura_C", lei=165.0, les=195.0, filtrar=False):
    st.subheader("1. Aplicación de SPC (Gráfico de Control, Cp y Cpk)")
    
    df_spc = df_datos.copy()
    if filtrar:
        df_spc = df_spc[df_spc[col_variable] <= 200.0]
        
    if len(df_spc) < 5:
        st.info("Se requieren al menos 5 registros válidos para calcular Cp y Cpk.")
        return

    serie = df_spc[col_variable]
    lotes = df_spc["id_lote"] if "id_lote" in df_spc.columns else df_spc.index

    media = serie.mean()
    sigma = serie.std()
    
    if sigma == 0 or np.isnan(sigma):
        st.warning("La desviación estándar es 0; esperando mayor variabilidad.")
        return

    lcs = media + (3 * sigma)
    lci = media - (3 * sigma)
    cp = (les - lei) / (6 * sigma)
    cpk = min((les - media) / (3 * sigma), (media - lei) / (3 * sigma))
    fuera_especificacion = ((serie > les) | (serie < lei)).sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Media Proceso (μ)", f"{media:.2f} °C")
    c2.metric("Desv. Estándar (σ)", f"{sigma:.2f} °C")
    c3.metric("LCI / LCS (±3σ)", f"{lci:.1f} / {lcs:.1f} °C")
    c4.metric("Índice Cp (Potencial)", f"{cp:.2f}")
    c5.metric("Índice Cpk (Real)", f"{cpk:.2f}")
    c6.metric("Lotes Fuera de Esp.", f"{fuera_especificacion} de {len(df_spc)}")

    if cpk >= 1.33:
        st.success(f"**DIAGNÓSTICO SPC - PROCESO CAPAZ Y CONTROLADO (Cpk = {cpk:.2f} ≥ 1.33)**")
    elif cpk >= 1.00:
        st.warning(f"**DIAGNÓSTICO SPC - PROCESO MARGINALMENTE CAPAZ (1.00 ≤ Cpk = {cpk:.2f} < 1.33)**")
    else:
        st.error(f"**DIAGNÓSTICO SPC - PROCESO NO CAPAZ (Cpk = {cpk:.2f} < 1.00):** {fuera_especificacion} lotes en riesgo de merma.")

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(lotes, serie, marker='o', linestyle='-', color='#1f77b4', markersize=4, label='Temperatura por Lote (°C)')
    ax.axhline(media, color='green', linestyle='-', linewidth=1.5, label=f'Media (μ = {media:.1f}°C)')
    ax.axhline(lcs, color='red', linestyle='--', linewidth=1.3, label=f'LCS (+3σ = {lcs:.1f}°C)')
    ax.axhline(lci, color='red', linestyle='--', linewidth=1.3, label=f'LCI (-3σ = {lci:.1f}°C)')
    ax.axhline(les, color='darkorange', linestyle=':', linewidth=1.5, label=f'LES Receta ({les:.1f}°C)')
    ax.axhline(lei, color='darkorange', linestyle=':', linewidth=1.5, label=f'LEI Receta ({lei:.1f}°C)')

    anomalos = df_spc[(df_spc[col_variable] > les) | (df_spc[col_variable] < lei)]
    if not anomalos.empty:
        lotes_anom = anomalos["id_lote"] if "id_lote" in anomalos.columns else anomalos.index
        ax.scatter(lotes_anom, anomalos[col_variable], color='red', s=55, zorder=5, label='Fuera de Especificación')

    ax.set_title("Gráfico de Control SPC X-Individual - Temperatura del Horno Principal")
    ax.set_xlabel("ID de Lote")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    st.pyplot(fig)
    plt.close(fig)

# =========================================================================
# MÓDULO 2: MODELO DE PREDICCIÓN (INFERENCIA Y EXPLICABILIDAD ML)
# =========================================================================
def renderizar_modulo_prediccion(df_datos, modelo):
    st.subheader("2. Modelo de Predicción (Inferencia Random Forest y Proyección de Tendencia)")

    if len(df_datos) < 5:
        return

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Curva de Probabilidad de Falla (%) y Proyección a Futuro (+5 Lotes)**")
        fig1, ax1 = plt.subplots(figsize=(6.5, 3.5))
        lotes_x = df_datos["id_lote"].values
        prob_y = df_datos["Probabilidad_Pct"].values
        
        ax1.plot(lotes_x, prob_y, marker='o', color='#d62728', linewidth=1.4, markersize=4, label='Prob. Falla Random Forest (%)')
        ax1.axhline(50.0, color='black', linestyle='--', linewidth=1.2, label='Umbral Crítico de Disparo (50%)')

        coef = np.polyfit(lotes_x, prob_y, 1)
        polinomio = np.poly1d(coef)
        lotes_futuros = np.arange(lotes_x[-1] + 1, lotes_x[-1] + 6)
        proyeccion_y = np.clip(polinomio(lotes_futuros), 0, 100)
        
        ax1.plot(
            np.concatenate([[lotes_x[-1]], lotes_futuros]),
            np.concatenate([[prob_y[-1]], proyeccion_y]),
            marker='x', linestyle=':', color='purple', linewidth=1.8, label='Tendencia Proyectada (+5 lotes)'
        )
        ax1.set_title("Inferencia de Riesgo de Falla Inminente por Lote")
        ax1.set_xlabel("ID de Lote")
        ax1.set_ylabel("Probabilidad de Falla (%)")
        ax1.set_ylim(-5, 105)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left', fontsize=8)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_p2:
        st.markdown("**Importancia de Variables Predictoras (Feature Importance del Modelo)**")
        fig2, ax2 = plt.subplots(figsize=(6.5, 3.5))
        nombres_vars = ["Temperatura (°C)", "Horas Uso Continuo", "Vibración (mm/s)"]
        importancias = modelo.feature_importances_ * 100.0
        barras = ax2.barh(nombres_vars, importancias, color=["#ff7f0e", "#1f77b4", "#2ca02c"], height=0.5)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Peso Relativo en la Decisión del Random Forest (%)")
        ax2.set_title("Explicabilidad del Artefacto (modelo_mantenimiento_horno.pkl)")
        ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

        for barra in barras:
            ancho = barra.get_width()
            ax2.text(ancho + 1.5, barra.get_y() + barra.get_height()/2, f"{ancho:.1f}%", va='center', fontsize=9, fontweight='bold')
        st.pyplot(fig2)
        plt.close(fig2)

# =========================================================================
# MÓDULO 3: DASHBOARD CON INDICADORES CLAVE (KPIs OEE, COSTOS Y PLANTA)
# =========================================================================
def renderizar_dashboard_kpis(df_datos, lei=165.0, les=195.0):
    st.subheader("3. Dashboard Ejecutivo con Indicadores Clave de Desempeño (KPIs Planta - ERP Odoo)")

    if len(df_datos) < 5:
        return

    total_lotes = len(df_datos)
    lotes_falla = int((df_datos["Falla_Inminente"] == 1).sum())
    lotes_optimos = total_lotes - lotes_falla

    disponibilidad_pct = (lotes_optimos / total_lotes) * 100.0
    desv_vib = np.abs(df_datos["Vibracion_mm_s"] - 2.5).mean()
    rendimiento_pct = float(np.clip(100.0 - (desv_vib * 5.0), 75.0, 99.5))

    lotes_calidad_ok = int(((df_datos["Temperatura_C"] >= lei) & (df_datos["Temperatura_C"] <= les)).sum())
    lotes_merma_termica = total_lotes - lotes_calidad_ok
    calidad_pct = (lotes_calidad_ok / total_lotes) * 100.0

    oee_global_pct = (disponibilidad_pct / 100.0) * (rendimiento_pct / 100.0) * (calidad_pct / 100.0) * 100.0
    delta_oee = oee_global_pct - META_OEE_ODOO_PCT

    mtbf_lotes = (total_lotes / lotes_falla) if lotes_falla > 0 else float(total_lotes)
    horas_evaluadas = total_lotes * TIEMPO_CICLO_HORAS
    costo_operativo_horno_cs = horas_evaluadas * COSTO_HORA_HORNO_CS
    riesgo_merma_cs = lotes_merma_termica * COSTO_ESTANDAR_TORTA_CS

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("OEE Global (Horno)", f"{oee_global_pct:.1f} %", f"{delta_oee:+.1f}% vs Meta Odoo (90%)")
    k2.metric("Disponibilidad Operativa", f"{disponibilidad_pct:.1f} %", f"{lotes_optimos}/{total_lotes} lotes activos")
    k3.metric("Calidad (FPY Receta)", f"{calidad_pct:.1f} %", f"{lotes_calidad_ok} lotes conformes")
    k4.metric("Confiabilidad (MTBF)", f"{mtbf_lotes:.1f} lotes", f"{lotes_falla} alertas críticas")
    k5.metric("Costo Op. Horno (Odoo)", f"C$ {costo_operativo_horno_cs:,.2f}", f"{horas_evaluadas:.1f} hrs a 90 C$/h")
    k6.metric("Riesgo Merma (PT-001)", f"C$ {riesgo_merma_cs:,.2f}", f"{lotes_merma_termica} lotes × 310.23 C$")

    col_k1, col_k2 = st.columns(2)

    with col_k1:
        st.markdown("**Desglose de Indicadores OEE vs. Meta Configurada en Odoo ERP (90%)**")
        fig_oee, ax_oee = plt.subplots(figsize=(6.5, 3.5))
        categorias_oee = ["Disponibilidad", "Rendimiento", "Calidad (SPC)", "OEE Global"]
        valores_oee = [disponibilidad_pct, rendimiento_pct, calidad_pct, oee_global_pct]
        barras_oee = ax_oee.bar(categorias_oee, valores_oee, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"], width=0.5)
        ax_oee.axhline(META_OEE_ODOO_PCT, color="red", linestyle="--", linewidth=1.5, label="Objetivo Odoo (90.0%)")
        ax_oee.set_ylim(0, 115)
        ax_oee.set_ylabel("Porcentaje (%)")
        ax_oee.set_title("Indicadores de Eficiencia del Centro de Trabajo Horno")
        ax_oee.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax_oee.legend(loc="lower right", fontsize=8)

        for b in barras_oee:
            h = b.get_height()
            ax_oee.text(b.get_x() + b.get_width()/2, h + 2.0, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")
        st.pyplot(fig_oee)
        plt.close(fig_oee)

    with col_k2:
        st.markdown("**Distribución de Calidad y Estado Operativo de Lotes Procesados**")
        fig_dist, ax_dist = plt.subplots(figsize=(6.5, 3.5))
        lotes_optimos_conformes = int(((df_datos["Falla_Inminente"] == 0) & (df_datos["Temperatura_C"] >= lei) & (df_datos["Temperatura_C"] <= les)).sum())
        lotes_desviacion_leve = max(0, total_lotes - lotes_optimos_conformes - lotes_falla)

        etiquetas = ["Óptimos y Conformes", "Desviación Térmica Leve", "Alerta Crítica (Falla ML)"]
        tamanos = [lotes_optimos_conformes, lotes_desviacion_leve, lotes_falla]
        colores_pie = ["#2ca02c", "#ffbb00", "#d62728"]

        datos_filtrados = [(e, t, c) for e, t, c in zip(etiquetas, tamanos, colores_pie) if t > 0]
        if datos_filtrados:
            etiq_f, tam_f, col_f = zip(*datos_filtrados)
            wedges, texts, autotexts = ax_dist.pie(
                tam_f, labels=etiq_f, autopct="%1.1f%%", startangle=140,
                colors=col_f, wedgeprops=dict(width=0.45, edgecolor="white"), textprops=dict(fontsize=8.5)
            )
            for autotext in autotexts:
                autotext.set_fontweight("bold")
        ax_dist.set_title(f"Balance Operativo sobre {total_lotes} Lotes Monitoreados")
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    st.markdown("**Matriz de Estado de Centros de Trabajo (Sincronización Parámetros Odoo ERP - Planta)**")
    df_centros_odoo = pd.DataFrame({
        "Centro de Trabajo (Odoo)": ["Área de mezclado (CT-01)", "Horno Principal (CT-02 - Monitoreado IIoT)", "Área de decoración (CT-03)"],
        "Operación Principal (BOM)": ["Batido Bizcocho (15m), Relleno (12m), Ganache (10m)", "Cocción Bizcocho SA-001 (30m a 180°C)", "Ensamble final PT-001 con perlas y cerezas"],
        "Costo Horario Odoo (C$/h)": ["60.00 C$", "90.00 C$", "55.00 C$"],
        "Meta Eficiencia Odoo (%)": ["90.0 %", "90.0 %", "90.0 %"],
        "Eficiencia Actual / OEE (%)": ["94.5 % (Estándar)", f"{oee_global_pct:.1f} % (Tiempo Real MQTT)", "92.0 % (Estándar)"],
        "Estado Operativo": ["Operativo", "ALERTA ACTIVA" if lotes_falla > 0 and df_datos.iloc[-1]["Falla_Inminente"] == 1 else "Monitoreo Activo OK", "Operativo"]
    })
    st.dataframe(df_centros_odoo, use_container_width=True, hide_index=True)

# =========================================================================
# MÓDULO 4: ANÁLISIS DE SOSTENIBILIDAD (HUELLA DE CARBONO, ECONOMÍA CIRCULAR Y EFICIENCIA)
# =========================================================================
def renderizar_modulo_sostenibilidad(df_datos, lei=165.0, les=195.0, simular_mejoras=True):
    st.subheader("4. Análisis de Sostenibilidad (Huella de Carbono, Economía Circular y Eficiencia Energética)")

    if len(df_datos) < 5:
        return

    df_sost = df_datos.copy()
    total_lotes = len(df_sost)

    # 1. Cálculo de Consumo Energético (kWh) y Huella de Carbono (kg CO2eq) por Lote
    # El consumo del horno (nominal 5 kW por 0.5h = 2.5 kWh/lote) varía según temperatura y fatiga mecánica
    factor_termico = df_sost["Temperatura_C"] / 180.0
    penalizacion_vib = np.where(df_sost["Vibracion_mm_s"] > 3.0, 1.12, 1.0)
    df_sost["KWh_Horno_Actual"] = POTENCIA_NOMINAL_HORNO_KW * factor_termico * penalizacion_vib * TIEMPO_CICLO_HORAS
    df_sost["KWh_Mezclado_Decorado"] = (POTENCIA_MEZCLADO_KW * (37.0 / 60.0)) + 0.15  # CT-01 + CT-03
    df_sost["KWh_Total_Actual"] = df_sost["KWh_Horno_Actual"] + df_sost["KWh_Mezclado_Decorado"]

    # Consumo optimizado con Estrategias de Eficiencia Energética (Control PID/IIoT + Aislamiento + Secuenciación MRP)
    factor_reduccion = 0.84 if simular_mejoras else 1.0
    df_sost["KWh_Horno_Optimizado"] = POTENCIA_NOMINAL_HORNO_KW * TIEMPO_CICLO_HORAS * factor_reduccion
    df_sost["KWh_Total_Optimizado"] = df_sost["KWh_Horno_Optimizado"] + (df_sost["KWh_Mezclado_Decorado"] * 0.92)

    energia_actual_kwh = df_sost["KWh_Total_Actual"].sum()
    energia_opt_kwh = df_sost["KWh_Total_Optimizado"].sum()
    ahorro_kwh = max(0.0, energia_actual_kwh - energia_opt_kwh)
    ahorro_energia_cs = ahorro_kwh * TARIFA_KWH_CS

    # Emisiones de Carbono: Alcance 2 (Electricidad Planta) + Alcance 3 (Materias Primas BOM PT-001)
    co2_alcance2_actual = energia_actual_kwh * FACTOR_EMISION_RED_CO2
    co2_alcance2_opt = energia_opt_kwh * FACTOR_EMISION_RED_CO2
    co2_alcance3_bom = total_lotes * HUELLA_BOM_INSUMOS_CO2 * (0.91 if simular_mejoras else 1.0)

    huella_total_actual_kg = (energia_actual_kwh * FACTOR_EMISION_RED_CO2) + (total_lotes * HUELLA_BOM_INSUMOS_CO2)
    huella_total_opt_kg = co2_alcance2_opt + co2_alcance3_bom
    huella_unitaria_torta = (huella_total_opt_kg if simular_mejoras else huella_total_actual_kg) / total_lotes
    reduccion_co2_kg = huella_total_actual_kg - huella_total_opt_kg

    # 2. Cálculos de Economía Circular (Valorización de Mermas y Subproductos)
    lotes_fuera_esp = int(((df_sost["Temperatura_C"] < lei) | (df_sost["Temperatura_C"] > les)).sum())
    peso_torta_kg = 2.62  # Peso neto sumado del BOM de PT-001 en kg
    merma_termica_kg = lotes_fuera_esp * 0.50  # 0.5 kg de bizcocho afectado por lote anómalo
    recortes_y_cascaras_kg = total_lotes * 0.14  # Recorte nivelación bizcocho + 6 cáscaras de huevo MP-003 por lote
    residuos_totales_kg = merma_termica_kg + recortes_y_cascaras_kg
    residuos_revalorizados_kg = residuos_totales_kg * (0.88 if simular_mejoras else 0.15)
    indice_circularidad_pct = (residuos_revalorizados_kg / residuos_totales_kg * 100.0) if residuos_totales_kg > 0 else 100.0

    # Tarjetas KPI de Sostenibilidad (6 métricas)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Consumo Eléctrico Total", f"{energia_actual_kwh:.1f} kWh", f"-{ahorro_kwh:.1f} kWh con Eficiencia")
    s2.metric("Huella Alcance 2 (Red)", f"{co2_alcance2_actual:.2f} kg CO₂e", f"Factor: {FACTOR_EMISION_RED_CO2} kg/kWh")
    s3.metric("Huella Total Proceso", f"{huella_total_actual_kg:.1f} kg CO₂e", f"-{reduccion_co2_kg:.1f} kg CO₂e mitigados")
    s4.metric("Huella Unitaria (PT-001)", f"{huella_unitaria_torta:.2f} kg CO₂e/ud", "Alcance 2 + BOM Alcance 3")
    s5.metric("Índice de Circularidad", f"{indice_circularidad_pct:.1f} %", f"{residuos_revalorizados_kg:.1f} kg revalorizados")
    s6.metric("Ahorro Energético (C$)", f"C$ {ahorro_energia_cs:,.2f}", f"Tarifa: {TARIFA_KWH_CS:.2f} C$/kWh")

    # Gráficos de Sostenibilidad y Eficiencia Energética (2 columnas)
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Desglose de Huella de Carbono por Fuente de Emisión (kg CO₂eq por Lote PT-001)**")
        fig_co2, ax_co2 = plt.subplots(figsize=(6.5, 3.6))
        fuentes_co2 = [
            "Lácteos BOM\n(Leche, Crema, Mant.)",
            "Cacao y Choc.\n(MP-004, 009, 011)",
            "Horno CT-02\n(Alcance 2 Elec.)",
            "Harina, Azúcar\ny Huevos BOM",
            "Mezclado y Dec.\n(CT-01 / CT-03)"
        ]
        valores_co2_ud = [
            1.65,
            1.35,
            (co2_alcance2_actual / total_lotes),
            0.85,
            (df_sost["KWh_Mezclado_Decorado"].mean() * FACTOR_EMISION_RED_CO2)
        ]
        colores_co2 = ["#2ca02c", "#8c564b", "#d62728", "#ffbb00", "#1f77b4"]
        barras_c = ax_co2.barh(fuentes_co2, valores_co2_ud, color=colores_co2, height=0.55)
        ax_co2.set_xlim(0, max(valores_co2_ud) * 1.28)
        ax_co2.set_xlabel("Emisiones Unitarias (kg CO₂eq / Torta producida)")
        ax_co2.set_title("Perfil de Huella de Carbono del Proceso (Alcance 2 + Alcance 3)")
        ax_co2.grid(True, axis="x", linestyle="--", alpha=0.5)

        for b in barras_c:
            w = b.get_width()
            ax_co2.text(w + 0.04, b.get_y() + b.get_height()/2, f"{w:.2f} kg", va="center", fontsize=8.5, fontweight="bold")
        st.pyplot(fig_co2)
        plt.close(fig_co2)

    with col_s2:
        st.markdown("**Estrategia de Eficiencia Energética: Consumo Actual vs. Optimizado IIoT (kWh/Lote)**")
        fig_ef, ax_ef = plt.subplots(figsize=(6.5, 3.6))
        lotes_id = df_sost["id_lote"].values
        ax_ef.plot(lotes_id, df_sost["KWh_Total_Actual"], marker="o", color="#d62728", markersize=3.5, linewidth=1.3, label="Consumo Real Planta (kWh/lote)")
        ax_ef.plot(lotes_id, df_sost["KWh_Total_Optimizado"], linestyle="--", color="#2ca02c", linewidth=1.8, label="Consumo Optimizado IIoT + MRP (kWh/lote)")
        ax_ef.fill_between(lotes_id, df_sost["KWh_Total_Optimizado"], df_sost["KWh_Total_Actual"], color="#2ca02c", alpha=0.18, label="Energía Ahorrada")
        ax_ef.set_title("Impacto de las Estrategias de Eficiencia Energética en el Horno")
        ax_ef.set_xlabel("ID de Lote")
        ax_ef.set_ylabel("Energía Consumida (kWh)")
        ax_ef.grid(True, linestyle="--", alpha=0.5)
        ax_ef.legend(loc="upper right", fontsize=8)
        st.pyplot(fig_ef)
        plt.close(fig_ef)

    # Tabla resumen de Economía Circular y Estrategias de Eficiencia Energética
    st.markdown("**Plan de Economía Circular y Estrategias de Eficiencia Energética en 'La Artesanal'**")
    df_circular = pd.DataFrame({
        "Eje de Sostenibilidad": [
            "Economía Circular (Subproductos)",
            "Economía Circular (Residuos Orgánicos)",
            "Economía Circular (Empaques Proveedor)",
            "Eficiencia Energética (Control IIoT / ML)",
            "Eficiencia Energética (Planificación MRP Odoo)"
        ],
        "Flujo / Recurso Intervenido (BOM / Planta)": [
            f"Recortes de Bizcocho SA-001 y lotes con desviación leve ({merma_termica_kg:.1f} kg)",
            f"Cáscaras de Huevos MP-003 (6 ud/torta) y mermas de crema ({recortes_y_cascaras_kg:.1f} kg)",
            "Sacos de Harina MP-001, Azúcar MP-002 y envases plásticos de Lácteos",
            "Horno Principal CT-02 (Potencia nominal 5.0 kW a 180°C)",
            "Secuenciación de Órdenes de Trabajo (MO) en Horno CT-02"
        ],
        "Propuesta / Estrategia Implementada": [
            "Reproceso interno para elaborar Cake Pops y bases de postres en vaso (Merma Cero)",
            "Entrega separada para compostaje agrícola y suplemento cálcico local",
            "Logística inversa con proveedor 'Molino' para retorno de sacos y compras a granel",
            "Corte preventivo por alerta Random Forest (evita sobrecalentamiento >200°C y vibración >3.5 mm/s)",
            "Agrupación de lotes continuos en Odoo para eliminar precalentamientos repetitivos en frío"
        ],
        "Impacto Cuantificado en el Proceso": [
            f"Recuperación del 85% de merma equivalente a C$ {(lotes_fuera_esp * 120.0):,.2f}",
            f"Desvío del 90% de residuos orgánicos del vertedero ({residuos_revalorizados_kg:.1f} kg)",
            "Reducción del 30% en residuos de embalaje de almacén (Alcance 3)",
            f"Ahorro directo de {ahorro_kwh:.1f} kWh (C$ {ahorro_energia_cs:,.2f}) en ventana actual",
            f"Mitigación de {reduccion_co2_kg:.2f} kg CO₂eq en emisiones de planta"
        ]
    })
    st.dataframe(df_circular, use_container_width=True, hide_index=True)

# =========================================================================
# 5. EJECUCIÓN SEGÚN EL MODO SELECCIONADO
# =========================================================================
if modo_fuente == "Flujo MQTT Local (Tiempo Real)":
    st.sidebar.success("Conectado al historial del Suscriptor MQTT (127.0.0.1:1883 -> la_artesanal/horno/telemetria)")
    auto_refresco = st.sidebar.checkbox("Actualizar automáticamente (cada 3s)", value=True)
    st.sidebar.caption("Tip: Desmarca la casilla superior cuando desees pausar la pantalla para tomar capturas.")

    contenedor_principal = st.empty()

    with contenedor_principal.container():
        if not os.path.exists(ARCHIVO_CSV_MQTT):
            st.warning(
                "Esperando datos del Broker MQTT Local... "
                "Asegúrese de tener en ejecución `mqtt_suscriptor_local.py` y `mqtt_publicador_local.py`."
            )
        else:
            df_historial = pd.read_csv(ARCHIVO_CSV_MQTT)
            if not df_historial.empty:
                ultima_fila = df_historial.iloc[-1]
                lote_actual = int(ultima_fila["id_lote"])
                temp_actual = float(ultima_fila["Temperatura_C"])
                horas_actual = float(ultima_fila["Horas_Uso_Continuo"])
                vib_actual = float(ultima_fila["Vibracion_mm_s"])
                prediccion = int(ultima_fila["Falla_Inminente"])
                prob_pct = float(ultima_fila["Probabilidad_Pct"])
                hora_reg = ultima_fila["timestamp"]

                st.caption(f"Última trama MQTT procesada: Lote #{lote_actual} | Fecha/Hora: {hora_reg}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Lote de Telemetría", f"#{lote_actual}")
                col2.metric("Temperatura Actual", f"{temp_actual:.2f} °C")
                col3.metric("Uso Continuo", f"{horas_actual:.2f} hrs")
                col4.metric("Vibración Motores", f"{vib_actual:.2f} mm/s")

                st.divider()
                st.subheader("Estado Predictivo Instantáneo (Inferencia Random Forest)")
                if prediccion == 1:
                    st.error(
                        f"**ALERTA CRÍTICA:** Alta probabilidad de falla inminente ({prob_pct:.1f}%). "
                        "Se recomienda detener producción en el Horno e iniciar mantenimiento predictivo."
                    )
                else:
                    st.success(
                        f"**ESTADO ÓPTIMO:** El equipo opera dentro de los parámetros seguros. "
                        f"Riesgo de falla bajo ({prob_pct:.1f}%)."
                    )

                st.divider()
                df_ventana = df_historial.tail(ventana_lotes)
                
                # 1. Bloque SPC
                renderizar_modulo_spc(df_ventana, "Temperatura_C", lei_temp, les_temp, filtrar_anomalias)

                st.divider()
                # 2. Bloque Modelo de Predicción
                renderizar_modulo_prediccion(df_ventana, modelo=modelo_rf)

                st.divider()
                # 3. Bloque Dashboard con Indicadores Clave (KPIs)
                renderizar_dashboard_kpis(df_ventana, lei=lei_temp, les=les_temp)

                st.divider()
                # 4. Bloque Análisis de Sostenibilidad (Fase 5.2)
                renderizar_modulo_sostenibilidad(df_ventana, lei=lei_temp, les=les_temp, simular_mejoras=aplicar_eficiencia)

                st.divider()
                with st.expander("Ver curvas nativas de telemetría y tabla de trazabilidad MQTT"):
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        st.markdown("**Tendencia de Temperatura (°C) por Lote MQTT**")
                        st.line_chart(df_ventana.set_index("id_lote")["Temperatura_C"])
                    with col_g2:
                        st.markdown("**Tendencia de Vibración (mm/s) por Lote MQTT**")
                        st.line_chart(df_ventana.set_index("id_lote")["Vibracion_mm_s"])
                    st.dataframe(df_ventana.tail(15), use_container_width=True)

    if auto_refresco:
        time.sleep(3)
        st.rerun()

else:
    # Modo Simulador Manual (Fase 3)
    st.sidebar.info("Modo de prueba manual activado (Sliders).")
    temp_actual = st.sidebar.slider("Temperatura (°C)", min_value=50, max_value=250, value=180)
    horas_actual = st.sidebar.slider("Horas de Uso Continuo", min_value=0, max_value=48, value=12)
    vib_actual = st.sidebar.slider("Vibración de ventiladores (mm/s)", min_value=0.0, max_value=5.0, value=2.1)

    datos_sensor = pd.DataFrame({
        "Temperatura_C": [temp_actual],
        "Horas_Uso_Continuo": [horas_actual],
        "Vibracion_mm_s": [vib_actual]
    })

    datos_escalados = scaler.transform(datos_sensor)
    prediccion = modelo_rf.predict(datos_escalados)
    probabilidad = modelo_rf.predict_proba(datos_escalados)[0][1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperatura", f"{temp_actual} °C")
    col2.metric("Uso Continuo", f"{horas_actual} hrs")
    col3.metric("Vibración", f"{vib_actual} mm/s")

    st.divider()
    st.subheader("Estado Predictivo del Equipo")
    if prediccion[0] == 1:
        st.error(f"**ALERTA CRÍTICA:** Alta probabilidad de falla inminente ({probabilidad*100:.1f}%).")
    else:
        st.success(f"**ESTADO ÓPTIMO:** Riesgo de falla bajo ({probabilidad*100:.1f}%).")

    st.divider()
    if os.path.exists(ARCHIVO_CSV_MQTT):
        df_hist_manual = pd.read_csv(ARCHIVO_CSV_MQTT).tail(ventana_lotes)
        renderizar_modulo_spc(df_hist_manual, "Temperatura_C", lei_temp, les_temp, filtrar_anomalias)
        st.divider()
        renderizar_modulo_prediccion(df_hist_manual, modelo=modelo_rf)
        st.divider()
        renderizar_dashboard_kpis(df_hist_manual, lei=lei_temp, les=les_temp)
        st.divider()
        renderizar_modulo_sostenibilidad(df_hist_manual, lei=lei_temp, les=les_temp, simular_mejoras=aplicar_eficiencia)