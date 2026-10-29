import os
import json
import joblib
import pandas as pd
import paho.mqtt.client as mqtt

# 1. Parámetros del Broker MQTT Local y archivo puente hacia Streamlit
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
MQTT_TOPIC  = "la_artesanal/horno/telemetria"
CLIENT_ID   = "Suscriptor_Motor_Predictivo"
ARCHIVO_CSV = "historial_mqtt_horno.csv"

# 2. Carga de artefactos entrenados en la Fase 3
print("=== CARGANDO MODELO RANDOM FOREST Y ESCALADOR (FASE 3) ===")
modelo_rf = joblib.load("modelo_mantenimiento_horno.pkl")
scaler    = joblib.load("escalador_horno.pkl")
print("[OK] Artefactos .pkl cargados correctamente en memoria.\n")

# Callback al conectar: suscribirse al tópico del horno
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[OK] Suscriptor conectado al Broker MQTT Local ({BROKER_HOST}:{BROKER_PORT})")
        client.subscribe(MQTT_TOPIC, qos=1)
        print(f"[ESCUCHANDO] Tópico suscrito: '{MQTT_TOPIC}' (QoS 1)\n" + "="*75)
    else:
        print(f"[ERROR] Fallo en conexión al Broker Local. Código: {rc}")

# Callback ejecutado automáticamente al recibir cada trama del Publicador
def on_message(client, userdata, msg):
    try:
        # 1. Decodificar mensaje JSON entrante
        datos = json.loads(msg.payload.decode("utf-8"))

        # 2. Construir DataFrame con las 3 columnas del entrenamiento (Fase 3)
        df_sensor = pd.DataFrame({
            "Temperatura_C": [datos["Temperatura_C"]],
            "Horas_Uso_Continuo": [datos["Horas_Uso_Continuo"]],
            "Vibracion_mm_s": [datos["Vibracion_mm_s"]]
        })

        # 3. Normalización (StandardScaler) e inferencia con Random Forest
        datos_escalados = scaler.transform(df_sensor)
        prediccion      = int(modelo_rf.predict(datos_escalados)[0])
        probabilidad    = float(modelo_rf.predict_proba(datos_escalados)[0][1]) * 100.0

        estado_equipo = "ALERTA CRÍTICA" if prediccion == 1 else "ESTADO ÓPTIMO"

        # 4. Mostrar resultado del procesamiento en consola
        print(f"[MENSAJE RECIBIDO - Lote #{datos['id_lote']} | {datos['timestamp']}]")
        print(f"  -> Telemetría : Temp={datos['Temperatura_C']}°C | Uso={datos['Horas_Uso_Continuo']}h | Vib={datos['Vibracion_mm_s']} mm/s")
        print(f"  -> Diagnóstico: {estado_equipo} (Probabilidad de falla: {probabilidad:.1f}%)")
        print("-" * 75)

        # 5. Almacenar registro procesado para alimentar el Dashboard Streamlit
        fila_procesada = {
            "id_lote": datos["id_lote"],
            "timestamp": datos["timestamp"],
            "Temperatura_C": datos["Temperatura_C"],
            "Horas_Uso_Continuo": datos["Horas_Uso_Continuo"],
            "Vibracion_mm_s": datos["Vibracion_mm_s"],
            "Falla_Inminente": prediccion,
            "Probabilidad_Pct": round(probabilidad, 1),
            "Estado_Equipo": estado_equipo
        }
        df_registro = pd.DataFrame([fila_procesada])
        incluir_cabecera = not os.path.exists(ARCHIVO_CSV)
        df_registro.to_csv(ARCHIVO_CSV, mode="a", index=False, header=incluir_cabecera)

    except Exception as e:
        print(f"[ERROR] Error procesando trama MQTT: {e}")

if __name__ == "__main__":
    try:
        suscriptor = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    except AttributeError:
        suscriptor = mqtt.Client(client_id=CLIENT_ID)

    suscriptor.on_connect = on_connect
    suscriptor.on_message = on_message
    suscriptor.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        suscriptor.loop_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Suscriptor detenido por el usuario.")
        suscriptor.disconnect()