import time
import json
import paho.mqtt.client as mqtt
from captura_sensores_horno import capturar_lectura_horno

# 1. Parámetros de conexión al Broker MQTT Local
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
MQTT_TOPIC  = "la_artesanal/horno/telemetria"
CLIENT_ID   = "Publicador_Horno_CT02"

# Callback de verificación de conexión con el Broker Local
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[OK] Publicador conectado al Broker MQTT Local ({BROKER_HOST}:{BROKER_PORT})")
        print(f"[INFO] Transmitiendo en el tópico: '{MQTT_TOPIC}'\n" + "-"*70)
    else:
        print(f"[ERROR] No se pudo conectar al Broker Local. Código: {rc}")

if __name__ == "__main__":
    print("=== INICIANDO PUBLICADOR MQTT LOCAL - HORNO LA ARTESANAL ===")

    # Inicializar cliente MQTT (compatible con paho-mqtt v1.x y v2.x)
    try:
        publicador = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    except AttributeError:
        publicador = mqtt.Client(client_id=CLIENT_ID)

    publicador.on_connect = on_connect
    publicador.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    publicador.loop_start()

    time.sleep(1)
    horas_operacion = 10.0
    lote_actual = 1

    try:
        while True:
            # Cada 4 lecturas se fuerza una anomalía para validar las alertas
            simular_falla = (lote_actual % 4 == 0)

            # Invocar la función del Paso 1 (Capa de Percepción)
            datos_horno = capturar_lectura_horno(
                numero_lote=lote_actual,
                horas_acumuladas=horas_operacion,
                forzar_anomalia=simular_falla
            )

            # Serializar a JSON y publicar con Calidad de Servicio QoS = 1
            payload_json = json.dumps(datos_horno, ensure_ascii=False)
            info_envio = publicador.publish(MQTT_TOPIC, payload_json, qos=1)
            info_envio.wait_for_publish()

            print(f"[PUBLICADO Lote #{lote_actual}] -> {payload_json}")

            horas_operacion += 0.25
            lote_actual += 1
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n[INFO] Publicador desconectado por el usuario.")
        publicador.loop_stop()
        publicador.disconnect()