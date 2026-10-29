import time
import json
import random

TEMP_NOMINAL_C = 180.0       # Temperatura ideal del horno (180°C)
VIB_NOMINAL_MMS = 2.5        # Vibración normal de motores de extracción (2.5 mm/s)

def capturar_lectura_horno(numero_lote, horas_acumuladas, forzar_anomalia=False):
    """
    Simula la lectura física de los sensores del Horno Principal y devuelve
    un diccionario con las 3 características exactas que utiliza el modelo
    RandomForestClassifier ('Temperatura_C', 'Horas_Uso_Continuo', 'Vibracion_mm_s').
    """
    if forzar_anomalia:
        # Simulación de condición crítica (Temp > 200°C y Horas > 18, o Vibración > 3.5 mm/s)
        temperatura = round(random.uniform(202.0, 220.0), 2)
        horas_uso = round(random.uniform(18.5, 23.5), 2)
        vibracion = round(random.uniform(3.6, 4.8), 2)
        estado_sensor = "ANOMALIA_SIMULADA"
    else:
        # Simulación de operación normal dentro de los límites de control
        temperatura = round(random.gauss(TEMP_NOMINAL_C, 4.5), 2)
        horas_uso = round(horas_acumuladas, 2)
        vibracion = round(random.gauss(VIB_NOMINAL_MMS, 0.3), 2)
        estado_sensor = "LECTURA_NORMAL"

    # Estructura de trama de telemetría (lista para serializar a JSON / MQTT)
    datos_capturados = {
        "id_lote": numero_lote,
        "centro_trabajo": "Horno",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "estado_captura": estado_sensor,
        "Temperatura_C": temperatura,
        "Horas_Uso_Continuo": horas_uso,
        "Vibracion_mm_s": max(0.0, vibracion)
    }
    return datos_capturados

# Bloque de prueba local del módulo de captura
if __name__ == "__main__":
    print("=====================================================================")
    print(" INICIANDO NODO DE CAPTURA DE SENSORES (PYTHON) - HORNO LA ARTESANAL ")
    print("=====================================================================\n")

    horas_operacion = 10.0  # Inicio simulado en 10 horas de uso continuo
    lote_actual = 1

    try:
        # Ejecutar prueba continua (presionar Ctrl + C para detener)
        while True:
            # Cada 4 lecturas se inyecta un pico anómalo para verificar la detección
            es_critico = (lote_actual % 4 == 0)

            lectura = capturar_lectura_horno(
                numero_lote=lote_actual,
                horas_acumuladas=horas_operacion,
                forzar_anomalia=es_critico
            )

            # Convertir diccionario a cadena JSON estándar para transmisión IIoT
            trama_json = json.dumps(lectura, ensure_ascii=False)
            print(f"[Captura #{lote_actual}] Payload JSON -> {trama_json}")

            # Incrementar tiempo de uso y esperar 3 segundos para el siguiente muestreo
            horas_operacion += 0.25
            lote_actual += 1
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n[INFO] Adquisición de sensores finalizada por el usuario.")