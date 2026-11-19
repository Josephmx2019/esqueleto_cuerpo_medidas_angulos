import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==========================================================
# CONFIGURACION
# ==========================================================

MODELO = "yolo26n-pose.pt"

CONFIANZA_PERSONA = 0.35
CONFIANZA_PUNTO = 0.30

GROSOR_LINEA = 3
RADIO_ARTICULACION = 5

COLOR_ALERTA = (0, 0, 255)       # Rojo BGR
COLOR_TEXTO = (255, 255, 255)    # Blanco BGR
COLOR_FONDO = (25, 25, 25)

COLORES = {
    "cabeza": (200, 60, 220),
    "torso": (50, 200, 70),
    "brazo_izquierdo": (0, 140, 255),
    "brazo_derecho": (220, 50, 220),
    "pierna_izquierda": (0, 220, 255),
    "pierna_derecha": (255, 130, 30),
}

# YOLO Pose COCO:
# 0 nariz, 1 ojo izq., 2 ojo der., 3 oreja izq., 4 oreja der.
# 5 hombro izq., 6 hombro der., 7 codo izq., 8 codo der.
# 9 muñeca izq., 10 muñeca der., 11 cadera izq., 12 cadera der.
# 13 rodilla izq., 14 rodilla der., 15 tobillo izq., 16 tobillo der.

NOMBRES_PUNTOS = {
    0: "Nariz", 1: "OjoI", 2: "OjoD", 3: "OrejaI", 4: "OrejaD",
    5: "HombroI", 6: "HombroD", 7: "CodoI", 8: "CodoD",
    9: "MunecaI", 10: "MunecaD", 11: "CaderaI", 12: "CaderaD",
    13: "RodillaI", 14: "RodillaD", 15: "TobilloI", 16: "TobilloD",
}

# Las 16 conexiones del esqueleto.
# Cada conexión tiene: punto inicial, punto final y región anatómica.
CONEXIONES = [
    (0, 1, "cabeza"), (0, 2, "cabeza"),
    (1, 3, "cabeza"), (2, 4, "cabeza"),
    (5, 6, "torso"), (5, 11, "torso"),
    (6, 12, "torso"), (11, 12, "torso"),
    (5, 7, "brazo_izquierdo"), (7, 9, "brazo_izquierdo"),
    (6, 8, "brazo_derecho"), (8, 10, "brazo_derecho"),
    (11, 13, "pierna_izquierda"), (13, 15, "pierna_izquierda"),
    (12, 14, "pierna_derecha"), (14, 16, "pierna_derecha"),
]

# Ángulos definidos mediante (punto anterior, vértice, punto siguiente).
ANGULOS = {
    "Hombro I": (7, 5, 11),
    "Hombro D": (8, 6, 12),
    "Codo I": (5, 7, 9),
    "Codo D": (6, 8, 10),
    "Cadera I": (5, 11, 13),
    "Cadera D": (6, 12, 14),
    "Rodilla I": (11, 13, 15),
    "Rodilla D": (12, 14, 16),
}

COLORES_PUNTOS = {
    0: "cabeza", 1: "cabeza", 2: "cabeza", 3: "cabeza", 4: "cabeza",
    5: "torso", 6: "torso",
    7: "brazo_izquierdo", 8: "brazo_derecho",
    9: "brazo_izquierdo", 10: "brazo_derecho",
    11: "torso", 12: "torso",
    13: "pierna_izquierda", 14: "pierna_derecha",
    15: "pierna_izquierda", 16: "pierna_derecha",
}


def punto_valido(puntos, indice):
    """Comprueba que una articulación tenga coordenadas y confianza válidas."""
    if indice >= len(puntos):
        return False

    x, y = puntos[indice, :2]
    if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0:
        return False

    if puntos.shape[1] >= 3:
        confianza = puntos[indice, 2]
        if not np.isfinite(confianza) or confianza < CONFIANZA_PUNTO:
            return False

    return True


def calcular_distancia_px(punto_a, punto_b):
    """Distancia euclidiana entre dos articulaciones en píxeles."""
    return float(np.linalg.norm(np.asarray(punto_a[:2]) - np.asarray(punto_b[:2])))


def calcular_angulo(punto_a, vertice, punto_c):
    """Calcula el ángulo 2D en el vértice, en grados."""
    vector_a = np.asarray(punto_a[:2], dtype=float) - np.asarray(vertice[:2], dtype=float)
    vector_c = np.asarray(punto_c[:2], dtype=float) - np.asarray(vertice[:2], dtype=float)

    norma_a = np.linalg.norm(vector_a)
    norma_c = np.linalg.norm(vector_c)

    if norma_a < 1e-6 or norma_c < 1e-6:
        return None

    coseno = np.dot(vector_a, vector_c) / (norma_a * norma_c)
    coseno = np.clip(coseno, -1.0, 1.0)

    return float(np.degrees(np.arccos(coseno)))


def obtener_altura_caja(caja):
    """Devuelve la altura en píxeles de la caja de detección."""
    if caja is None:
        return None

    alto = float(caja[3] - caja[1])
    return alto if alto > 1 else None


def calcular_medida_cm(distancia_px, altura_caja_px, altura_real_cm):
    """
    Estimación proporcional en cm.
    Requiere altura real conocida y cuerpo suficientemente visible.
    """
    if altura_caja_px is None or altura_real_cm is None:
        return None

    return distancia_px * altura_real_cm / altura_caja_px


def dibujar_texto_con_fondo(frame, texto, posicion, escala=0.48,
                            color=COLOR_TEXTO, grosor=1):
    """Escribe texto legible con un pequeño fondo oscuro."""
    x, y = posicion
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    (ancho, alto), base = cv2.getTextSize(texto, fuente, escala, grosor)

    x = max(0, min(x, frame.shape[1] - ancho - 6))
    y = max(alto + 5, min(y, frame.shape[0] - base - 2))

    cv2.rectangle(
        frame, (x - 2, y - alto - 4), (x + ancho + 3, y + base),
        COLOR_FONDO, -1
    )
    cv2.putText(frame, texto, (x, y), fuente, escala, color, grosor, cv2.LINE_AA)


def procesar_persona(frame, puntos, caja, altura_real_cm, umbral_angulo, persona_id):
    """Calcula medidas/ángulos y dibuja la información de una persona."""
    altura_px = obtener_altura_caja(caja)

    # Calcular ángulos y determinar qué segmentos/articulaciones están en alerta.
    resultados_angulos = {}
    segmentos_alerta = set()
    puntos_alerta = set()

    for nombre, (a, b, c) in ANGULOS.items():
        if all(punto_valido(puntos, idx) for idx in (a, b, c)):
            angulo = calcular_angulo(puntos[a], puntos[b], puntos[c])
        else:
            angulo = None

        resultados_angulos[nombre] = angulo

        if angulo is not None and angulo >= umbral_angulo:
            segmentos_alerta.add(tuple(sorted((a, b))))
            segmentos_alerta.add(tuple(sorted((b, c))))
            puntos_alerta.update((a, b, c))

    # Dibujar conexiones y mostrar longitud aproximada.
    for inicio, fin, region in CONEXIONES:
        if not (punto_valido(puntos, inicio) and punto_valido(puntos, fin)):
            continue

        p1 = tuple(puntos[inicio, :2].astype(int))
        p2 = tuple(puntos[fin, :2].astype(int))
        clave = tuple(sorted((inicio, fin)))
        en_alerta = clave in segmentos_alerta
        color = COLOR_ALERTA if en_alerta else COLORES[region]

        cv2.line(frame, p1, p2, color, GROSOR_LINEA, cv2.LINE_AA)

        distancia_px = calcular_distancia_px(puntos[inicio], puntos[fin])
        medida_cm = calcular_medida_cm(distancia_px, altura_px, altura_real_cm)

        if medida_cm is not None:
            etiqueta = f"{medida_cm:.1f} cm"
        else:
            etiqueta = f"{distancia_px:.0f} px"

        medio = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        dibujar_texto_con_fondo(frame, etiqueta, (medio[0] + 4, medio[1] - 4),
                                escala=0.42, color=COLOR_ALERTA if en_alerta else COLOR_TEXTO)

    # Dibujar articulaciones.
    for indice in range(min(len(puntos), 17)):
        if not punto_valido(puntos, indice):
            continue

        x, y = puntos[indice, :2].astype(int)
        color = COLOR_ALERTA if indice in puntos_alerta else COLORES[COLORES_PUNTOS[indice]]
        cv2.circle(frame, (x, y), RADIO_ARTICULACION, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), RADIO_ARTICULACION, (20, 20, 20), 1, cv2.LINE_AA)

    # Mostrar cada ángulo cerca de su articulación central.
    alertas_activas = []
    for nombre, (a, b, c) in ANGULOS.items():
        angulo = resultados_angulos[nombre]
        if angulo is None:
            continue

        x, y = puntos[b, :2].astype(int)
        en_alerta = angulo >= umbral_angulo
        color = COLOR_ALERTA if en_alerta else COLOR_TEXTO
        dibujar_texto_con_fondo(
            frame, f"{nombre}: {angulo:.0f} grados",
            (x + 8, y - 12), escala=0.43, color=color
        )

        if en_alerta:
            alertas_activas.append((nombre, angulo))

    # Caja y encabezado de persona.
    if caja is not None:
        x1, y1, x2, y2 = np.asarray(caja).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        dibujar_texto_con_fondo(frame, f"Persona {persona_id}",
                                (x1, max(22, y1 - 8)), escala=0.55)

    # Aviso visible si algún ángulo alcanza o supera el umbral.
    if alertas_activas:
        nombres = ", ".join(f"{nombre} {angulo:.0f}°" for nombre, angulo in alertas_activas)
        mensaje = f"ALERTA: {nombres}"
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 780), 42), (0, 0, 150), -1)
        cv2.putText(frame, mensaje[:95], (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return resultados_angulos, alertas_activas


def procesar_frame(frame, modelo, altura_real_cm, umbral_angulo):
    """Ejecuta YOLO Pose y procesa todas las personas detectadas."""
    resultados = modelo.predict(
        source=frame,
        conf=CONFIANZA_PERSONA,
        imgsz=640,
        verbose=False
    )

    resultado = resultados[0]

    if resultado.keypoints is not None and resultado.boxes is not None:
        coordenadas = resultado.keypoints.data.cpu().numpy()
        cajas = resultado.boxes.xyxy.cpu().numpy()

        for i, puntos in enumerate(coordenadas):
            caja = cajas[i] if i < len(cajas) else None
            procesar_persona(
                frame, puntos, caja, altura_real_cm, umbral_angulo, i + 1
            )

    return frame


def procesar_camara(indice=0, salida=None, altura_real_cm=None, umbral_angulo=120):
    captura = cv2.VideoCapture(indice)

    if not captura.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara con índice {indice}. Prueba con otro índice."
        )

    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    escritor = None
    if salida:
        ruta_salida = Path(salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        escritor = cv2.VideoWriter(
            str(ruta_salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
        )
        if not escritor.isOpened():
            captura.release()
            raise RuntimeError("No se pudo crear el archivo de salida.")

    modelo = YOLO(MODELO)
    print("Cámara activa. Pulsa Q o ESC para salir.")

    try:
        while True:
            correcto, frame = captura.read()
            if not correcto:
                print("No se pudo leer un fotograma de la cámara.")
                break

            procesar_frame(frame, modelo, altura_real_cm, umbral_angulo)

            if escritor is not None:
                escritor.write(frame)

            cv2.imshow("YOLO Pose - Medidas y angulos", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q") or tecla == 27:
                break

    finally:
        captura.release()
        if escritor is not None:
            escritor.release()
        cv2.destroyAllWindows()

    if salida:
        print(f"Video guardado en: {salida}")
    print("Cámara desconectada.")


def procesar_video(entrada, salida, altura_real_cm=None, umbral_angulo=120):
    entrada = Path(entrada)
    salida = Path(salida)

    if not entrada.is_file():
        raise FileNotFoundError(f"No existe el video de entrada: {entrada}")

    salida.parent.mkdir(parents=True, exist_ok=True)

    captura = cv2.VideoCapture(str(entrada))
    if not captura.isOpened():
        raise RuntimeError("No se pudo abrir el video. Comprueba su formato.")

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    if ancho <= 0 or alto <= 0:
        captura.release()
        raise RuntimeError("El video no tiene dimensiones válidas.")

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    escritor = cv2.VideoWriter(
        str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
    )
    if not escritor.isOpened():
        captura.release()
        raise RuntimeError("No se pudo crear el video de salida.")

    modelo = YOLO(MODELO)
    numero_fotograma = 0

    try:
        while True:
            correcto, frame = captura.read()
            if not correcto:
                break

            procesar_frame(frame, modelo, altura_real_cm, umbral_angulo)
            escritor.write(frame)
            numero_fotograma += 1

            if numero_fotograma % 30 == 0:
                if total > 0:
                    porcentaje = numero_fotograma / total * 100
                    print(f"Procesados: {numero_fotograma}/{total} ({porcentaje:.1f}%)")
                else:
                    print(f"Fotogramas procesados: {numero_fotograma}")

    finally:
        captura.release()
        escritor.release()

    print(f"\nVideo generado correctamente: {salida}")
    print(f"Fotogramas procesados: {numero_fotograma}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Pose: esqueletización, longitudes aproximadas y ángulos."
    )

    parser.add_argument("entrada", nargs="?", help="Ruta del video que se desea procesar")
    parser.add_argument("--camara", type=int, metavar="INDICE",
                        help="Usar una cámara en lugar de un archivo de video")
    parser.add_argument("-o", "--salida", default=None,
                        help="Ruta del MP4 de salida")
    parser.add_argument("--altura", type=float, default=None,
                        help="Altura real de la persona en centímetros (necesaria para estimar cm)")
    parser.add_argument("--umbral", type=float, default=120.0,
                        help="Ángulo de alerta en grados (por defecto: 120)")

    args = parser.parse_args()

    if not 0 < args.umbral < 180:
        parser.error("El umbral debe ser mayor que 0 y menor que 180 grados.")

    altura_real_cm = args.altura
    if altura_real_cm is not None and altura_real_cm <= 0:
        parser.error("La altura debe ser un número positivo.")

    if altura_real_cm is None:
        respuesta = input(
            "Altura real de la persona en cm (Enter para mostrar solo píxeles): "
        ).strip()
        if respuesta:
            try:
                altura_real_cm = float(respuesta)
                if altura_real_cm <= 0:
                    parser.error("La altura debe ser un número positivo.")
            except ValueError:
                parser.error("La altura debe ser un número válido.")

    if args.camara is not None:
        procesar_camara(
            indice=args.camara,
            salida=args.salida,
            altura_real_cm=altura_real_cm,
            umbral_angulo=args.umbral
        )
    elif args.entrada:
        salida = args.salida or "resultado_esqueleto_medidas.mp4"
        procesar_video(
            args.entrada, salida, altura_real_cm, args.umbral
        )
    else:
        parser.error("Indica un video o utiliza --camara INDICE.")


if __name__ == "__main__":
    main()
