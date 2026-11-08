"""
preparar_datos.py — BatGAN Lab
==============================

Convierte los dibujos de murciélago (categoría "bat") del dataset Quick, Draw!
desde el formato RAW (.ndjson, trazos vectoriales) a imágenes de 28x28 en
escala de grises, guardadas como un arreglo NumPy (.npy) que la página web
carga automáticamente desde data/bat_28x28.npy.

Formato RAW de cada línea del .ndjson:
    {"word": "bat", "recognized": true, "drawing": [[[x...], [y...], [t...]], ...]}
    - cada trazo es una lista de coordenadas x, coordenadas y y tiempos (ms)

Proceso por dibujo:
    1. Se calcula la caja que encierra todos los trazos.
    2. Se escala para que quepa en un lienzo de 256x256 (con margen), centrado.
    3. Se dibujan los trazos en blanco sobre negro (igual que los .npy oficiales).
    4. Se reduce a 28x28 con un filtro de área (antialiasing).

Uso:
    python preparar_datos.py --entrada full_raw_bat.ndjson --cantidad 20000
    python preparar_datos.py            # descarga el RAW directamente de Google

Requisitos: pip install numpy pillow
Datos: The Quick, Draw! Dataset — Google, licencia CC BY 4.0.
"""

import argparse
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

URL_RAW = "https://storage.googleapis.com/quickdraw_dataset/full/raw/bat.ndjson"

LIENZO = 256      # tamaño del lienzo intermedio en píxeles
MARGEN = 16       # margen alrededor del dibujo dentro del lienzo
GROSOR = 14       # grosor del trazo en el lienzo de 256 (≈1.5 px al reducir a 28)
SALIDA_PX = 28    # tamaño final de la imagen (28x28, como MNIST)


def rasterizar(trazos):
    """Convierte la lista de trazos RAW de un dibujo en una imagen 28x28 (uint8)."""
    xs = [x for t in trazos for x in t[0]]
    ys = [y for t in trazos for y in t[1]]
    if not xs:
        return None
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    ancho, alto = max(max_x - min_x, 1), max(max_y - min_y, 1)

    # Escala uniforme (conserva la proporción) y centrado en el lienzo
    escala = (LIENZO - 2 * MARGEN) / max(ancho, alto)
    off_x = (LIENZO - ancho * escala) / 2
    off_y = (LIENZO - alto * escala) / 2

    img = Image.new("L", (LIENZO, LIENZO), 0)          # fondo negro
    lapiz = ImageDraw.Draw(img)
    for t in trazos:
        puntos = [((x - min_x) * escala + off_x, (y - min_y) * escala + off_y)
                  for x, y in zip(t[0], t[1])]
        if len(puntos) == 1:                          # un solo punto: dibujar un círculo
            px, py = puntos[0]
            r = GROSOR / 2
            lapiz.ellipse([px - r, py - r, px + r, py + r], fill=255)
        else:
            lapiz.line(puntos, fill=255, width=GROSOR, joint="curve")
            for px, py in (puntos[0], puntos[-1]):    # extremos redondeados
                r = GROSOR / 2
                lapiz.ellipse([px - r, py - r, px + r, py + r], fill=255)

    # Reducir a 28x28 promediando áreas (suaviza los bordes)
    pequeña = img.resize((SALIDA_PX, SALIDA_PX), Image.Resampling.BOX)
    return np.asarray(pequeña, dtype=np.uint8).reshape(-1)   # 784 valores


def lineas_desde(entrada):
    """Devuelve un iterador de líneas: del archivo local o descargando de Google."""
    if entrada and os.path.exists(entrada):
        return open(entrada, "r", encoding="utf-8")
    print(f"Descargando {URL_RAW} (se detiene al tener suficientes dibujos)...")
    respuesta = urllib.request.urlopen(URL_RAW)
    return io.TextIOWrapper(respuesta, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Prepara los murciélagos de Quick, Draw! para BatGAN Lab")
    p.add_argument("--entrada", default="full_raw_bat.ndjson", help="archivo .ndjson RAW de la categoría bat")
    p.add_argument("--salida", default=os.path.join("data", "bat_28x28.npy"))
    p.add_argument("--cantidad", type=int, default=20000, help="cuántos dibujos guardar")
    p.add_argument("--incluir-no-reconocidos", action="store_true",
                   help="por defecto solo se usan los dibujos que el juego reconoció como 'bat'")
    args = p.parse_args()

    imagenes, leidos = [], 0
    with lineas_desde(args.entrada) as f:
        for linea in f:
            leidos += 1
            d = json.loads(linea)
            if not d.get("recognized", False) and not args.incluir_no_reconocidos:
                continue
            img = rasterizar(d["drawing"])
            if img is not None:
                imagenes.append(img)
            if len(imagenes) % 2000 == 0 and imagenes:
                print(f"  {len(imagenes)} dibujos convertidos...", flush=True)
            if len(imagenes) >= args.cantidad:
                break

    datos = np.stack(imagenes).astype(np.uint8)          # forma (N, 784), igual que los .npy oficiales
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    np.save(args.salida, datos)
    print(f"Listo: {datos.shape[0]} dibujos ({leidos} líneas leídas) -> {args.salida} "
          f"[{os.path.getsize(args.salida) / 1e6:.1f} MB]")


if __name__ == "__main__":
    sys.exit(main())
