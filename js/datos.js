/* =============================================================================
 * datos.js — Carga de los dibujos de murciélago (Quick, Draw!, categoría "bat")
 * =============================================================================
 * Acepta tres fuentes:
 *   1. data/bat_28x28.npy  → generado por preparar_datos.py (se carga solo al abrir)
 *   2. Un .npy oficial (numpy_bitmap/bat.npy) que el usuario arrastra a la página
 *   3. Un .ndjson RAW o simplificado (full_raw_bat.ndjson): los trazos se dibujan
 *      aquí mismo en un canvas y se reducen a 28x28. Solo se lee la parte del
 *      archivo necesaria (no hace falta cargar los 566 MB completos).
 *
 * En todos los casos el resultado es un Uint8Array de N*784 valores (0 = fondo,
 * 255 = trazo), el mismo formato de los .npy oficiales de Google.
 * ========================================================================== */

const PIXELES = 28 * 28;   // cada dibujo es una imagen de 28x28 = 784 valores

/** Lee la cabecera de un archivo .npy y devuelve {desplazamiento, forma, dtype}. */
function leerCabeceraNpy(bytes) {
  // Los .npy empiezan con el "número mágico" \x93NUMPY
  const magico = String.fromCharCode(...bytes.slice(1, 6));
  if (bytes[0] !== 0x93 || magico !== 'NUMPY') throw new Error('No es un archivo .npy válido');
  const version = bytes[6];
  const vista = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // Versión 1.x guarda el largo de la cabecera en 2 bytes; versiones 2.x/3.x en 4 bytes
  const largo = version === 1 ? vista.getUint16(8, true) : vista.getUint32(8, true);
  const inicio = version === 1 ? 10 : 12;
  const texto = new TextDecoder().decode(bytes.slice(inicio, inicio + largo));
  const forma = (texto.match(/'shape':\s*\(([^)]*)\)/) || [])[1];
  const dtype = (texto.match(/'descr':\s*'([^']*)'/) || [])[1];
  if (!dtype || !/u1|\|u1|uint8/.test(dtype)) throw new Error(`Se esperaba uint8 y llegó ${dtype}`);
  const dims = forma.split(',').map(s => s.trim()).filter(Boolean).map(Number);
  return { desplazamiento: inicio + largo, forma: dims, dtype };
}

/** Lee hasta `maximo` dibujos de un .npy (File/Blob), leyendo solo los bytes necesarios. */
async function leerNpyDeBlob(blob, maximo) {
  const cabecera = leerCabeceraNpy(new Uint8Array(await blob.slice(0, 512).arrayBuffer()));
  const total = cabecera.forma[0];
  const porFila = cabecera.forma.slice(1).reduce((a, b) => a * b, 1);
  if (porFila !== PIXELES) throw new Error(`Cada fila debería tener 784 píxeles y tiene ${porFila}`);
  const n = Math.min(total, maximo);
  const inicio = cabecera.desplazamiento;
  const datos = new Uint8Array(await blob.slice(inicio, inicio + n * PIXELES).arrayBuffer());
  return { pixeles: datos, n, total };
}

/* ---------- Rasterizado de trazos (para archivos .ndjson) ---------- */
const LIENZO = 256, MARGEN = 16, GROSOR = 14;
let _lienzoGrande = null, _lienzoMedio = null, _lienzoPeq = null;

/** Dibuja los trazos de un dibujo en un canvas y lo reduce a 28x28 (mismo método que preparar_datos.py). */
function rasterizarDibujo(trazos) {
  if (!_lienzoGrande) {
    const crear = (t) => { const c = document.createElement('canvas'); c.width = c.height = t; return c; };
    _lienzoGrande = crear(LIENZO); _lienzoMedio = crear(64); _lienzoPeq = crear(28);
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [xs, ys] of trazos) {
    for (const x of xs) { if (x < minX) minX = x; if (x > maxX) maxX = x; }
    for (const y of ys) { if (y < minY) minY = y; if (y > maxY) maxY = y; }
  }
  const ancho = Math.max(maxX - minX, 1), alto = Math.max(maxY - minY, 1);
  const escala = (LIENZO - 2 * MARGEN) / Math.max(ancho, alto);
  const offX = (LIENZO - ancho * escala) / 2, offY = (LIENZO - alto * escala) / 2;

  const g = _lienzoGrande.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, LIENZO, LIENZO);
  g.strokeStyle = '#fff'; g.fillStyle = '#fff';
  g.lineWidth = GROSOR; g.lineCap = 'round'; g.lineJoin = 'round';
  for (const [xs, ys] of trazos) {
    g.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = (xs[i] - minX) * escala + offX, py = (ys[i] - minY) * escala + offY;
      if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
    }
    if (xs.length === 1) { g.arc((xs[0] - minX) * escala + offX, (ys[0] - minY) * escala + offY, GROSOR / 2, 0, 7); g.fill(); }
    else g.stroke();
  }
  // Reducción en dos pasos (256 → 64 → 28) para que quede suave
  const m = _lienzoMedio.getContext('2d');
  m.imageSmoothingEnabled = true; m.imageSmoothingQuality = 'high';
  m.drawImage(_lienzoGrande, 0, 0, 64, 64);
  const p = _lienzoPeq.getContext('2d');
  p.imageSmoothingEnabled = true; p.imageSmoothingQuality = 'high';
  p.drawImage(_lienzoMedio, 0, 0, 28, 28);
  const rgba = p.getImageData(0, 0, 28, 28).data;
  const salida = new Uint8Array(PIXELES);
  for (let i = 0; i < PIXELES; i++) salida[i] = rgba[i * 4];
  return salida;
}

/** Lee un .ndjson por trozos de 4 MB hasta juntar `maximo` dibujos reconocidos. */
async function leerNdjsonDeBlob(blob, maximo, alProgresar) {
  const TROZO = 4 * 1024 * 1024;
  const datos = new Uint8Array(maximo * PIXELES);
  let n = 0, pos = 0, resto = '';
  const decodificador = new TextDecoder();
  while (n < maximo && pos < blob.size) {
    const texto = resto + decodificador.decode(await blob.slice(pos, pos + TROZO).arrayBuffer(), { stream: true });
    pos += TROZO;
    const lineas = texto.split('\n');
    resto = lineas.pop();                       // la última línea puede venir cortada
    for (const linea of lineas) {
      if (!linea.trim()) continue;
      const d = JSON.parse(linea);
      if (d.recognized === false) continue;     // solo dibujos que el juego reconoció como "bat"
      datos.set(rasterizarDibujo(d.drawing), n * PIXELES);
      if (++n >= maximo) break;
    }
    if (alProgresar) alProgresar(n, maximo);
    await new Promise(r => setTimeout(r, 0));   // deja respirar a la interfaz
  }
  return { pixeles: datos.slice(0, n * PIXELES), n, total: null };
}

/** Punto de entrada: decide cómo leer según la extensión del archivo. */
async function cargarDibujos(blob, nombre, maximo, alProgresar) {
  if (/\.npy$/i.test(nombre)) return leerNpyDeBlob(blob, maximo);
  if (/\.ndjson$/i.test(nombre)) return leerNdjsonDeBlob(blob, maximo, alProgresar);
  throw new Error('Formato no soportado: usa .npy o .ndjson');
}

/** Intenta descargar un archivo por URL (relativa o absoluta) y lo devuelve como Blob. */
async function descargarComoBlob(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.blob();
}

/**
 * Convierte los píxeles (0..255) en un tensor [N, 28, 28, 1] con valores en [-1, 1].
 * Se normaliza a [-1, 1] porque el generador termina en tanh, cuyo rango es [-1, 1].
 */
function crearTensorDatos(pixeles, n) {
  const normalizados = new Float32Array(n * PIXELES);
  for (let i = 0; i < normalizados.length; i++) normalizados[i] = pixeles[i] / 127.5 - 1;
  return tf.tensor4d(normalizados, [n, 28, 28, 1]);
}
