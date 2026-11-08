/* =============================================================================
 * visualizacion.js — Dibujos en canvas: grillas, curvas, histogramas, filtros
 * ========================================================================== */

const COLOR_REAL = [27, 140, 84];     // verde  (como en GAN Lab: "real")
const COLOR_FALSO = [123, 50, 148];   // morado (como en GAN Lab: "falso")
const COLOR_NEUTRO = [170, 170, 170];

/** Mezcla morado→gris→verde según t ∈ [0,1] (0 = parece falso, 1 = parece real). */
function colorOpinion(t) {
  t = Math.max(0, Math.min(1, t));
  const [a, b, u] = t < 0.5 ? [COLOR_FALSO, COLOR_NEUTRO, t * 2] : [COLOR_NEUTRO, COLOR_REAL, (t - 0.5) * 2];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * u)).join(',')})`;
}

const _celda = document.createElement('canvas');
_celda.width = _celda.height = 28;

/** Pinta una imagen 28x28 (valores en [-1,1], trazo = +1) como trazo oscuro sobre papel. */
function pintarImagen(ctx, valores, desde, x, y, tam) {
  const c = _celda.getContext('2d');
  const img = c.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = 255 - Math.round((valores[desde + i] + 1) * 127.5);
    const p = Math.max(0, Math.min(255, v));
    img.data[i * 4] = p; img.data[i * 4 + 1] = p; img.data[i * 4 + 2] = p; img.data[i * 4 + 3] = 255;
  }
  c.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(_celda, x, y, tam, tam);
}

/**
 * Grilla de dibujos. valores: Float32Array de n*784 en [-1,1].
 * opiniones (opcional): array de t ∈ [0,1] para colorear el borde de cada celda.
 */
function dibujarGrilla(canvas, valores, n, cols, opiniones) {
  const ctx = canvas.getContext('2d');
  const tam = canvas.width / cols;
  const borde = opiniones ? Math.max(2, Math.round(tam * 0.07)) : 0;
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < n; i++) {
    const x = (i % cols) * tam, y = Math.floor(i / cols) * tam;
    if (opiniones) { ctx.fillStyle = colorOpinion(opiniones[i]); ctx.fillRect(x, y, tam, tam); }
    pintarImagen(ctx, valores, i * 784, x + borde, y + borde, tam - 2 * borde);
  }
}

/** Gráfica de líneas. series = [{nombre, color, valores:[...]}], pasos = [...] (eje x). */
function dibujarCurvas(canvas, pasos, series, { lineaRef = null, minY = null, maxY = null } = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, m = { i: 40, d: 10, s: 22, b: 22 };
  ctx.clearRect(0, 0, W, H);
  ctx.font = '11px Inter, sans-serif';
  if (!pasos.length) { ctx.fillStyle = '#94a3b8'; ctx.fillText('Entrena para ver la curva', m.i, H / 2); return; }
  let lo = Infinity, hi = -Infinity;
  series.forEach(s => s.valores.forEach(v => { if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  if (lineaRef !== null) { lo = Math.min(lo, lineaRef); hi = Math.max(hi, lineaRef); }
  if (minY !== null) lo = minY; if (maxY !== null) hi = maxY;
  if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
  const x0 = pasos[0], x1 = Math.max(pasos[pasos.length - 1], x0 + 1);
  const X = (p) => m.i + (p - x0) / (x1 - x0) * (W - m.i - m.d);
  const Y = (v) => m.s + (1 - (v - lo) / (hi - lo)) * (H - m.s - m.b);
  // ejes y guías
  ctx.strokeStyle = '#e2e8f0'; ctx.fillStyle = '#64748b'; ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(m.i, y); ctx.lineTo(W - m.d, y); ctx.stroke();
    ctx.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2), 2, y + 4);
  }
  ctx.fillText(`paso ${x0}`, m.i, H - 6);
  const t1 = `paso ${pasos[pasos.length - 1]}`;
  ctx.fillText(t1, W - m.d - ctx.measureText(t1).width, H - 6);
  if (lineaRef !== null) {
    ctx.setLineDash([4, 4]); ctx.strokeStyle = '#94a3b8';
    ctx.beginPath(); ctx.moveTo(m.i, Y(lineaRef)); ctx.lineTo(W - m.d, Y(lineaRef)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // líneas
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.8; ctx.beginPath();
    s.valores.forEach((v, i) => { const x = X(pasos[i]), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  });
  // leyenda
  let lx = m.i;
  series.forEach(s => {
    ctx.fillStyle = s.color; ctx.fillRect(lx, 6, 10, 10);
    ctx.fillStyle = '#334155'; ctx.fillText(s.nombre, lx + 14, 15);
    lx += ctx.measureText(s.nombre).width + 30;
  });
}

/** Histograma de los valores de los pesos de una capa. */
function dibujarHistograma(canvas, valores, color) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, BINS = 50;
  ctx.clearRect(0, 0, W, H);
  let lo = Infinity, hi = -Infinity;
  for (const v of valores) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-9) { hi = lo + 1e-3; }
  const cuentas = new Array(BINS).fill(0);
  for (const v of valores) cuentas[Math.min(BINS - 1, Math.floor((v - lo) / (hi - lo) * BINS))]++;
  const max = Math.max(...cuentas);
  const ancho = (W - 20) / BINS;
  ctx.fillStyle = color;
  cuentas.forEach((c, i) => {
    const h = c / max * (H - 26);
    ctx.fillRect(10 + i * ancho, H - 16 - h, ancho - 1, h);
  });
  ctx.fillStyle = '#64748b'; ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillText(lo.toFixed(3), 10, H - 3);
  const t = hi.toFixed(3); ctx.fillText(t, W - 10 - ctx.measureText(t).width, H - 3);
  if (lo < 0 && hi > 0) {   // marca del cero
    const x0 = 10 + (-lo) / (hi - lo) * (W - 20);
    ctx.strokeStyle = '#94a3b8'; ctx.beginPath(); ctx.moveTo(x0, 4); ctx.lineTo(x0, H - 16); ctx.stroke();
    ctx.fillText('0', x0 - 3, H - 3);
  }
}

/** Mapa de calor divergente (negativo = morado, 0 = blanco, positivo = verde). */
function colorDivergente(v) {
  const t = Math.max(-1, Math.min(1, v));
  const base = t < 0 ? COLOR_FALSO : COLOR_REAL, u = Math.abs(t);
  return [255 + (base[0] - 255) * u, 255 + (base[1] - 255) * u, 255 + (base[2] - 255) * u];
}

/**
 * Filtros de una capa.
 *  - Kernel 4D [alto, ancho, a, b] (conv): un mosaico de filtros 5x5 (canal 0 de la 3a dimensión).
 *  - Kernel 2D [entradas, salidas] (densa): mapa de calor de la matriz de pesos.
 */
function dibujarFiltros(canvas, valores, forma) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  let maxAbs = 1e-9;
  for (const v of valores) maxAbs = Math.max(maxAbs, Math.abs(v));
  const tmp = document.createElement('canvas');
  if (forma.length === 4) {
    const [kh, kw, a, b] = forma;
    const cantidad = Math.min(b, 32), cols = 16;
    const tam = Math.floor(Math.min(W / cols, H / Math.ceil(cantidad / cols))) - 3;
    tmp.width = kw; tmp.height = kh;
    const tc = tmp.getContext('2d'), img = tc.createImageData(kw, kh);
    for (let f = 0; f < cantidad; f++) {
      for (let y = 0; y < kh; y++) for (let x = 0; x < kw; x++) {
        const v = valores[((y * kw + x) * a + 0) * b + f] / maxAbs;
        const [r, g, bl] = colorDivergente(v), p = (y * kw + x) * 4;
        img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = bl; img.data[p + 3] = 255;
      }
      tc.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      const cx = (f % cols) * (tam + 3), cy = Math.floor(f / cols) * (tam + 3);
      ctx.drawImage(tmp, cx, cy, tam, tam);
      ctx.strokeStyle = '#cbd5e1'; ctx.strokeRect(cx + 0.5, cy + 0.5, tam - 1, tam - 1);
    }
    return `${cantidad} de ${b} filtros ${kh}×${kw} (verde = peso positivo, morado = negativo)`;
  }
  if (forma.length === 2) {
    const [filas, columnas] = forma;
    tmp.width = Math.min(columnas, W); tmp.height = Math.min(filas, H);
    const tc = tmp.getContext('2d'), img = tc.createImageData(tmp.width, tmp.height);
    for (let y = 0; y < tmp.height; y++) for (let x = 0; x < tmp.width; x++) {
      const fi = Math.floor(y * filas / tmp.height), co = Math.floor(x * columnas / tmp.width);
      const [r, g, bl] = colorDivergente(valores[fi * columnas + co] / maxAbs), p = (y * tmp.width + x) * 4;
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = bl; img.data[p + 3] = 255;
    }
    tc.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, W, H);
    return `Matriz de pesos ${filas}×${columnas} (filas = entradas, columnas = neuronas)`;
  }
  return 'Esta capa tiene pesos de una dimensión (vector)';
}

/** Pinta un mapa de activación 2D normalizado (escala de grises) en un canvas nuevo. */
function canvasMapa(valores, alto, ancho, canales, canal, tamPx) {
  const c = document.createElement('canvas');
  c.width = ancho; c.height = alto;
  c.style.width = c.style.height = tamPx + 'px';
  const ctx = c.getContext('2d'), img = ctx.createImageData(ancho, alto);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < alto * ancho; i++) { const v = valores[i * canales + canal]; lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const r = hi - lo || 1;
  for (let i = 0; i < alto * ancho; i++) {
    const p = 255 - Math.round((valores[i * canales + canal] - lo) / r * 255);
    img.data[i * 4] = p; img.data[i * 4 + 1] = p; img.data[i * 4 + 2] = p; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  c.className = 'mapa';
  return c;
}
