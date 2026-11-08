/* =============================================================================
 * app.js — Interfaz: conecta los controles con la GAN y actualiza las vistas
 * ========================================================================== */

const $ = (id) => document.getElementById(id);
const PERDIDAS = definirPerdidas();

const estado = {
  // datos
  pixeles: null, n: 0, datos: null, fuente: null,
  // modelos
  G: null, D: null, optG: null, optD: null, actG: null, actD: null,
  cfg: null, tipoPerdida: 'no_saturante',
  // entrenamiento
  paso: 0, entrenando: false,
  historia: { pasos: [], lD: [], lG: [], dx: [], dgz: [] },
  ultimo: null,
  // vectores de ruido fijos (para ver cómo cambian los mismos z con el tiempo)
  zFijo: null, zA: null, zB: null, zMuestra: null,
  capturas: [],
};

/* ---------------- Configuración desde la interfaz ---------------- */
function leerConfig() {
  return {
    dimLatente: +$('gDimLatente').value,
    filtrosG: +$('gFiltros').value,
    filtrosD: +$('dFiltros').value,
    dropout: +$('dDropout').value,
    tamLote: +$('tamLote').value,
    pasosD: +$('dPasos').value,
    optG: $('gOpt').value, lrG: +$('gLR').value, beta1G: +$('gBeta1').value,
    optD: $('dOpt').value, lrD: +$('dLR').value, beta1D: +$('dBeta1').value,
  };
}

function aviso(texto, ms = 4500) {
  const t = $('toast');
  t.textContent = texto; t.hidden = false;
  clearTimeout(aviso._t); aviso._t = setTimeout(() => (t.hidden = true), ms);
}

/* ---------------- Carga de datos ---------------- */
async function cargarFuente(blob, nombre) {
  const maximo = +$('selMuestras').value;
  $('estadoDatos').textContent = 'Leyendo ' + nombre + '…';
  $('estadoDatos').className = 'estado cargando';
  try {
    const r = await cargarDibujos(blob, nombre, maximo, (n, m) => {
      $('estadoDatos').textContent = `Dibujando trazos: ${n.toLocaleString('es')} / ${m.toLocaleString('es')}`;
    });
    if (!r.n) throw new Error('El archivo no tiene dibujos');
    estado.fuente = { blob, nombre };
    if (estado.datos) estado.datos.dispose();
    estado.pixeles = r.pixeles; estado.n = r.n;
    estado.datos = crearTensorDatos(r.pixeles, r.n);
    const total = r.total ? ` de ${r.total.toLocaleString('es')}` : '';
    $('estadoDatos').textContent = `✔ ${r.n.toLocaleString('es')}${total} murciélagos · ${nombre} · backend ${tf.getBackend()}`;
    $('estadoDatos').className = 'estado ok';
    dibujarReales();
    construirModelos();
  } catch (err) {
    console.error(err);
    $('estadoDatos').textContent = 'Error: ' + err.message;
    $('estadoDatos').className = 'estado error';
  }
}

function dibujarReales(opiniones) {
  if (!estado.pixeles) return;
  if (!estado.muestraReal) {
    // 64 dibujos reales al azar (fijos para poder comparar)
    const v = new Float32Array(64 * 784);
    for (let i = 0; i < 64; i++) {
      const k = Math.floor(Math.random() * estado.n);
      for (let j = 0; j < 784; j++) v[i * 784 + j] = estado.pixeles[k * 784 + j] / 127.5 - 1;
    }
    estado.muestraReal = v;
  }
  dibujarGrilla($('cvReales'), estado.muestraReal, 64, 8, opiniones);
  dibujarGrilla($('cvDiagReal'), estado.muestraReal, 1, 1);
}

/* ---------------- Crear / reiniciar las redes ---------------- */
function construirModelos() {
  pausar();
  // Primero los modelos auxiliares (comparten capas con G y D), después las redes
  ['actG', 'actD', 'G', 'D'].forEach(k => { try { if (estado[k]) estado[k].dispose(); } catch (e) { /* capa ya liberada */ } estado[k] = null; });
  ['zFijo', 'zA', 'zB', 'zMuestra'].forEach(k => estado[k] && estado[k].dispose());
  estado.cfg = leerConfig();
  const cfg = estado.cfg;

  // ===> Aquí se llaman las funciones que CREAN cada red neuronal <===
  estado.G = crearGenerador(cfg);       // js/generador.js
  estado.D = crearDiscriminador(cfg);   // js/discriminador.js

  // Modelos auxiliares que devuelven las salidas de TODAS las capas (para ver valores internos)
  estado.actG = tf.model({ inputs: estado.G.inputs, outputs: estado.G.layers.map(l => l.output) });
  estado.actD = tf.model({ inputs: estado.D.inputs, outputs: estado.D.layers.map(l => l.output) });

  crearOptimizadores();
  estado.perdida = PERDIDAS[estado.tipoPerdida];
  estado.zFijo = tf.randomNormal([64, cfg.dimLatente]);
  estado.zA = tf.randomNormal([1, cfg.dimLatente]);
  estado.zB = tf.randomNormal([1, cfg.dimLatente]);
  estado.zMuestra = tf.randomNormal([1, cfg.dimLatente]);
  estado.paso = 0; estado.ultimo = null; estado.capturas = [];
  estado.historia = { pasos: [], lD: [], lG: [], dx: [], dgz: [] };
  $('historial').innerHTML = '<p class="vacio">Aquí aparecen capturas cada 100 pasos.</p>';

  llenarTablaCapas(estado.G, $('tablaG'), $('gTotal'), $('gCapa'));
  llenarTablaCapas(estado.D, $('tablaD'), $('dTotal'), $('dCapa'));
  $('dgZ').textContent = `${cfg.dimLatente} valores`;
  $('dgG').textContent = `${estado.G.countParams().toLocaleString('es')} parámetros`;
  $('dgD').textContent = `${estado.D.countParams().toLocaleString('es')} parámetros`;
  ['btnEntrenar', 'btnPaso', 'btnReiniciar', 'btnMuestrasNuevas', 'btnNuevaMuestra'].forEach(id => ($(id).disabled = false));
  actualizarTodo();
}

function crearOptimizadores() {
  const c = estado.cfg;
  if (estado.optG) estado.optG.dispose();
  if (estado.optD) estado.optD.dispose();
  estado.optG = crearOptimizador(c.optG, c.lrG, c.beta1G);
  estado.optD = crearOptimizador(c.optD, c.lrD, c.beta1D);
}

const NOMBRES_CAPA = {
  Dense: 'Densa', BatchNormalization: 'Normalización por lotes', LeakyReLU: 'LeakyReLU',
  Reshape: 'Reshape', Conv2DTranspose: 'Conv. transpuesta', Conv2D: 'Convolución',
  Dropout: 'Dropout', Flatten: 'Aplanar',
};

function llenarTablaCapas(modelo, tabla, total, selector) {
  const filas = modelo.layers.map(l => {
    const forma = '[' + l.outputShape.slice(1).join('×') + ']';
    const extra = l.getConfig().activation && l.getConfig().activation !== 'linear' ? ` (${l.getConfig().activation})` : '';
    return `<tr><td class="mono">${l.name}</td><td>${NOMBRES_CAPA[l.getClassName()] || l.getClassName()}${extra}</td>` +
           `<td class="mono">${forma}</td><td class="cifra">${l.countParams().toLocaleString('es')}</td></tr>`;
  }).join('');
  tabla.innerHTML = '<tr><th>Capa</th><th>Tipo</th><th>Salida</th><th>Parámetros</th></tr>' + filas;
  total.textContent = `· ${modelo.countParams().toLocaleString('es')} parámetros`;
  // Selector de pesos: todas las variables de las capas que tienen pesos
  selector.innerHTML = '';
  modelo.layers.forEach(l => l.weights.forEach((w, i) => {
    const corto = (w.originalName || w.name).split('/').pop().replace(/_\d+$/, '');
    const o = document.createElement('option');
    o.value = `${l.name}|${i}`;
    o.textContent = `${l.name} · ${corto} [${w.shape.join('×')}]`;
    selector.appendChild(o);
  }));
}

/* ---------------- Entrenamiento ---------------- */
function entrenarUnPaso() {
  const m = pasoEntrenamiento({
    G: estado.G, D: estado.D, optG: estado.optG, optD: estado.optD,
    datos: estado.datos, n: estado.n, cfg: estado.cfg, perdida: estado.perdida,
  });
  estado.paso++;
  estado.ultimo = m;
  const h = estado.historia;
  h.pasos.push(estado.paso); h.lD.push(m.perdidaD); h.lG.push(m.perdidaG);
  h.dx.push(m.puntajeReal); h.dgz.push(m.puntajeFalso);
  if (h.pasos.length > 1500) {        // conservar la historia compacta: un punto de cada dos
    for (const k of Object.keys(h)) h[k] = h[k].filter((_, i) => i % 2 === 0);
  }
}

async function bucle() {
  while (estado.entrenando) {
    const porCuadro = +$('selVelocidad').value;
    try {
      for (let i = 0; i < porCuadro; i++) {
        entrenarUnPaso();
        const p = estado.paso;
        if (p % 10 === 0) actualizarResultado();
        if (p % 25 === 0) { actualizarPesos(); actualizarCurvas(); }
        if (p % 50 === 0) { actualizarRecorrido(); actualizarInterpolacion(); }
        if (p % 100 === 0) capturar();
      }
    } catch (err) {
      console.error(err); aviso('Error al entrenar: ' + err.message); pausar(); break;
    }
    actualizarMetricas();
    await tf.nextFrame();
  }
}

function entrenar() {
  if (estado.entrenando || !estado.G) return;
  estado.entrenando = true;
  $('btnEntrenar').textContent = '⏸'; $('btnEntrenar').classList.add('activo');
  bucle();
}
function pausar() {
  estado.entrenando = false;
  $('btnEntrenar').textContent = '▶'; $('btnEntrenar').classList.remove('activo');
  if (estado.G) actualizarTodo();
}

/* ---------------- Actualización de vistas ---------------- */
function actualizarTodo() {
  actualizarMetricas(); actualizarResultado(); actualizarPesos(); actualizarCurvas();
  actualizarRecorrido(); actualizarInterpolacion(); actualizarPerdidaUI(); mostrarCodigo();
}

/** Convierte el puntaje de D en t ∈ [0,1] (0 = "falso", 1 = "real") para colorear. */
function aOpinion(p) {
  if (estado.tipoPerdida === 'wasserstein') {
    const u = estado.ultimo;
    if (!u || Math.abs(u.puntajeReal - u.puntajeFalso) < 1e-6) return 0.5;
    return (p - u.puntajeFalso) / (u.puntajeReal - u.puntajeFalso) * 0.5 + 0.25;
  }
  return p;   // sigmoide (0..1) o LSGAN (objetivo 0..1)
}

function generarImagenes(z) {
  return tf.tidy(() => estado.G.apply(z, { training: false }));
}
function opinionesDe(imagenes) {
  const t = tf.tidy(() => estado.perdida.puntaje(estado.D.apply(imagenes, { training: false })));
  const valores = t.dataSync(); t.dispose();
  return valores;
}

function actualizarResultado() {
  if (!estado.G) return;
  const imgs = generarImagenes(estado.zFijo);
  const ops = Array.from(opinionesDe(imgs)).map(aOpinion);
  const valores = imgs.dataSync();
  dibujarGrilla($('cvFalsos'), valores, 64, 8, ops);
  dibujarGrilla($('cvDiagFalso'), valores, 1, 1);
  imgs.dispose();
  // Opinión de D sobre los reales mostrados arriba
  const reales = tf.tensor4d(estado.muestraReal, [64, 28, 28, 1]);
  dibujarReales(Array.from(opinionesDe(reales)).map(aOpinion));
  reales.dispose();
}

function actualizarMetricas() {
  $('txtPaso').textContent = estado.paso.toLocaleString('es');
  const epoca = estado.cfg ? estado.paso * estado.cfg.tamLote * estado.cfg.pasosD / Math.max(estado.n, 1) : 0;
  $('txtEpoca').textContent = epoca.toFixed(2);
  const u = estado.ultimo, f = (v) => (v === undefined || v === null ? '—' : v.toFixed(4));
  $('mLD').textContent = u ? f(u.perdidaD) : '—';
  $('mLG').textContent = u ? f(u.perdidaG) : '—';
  $('mDx').textContent = u ? f(u.puntajeReal) : '—';
  $('mDgz').textContent = u ? f(u.puntajeFalso) : '—';
  $('dgLD').textContent = 'L_D = ' + (u ? u.perdidaD.toFixed(3) : '—');
  $('dgLG').textContent = 'L_G = ' + (u ? u.perdidaG.toFixed(3) : '—');
}

function actualizarCurvas() {
  const h = estado.historia, info = INFO_PERDIDAS[estado.tipoPerdida];
  dibujarCurvas($('cvPerdidas'), h.pasos, [
    { nombre: 'Pérdida D', color: 'rgb(27,140,84)', valores: suavizar(h.lD) },
    { nombre: 'Pérdida G', color: 'rgb(123,50,148)', valores: suavizar(h.lG) },
  ]);
  dibujarCurvas($('cvPuntajes'), h.pasos, [
    { nombre: info.etiquetas[0], color: 'rgb(27,140,84)', valores: suavizar(h.dx) },
    { nombre: info.etiquetas[1], color: 'rgb(123,50,148)', valores: suavizar(h.dgz) },
  ], info.rango ? { lineaRef: info.equilibrio, minY: 0, maxY: 1 } : { lineaRef: info.equilibrio });
}

/** Media móvil exponencial para que las curvas se lean mejor. */
function suavizar(v, a = 0.9) {
  let s = v[0]; return v.map(x => (s = a * s + (1 - a) * x));
}

function actualizarPesos() {
  [['G', 'gCapa', 'cvHistG', 'gStats', 'cvFiltrosG', 'gFiltrosTxt', 'rgb(123,50,148)'],
   ['D', 'dCapa', 'cvHistD', 'dStats', 'cvFiltrosD', 'dFiltrosTxt', 'rgb(27,140,84)']].forEach(
    ([red, sel, hist, stats, filtros, txt, color]) => {
      const modelo = estado[red];
      if (!modelo || !$(sel).value) return;
      const [nombre, i] = $(sel).value.split('|');
      const variable = modelo.getLayer(nombre).weights[+i];
      const valores = variable.read().dataSync();
      let suma = 0, suma2 = 0, lo = Infinity, hi = -Infinity;
      for (const v of valores) { suma += v; suma2 += v * v; lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const media = suma / valores.length, desv = Math.sqrt(Math.max(0, suma2 / valores.length - media * media));
      $(stats).innerHTML = `n=${valores.length.toLocaleString('es')} · media ${media.toFixed(4)} · desv ${desv.toFixed(4)} · [${lo.toFixed(3)}, ${hi.toFixed(3)}]`;
      dibujarHistograma($(hist), valores, color);
      $(txt).textContent = dibujarFiltros($(filtros), valores, variable.shape);
    });
}

/** Muestra los valores internos (activaciones) de G y D para un z. */
function actualizarRecorrido() {
  if (!estado.G) return;
  const cont = $('recorrido');
  cont.innerHTML = '';
  const etapa = (titulo, forma, clase) => {
    const d = document.createElement('div'); d.className = 'etapa ' + (clase || '');
    d.innerHTML = `<div class="etapa-t">${titulo}</div><div class="etapa-f">${forma}</div><div class="mapas"></div>`;
    cont.appendChild(d); return d.querySelector('.mapas');
  };
  const flecha = () => { const s = document.createElement('div'); s.className = 'flechita'; s.textContent = '→'; cont.appendChild(s); };
  tf.tidy(() => {
    // vector z como cuadrícula
    const z = estado.zMuestra.dataSync(), lado = Math.ceil(Math.sqrt(z.length));
    const zp = new Float32Array(lado * lado); zp.set(z);
    etapa('Ruido z', `[${z.length}]`, 'ruido').appendChild(canvasMapa(zp, lado, lado, 1, 0, 64));
    const salidasG = estado.actG.predict(estado.zMuestra);
    const mostrarG = ['g_reshape', 'g_activ2', 'g_salida'];
    estado.G.layers.forEach((l, i) => {
      if (!mostrarG.includes(l.name)) return;
      flecha();
      const t = salidasG[i], [, h, w, c] = t.shape, v = t.dataSync();
      const m = etapa(l.name, `[${h}×${w}×${c}]`, 'g');
      for (let k = 0; k < Math.min(c, 8); k++) m.appendChild(canvasMapa(v, h, w, c, k, c === 1 ? 84 : 40));
    });
    flecha();
    const imagen = salidasG[salidasG.length - 1];
    const salidasD = estado.actD.predict(imagen);
    const mostrarD = ['d_activ1', 'd_activ2'];
    estado.D.layers.forEach((l, i) => {
      if (!mostrarD.includes(l.name)) return;
      const t = salidasD[i], [, h, w, c] = t.shape, v = t.dataSync();
      const m = etapa(l.name, `[${h}×${w}×${c}]`, 'd');
      for (let k = 0; k < Math.min(c, 8); k++) m.appendChild(canvasMapa(v, h, w, c, k, 40));
      flecha();
    });
    const logit = salidasD[salidasD.length - 1].dataSync()[0];
    const p = estado.perdida.puntaje(tf.scalar(logit)).dataSync()[0];
    const t = aOpinion(p);
    const m = etapa('d_salida', '[1]', 'd');
    m.innerHTML = `<div class="veredicto" style="border-color:${colorOpinion(t)}">
      <div>logit = <b>${logit.toFixed(3)}</b></div>
      <div>${['wasserstein', 'lsgan'].includes(estado.tipoPerdida) ? 'puntaje' : 'D = σ(logit)'} = <b>${p.toFixed(3)}</b></div>
      <div class="pill" style="background:${colorOpinion(t)}">${t >= 0.5 ? 'D cree: REAL' : 'D cree: FALSO'}</div></div>`;
  });
}

function actualizarInterpolacion() {
  if (!estado.G) return;
  const PASOS = 10;
  const valores = tf.tidy(() => {
    const t = tf.linspace(0, 1, PASOS).reshape([PASOS, 1]);
    const z = estado.zA.mul(tf.sub(1, t)).add(estado.zB.mul(t));   // z = (1−t)·zA + t·zB
    return estado.G.apply(z, { training: false }).dataSync();
  });
  dibujarGrilla($('cvInterp'), valores, PASOS, PASOS);
}

function capturar() {
  const cont = $('historial');
  if (!estado.capturas.length) cont.innerHTML = '';
  const valores = tf.tidy(() => estado.G.apply(estado.zFijo.slice([0, 0], [4, -1]), { training: false }).dataSync());
  const fig = document.createElement('figure');
  const c = document.createElement('canvas'); c.width = c.height = 72;
  dibujarGrilla(c, valores, 4, 2);
  fig.appendChild(c);
  const cap = document.createElement('figcaption'); cap.textContent = 'paso ' + estado.paso;
  fig.appendChild(cap);
  cont.appendChild(fig);
  estado.capturas.push(estado.paso);
  while (cont.children.length > 16) cont.removeChild(cont.children[1]);   // conserva la primera
}

/* ---------------- Panel de pérdidas ---------------- */
function actualizarPerdidaUI() {
  const info = INFO_PERDIDAS[estado.tipoPerdida];
  $('formulas').innerHTML = `<div class="f d">${info.formulaD}</div><div class="f g">${info.formulaG}</div>`;
  $('explicacionPerdida').innerHTML = info.explicacion;
  $('notaPerdida').textContent = info.nota;
  $('etqDx').textContent = info.etiquetas[0];
  $('etqDgz').textContent = info.etiquetas[1];
  $('dgTipo').textContent = PERDIDAS[estado.tipoPerdida].nombre.toLowerCase();
}

function cambiarPerdida() {
  const anterior = estado.tipoPerdida;
  estado.tipoPerdida = $('selPerdida').value;
  estado.perdida = PERDIDAS[estado.tipoPerdida];
  // WGAN funciona mal con Adam y tasas altas: aplicar los valores recomendados
  if (estado.tipoPerdida === 'wasserstein') {
    $('gOpt').value = $('dOpt').value = 'rmsprop';
    $('gLR').value = $('dLR').value = '0.00005';
    $('dPasos').value = '5';
  } else if (anterior === 'wasserstein') {
    $('gOpt').value = $('dOpt').value = 'adam';
    $('gLR').value = $('dLR').value = '0.0002';
    $('dPasos').value = '1';
  }
  if (estado.cfg) { Object.assign(estado.cfg, leerConfig()); crearOptimizadores(); }
  actualizarPerdidaUI(); actualizarCurvas();
  if (estado.paso > 0) aviso('Pérdida cambiada. Para compararla desde cero usa ↺ Reiniciar.');
}

/* ---------------- Código fuente en vivo ---------------- */
function mostrarCodigo(fn) {
  fn = fn || document.querySelector('#pestanasCodigo .activa').dataset.fn;
  const escapar = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const html = escapar(window[fn].toString()).split('\n').map(linea => {
    const i = linea.indexOf('//');
    let codigo = i >= 0 ? linea.slice(0, i) : linea;
    const comentario = i >= 0 ? `<span class="c">${linea.slice(i)}</span>` : '';
    codigo = codigo.replace(/\b(function|const|let|return|for|if|else|new)\b/g, '<span class="k">$1</span>')
                   .replace(/'([^']*)'/g, '<span class="s">\'$1\'</span>');
    return codigo + comentario;
  }).join('\n');
  $('codigo').innerHTML = html;
}

/* ---------------- Eventos ---------------- */
function conectarEventos() {
  $('btnEntrenar').onclick = () => (estado.entrenando ? pausar() : entrenar());
  $('btnPaso').onclick = () => { pausar(); entrenarUnPaso(); actualizarTodo(); };
  $('btnReiniciar').onclick = () => construirModelos();
  $('btnMuestrasNuevas').onclick = () => { estado.zFijo.dispose(); estado.zFijo = tf.randomNormal([64, estado.cfg.dimLatente]); actualizarResultado(); };
  $('btnNuevaMuestra').onclick = () => {
    estado.zMuestra.dispose(); estado.zMuestra = tf.randomNormal([1, estado.cfg.dimLatente]);
    estado.zA.dispose(); estado.zA = estado.zB; estado.zB = tf.randomNormal([1, estado.cfg.dimLatente]);
    actualizarRecorrido(); actualizarInterpolacion();
  };
  $('selPerdida').onchange = cambiarPerdida;
  $('gCapa').onchange = $('dCapa').onchange = actualizarPesos;

  // Hiperparámetros: los de arquitectura reinician las redes, los demás solo los optimizadores
  document.querySelectorAll('.params select').forEach(sel => {
    sel.onchange = () => {
      if (!estado.G) return;
      if (sel.dataset.reinicia) { construirModelos(); aviso('Arquitectura cambiada: las redes se crearon de nuevo.'); }
      else { Object.assign(estado.cfg, leerConfig()); crearOptimizadores(); }
    };
  });

  $('selMuestras').onchange = () => { if (estado.fuente) { estado.muestraReal = null; cargarFuente(estado.fuente.blob, estado.fuente.nombre); } };
  $('inputArchivo').onchange = (e) => { const f = e.target.files[0]; if (f) { estado.muestraReal = null; cargarFuente(f, f.name); } };
  const zona = $('zonaArchivo');
  zona.ondragover = (e) => { e.preventDefault(); zona.classList.add('encima'); };
  zona.ondragleave = () => zona.classList.remove('encima');
  zona.ondrop = (e) => {
    e.preventDefault(); zona.classList.remove('encima');
    const f = e.dataTransfer.files[0]; if (f) { estado.muestraReal = null; cargarFuente(f, f.name); }
  };
  $('btnDescargar').onclick = async () => {
    const url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/bat.npy';
    $('estadoDatos').textContent = 'Descargando bat.npy de Google (~95 MB)…';
    try { const b = await descargarComoBlob(url); estado.muestraReal = null; cargarFuente(b, 'bat.npy'); }
    catch (err) {
      $('estadoDatos').textContent = 'Sin datos cargados';
      aviso('El navegador bloqueó la descarga directa. Descarga bat.npy con el enlace y arrástralo aquí.', 7000);
    }
  };
  document.querySelectorAll('#pestanasCodigo button').forEach(b => b.onclick = () => {
    document.querySelectorAll('#pestanasCodigo button').forEach(x => x.classList.remove('activa'));
    b.classList.add('activa'); mostrarCodigo(b.dataset.fn);
  });
  document.querySelectorAll('[data-ir]').forEach(n => n.addEventListener('click', () => {
    const destino = $(n.dataset.ir);
    destino.scrollIntoView({ behavior: 'smooth', block: 'start' });
    destino.classList.add('resaltar'); setTimeout(() => destino.classList.remove('resaltar'), 1200);
  }));
}

/* ---------------- Inicio ---------------- */
async function iniciar() {
  conectarEventos();
  actualizarPerdidaUI();
  mostrarCodigo('crearGenerador');
  dibujarCurvas($('cvPerdidas'), [], []);
  dibujarCurvas($('cvPuntajes'), [], []);
  await tf.ready();
  // Intentar cargar automáticamente los datos preparados con preparar_datos.py
  // (si no existe el .npy, se intenta el mismo archivo con extensión .bin)
  try {
    let blob;
    try { blob = await descargarComoBlob('data/bat_28x28.npy'); }
    catch (e) { blob = await descargarComoBlob('data/bat_28x28.bin'); }
    await cargarFuente(blob, 'bat_28x28.npy');
  } catch (err) {
    $('estadoDatos').textContent = 'Sin datos: arrastra bat.npy o full_raw_bat.ndjson';
    $('estadoDatos').className = 'estado';
  }
}

if (typeof tf === 'undefined') {
  document.body.insertAdjacentHTML('afterbegin',
    '<div class="toast" style="position:static">No se pudo cargar TensorFlow.js. Revisa tu conexión a internet.</div>');
} else {
  iniciar();
}
