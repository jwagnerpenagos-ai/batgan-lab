/* =============================================================================
 * resultados.js — Muestra lo que exportó gan.py (resultados/resultados.js)
 * =============================================================================
 * gan.py deja en window.RESULTADOS: tablas de capas, estadísticas de los pesos,
 * historia de pérdidas, valores por época, e imágenes en resultados/img/.
 * ========================================================================== */

(function mostrarResultadosPython() {
  const R = window.RESULTADOS;
  const $ = (id) => document.getElementById(id);
  if (!R) { $('pyVacio').hidden = false; $('pyContenido').hidden = true; $('pyMeta').hidden = true; return; }

  const nombrePerdida = { no_saturante: 'no saturante', minimax: 'minimax', lsgan: 'LSGAN' }[R.config.perdida] || R.config.perdida;
  const min = Math.floor(R.segundos / 60), seg = Math.round(R.segundos % 60);
  $('pyMeta').textContent = `${R.dibujos.toLocaleString('es')} dibujos · ${R.por_epoca.length} épocas · ` +
    `${R.pasos.toLocaleString('es')} pasos · pérdida ${nombrePerdida} · ${min} min ${seg} s en ${R.dispositivo.toUpperCase()}`;

  // ---------- Imágenes: reales y evolución por época ----------
  const imgEpoca = (e) => `resultados/img/epoca_${String(e).padStart(2, '0')}.png`;
  $('pyReales').src = 'resultados/img/reales.png';
  const slider = $('pySlider'), total = R.por_epoca.length;
  slider.max = total; slider.value = total;
  const verEpoca = (e) => { $('pyEpoca').src = imgEpoca(e); $('pyEpocaTxt').textContent = `${e} de ${total}`; };
  slider.oninput = () => verEpoca(+slider.value);
  verEpoca(total);
  let animando = null;
  $('pyAnimar').onclick = () => {
    if (animando) { clearInterval(animando); animando = null; $('pyAnimar').textContent = '▶ Animar épocas'; return; }
    let e = 1; $('pyAnimar').textContent = '⏸ Detener';
    animando = setInterval(() => {
      slider.value = e; verEpoca(e);
      if (++e > total) { clearInterval(animando); animando = null; $('pyAnimar').textContent = '▶ Animar épocas'; }
    }, 450);
  };

  // ---------- Gráficas de pérdidas y de D(x), D(G(z)) ----------
  const h = R.historia, s = (v) => suavizar(v, 0.95);
  dibujarCurvas($('pyPerdidas'), h.paso, [
    { nombre: 'Pérdida D', color: 'rgb(27,140,84)', valores: s(h.perdida_D) },
    { nombre: 'Pérdida G', color: 'rgb(123,50,148)', valores: s(h.perdida_G) },
  ]);
  dibujarCurvas($('pyPuntajes'), h.paso, [
    { nombre: 'D(x) reales', color: 'rgb(27,140,84)', valores: s(h.D_x) },
    { nombre: 'D(G(z)) falsos', color: 'rgb(123,50,148)', valores: s(h.D_G_z) },
  ], R.config.perdida === 'lsgan' ? { lineaRef: 0.5 } : { lineaRef: 0.5, minY: 0, maxY: 1 });

  // ---------- Tabla de valores por época ----------
  const f = (v) => v.toFixed(3);
  $('pyTablaEpocas').innerHTML = '<tr><th>Época</th><th>Pérdida D</th><th>Pérdida G</th><th>D(x)</th><th>D(G(z))</th></tr>' +
    R.por_epoca.map(e => `<tr data-epoca="${e.epoca}"><td>${e.epoca}</td><td class="cifra">${f(e.perdida_D)}</td>` +
      `<td class="cifra">${f(e.perdida_G)}</td><td class="cifra">${f(e.D_x)}</td><td class="cifra">${f(e.D_G_z)}</td></tr>`).join('');
  $('pyTablaEpocas').querySelectorAll('tr[data-epoca]').forEach(tr =>
    tr.onclick = () => { slider.value = tr.dataset.epoca; verEpoca(+tr.dataset.epoca); });

  // ---------- Arquitectura y valores de los pesos de cada red ----------
  const TIPOS = { Linear: 'Densa', BatchNorm1d: 'Normalización por lotes', BatchNorm2d: 'Normalización por lotes',
    ConvTranspose2d: 'Conv. transpuesta', Conv2d: 'Convolución', Dropout: 'Dropout' };
  [['G', R.generador, 'rgb(123,50,148)'], ['D', R.discriminador, 'rgb(27,140,84)']].forEach(([red, datos, color]) => {
    $('pyTotal' + red).textContent = `${datos.total.toLocaleString('es')} parámetros`;
    $('pyCapas' + red).innerHTML = '<tr><th>Capa</th><th>Tipo</th><th>Salida</th><th>Parámetros</th></tr>' +
      datos.capas.map(c => `<tr><td class="mono">${c.capa}</td><td>${TIPOS[c.tipo] || c.tipo}</td>` +
        `<td class="mono">[${c.salida.join('×')}]</td><td class="cifra">${c.parametros.toLocaleString('es')}</td></tr>`).join('');
    const nombres = Object.keys(datos.pesos);
    const tabla = $('pyPesos' + red);
    tabla.innerHTML = '<tr><th>Tensor</th><th>Forma</th><th>Media</th><th>Desv.</th><th>Mín</th><th>Máx</th></tr>' +
      nombres.map(n => {
        const p = datos.pesos[n];
        return `<tr data-n="${n}"><td class="mono">${n}</td><td class="mono">[${p.forma.join('×')}]</td>` +
          `<td class="cifra">${p.media.toFixed(4)}</td><td class="cifra">${p.desv.toFixed(4)}</td>` +
          `<td class="cifra">${p.min.toFixed(3)}</td><td class="cifra">${p.max.toFixed(3)}</td></tr>`;
      }).join('');
    const elegir = (n) => {
      tabla.querySelectorAll('tr').forEach(tr => tr.classList.toggle('elegida', tr.dataset.n === n));
      dibujarHistograma($('pyHist' + red), datos.pesos[n].muestra, color);
    };
    tabla.querySelectorAll('tr[data-n]').forEach(tr => tr.onclick = () => elegir(tr.dataset.n));
    elegir(nombres[0]);
  });
})();

/* ---------- Pestañas: #resultado (por defecto) y #laboratorio ---------- */
(function pestanas() {
  const VISTAS = ['resultado', 'laboratorio'];
  function mostrar() {
    const actual = VISTAS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'resultado';
    VISTAS.forEach(v => { document.getElementById('vista-' + v).hidden = v !== actual; });
    document.querySelectorAll('.pestanas-vista a').forEach(a => a.classList.toggle('activa', a.dataset.vista === actual));
    document.body.dataset.vista = actual;   // el CSS oculta los controles de entrenamiento fuera del laboratorio
    // Al salir del laboratorio se pausa el entrenamiento en vivo
    if (actual !== 'laboratorio' && typeof pausar === 'function' && typeof estado !== 'undefined' && estado.entrenando) pausar();
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', mostrar);
  mostrar();
})();
