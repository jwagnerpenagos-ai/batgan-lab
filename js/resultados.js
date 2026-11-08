/* =============================================================================
 * resultados.js — PESTAÑA "🦇 RESULTADO": muestra lo que produjo gan.py
 * =============================================================================
 * Flujo completo:
 *   1. gan.py entrena la GAN en Python (PyTorch).
 *   2. Al terminar, gan.py exporta todo a la carpeta resultados/:
 *        - resultados/resultados.js → define window.RESULTADOS con los números
 *        - resultados/img/*.png      → dibujos reales y generados por época
 *   3. index.html carga resultados/resultados.js y después ESTE archivo,
 *      que lee window.RESULTADOS y llena la pestaña.
 *
 * Qué trae window.RESULTADOS (lo arma la sección 8 de gan.py):
 *   categoria, dibujos, pasos, segundos, dispositivo → datos generales
 *   config         → hiperparámetros usados (épocas, lote, tasa, pérdida…)
 *   historia       → pérdidas y D(x), D(G(z)) cada 10 pasos (para las gráficas)
 *   por_epoca      → promedios de cada época (para la tabla)
 *   generador      → { capas, total, pesos } de la red generadora
 *   discriminador  → { capas, total, pesos } de la red discriminadora
 *
 * Funciones que se reutilizan de otros archivos:
 *   dibujarCurvas, dibujarHistograma → js/visualizacion.js
 *   suavizar                         → js/app.js
 * ========================================================================== */

(function mostrarResultadosPython() {
  const R = window.RESULTADOS;                         // datos exportados por gan.py
  const $ = (id) => document.getElementById(id);      // atajo para buscar elementos

  // Si no existe resultados/resultados.js (aún no se ha corrido gan.py),
  // se muestra un aviso con la instrucción y se esconde el resto de la sección.
  if (!R) { $('pyVacio').hidden = false; $('pyContenido').hidden = true; $('pyMeta').hidden = true; return; }

  // ---------------------------------------------------------------------------
  // 1. RESUMEN DEL ENTRENAMIENTO (etiqueta verde arriba a la derecha)
  // ---------------------------------------------------------------------------
  // Ej.: "20.000 dibujos · 15 épocas · 4680 pasos · pérdida no saturante · 4 min 53 s en CPU"
  const nombrePerdida = { no_saturante: 'no saturante', minimax: 'minimax', lsgan: 'LSGAN' }[R.config.perdida] || R.config.perdida;
  const min = Math.floor(R.segundos / 60), seg = Math.round(R.segundos % 60);
  $('pyMeta').textContent = `${R.dibujos.toLocaleString('es')} dibujos · ${R.por_epoca.length} épocas · ` +
    `${R.pasos.toLocaleString('es')} pasos · pérdida ${nombrePerdida} · ${min} min ${seg} s en ${R.dispositivo.toUpperCase()}`;

  // ---------------------------------------------------------------------------
  // 2. IMÁGENES: dibujos reales vs. dibujos generados en cada época
  // ---------------------------------------------------------------------------
  // gan.py usa SIEMPRE el mismo vector de ruido (z_fijo) para generar las 64
  // imágenes de cada época. Así, al mover el deslizador se ve cómo el mismo z
  // pasa de ser ruido a parecerse a un murciélago: es el aprendizaje de G.
  const imgEpoca = (e) => `resultados/img/epoca_${String(e).padStart(2, '0')}.png`;   // epoca_01.png, epoca_02.png…
  $('pyReales').src = 'resultados/img/reales.png';    // 64 dibujos reales al azar, para comparar

  const slider = $('pySlider'), total = R.por_epoca.length;
  slider.max = total; slider.value = total;           // el deslizador arranca en la última época

  // Cambia la imagen y el texto "Época X de N"
  const verEpoca = (e) => { $('pyEpoca').src = imgEpoca(e); $('pyEpocaTxt').textContent = `${e} de ${total}`; };
  slider.oninput = () => verEpoca(+slider.value);
  verEpoca(total);

  // Botón "▶ Animar épocas": recorre las épocas de la 1 a la última cada 450 ms.
  // Si se pulsa mientras anima, se detiene.
  let animando = null;
  $('pyAnimar').onclick = () => {
    if (animando) { clearInterval(animando); animando = null; $('pyAnimar').textContent = '▶ Animar épocas'; return; }
    let e = 1; $('pyAnimar').textContent = '⏸ Detener';
    animando = setInterval(() => {
      slider.value = e; verEpoca(e);
      if (++e > total) { clearInterval(animando); animando = null; $('pyAnimar').textContent = '▶ Animar épocas'; }
    }, 450);
  };

  // ---------------------------------------------------------------------------
  // 3. GRÁFICAS: funciones de pérdida y valores del discriminador
  // ---------------------------------------------------------------------------
  // Las curvas crudas oscilan mucho de un paso a otro, así que se suavizan con
  // una media móvil exponencial (factor 0.95) para que se lea la tendencia.
  const h = R.historia, s = (v) => suavizar(v, 0.95);

  // Gráfica izquierda: pérdida de D (verde) y pérdida de G (morado).
  // Si una de las dos se fuera a 0 y la otra se disparara, una red le estaría
  // ganando a la otra. Que ambas se estabilicen indica un entrenamiento sano.
  dibujarCurvas($('pyPerdidas'), h.paso, [
    { nombre: 'Pérdida D', color: 'rgb(27,140,84)', valores: s(h.perdida_D) },
    { nombre: 'Pérdida G', color: 'rgb(123,50,148)', valores: s(h.perdida_G) },
  ]);

  // Gráfica derecha: D(x) = probabilidad que D le da a los dibujos REALES y
  // D(G(z)) = la que le da a los FALSOS. La línea punteada en 0.5 es el
  // equilibrio ideal: D ya no distingue unos de otros.
  // (En LSGAN la salida de D no es una probabilidad, por eso no se fija el rango 0–1.)
  dibujarCurvas($('pyPuntajes'), h.paso, [
    { nombre: 'D(x) reales', color: 'rgb(27,140,84)', valores: s(h.D_x) },
    { nombre: 'D(G(z)) falsos', color: 'rgb(123,50,148)', valores: s(h.D_G_z) },
  ], R.config.perdida === 'lsgan' ? { lineaRef: 0.5 } : { lineaRef: 0.5, minY: 0, maxY: 1 });

  // ---------------------------------------------------------------------------
  // 4. TABLA DE VALORES POR ÉPOCA (plegable)
  // ---------------------------------------------------------------------------
  // Cada fila es el promedio de la época: pérdida D, pérdida G, D(x) y D(G(z)).
  // Al hacer clic en una fila se muestran los dibujos de esa época.
  const f = (v) => v.toFixed(3);                      // 3 decimales
  $('pyTablaEpocas').innerHTML = '<tr><th>Época</th><th>Pérdida D</th><th>Pérdida G</th><th>D(x)</th><th>D(G(z))</th></tr>' +
    R.por_epoca.map(e => `<tr data-epoca="${e.epoca}"><td>${e.epoca}</td><td class="cifra">${f(e.perdida_D)}</td>` +
      `<td class="cifra">${f(e.perdida_G)}</td><td class="cifra">${f(e.D_x)}</td><td class="cifra">${f(e.D_G_z)}</td></tr>`).join('');
  $('pyTablaEpocas').querySelectorAll('tr[data-epoca]').forEach(tr =>
    tr.onclick = () => { slider.value = tr.dataset.epoca; verEpoca(+tr.dataset.epoca); });

  // ---------------------------------------------------------------------------
  // 5. PARÁMETROS DE LAS REDES: arquitectura y valores de los pesos
  // ---------------------------------------------------------------------------
  // Nombres en español para los tipos de capa de PyTorch
  const TIPOS = { Linear: 'Densa', BatchNorm1d: 'Normalización por lotes', BatchNorm2d: 'Normalización por lotes',
    ConvTranspose2d: 'Conv. transpuesta', Conv2d: 'Convolución', Dropout: 'Dropout' };

  // Se hace lo mismo para las dos redes: G en morado y D en verde
  [['G', R.generador, 'rgb(123,50,148)'], ['D', R.discriminador, 'rgb(27,140,84)']].forEach(([red, datos, color]) => {

    // 5a. Total de parámetros entrenables de la red (ej. "371.937 parámetros")
    $('pyTotal' + red).textContent = `${datos.total.toLocaleString('es')} parámetros`;

    // 5b. Tabla de capas: nombre, tipo, forma de la salida y parámetros de cada capa.
    //     Esta tabla la generó gan.py pasando una entrada de prueba por la red.
    $('pyCapas' + red).innerHTML = '<tr><th>Capa</th><th>Tipo</th><th>Salida</th><th>Parámetros</th></tr>' +
      datos.capas.map(c => `<tr><td class="mono">${c.capa}</td><td>${TIPOS[c.tipo] || c.tipo}</td>` +
        `<td class="mono">[${c.salida.join('×')}]</td><td class="cifra">${c.parametros.toLocaleString('es')}</td></tr>`).join('');

    // 5c. Valores de los pesos AL FINAL del entrenamiento. Cada tensor es una
    //     matriz de pesos (weight) o un vector de sesgos (bias). Para cada uno:
    //       media, desviación estándar, mínimo y máximo de todos sus valores.
    //     Ej.: los pesos empiezan ~ N(0, 0.02); si la desviación crece, la red
    //     se alejó de la inicialización al aprender.
    const nombres = Object.keys(datos.pesos);
    const tabla = $('pyPesos' + red);
    tabla.innerHTML = '<tr><th>Tensor</th><th>Forma</th><th>Media</th><th>Desv.</th><th>Mín</th><th>Máx</th></tr>' +
      nombres.map(n => {
        const p = datos.pesos[n];
        return `<tr data-n="${n}"><td class="mono">${n}</td><td class="mono">[${p.forma.join('×')}]</td>` +
          `<td class="cifra">${p.media.toFixed(4)}</td><td class="cifra">${p.desv.toFixed(4)}</td>` +
          `<td class="cifra">${p.min.toFixed(3)}</td><td class="cifra">${p.max.toFixed(3)}</td></tr>`;
      }).join('');

    // 5d. Histograma: al hacer clic en una fila se dibuja la distribución de los
    //     valores de ese tensor (gan.py guarda una muestra de hasta 3 000 valores).
    const elegir = (n) => {
      tabla.querySelectorAll('tr').forEach(tr => tr.classList.toggle('elegida', tr.dataset.n === n));   // resalta la fila
      dibujarHistograma($('pyHist' + red), datos.pesos[n].muestra, color);
    };
    tabla.querySelectorAll('tr[data-n]').forEach(tr => tr.onclick = () => elegir(tr.dataset.n));
    elegir(nombres[0]);                                // al abrir, muestra el primer tensor
  });
})();

/* =============================================================================
 * PESTAÑAS: "🦇 Resultado" (#resultado, por defecto) y "🧪 Laboratorio" (#laboratorio)
 * =============================================================================
 * La pestaña activa se guarda en el # de la URL, así se puede compartir el link
 * directo a cada una: .../batgan-lab/#resultado o .../batgan-lab/#laboratorio
 * ========================================================================== */
(function pestanas() {
  const VISTAS = ['resultado', 'laboratorio'];

  function mostrar() {
    // Lee el # de la URL; si no es una vista conocida, abre "resultado"
    const actual = VISTAS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'resultado';
    // Muestra solo el contenedor de la vista activa (#vista-resultado o #vista-laboratorio)
    VISTAS.forEach(v => { document.getElementById('vista-' + v).hidden = v !== actual; });
    // Marca en amarillo la pestaña activa de la barra superior
    document.querySelectorAll('.pestanas-vista a').forEach(a => a.classList.toggle('activa', a.dataset.vista === actual));
    // El CSS usa este atributo para ocultar los controles de entrenamiento (▶, Paso +1…)
    // cuando no se está en el laboratorio
    document.body.dataset.vista = actual;
    // Al salir del laboratorio se pausa el entrenamiento en vivo (pausar y estado vienen de app.js)
    if (actual !== 'laboratorio' && typeof pausar === 'function' && typeof estado !== 'undefined' && estado.entrenando) pausar();
    window.scrollTo(0, 0);                             // cada pestaña empieza desde arriba
  }

  window.addEventListener('hashchange', mostrar);      // se ejecuta al hacer clic en una pestaña
  mostrar();                                           // y una vez al cargar la página
})();
