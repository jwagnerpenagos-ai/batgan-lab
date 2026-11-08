/* =============================================================================
 * entrenamiento.js — Optimizadores y BUCLE DE ENTRENAMIENTO adversario
 * ========================================================================== */

/** Crea el optimizador elegido en la interfaz (Adam, RMSprop o SGD). */
function crearOptimizador(tipo, tasa, beta1) {
  if (tipo === 'rmsprop') return tf.train.rmsprop(tasa);
  if (tipo === 'sgd') return tf.train.sgd(tasa);
  return tf.train.adam(tasa, beta1, 0.999);   // Adam con β1=0.5 es lo típico en GANs
}

/** Toma un lote aleatorio de dibujos reales del tensor de datos. */
function loteReal(datos, n, tamLote) {
  const indices = new Int32Array(tamLote);
  for (let i = 0; i < tamLote; i++) indices[i] = Math.floor(Math.random() * n);
  return tf.tidy(() => tf.gather(datos, tf.tensor1d(indices, 'int32')));
}

function pasoEntrenamiento(e) {
  // ╔══════════════════════════════════════════════════════════════════════╗
  // ║              UN PASO DE ENTRENAMIENTO DE LA GAN                       ║
  // ╚══════════════════════════════════════════════════════════════════════╝
  // e = { G, D, optG, optD, datos, n, cfg, perdida }
  // Cada paso tiene dos fases que se alternan, como en el curso de Google:
  //   FASE 1: se entrena D (G queda congelado)  → aprende a separar real/falso
  //   FASE 2: se entrena G (D queda congelado)  → aprende a engañar a D
  const { G, D, cfg, perdida } = e;
  const varsD = D.trainableWeights.map(w => w.val);   // pesos que cambia la fase 1
  const varsG = G.trainableWeights.map(w => w.val);   // pesos que cambia la fase 2
  // Los valores de cada fase se guardan como tensores y se leen de la GPU
  // UNA sola vez al final (leer de la GPU es lo más lento de todo el paso).
  let perdidaD = null, puntajeReal = null, puntajeFalso = null;

  // ---------------- FASE 1: entrenar el DISCRIMINADOR ----------------
  for (let k = 0; k < cfg.pasosD; k++) {
    tf.tidy(() => {
      const reales = loteReal(e.datos, e.n, cfg.tamLote);           // x ~ datos reales
      const z = tf.randomNormal([cfg.tamLote, cfg.dimLatente]);      // z ~ N(0, 1)
      const falsos = G.apply(z, { training: true });                 // G(z), sin actualizar G
      let pr, pf;
      const costo = e.optD.minimize(() => {
        const logitsReales = D.apply(reales, { training: true });    // D(x)
        const logitsFalsos = D.apply(falsos, { training: true });    // D(G(z))
        // tf.keep: minimize() limpia los tensores intermedios; estos los necesitamos después
        pr = tf.keep(tf.mean(perdida.puntaje(logitsReales)));
        pf = tf.keep(tf.mean(perdida.puntaje(logitsFalsos)));
        return perdida.perdidaD(logitsReales, logitsFalsos);         // L_D
      }, true, varsD);                                               // ← solo pesos de D
      [perdidaD, puntajeReal, puntajeFalso].forEach(t => t && t.dispose());
      perdidaD = tf.keep(costo); puntajeReal = tf.keep(pr); puntajeFalso = tf.keep(pf);

      // Solo WGAN: recortar los pesos de D para mantenerlo 1-Lipschitz
      if (perdida.recortePesosD) {
        const c = perdida.recortePesosD;
        varsD.forEach(v => v.assign(tf.clipByValue(v, -c, c)));
      }
    });
  }

  // ---------------- FASE 2: entrenar el GENERADOR ----------------
  const perdidaG = tf.tidy(() => {
    const z = tf.randomNormal([cfg.tamLote, cfg.dimLatente]);
    return e.optG.minimize(() => {
      const falsos = G.apply(z, { training: true });                 // G(z)
      const logitsFalsos = D.apply(falsos, { training: true });      // D(G(z))
      return perdida.perdidaG(logitsFalsos);                         // L_G
    }, true, varsG);                                                 // ← solo pesos de G
  });

  // Una sola lectura GPU → CPU con los 4 valores que la interfaz grafica
  const todos = [perdidaD, perdidaG, puntajeReal, puntajeFalso];
  const apilados = tf.tidy(() => tf.stack(todos.map(t => t.reshape([]))));
  const v = apilados.dataSync();
  apilados.dispose(); todos.forEach(t => t.dispose());
  return { perdidaD: v[0], perdidaG: v[1], puntajeReal: v[2], puntajeFalso: v[3] };
}
