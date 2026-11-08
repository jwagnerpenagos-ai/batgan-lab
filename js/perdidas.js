/* =============================================================================
 * perdidas.js — FUNCIONES DE PÉRDIDA de la GAN
 * =============================================================================
 * Todo está dentro de definirPerdidas() para que la página pueda mostrarlo.
 * ========================================================================== */

function definirPerdidas() {
  // ╔══════════════════════════════════════════════════════════════════════╗
  // ║                AQUÍ SE DEFINEN LAS FUNCIONES DE PÉRDIDA               ║
  // ╚══════════════════════════════════════════════════════════════════════╝
  // Notación:
  //   x        → dibujo real de murciélago
  //   z        → vector de ruido
  //   G(z)     → dibujo falso creado por el generador
  //   l = D(·) → logit que devuelve el discriminador (número sin acotar)
  //   σ(l)     → sigmoide: convierte el logit en probabilidad de "real" (0..1)
  //
  // Truco numérico: log σ(l) = −softplus(−l)   y   log(1 − σ(l)) = −softplus(l)
  // donde softplus(a) = log(1 + eᵃ). Así nunca calculamos log(0).
  //
  // Cada pérdida devuelve:
  //   perdidaD(logitsReales, logitsFalsos) → lo que el DISCRIMINADOR minimiza
  //   perdidaG(logitsFalsos)               → lo que el GENERADOR minimiza
  //   puntaje(logits)                      → valor que se muestra como "D(x)"

  const media = (t) => tf.mean(t);
  const sigmoide = (l) => tf.sigmoid(l);

  return {

    // ---------------------------------------------------------------------
    // 1) NO SATURANTE (la que se usa en la práctica, por defecto)
    // ---------------------------------------------------------------------
    // D: entropía cruzada binaria → quiere D(x)=1 para reales y D(G(z))=0 para falsos
    //    L_D = −E[log D(x)] − E[log(1 − D(G(z)))]
    // G: en vez de minimizar log(1 − D(G(z))), MAXIMIZA log D(G(z)):
    //    L_G = −E[log D(G(z))]
    //    Al inicio D gana fácil y D(G(z))≈0; con esta versión el gradiente de G
    //    sigue siendo grande, así G no se queda "estancado".
    no_saturante: {
      nombre: 'No saturante',
      perdidaD: (lr, lf) => tf.add(media(tf.softplus(tf.neg(lr))), media(tf.softplus(lf))),
      perdidaG: (lf) => media(tf.softplus(tf.neg(lf))),
      puntaje: sigmoide,
    },

    // ---------------------------------------------------------------------
    // 2) MINIMAX (la formulación original de Goodfellow et al., 2014)
    // ---------------------------------------------------------------------
    // min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]
    // D: igual que arriba.
    // G: minimiza directamente   L_G = E[log(1 − D(G(z)))]   (valor negativo)
    //    Problema: cuando D rechaza los falsos con seguridad, esta curva es
    //    plana ("se satura") y G casi no aprende. Pruébala y compara.
    minimax: {
      nombre: 'Minimax',
      perdidaD: (lr, lf) => tf.add(media(tf.softplus(tf.neg(lr))), media(tf.softplus(lf))),
      perdidaG: (lf) => tf.neg(media(tf.softplus(lf))),
      puntaje: sigmoide,
    },

    // ---------------------------------------------------------------------
    // 3) MÍNIMOS CUADRADOS — LSGAN (Mao et al., 2017)
    // ---------------------------------------------------------------------
    // Cambia la entropía cruzada por error cuadrático. D da un número y se le
    // pide que sea 1 para reales y 0 para falsos:
    //    L_D = ½ E[(D(x) − 1)²] + ½ E[D(G(z))²]
    //    L_G = ½ E[(D(G(z)) − 1)²]
    // Castiga también a los falsos que D ya clasifica bien pero están lejos de
    // los reales, lo que da gradientes más suaves.
    lsgan: {
      nombre: 'LSGAN',
      perdidaD: (lr, lf) => tf.add(
        tf.mul(0.5, media(tf.square(tf.sub(lr, 1)))),
        tf.mul(0.5, media(tf.square(lf)))),
      perdidaG: (lf) => tf.mul(0.5, media(tf.square(tf.sub(lf, 1)))),
      puntaje: (l) => l,
    },

    // ---------------------------------------------------------------------
    // 4) WASSERSTEIN — WGAN (Arjovsky et al., 2017)
    // ---------------------------------------------------------------------
    // D pasa a llamarse "crítico": no da probabilidades, da una puntuación.
    //    L_D = E[D(G(z))] − E[D(x)]     (separar puntuaciones reales y falsas)
    //    L_G = −E[D(G(z))]               (subir la puntuación de los falsos)
    // −L_D aproxima la distancia de Wasserstein ("cuánta tierra hay que mover"
    // para convertir la distribución falsa en la real). Para que funcione, D
    // debe ser 1-Lipschitz: aquí se logra recortando sus pesos a [−c, c].
    // Recomendado: RMSprop, tasa 0.00005 y 5 pasos de D por paso de G.
    wasserstein: {
      nombre: 'Wasserstein',
      perdidaD: (lr, lf) => tf.sub(media(lf), media(lr)),
      perdidaG: (lf) => tf.neg(media(lf)),
      puntaje: (l) => l,
      recortePesosD: 0.01,
    },
  };
}

/* Textos para la interfaz (fórmulas legibles y explicaciones) */
const INFO_PERDIDAS = {
  no_saturante: {
    formulaD: 'L<sub>D</sub> = −𝔼<sub>x</sub>[ log D(x) ] − 𝔼<sub>z</sub>[ log(1 − D(G(z))) ]',
    formulaG: 'L<sub>G</sub> = −𝔼<sub>z</sub>[ log D(G(z)) ]',
    explicacion: 'Entropía cruzada binaria. D quiere decir 1 a los reales y 0 a los falsos. G, en vez de minimizar log(1 − D(G(z))), maximiza log D(G(z)): así recibe gradientes fuertes incluso al inicio, cuando D lo descubre con facilidad. Es la versión que se usa en la práctica.',
    etiquetas: ['D(x) reales', 'D(G(z)) falsos'], rango: [0, 1], equilibrio: 0.5,
    nota: 'Equilibrio ideal: D(x) ≈ D(G(z)) ≈ 0.5 (D ya no distingue). L_D ≈ 2·ln2 ≈ 1.386 y L_G ≈ ln2 ≈ 0.693.',
  },
  minimax: {
    formulaD: 'L<sub>D</sub> = −𝔼<sub>x</sub>[ log D(x) ] − 𝔼<sub>z</sub>[ log(1 − D(G(z))) ]',
    formulaG: 'L<sub>G</sub> = 𝔼<sub>z</sub>[ log(1 − D(G(z))) ]',
    explicacion: 'Juego de suma cero del artículo original: min<sub>G</sub> max<sub>D</sub> V(D,G). G minimiza directamente la probabilidad de que D detecte sus falsos. Si D gana con facilidad, D(G(z)) ≈ 0 y la pérdida de G se aplana (satura): G aprende muy lento. Por eso su pérdida es negativa.',
    etiquetas: ['D(x) reales', 'D(G(z)) falsos'], rango: [0, 1], equilibrio: 0.5,
    nota: 'Compárala con la no saturante: suele tardar mucho más en producir murciélagos reconocibles.',
  },
  lsgan: {
    formulaD: 'L<sub>D</sub> = ½ 𝔼<sub>x</sub>[ (D(x) − 1)² ] + ½ 𝔼<sub>z</sub>[ D(G(z))² ]',
    formulaG: 'L<sub>G</sub> = ½ 𝔼<sub>z</sub>[ (D(G(z)) − 1)² ]',
    explicacion: 'Mínimos cuadrados: la salida de D es un número (sin sigmoide) y se le pide valer 1 en reales y 0 en falsos. A diferencia de la entropía cruzada, también castiga a los falsos que D ya clasifica bien pero que están lejos de los reales, lo que da gradientes más estables.',
    etiquetas: ['D(x) reales', 'D(G(z)) falsos'], rango: null, equilibrio: 0.5,
    nota: 'Equilibrio: D(x) ≈ D(G(z)) ≈ 0.5, L_D ≈ 0.25 y L_G ≈ 0.125.',
  },
  wasserstein: {
    formulaD: 'L<sub>D</sub> = 𝔼<sub>z</sub>[ D(G(z)) ] − 𝔼<sub>x</sub>[ D(x) ]',
    formulaG: 'L<sub>G</sub> = −𝔼<sub>z</sub>[ D(G(z)) ]',
    explicacion: 'D se vuelve un "crítico" que puntúa en vez de clasificar. −L<sub>D</sub> estima la distancia de Wasserstein entre dibujos reales y falsos (cuánto hay que "mover" una distribución para obtener la otra). Para que la estimación sea válida, los pesos de D se recortan a [−0.01, 0.01] después de cada paso.',
    etiquetas: ['D(x) reales (puntaje)', 'D(G(z)) falsos (puntaje)'], rango: null, equilibrio: null,
    nota: 'Al elegir WGAN se ajustan solos los hiperparámetros recomendados: RMSprop, tasa 0.00005 y 5 pasos de D por paso de G.',
  },
};
