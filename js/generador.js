/* =============================================================================
 * generador.js — RED NEURONAL GENERADORA (G)
 * =============================================================================
 * Ver la función crearGenerador() abajo. Los comentarios están dentro de la
 * función para que la sección "Código fuente" de la página también los muestre.
 * ========================================================================== */

function crearGenerador(cfg) {
  // ╔══════════════════════════════════════════════════════════════════════╗
  // ║          AQUÍ SE CREA LA RED NEURONAL GENERADORA  (G)                 ║
  // ╚══════════════════════════════════════════════════════════════════════╝
  // Objetivo: transformar un vector de ruido aleatorio z (por ejemplo, 100
  // números sacados de una normal N(0,1)) en una imagen de 28x28 que parezca
  // un murciélago dibujado por una persona.
  //
  // Arquitectura tipo DCGAN (Radford et al., 2015):
  //   z [dimLatente]
  //     → Densa → 7·7·(2·f) neuronas      (de "ruido" a un mapa pequeño)
  //     → Normalización por lotes + LeakyReLU
  //     → Reshape a 7×7×(2·f)              (ya tiene forma de imagen)
  //     → Conv2D transpuesta, paso 2  →  14×14×f   (duplica el tamaño)
  //     → Normalización por lotes + LeakyReLU
  //     → Conv2D transpuesta, paso 2  →  28×28×1   (imagen final)
  //     → tanh: deja cada píxel en [-1, 1], igual que los datos reales
  //
  // Parámetros de cfg:
  //   cfg.dimLatente → tamaño del vector de ruido z
  //   cfg.filtrosG   → número de filtros base f (más filtros = más capacidad)

  const f = cfg.filtrosG;
  const G = tf.sequential({ name: 'Generador' });

  // Capa 1 — Densa: cada uno de los dimLatente números de z se conecta con
  // 7·7·2f neuronas. Sin sesgo (bias) porque la normalización siguiente lo anula.
  G.add(tf.layers.dense({
    name: 'g_densa',
    inputShape: [cfg.dimLatente],
    units: 7 * 7 * f * 2,
    useBias: false,
    kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 }),
  }));
  // Normalización por lotes: estabiliza el entrenamiento (media 0, desviación 1)
  G.add(tf.layers.batchNormalization({ name: 'g_norm1', momentum: 0.9 }));
  // LeakyReLU: como ReLU pero deja pasar un poco los negativos (evita neuronas "muertas")
  G.add(tf.layers.leakyReLU({ name: 'g_activ1', alpha: 0.2 }));

  // Capa 2 — Reshape: el vector de 7·7·2f valores se acomoda como 7×7 con 2f canales
  G.add(tf.layers.reshape({ name: 'g_reshape', targetShape: [7, 7, f * 2] }));

  // Capa 3 — Convolución transpuesta ("deconvolución"): con strides=2 duplica
  // el ancho y el alto: 7×7 → 14×14. Aprende a "pintar" detalles.
  G.add(tf.layers.conv2dTranspose({
    name: 'g_deconv1',
    filters: f, kernelSize: 5, strides: 2, padding: 'same', useBias: false,
    kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 }),
  }));
  G.add(tf.layers.batchNormalization({ name: 'g_norm2', momentum: 0.9 }));
  G.add(tf.layers.leakyReLU({ name: 'g_activ2', alpha: 0.2 }));

  // Capa 4 — Salida: otra convolución transpuesta 14×14 → 28×28 con 1 canal
  // (escala de grises). tanh deja los píxeles en [-1, 1].
  G.add(tf.layers.conv2dTranspose({
    name: 'g_salida',
    filters: 1, kernelSize: 5, strides: 2, padding: 'same', activation: 'tanh',
    kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 }),
  }));

  return G;   // entrada: [lote, dimLatente]  →  salida: [lote, 28, 28, 1]
}
