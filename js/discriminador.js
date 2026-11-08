/* =============================================================================
 * discriminador.js — RED NEURONAL DISCRIMINADORA (D)
 * =============================================================================
 * Ver la función crearDiscriminador() abajo. Los comentarios están dentro de la
 * función para que la sección "Código fuente" de la página también los muestre.
 * ========================================================================== */

function crearDiscriminador(cfg) {
  // ╔══════════════════════════════════════════════════════════════════════╗
  // ║        AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA  (D)               ║
  // ╚══════════════════════════════════════════════════════════════════════╝
  // Objetivo: recibir una imagen de 28x28 y devolver UN número que indica qué
  // tan "real" le parece (un murciélago dibujado por una persona) o qué tan
  // "falsa" (un murciélago inventado por el generador).
  //
  // Es un clasificador binario convolucional:
  //   imagen [28×28×1]
  //     → Conv2D, paso 2  → 14×14×f    (detecta trazos y bordes)
  //     → LeakyReLU + Dropout
  //     → Conv2D, paso 2  →  7×7×2f    (detecta partes: alas, orejas, cuerpo)
  //     → LeakyReLU + Dropout
  //     → Flatten → Densa(1)           (un solo número: el "logit")
  //
  // OJO: la salida NO tiene sigmoide. Es un logit (−∞, +∞). La sigmoide se
  // aplica dentro de la función de pérdida cuando corresponde (ver perdidas.js),
  // porque así es numéricamente más estable y además permite usar pérdidas
  // como LSGAN o Wasserstein, que necesitan la salida "cruda".
  //
  // Parámetros de cfg:
  //   cfg.filtrosD → número de filtros base f
  //   cfg.dropout  → probabilidad de apagar neuronas al entrenar (regularización)

  const f = cfg.filtrosD;
  const D = tf.sequential({ name: 'Discriminador' });

  // Capa 1 — Convolución: 5×5 con paso 2, reduce 28×28 → 14×14.
  // Sin normalización por lotes en la primera capa (recomendación DCGAN).
  D.add(tf.layers.conv2d({
    name: 'd_conv1',
    inputShape: [28, 28, 1],
    filters: f, kernelSize: 5, strides: 2, padding: 'same',
    kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 }),
  }));
  D.add(tf.layers.leakyReLU({ name: 'd_activ1', alpha: 0.2 }));
  // Dropout: apaga al azar una fracción de neuronas durante el entrenamiento
  // para que D no se vuelva demasiado fuerte demasiado rápido.
  D.add(tf.layers.dropout({ name: 'd_dropout1', rate: cfg.dropout }));

  // Capa 2 — Convolución: 14×14 → 7×7, con el doble de filtros
  D.add(tf.layers.conv2d({
    name: 'd_conv2',
    filters: f * 2, kernelSize: 5, strides: 2, padding: 'same',
    kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 }),
  }));
  D.add(tf.layers.leakyReLU({ name: 'd_activ2', alpha: 0.2 }));
  D.add(tf.layers.dropout({ name: 'd_dropout2', rate: cfg.dropout }));

  // Capa 3 — Aplanar: 7×7×2f valores → un vector
  D.add(tf.layers.flatten({ name: 'd_aplanar' }));

  // Capa 4 — Salida: una sola neurona. Logit alto = "parece real",
  // logit bajo = "parece falso".
  D.add(tf.layers.dense({ name: 'd_salida', units: 1 }));

  return D;   // entrada: [lote, 28, 28, 1]  →  salida: [lote, 1]
}
