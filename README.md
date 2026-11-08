# 🦇 BatGAN Lab

**Una Red Generativa Adversaria (GAN) interactiva que aprende a dibujar murciélagos, entrenada en vivo en el navegador.**

Proyecto de IA Generativa, inspirado en [GAN Lab](https://poloclub.github.io/ganlab/) (Georgia Tech + Google PAIR). La diferencia es que, en vez de puntos en 2D, esta GAN genera **dibujos reales de 28×28 píxeles** usando el dataset [Quick, Draw!](https://quickdraw.withgoogle.com/data) de Google.

> **Categoría usada: `bat` (murciélago)**: 20 000 dibujos del archivo RAW `full_raw_bat.ndjson`.

---

## ¿Qué muestra la página?

| Sección | Qué se ve |
|---|---|
| **0 · Datos reales** | Muestra de los murciélagos de Quick, Draw! y cuántos se usan para entrenar. |
| **1 · Vista general** | Diagrama del flujo: ruido z → Generador → imagen falsa → Discriminador ← imagen real → pérdidas → gradientes. |
| **2 · Red Generadora** | Hiperparámetros editables, tabla de capas con la salida y los parámetros de cada una, histograma de **valores de los pesos** y sus filtros. |
| **3 · Red Discriminadora** | Lo mismo para D: arquitectura, parámetros, valores de los pesos y filtros convolucionales 5×5. |
| **4 · Funciones de pérdida** | 4 pérdidas a elegir (no saturante, minimax, LSGAN y Wasserstein) con su fórmula y explicación; **valores en vivo** de L_D, L_G, D(x) y D(G(z)) y sus gráficas. |
| **5 · Recorrido de una muestra** | Los valores internos (activaciones) de cada capa cuando un z atraviesa G y luego D, hasta el veredicto final de D. |
| **6 · Resultado de la GAN** | 64 murciélagos generados (con borde verde o morado según la opinión de D), la evolución cada 100 pasos y una interpolación en el espacio latente. |
| **7 · Código fuente** | El código que se está ejecutando, con comentarios, leído en vivo desde la página. |

## ¿Dónde se crea cada red neuronal?

| Componente | Archivo | Función |
|---|---|---|
| **Red Generadora (G)** | [`js/generador.js`](js/generador.js) | `crearGenerador(cfg)` |
| **Red Discriminadora (D)** | [`js/discriminador.js`](js/discriminador.js) | `crearDiscriminador(cfg)` |
| **Funciones de pérdida** | [`js/perdidas.js`](js/perdidas.js) | `definirPerdidas()` |
| **Bucle de entrenamiento** | [`js/entrenamiento.js`](js/entrenamiento.js) | `pasoEntrenamiento(e)` |
| Llamado a las funciones que crean G y D | [`js/app.js`](js/app.js) | `construirModelos()` |

Cada bloque donde se crea una red está marcado en el código con un recuadro `AQUÍ SE CREA LA RED NEURONAL ...`.

### Arquitectura (tipo DCGAN)

**Generador**: `z [100] → Densa 3136 → BatchNorm → LeakyReLU → Reshape 7×7×64 → ConvTranspuesta 5×5 /2 → 14×14×32 → BatchNorm → LeakyReLU → ConvTranspuesta 5×5 /2 → 28×28×1 (tanh)`, con unos 378 mil parámetros.

**Discriminador**: `28×28×1 → Conv 5×5 /2 → 14×14×32 → LeakyReLU → Dropout → Conv 5×5 /2 → 7×7×64 → LeakyReLU → Dropout → Aplanar → Densa 1 (logit)`, con unos 55 mil parámetros.

### Funciones de pérdida

| Pérdida | Discriminador | Generador |
|---|---|---|
| No saturante (por defecto) | −E[log D(x)] − E[log(1 − D(G(z)))] | −E[log D(G(z))] |
| Minimax (original) | igual | E[log(1 − D(G(z)))] |
| LSGAN | ½E[(D(x)−1)²] + ½E[D(G(z))²] | ½E[(D(G(z))−1)²] |
| Wasserstein | E[D(G(z))] − E[D(x)] (+ recorte de pesos) | −E[D(G(z))] |

## Cómo ejecutarlo

La página es estática (HTML + CSS + JavaScript con [TensorFlow.js](https://www.tensorflow.org/js)) y entrena usando la GPU del navegador. **No hay que instalar nada para la página**, pero hay que abrirla desde un servidor (abrirla con doble clic bloquea la carga de `data/`):

```bash
# Opción 1: Python
python -m http.server 8000
# luego abrir http://localhost:8000

# Opción 2: VS Code → extensión "Live Server" → "Open with Live Server"
```

**Opción 3: GitHub Pages.** En el repo, ve a *Settings → Pages → Branch: main / root*, y en un minuto la página queda publicada en `https://<usuario>.github.io/<repo>/`.

Si la abres con doble clic, igual funciona: arrastra `data/bat_28x28.npy` (o el `.ndjson` RAW) a la zona de datos.

### Regenerar los datos

`data/bat_28x28.npy` se genera desde el archivo RAW de Quick, Draw!:

```bash
pip install numpy pillow
python preparar_datos.py --entrada full_raw_bat.ndjson --cantidad 20000
# sin --entrada, descarga el RAW directamente de Google
```

El script toma cada dibujo (trazos x, y, t), lo centra y escala en un lienzo de 256×256, dibuja los trazos y lo reduce a 28×28. Solo usa dibujos que el juego reconoció como "bat".

## Consejos para experimentar

- Con la pérdida **no saturante**, a los ~1 000–3 000 pasos ya aparecen siluetas con alas.
- Prueba **minimax** desde cero (↺ Reiniciar) y compara: G aprende mucho más lento porque su gradiente se satura.
- Sube la tasa de aprendizaje de D a 0.001 y mira cómo D(x) → 1 y D(G(z)) → 0: D "gana" y G deja de aprender.
- Baja la dimensión latente a 16 y busca **mode collapse**, es decir, que muchos murciélagos salgan iguales.

## Estructura

```
batgan-lab/
├── index.html            # página principal
├── css/estilos.css
├── js/
│   ├── datos.js          # carga .npy / .ndjson y normalización a [-1, 1]
│   ├── generador.js      # ← RED GENERADORA
│   ├── discriminador.js  # ← RED DISCRIMINADORA
│   ├── perdidas.js       # ← FUNCIONES DE PÉRDIDA
│   ├── entrenamiento.js  # optimizadores y paso de entrenamiento
│   ├── visualizacion.js  # grillas, curvas, histogramas y filtros en canvas
│   └── app.js            # interfaz
├── data/bat_28x28.npy    # 20 000 murciélagos 28×28 (uint8)
└── preparar_datos.py     # RAW .ndjson → .npy
```

## Créditos

- Datos: [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset), Google, licencia CC BY 4.0.
- Inspiración visual: [GAN Lab](https://poloclub.github.io/ganlab/), Kahng et al., Georgia Tech y Google PAIR.
- Teoría: [Curso de Introducción a las GAN, Google for Developers](https://developers.google.com/machine-learning/gan?hl=es-419).
- Goodfellow et al. (2014), *Generative Adversarial Nets*; Radford et al. (2015), *DCGAN*; Mao et al. (2017), *LSGAN*; Arjovsky et al. (2017), *Wasserstein GAN*.
