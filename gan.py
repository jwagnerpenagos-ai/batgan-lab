"""
gan.py — BatGAN: una GAN convolucional (DCGAN) que aprende a dibujar murciélagos
================================================================================

Proyecto de IA Generativa. Categoría usada del dataset Quick, Draw!: **bat**.

Este archivo contiene TODO el modelo, en este orden:

    1. Configuración
    2. Datos            → carga data/bat_28x28.npy (hecho con preparar_datos.py)
    3. RED GENERADORA      (class Generador)       ← aquí se crea G
    4. RED DISCRIMINADORA  (class Discriminador)   ← aquí se crea D
    5. FUNCIONES DE PÉRDIDA
    6. Instanciación de las redes y optimizadores  (G = Generador(), D = Discriminador())
    7. Entrenamiento
    8. Exportación de resultados para la página web (resultados/)

Uso:
    pip install torch numpy pillow
    python gan.py                       # 15 épocas con los valores por defecto
    python gan.py --epocas 30 --perdida lsgan

Al terminar, abre index.html: la sección "Resultado del entrenamiento" muestra
lo que se exportó en resultados/.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
p = argparse.ArgumentParser(description="Entrena una DCGAN con los murciélagos de Quick, Draw!")
p.add_argument("--datos", default=os.path.join("data", "bat_28x28.npy"))
p.add_argument("--epocas", type=int, default=15)
p.add_argument("--lote", type=int, default=64, help="tamaño de lote (batch size)")
p.add_argument("--dim-latente", type=int, default=100, help="tamaño del vector de ruido z")
p.add_argument("--filtros", type=int, default=32, help="filtros base de las convoluciones")
p.add_argument("--lr", type=float, default=2e-4, help="tasa de aprendizaje de G y D (Adam)")
p.add_argument("--beta1", type=float, default=0.5, help="β1 de Adam (0.5 es lo usual en GANs)")
p.add_argument("--perdida", default="no_saturante", choices=["no_saturante", "minimax", "lsgan"])
p.add_argument("--salida", default="resultados")
p.add_argument("--semilla", type=int, default=42)
args = p.parse_args()

torch.manual_seed(args.semilla)
np.random.seed(args.semilla)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"   # usa la GPU si existe
CATEGORIA = "bat"

# =============================================================================
# 2. DATOS
# =============================================================================
# Cada fila del .npy es un dibujo de 28x28 = 784 píxeles (0 = fondo, 255 = trazo).
# Se normalizan a [-1, 1] porque la salida del generador es tanh, cuyo rango es [-1, 1].
pixeles = np.load(args.datos)                                    # forma (N, 784), uint8
reales = torch.tensor(pixeles, dtype=torch.float32).view(-1, 1, 28, 28) / 127.5 - 1.0
N = len(reales)
cargador = torch.utils.data.DataLoader(reales, batch_size=args.lote, shuffle=True, drop_last=True)
print(f"Categoría: {CATEGORIA} · {N} dibujos de 28x28 · dispositivo: {dispositivo}")


def iniciar_pesos(capa):
    """Pesos iniciales ~ N(0, 0.02), como recomienda el artículo de DCGAN."""
    if isinstance(capa, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(capa.weight, 0.0, 0.02)
        if capa.bias is not None:
            nn.init.zeros_(capa.bias)
    elif isinstance(capa, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(capa.weight, 1.0, 0.02)
        nn.init.zeros_(capa.bias)


# =============================================================================
# 3. ╔══════════════════════════════════════════════════════════════════╗
#    ║          AQUÍ SE CREA LA RED NEURONAL GENERADORA  (G)             ║
#    ╚══════════════════════════════════════════════════════════════════╝
# =============================================================================
class Generador(nn.Module):
    """
    Transforma un vector de ruido z (dim_latente números ~ N(0,1)) en un dibujo 28x28.

        z [100]
          → Lineal → 7·7·64 neuronas → BatchNorm → LeakyReLU
          → se reacomoda como un "mapa" de 7×7 con 64 canales
          → ConvTranspuesta 5×5, paso 2 → 14×14×32 → BatchNorm → LeakyReLU
          → ConvTranspuesta 5×5, paso 2 → 28×28×1  → tanh  (píxeles en [-1, 1])

    La convolución transpuesta con paso 2 DUPLICA el ancho y el alto de la imagen:
    es la operación que "pinta" detalles a partir de un mapa pequeño.
    """

    def __init__(self, dim_latente=100, f=32):
        super().__init__()
        self.f = f
        # Capa densa: conecta cada número de z con 7·7·2f neuronas.
        # bias=False porque la BatchNorm siguiente ya resta la media.
        self.densa = nn.Linear(dim_latente, 7 * 7 * 2 * f, bias=False)
        self.norm1 = nn.BatchNorm1d(7 * 7 * 2 * f)          # estabiliza el entrenamiento
        # 7×7×2f → 14×14×f
        self.deconv1 = nn.ConvTranspose2d(2 * f, f, kernel_size=5, stride=2, padding=2,
                                          output_padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(f)
        # 14×14×f → 28×28×1 (imagen final en escala de grises)
        self.salida = nn.ConvTranspose2d(f, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z):
        x = F.leaky_relu(self.norm1(self.densa(z)), 0.2)     # LeakyReLU deja pasar un poco los negativos
        x = x.view(-1, 2 * self.f, 7, 7)                      # vector → mapa 7×7
        x = F.leaky_relu(self.norm2(self.deconv1(x)), 0.2)
        return torch.tanh(self.salida(x))                     # [-1, 1], igual que los datos reales


# =============================================================================
# 4. ╔══════════════════════════════════════════════════════════════════╗
#    ║        AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA  (D)           ║
#    ╚══════════════════════════════════════════════════════════════════╝
# =============================================================================
class Discriminador(nn.Module):
    """
    Recibe un dibujo de 28x28 y devuelve UN número (logit): alto = "parece real",
    bajo = "parece falso". Es un clasificador binario convolucional.

        imagen [1×28×28]
          → Conv 5×5, paso 2 → 32×14×14 → LeakyReLU → Dropout   (trazos y bordes)
          → Conv 5×5, paso 2 → 64×7×7   → LeakyReLU → Dropout   (alas, orejas, cuerpo)
          → Aplanar → Lineal → 1 logit

    No termina en sigmoide: la sigmoide va dentro de la función de pérdida
    (BCEWithLogits), que es numéricamente más estable. D(x) = σ(logit).
    """

    def __init__(self, f=32, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, f, kernel_size=5, stride=2, padding=2)        # 28 → 14
        self.conv2 = nn.Conv2d(f, 2 * f, kernel_size=5, stride=2, padding=2)    # 14 → 7
        self.dropout = nn.Dropout(dropout)   # apaga neuronas al azar para que D no gane demasiado rápido
        self.salida = nn.Linear(2 * f * 7 * 7, 1)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.conv1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.conv2(x), 0.2))
        return self.salida(x.flatten(1))                      # logit (sin sigmoide)


# =============================================================================
# 5. FUNCIONES DE PÉRDIDA
# =============================================================================
# Notación: x = dibujo real, z = ruido, G(z) = dibujo falso, D(·) = σ(logit) ∈ (0, 1)
#
# Discriminador (entropía cruzada binaria): quiere D(x) = 1 y D(G(z)) = 0
#     L_D = −E[log D(x)] − E[log(1 − D(G(z)))]
#
# Generador:
#   • minimax (artículo original de Goodfellow, 2014):  L_G = E[log(1 − D(G(z)))]
#     Problema: al inicio D rechaza los falsos con facilidad y esta curva se aplana
#     ("satura"), así que G casi no aprende.
#   • no saturante (la que se usa en la práctica):      L_G = −E[log D(G(z))]
#     Mismo objetivo, pero con gradientes fuertes cuando G lo está haciendo mal.
#   • LSGAN (mínimos cuadrados, Mao et al. 2017): usa error cuadrático sobre el logit
#     L_D = ½E[(D(x) − 1)²] + ½E[D(G(z))²]      L_G = ½E[(D(G(z)) − 1)²]
bce = nn.BCEWithLogitsLoss()      # aplica σ y luego la entropía cruzada, de forma estable


def perdida_discriminador(logits_reales, logits_falsos):
    if args.perdida == "lsgan":
        return 0.5 * ((logits_reales - 1) ** 2).mean() + 0.5 * (logits_falsos ** 2).mean()
    return (bce(logits_reales, torch.ones_like(logits_reales)) +      # reales → etiqueta 1
            bce(logits_falsos, torch.zeros_like(logits_falsos)))      # falsos → etiqueta 0


def perdida_generador(logits_falsos):
    if args.perdida == "lsgan":
        return 0.5 * ((logits_falsos - 1) ** 2).mean()
    if args.perdida == "minimax":
        return -bce(logits_falsos, torch.zeros_like(logits_falsos))   # = E[log(1 − D(G(z)))]
    return bce(logits_falsos, torch.ones_like(logits_falsos))         # = −E[log D(G(z))]


def opinion(logits):
    """Valor que se reporta como D(·): probabilidad de 'real' (o el puntaje crudo en LSGAN)."""
    return logits if args.perdida == "lsgan" else torch.sigmoid(logits)


# =============================================================================
# 6. INSTANCIACIÓN: aquí se construyen G y D y sus optimizadores
# =============================================================================
G = Generador(args.dim_latente, args.filtros).to(dispositivo)       # ← red generadora
D = Discriminador(args.filtros).to(dispositivo)                     # ← red discriminadora
G.apply(iniciar_pesos)
D.apply(iniciar_pesos)

# Adam con β1 = 0.5 (recomendación DCGAN): un optimizador independiente para cada red
opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


def resumen(modelo, entrada):
    """Tabla de capas: nombre, tipo, forma de salida y número de parámetros."""
    filas, ganchos = [], []

    def gancho(nombre):
        def f(mod, _, out):
            filas.append({"capa": nombre, "tipo": mod.__class__.__name__,
                          "salida": list(out.shape[1:]),
                          "parametros": sum(t.numel() for t in mod.parameters(recurse=False))})
        return f

    for nombre, mod in modelo.named_children():
        ganchos.append(mod.register_forward_hook(gancho(nombre)))
    modelo.eval()
    with torch.no_grad():
        modelo(entrada)
    modelo.train()
    for g in ganchos:
        g.remove()
    return filas


tabla_G = resumen(G, torch.randn(2, args.dim_latente, device=dispositivo))
tabla_D = resumen(D, torch.randn(2, 1, 28, 28, device=dispositivo))
for nombre, red, tabla in (("GENERADOR", G, tabla_G), ("DISCRIMINADOR", D, tabla_D)):
    print(f"\n{nombre}: {sum(t.numel() for t in red.parameters()):,} parámetros")
    for c in tabla:
        print(f"  {c['capa']:<10} {c['tipo']:<16} salida {str(c['salida']):<14} {c['parametros']:>9,} parámetros")

# =============================================================================
# 7. ENTRENAMIENTO
# =============================================================================
# En cada paso se alternan dos fases (como explica el curso de GANs de Google):
#   Fase 1: se actualiza D con G fijo  → aprende a separar reales de falsos
#   Fase 2: se actualiza G con D fijo  → aprende a engañar a D
os.makedirs(os.path.join(args.salida, "img"), exist_ok=True)
z_fijo = torch.randn(64, args.dim_latente, device=dispositivo)   # el mismo ruido en todas las épocas


def guardar_grilla(imagenes, ruta, columnas=8, escala=2):
    """Guarda una grilla de dibujos (trazo oscuro sobre fondo blanco)."""
    v = imagenes.detach().cpu().numpy().reshape(-1, 28, 28)
    filas = int(np.ceil(len(v) / columnas))
    lienzo = np.ones((filas * 28, columnas * 28), dtype=np.float32)
    for i, img in enumerate(v):
        f, c = divmod(i, columnas)
        lienzo[f * 28:(f + 1) * 28, c * 28:(c + 1) * 28] = 1 - (img + 1) / 2
    im = Image.fromarray((lienzo.clip(0, 1) * 255).astype(np.uint8))
    im.resize((im.width * escala, im.height * escala), Image.NEAREST).save(ruta)


def estadisticas_pesos(modelo):
    """Media, desviación, mínimo, máximo y una muestra de valores de cada tensor de pesos."""
    salida = {}
    for nombre, t in modelo.named_parameters():
        v = t.detach().cpu().flatten().numpy()
        muestra = v if len(v) <= 3000 else np.random.choice(v, 3000, replace=False)
        salida[nombre] = {"forma": list(t.shape), "media": float(v.mean()), "desv": float(v.std()),
                          "min": float(v.min()), "max": float(v.max()),
                          "muestra": [round(float(x), 5) for x in muestra]}
    return salida


guardar_grilla(reales[np.random.choice(N, 64, replace=False)], os.path.join(args.salida, "img", "reales.png"))
historia = {"paso": [], "perdida_D": [], "perdida_G": [], "D_x": [], "D_G_z": []}
por_epoca = []
paso, inicio = 0, time.time()

for epoca in range(1, args.epocas + 1):
    acum = np.zeros(4)
    for x in cargador:
        x = x.to(dispositivo)
        z = torch.randn(len(x), args.dim_latente, device=dispositivo)

        # ---------- FASE 1: entrenar el DISCRIMINADOR ----------
        falsos = G(z).detach()                       # detach: en esta fase G no se actualiza
        logits_reales, logits_falsos = D(x), D(falsos)
        loss_D = perdida_discriminador(logits_reales, logits_falsos)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---------- FASE 2: entrenar el GENERADOR ----------
        logits_falsos_G = D(G(z))                    # ahora sí fluye el gradiente hasta G
        loss_G = perdida_generador(logits_falsos_G)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()                                 # solo cambian los pesos de G

        # ---------- Valores para monitorear ----------
        valores = [loss_D.item(), loss_G.item(),
                   opinion(logits_reales).mean().item(), opinion(logits_falsos).mean().item()]
        acum += valores
        paso += 1
        if paso % 10 == 0:
            for k, v in zip(["perdida_D", "perdida_G", "D_x", "D_G_z"], valores):
                historia[k].append(round(v, 4))
            historia["paso"].append(paso)

    # ---------- Fin de época: promedio y dibujos del ruido fijo ----------
    m = acum / len(cargador)
    G.eval()
    with torch.no_grad():
        guardar_grilla(G(z_fijo), os.path.join(args.salida, "img", f"epoca_{epoca:02d}.png"))
    G.train()
    por_epoca.append({"epoca": epoca, "perdida_D": m[0], "perdida_G": m[1], "D_x": m[2], "D_G_z": m[3]})
    print(f"Época {epoca:2d}/{args.epocas}  L_D={m[0]:.3f}  L_G={m[1]:.3f}  "
          f"D(x)={m[2]:.2f}  D(G(z))={m[3]:.2f}  [{time.time() - inicio:.0f}s]", flush=True)

# =============================================================================
# 8. EXPORTAR RESULTADOS PARA LA PÁGINA WEB
# =============================================================================
G.eval()
with torch.no_grad():
    final = G(z_fijo)
    guardar_grilla(final, os.path.join(args.salida, "img", "final.png"), escala=3)
    opiniones_finales = opinion(D.eval()(final)).flatten().cpu().numpy().round(3).tolist()

resultados = {
    "categoria": CATEGORIA,
    "dibujos": N,
    "dispositivo": dispositivo,
    "segundos": round(time.time() - inicio, 1),
    "pasos": paso,
    "config": vars(args),
    "generador": {"capas": tabla_G, "total": sum(t.numel() for t in G.parameters()),
                  "pesos": estadisticas_pesos(G)},
    "discriminador": {"capas": tabla_D, "total": sum(t.numel() for t in D.parameters()),
                      "pesos": estadisticas_pesos(D)},
    "historia": historia,
    "por_epoca": por_epoca,
    "opiniones_finales": opiniones_finales,
}
# Se guarda como .js (no .json) para que la página funcione también abriéndola con doble clic
with open(os.path.join(args.salida, "resultados.js"), "w", encoding="utf-8") as f:
    f.write("window.RESULTADOS = " + json.dumps(resultados, ensure_ascii=False) + ";\n")
torch.save({"G": G.state_dict(), "D": D.state_dict()}, os.path.join(args.salida, "modelos.pt"))
print(f"\nListo: resultados en {args.salida}/ ({paso} pasos, {resultados['segundos']}s)")
