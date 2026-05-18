import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============================================================================
# DATOS Y SIMULACION (tu codigo original)
# =============================================================================
rho = 7850  # kg/m3

# Barra 1
L1 = 2      # m
D1 = 0.5    # m
R1 = D1 / 2
m1 = rho * np.pi * R1**2 * L1

# Barra 2
L2 = 1      # m
D2 = 0.25   # m
R2 = D2 / 2
m2 = rho * np.pi * R2**2 * L2

# Posicion del centro de masa
r_CM1 = m2 * (R1 + L2 / 2) / (m1 + m2)
r_CM2 = R1 + L2 / 2 - m2 * (R1 + L2 / 2) / (m1 + m2)

# Tensor de inercia - Barra 1 en sistema 1
Ix1x1 = m1 / 12 * (L1**2 + 3 * R1**2)
Iy1y1 = Ix1x1
Iz1z1 = m1 * R1**2 / 2

# Tensor de inercia - Barra 2 en sistema 2
Ix2x2 = m2 / 12 * (L2**2 + 3 * R2**2)
Iy2y2 = Ix2x2
Iz2z2 = m2 * R2**2 / 2

# Traslado al CM
Ix1x1_CM = Iy1y1 + m1 * r_CM1**2
Iy1y1_CM = Ix1x1
Iz1z1_CM = Iz1z1 + m1 * r_CM1**2

Ix2x2_CM = Iy2y2 + m2 * r_CM2**2
Iy2y2_CM = Iz2z2
Iz2z2_CM = Ix2x2 + m2 * r_CM2**2

# Momentos principales de inercia totales
Ixx = Ix1x1_CM + Ix2x2_CM
Iyy = Iy1y1_CM + Iy2y2_CM
Izz = Iz1z1_CM + Iz2z2_CM

lista = sorted([Ixx, Iyy, Izz])
I1 = max([Ixx, Iyy, Izz])
I2 = lista[1]
I3 = min([Ixx, Iyy, Izz])

# Condiciones iniciales
w1_0 = 1e-3  # rad/s  (eje perpendicular: x)
w2_0 = 1.0   # rad/s  (eje horizontal: y, paralelo a barra 2)
w3_0 = 1e-3  # rad/s  (eje vertical: z, paralelo a barra 1)

w1_lista = [w1_0]
w2_lista = [w2_0]
w3_lista = [w3_0]
t_lista  = [0.0]

dt = 0.1   # s
tf = 60.0  # s

t = 0.0
i = 0
while t < tf:
    f1 = ((I3 - I2) / I1) * w2_lista[i] * w3_lista[i]
    w1 = w1_lista[i] + dt * f1

    f2 = ((I1 - I3) / I2) * w1_lista[i] * w3_lista[i]
    w2 = w2_lista[i] + dt * f2

    f3 = ((I2 - I1) / I3) * w2_lista[i] * w1_lista[i]
    w3 = w3_lista[i] + dt * f3

    w1_lista.append(w1)
    w2_lista.append(w2)
    w3_lista.append(w3)

    i += 1
    t += dt
    t_lista.append(t)

print(f"Simulacion terminada: {len(t_lista)} pasos, tf = {t_lista[-1]:.1f} s")

# =============================================================================
# INTEGRACION DE LA ORIENTACION (cuaterniones)
# Los ejes cuerpo son:
#   e1 → eje perpendicular (x)   corresponde a w1
#   e2 → eje horizontal   (y)   corresponde a w2  (paralelo a barra 2 inicial)
#   e3 → eje vertical     (z)   corresponde a w3  (paralelo a barra 1 inicial)
# =============================================================================

def quat_mult(q, p):
    """Producto de cuaterniones q*p. Formato: [w, x, y, z]."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = p
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])

def quat_rotate(q, v):
    """Rota el vector v con el cuaternion q. Devuelve vector rotado."""
    qv   = np.array([0.0, v[0], v[1], v[2]])
    qconj = np.array([q[0], -q[1], -q[2], -q[3]])
    res  = quat_mult(quat_mult(q, qv), qconj)
    return res[1:]

# Integrar cuaternion de orientacion frame a frame
q = np.array([1.0, 0.0, 0.0, 0.0])   # identidad
orientaciones = [q.copy()]

for i in range(len(t_lista) - 1):
    w1 = w1_lista[i]
    w2 = w2_lista[i]
    w3 = w3_lista[i]

    # dq/dt = 0.5 * q * [0, w1, w2, w3]
    omega_q = np.array([0.0, w1, w2, w3])
    dq = 0.5 * quat_mult(q, omega_q)
    q  = q + dt * dq
    q  = q / np.linalg.norm(q)   # renormalizar
    orientaciones.append(q.copy())

print(f"Orientaciones calculadas: {len(orientaciones)}")

# =============================================================================
# GEOMETRIA DE LA LLAVE EN T  (en el sistema cuerpo, centrada en CM)
# =============================================================================
# La barra 1 va a lo largo de e3 (z), la barra 2 a lo largo de e2 (y).
# El CM de la union queda desplazado r_CM1 hacia abajo del centro de la barra 1.

def cilindro_mesh(eje, centro, radio, longitud, n=30):
    """
    Genera la malla de superficie de un cilindro.
    eje      : 'x', 'y' o 'z'
    centro   : np.array([cx, cy, cz]) en el sistema cuerpo
    radio    : radio del cilindro
    longitud : longitud total del cilindro
    Devuelve X, Y, Z arrays para plot_surface.
    """
    theta = np.linspace(0, 2 * np.pi, n)
    s     = np.linspace(-longitud / 2, longitud / 2, 2)
    T, S  = np.meshgrid(theta, s)

    circ1 = radio * np.cos(T)
    circ2 = radio * np.sin(T)

    if eje == 'z':
        X = circ1 + centro[0]
        Y = circ2 + centro[1]
        Z = S      + centro[2]
    elif eje == 'y':
        X = circ1 + centro[0]
        Y = S      + centro[1]
        Z = circ2 + centro[2]
    elif eje == 'x':
        X = S      + centro[0]
        Y = circ1 + centro[1]
        Z = circ2 + centro[2]

    return X, Y, Z


# Centro de barra 1 en sistema cuerpo (eje z), desplazado de CM
c1_body = np.array([0.0, 0.0, r_CM1])        # CM de barra 1 respecto al CM global

# Centro de barra 2 en sistema cuerpo (eje y)
# La barra 2 está cruzando la barra 1 en su extremo superior
c2_body = np.array([0.0, 0.0, r_CM1 + L1/2 - r_CM1])   # simplificado abajo
# Distancia vertical desde CM global al eje de la barra 2
z_barra2 = L1 / 2 - r_CM1   # borde superior de barra 1 menos desplazamiento CM
c2_body  = np.array([0.0, 0.0, z_barra2])

print(f"CM global desplazado {r_CM1:.4f} m del centro de barra 1")
print(f"Centro barra 2 en z = {z_barra2:.4f} m (sistema cuerpo)")

# Mallas en sistema cuerpo (sin rotar)
X1b, Y1b, Z1b = cilindro_mesh('z', c1_body, R1, L1)
X2b, Y2b, Z2b = cilindro_mesh('y', c2_body, R2, L2)

def rotar_malla(Xb, Yb, Zb, q):
    """Aplica la rotacion q a cada punto de la malla."""
    sh = Xb.shape
    pts = np.stack([Xb.ravel(), Yb.ravel(), Zb.ravel()], axis=1)
    pts_r = np.array([quat_rotate(q, p) for p in pts])
    return (pts_r[:, 0].reshape(sh),
            pts_r[:, 1].reshape(sh),
            pts_r[:, 2].reshape(sh))

# =============================================================================
# ANIMACION
# =============================================================================
STEP    = 3          # cuantos pasos de simulacion por frame (velocidad visual)
N_frames = len(orientaciones) // STEP

fig = plt.figure(figsize=(9, 7), facecolor="#0a0a1a")
fig.suptitle("Rotacion libre — Llave en T (sin par externo)",
             color="white", fontsize=12, fontweight="bold", y=0.98)

ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0a0a1a")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#222244")

lim = 1.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_xlabel("X", color="#8888cc")
ax.set_ylabel("Y", color="#8888cc")
ax.set_zlabel("Z", color="#8888cc")
ax.tick_params(colors="#555577", labelsize=7)

# Ejes inerciales de referencia (fijos)
for vec, col, lbl in [([1,0,0],"#ff4444","X"),
                       ([0,1,0],"#44ff44","Y"),
                       ([0,0,1],"#4444ff","Z")]:
    ax.quiver(0,0,0, *[v*lim*0.6 for v in vec],
              color=col, linewidth=1, alpha=0.4, arrow_length_ratio=0.15)

# Texto de tiempo y velocidades
txt_t  = ax.text2D(0.02, 0.97, "", transform=ax.transAxes,
                   color="white", fontsize=9, va="top")
txt_w  = ax.text2D(0.02, 0.90, "", transform=ax.transAxes,
                   color="#aaccff", fontsize=8, va="top")

surf1 = [None]
surf2 = [None]

def init():
    return []

def update(frame):
    idx = frame * STEP
    q   = orientaciones[idx]

    # Rotar mallas
    X1, Y1, Z1 = rotar_malla(X1b, Y1b, Z1b, q)
    X2, Y2, Z2 = rotar_malla(X2b, Y2b, Z2b, q)

    # Limpiar superficies anteriores
    if surf1[0] is not None:
        surf1[0].remove()
    if surf2[0] is not None:
        surf2[0].remove()

    surf1[0] = ax.plot_surface(X1, Y1, Z1, color="steelblue",
                               alpha=0.85, linewidth=0, antialiased=True)
    surf2[0] = ax.plot_surface(X2, Y2, Z2, color="darkorange",
                               alpha=0.85, linewidth=0, antialiased=True)

    t_actual = t_lista[idx]
    w1v = w1_lista[idx]
    w2v = w2_lista[idx]
    w3v = w3_lista[idx]

    txt_t.set_text(f"t = {t_actual:.1f} s")
    txt_w.set_text(f"ω₁={w1v:.3f}  ω₂={w2v:.3f}  ω₃={w3v:.3f}  rad/s")

    return [surf1[0], surf2[0]]


ani = animation.FuncAnimation(
    fig, update,
    frames=N_frames,
    init_func=init,
    interval=40,      # ms entre frames (~25 fps visual)
    blit=False,
    repeat=True,
)

plt.tight_layout()
plt.show()