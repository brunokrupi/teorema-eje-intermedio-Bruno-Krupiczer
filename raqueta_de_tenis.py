import numpy as np
import matplotlib.pyplot as plt

# Datos
rho = 7850 #kg/m3
# Barra 1
L1 = 2 #m
D1 = 0.5 #m
R1 = D1/2
m1 = rho*np.pi*R1**2*L1
# Barra 2
L2 = 1 #m
D2 = 0.25 #m
R2 = D2/2
m2 = rho*np.pi*R2**2*L2

# Posicion del centro de masa desde sistema 1
r_CM1 = m2*(R1+L2/2)/(m1+m2)
# Posicion del centro de masa desde sistema 2
r_CM2 = R1+L2/2 - m2*(R1+L2/2)/(m1+m2)

#Tensor de inercia
# Barra 1 en sistema 1
Ix1x1 = m1/12*(L1**2+3*R1**2)
Iy1y1 = Ix1x1
Iz1z1 = m1*R1**2/2
# Barra 2 en sistema 2
Ix2x2 = m2/12*(L2**2+3*R2**2)
Iy2y2 = Ix2x2
Iz2z2 = m2*R2**2/2
# Barra 1 en CM
Ix1x1_CM = Iy1y1 + m1*r_CM1**2
Iy1y1_CM = Ix1x1
Iz1z1_CM = Iz1z1 + m1*r_CM1**2
# Barra 2 en CM
Ix2x2_CM = Iy2y2 + m2*r_CM2**2
Iy2y2_CM = Iz2z2
Iz2z2_CM = Ix2x2 + m2*r_CM2**2
# TOTAL en CM
Ixx = Ix1x1_CM + Ix2x2_CM
Iyy = Iy1y1_CM + Iy2y2_CM
Izz = Iz1z1_CM + Iz2z2_CM

# Momentos principales de inercia
lista = sorted([Ixx,Iyy,Izz])
I1 = max([Ixx,Iyy,Izz])
I2 = lista[1]
I3 = min([Ixx,Iyy,Izz])

# Condiciones iniciales
w1_0 = 1e-3 #rad/s
w2_0 = 1 #rad/s
w3_0 = 1e-3 #rad/s

# Listas de velocidades angulares
w1_lista = [w1_0]
w2_lista = [w2_0]
w3_lista = [w3_0]

#Listas de angulos
theta2_lista = [0]

t_lista = [0]
# Paso temporal
dt = 0.1 #s
# Tiempo final
tf = 60 #s

# Metodo de FOWARD-EULER
t = 0
i = 0
while t<tf:
    
    f1 = ((I3-I2)/I1) * w2_lista[i]*w3_lista[i]
    w1 = w1_lista[i] + dt*f1
    
    f2 = ((I1-I3)/I2) * w1_lista[i]*w3_lista[i]
    w2 = w2_lista[i] + dt*f2

    f3 = ((I2-I1)/I3) * w2_lista[i]*w1_lista[i]
    w3 = w3_lista[i] + dt*f3
    
    theta2 = dt*w2 + theta2_lista[i]
    theta2_lista.append(theta2)
    
    w1_lista.append(w1)
    w2_lista.append(w2)
    w3_lista.append(w3)
    
    i += 1
    t += dt
    t_lista.append(t)

# Ventana temporal entre cambios de orientacion consecutivos
w3_lista = np.array(w3_lista)
idx1 = np.argmin(w3_lista)
idx2 = np.argmax(w3_lista)
tmin = t_lista[idx1]
tmax = t_lista[idx2]
DT = tmax - tmin

# Velocidad angular en eje 2
w2_lista = np.array(w2_lista)
w2_cte = abs(np.min(w2_lista))

# Vueltas entre cambios de orientacion consecutivos (METODO APROXIMADO)
Dtheta = w2_cte*DT
N = Dtheta/(2*np.pi)
print(f"Vueltas entre cambios de orientacion consecutivos (APROX): {round(N,1)}")

# Angulo en funcion del tiempo
theta2 = np.array(theta2_lista)
# Frecuencia
f2 = theta2/(2*np.pi)
# Numero de vueltas entre cambios de orientacion consecutivos
N = np.max(f2)-np.min(f2)
print(f"Vueltas entre cambios de orientacion consecutivos: {round(N,1)}")

# Graficos de las velocidades angulares
#plt.plot(t_lista,w1_lista,label=r'$\omega_1$',linestyle='-.', color='limegreen')
plt.plot(t_lista,w2_lista,label=r'$\omega_2$',linestyle='--', color='darkcyan')
plt.plot(t_lista,w3_lista,label=r'$\omega_3$',linewidth=2, color='orangered')

plt.xlabel('Tiempo, t [s]')
plt.ylabel(r'Velocidad Angular, $\omega$ [rad/s]')
plt.grid(color='darkgray')
plt.legend()
plt.title('Velocidades Angulares')
plt.xlim(0,tf)
plt.show()

plt.plot(t_lista,f2,label=r'$n_2$', color='darkcyan')
plt.xlabel('Tiempo, t [s]')
plt.ylabel(r'Numero de vueltas, $n_2$ ')
plt.grid(color='darkgray')
plt.legend()
plt.title('Numero de vueltas en el eje 2')
plt.xlim(0,tf)
plt.show()