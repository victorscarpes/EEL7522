import numpy as np
from matplotlib import pyplot as plt
from cmath import exp, pi, phase

j = complex(0, 1)

def x(t):
    """Função onda quadrada"""
    if 0 < t <= 10:
        return 1
    elif -10 < t <= 0:
        return -1
    elif t > 10:
        return x(t-20)
    else:
        return x(t+20)

# Parâmetros fundamentais (período e frequência e constante de tempo)
T0 = 20
w0 = 2*pi/T0
tau = 1

def a(k):
    """Coeficientes da série de Fourier de x(t)"""
    if k%2 == 0:
        return 0
    else:
        return 2/(j*k*pi)

def H(s):
    """Função de transferencia"""
    return 1/(tau*s + 1)

def yf(t, n):
    """Série truncada de Fourier de k=-n até k=n"""
    z = sum([H(j*k*w0)*a(k)*exp(j*k*w0*t) for k in range(-n, n+1)])
    return z.real

# Simulação de t = ti até t = tf com passo dt com N harmônicas
dt = 1.0e-3
ti = -30
tf = 30
N = 10

T = np.arange(ti, tf+dt, dt) # Vetor de tempo
X = [x(t) for t in T] # Vetor de x(t)
Yf = [yf(t, N) for t in T] # Vetor da série de Fourier truncada
K = list(range(-N, N+1)) # Vetor de k
A = [abs(H(j*k*w0)) for k in K] # Vetor de magnitude do filtro
P = [phase(H(j*k*w0)) for k in K] # Vetor de fase do filtro


plt.title(f"Filtro RC com τ = {tau}")
plt.plot(T, X, label="x(t)")
plt.plot(T, Yf, label=f"yₙ(t), n={N}")
plt.legend(loc="upper right")
plt.xlabel("t")
plt.show()

fig_frequencia, (ax_mag, ax_phas) = plt.subplots(2, sharex=True)
ax_mag.stem(K, A, "--.", basefmt="k")
ax_mag.set_title("Espectro de magnitude")
ax_mag.set_ylabel("|H(jkω₀)|")
ax_phas.stem(K, P, "--.", basefmt="k")
ax_phas.set_title("Espectro de fase")
ax_phas.set_ylabel("Arg{H(jkω₀)}")
plt.xlabel("k")
plt.xticks(range(-N+1-N%2, N+2-N%2, 2))
plt.show()