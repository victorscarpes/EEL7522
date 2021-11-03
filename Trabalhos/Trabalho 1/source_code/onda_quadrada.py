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

def a(k):
    """Coeficientes da série de Fourier exponencial"""
    if k%2 == 0:
        return 0
    else:
        return 2/(j*k*pi)

# Parâmetros fundamentais (período e frequência)
T0 = 20
w0 = 2*pi/T0
    
def xf(t, n):
    """Série truncada de Fourier de k=-n até k=n"""
    z = sum([a(k)*exp(j*k*w0*t) for k in range(-n, n+1)])
    return z.real

# Simulação de t = ti até t = tf com passo dt com N harmônicas
dt = 1.0e-2
ti = -30
tf = 30
N = 10

T = np.arange(ti, tf+dt, dt) # Vetor de tempo
X = [x(t) for t in T] # Vetor de x(t)
Xf = [xf(t, N) for t in T] # Vetor da série de Fourier truncada
dX = [xf(t, N) - x(t) for t in T] # Vetor do erro
dX2 = [(xf(t, N) - x(t))**2 for t in T] # Vetor do erro quadrático
dX2c = [(xf(t, N) - x(t))**2 for t in T if 0 < t <= T0] # Vetor do erro quadrático dentro de apenas um período
K = list(range(-N, N+1)) # Vetor de k
A = [abs(a(k)) for k in K] # Vetor de magnitude
P = [phase(a(k)) for k in K] # Vetor de fase

RMS = np.sqrt(sum(dX2c)*dt/T0) # Valor RMS do erro
SEQ = sum(dX2c) # Somatório dos erros quadráticos dentro de um período

fig_tempo, (ax_approx, ax_erro) = plt.subplots(2, sharex=True)
ax_approx.plot(T, X, label='x(t)')
ax_approx.plot(T, Xf, label=f'xₙ(t), n={N}')
ax_approx.set_title('Aproximação')
ax_approx.legend(loc='upper right')
ax_erro.plot(T, dX, label='xₙ(t) - x(t)', color='green')
ax_erro.plot(T, dX2, label='[xₙ(t) - x(t)]²', color='magenta')
ax_erro.set_title(f'Erro (RMS = {round(RMS, 3)} e SEQ = {round(SEQ, 3)})')
ax_erro.legend(loc='upper right')
plt.xlabel("t")
plt.show()

fig_frequencia, (ax_mag, ax_phas) = plt.subplots(2, sharex=True)
ax_mag.stem(K, A, "--.", basefmt="k")
ax_mag.set_title("Espectro de magnitude")
ax_mag.set_ylabel("|aₖ|")
ax_phas.stem(K, P, "--.", basefmt="k")
ax_phas.set_title("Espectro de fase")
ax_phas.set_ylabel(r"Arg{aₖ}")
plt.xlabel("k")
plt.xticks(range(-N+1-N%2, N+2-N%2, 2))
plt.show()
