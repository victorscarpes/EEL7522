import numpy as np
from numpy import pi, cos
from matplotlib import pyplot as plt


def x(t):
    """Função dente de serra"""
    if 0 < t <= 2:
        return t/2
    elif t > 2:
        return x(t-2)
    else:
        return x(t+2)

# Parâmetros fundamentais (período e frequência)
T0 = 2
w0 = 2*pi/T0



def A(k):
    """Coeficientes das componentes cossenoidais da série de Fourier"""
    if k==0:
        return 1/2
    else:
        return 1/(k*pi)


def phi(k):
    """Fases das componentes cossenoidais da série de Fourier"""
    if k==0:
        return 0
    else:
        return pi/2


def xf(t, n):
    """Série truncada de Fourier até o termo n"""
    seq = [A(k)*cos(k*w0*t + phi(k)) for k in range(0, n+1)]
    return sum(seq)

# Simulação de t = ti até t = tf com passo dt com N harmônicas
dt = 1.0e-3
ti = -3
tf = 3
N = 2

T = np.arange(ti, tf+dt, dt) # Vetor de tempo
X = [x(t) for t in T] # Vetor de x(t)
Xf = [xf(t, N) for t in T] # Vetor da série de Fourier truncada
dX = [xf(t, N) - x(t) for t in T] # Vetor do erro
dX2 = [(xf(t, N) - x(t))**2 for t in T] # Vetor do erro quadrático
dX2c = [(xf(t, N) - x(t))**2 for t in T if 0 < t <= T0] # Vetor do erro quadrático dentro de apenas um período

RMS = np.sqrt(sum(dX2c)*dt/T0) # Valor RMS do erro
SEQ = sum(dX2c) # Somatório dos erros quadráticos dentro de um período

fig1, (ax_approx, ax_erro) = plt.subplots(2, sharex=True)
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