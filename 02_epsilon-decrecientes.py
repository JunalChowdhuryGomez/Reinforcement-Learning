# Función para simular el bandit problem con secuencias de epsilon decrecientes
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
n = 1000
P_A = 0.4
P_B = 0.6


def simulate_epsilon_decreasing(n, P_A, P_B, epsilon_func):
    X = np.zeros(n)
    clicks_A = 0
    clicks_B = 0
    counts_A = 0
    counts_B = 0

    for i in range(n):
        epsilon = epsilon_func(i + 1)  # Epsilon depende de i
        if i < 2:  # Exploración inicial
            E = 'A' if i == 0 else 'B'
        else:
            if np.random.rand() < epsilon:  # Exploración
                E = 'A' if np.random.rand() < 0.5 else 'B'
            else:  # Explotación
                E = 'A' if clicks_A / max(1, counts_A) > clicks_B / max(1, counts_B) else 'B'

        # Generar el clic basado en la elección
        if E == 'A':
            X[i] = 1 if np.random.rand() < P_A else 0
            clicks_A += X[i]
            counts_A += 1
        else:
            X[i] = 1 if np.random.rand() < P_B else 0
            clicks_B += X[i]
            counts_B += 1

    return np.cumsum(X) / np.arange(1, n+1)

# Funciones para epsilon decreciente
epsilon_funcs = [
    lambda i: 1 / i**2,
    lambda i: 1 / np.sqrt(i),
    lambda i: 1 / np.log(i + 1)  # Evitar log(0)
]

labels = ["1/i^2", "1/sqrt(i)", "1/log(i)"]

plt.figure(figsize=(10, 6))

for epsilon_func, label in zip(epsilon_funcs, labels):
    trajectory = simulate_epsilon_decreasing(n, P_A, P_B, epsilon_func)
    plt.plot(trajectory, label=label)

# Etiquetas y leyenda
plt.axhline(P_A, color='r', linestyle='--', label='P_A = 0.4')
plt.axhline(P_B, color='g', linestyle='--', label='P_B = 0.6')
plt.xlabel('Clientes')
plt.ylabel('Proporción de clics')
plt.legend()
plt.title('Simulación de estrategias con epsilon decreciente')
plt.show()
