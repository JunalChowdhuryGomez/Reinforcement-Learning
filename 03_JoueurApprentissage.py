import numpy as np
import random

# Función para calcular la siguiente jugada basándose en aprendizaje por refuerzo
def JoueurApprentissage(Historique, epsilon, i):
    # Si es la primera jugada, elige aleatoriamente entre Pi, Fe, Ci
    if len(Historique) < 2:
        return random.choice(['Pi', 'Fe', 'Ci'])

    # Con probabilidad epsilon exploramos
    if random.random() < epsilon:
        return random.choice(['Pi', 'Fe', 'Ci'])

    # Explotación: basándonos en el historial, elegimos la mejor opción
    # Historial de las últimas jugadas (X_t-2, X_t-1) y (Y_t-2, Y_t-1)
    pase_reciente = Historique[-2:]  # Tomamos las dos últimas jugadas

    # Contabilizamos las jugadas del adversario cuando se ha jugado la misma secuencia
    freq_pi, freq_fe, freq_ci = 0, 0, 0
    for jugada in Historique[:-1]:
        if jugada[0] == pase_reciente[0][0] and jugada[1] == pase_reciente[0][1]:
            if jugada[1] == 'Pi':
                freq_pi += 1
            elif jugada[1] == 'Fe':
                freq_fe += 1
            else:
                freq_ci += 1

    # Tomamos la jugada con mayor frecuencia
    if freq_pi > freq_fe and freq_pi > freq_ci:
        return 'Fe'  # Contra Pi, lo mejor es jugar Fe
    elif freq_fe > freq_ci:
        return 'Ci'  # Contra Fe, lo mejor es jugar Ci
    else:
        return 'Pi'  # Contra Ci, lo mejor es jugar Pi

# Simulación de partidas
def SimularPartida(n, epsilon):
    Historique = []
    for t in range(n):
        jugador_X = JoueurApprentissage(Historique, epsilon, 0)
        jugador_Y = random.choice(['Pi', 'Fe', 'Ci'])  # El adversario juega aleatoriamente
        Historique.append([jugador_X, jugador_Y])
    return Historique

N_RONDAS = 1000
# Ejecutamos una simulación con 1000 rondas y epsilon = 0.1
historique = SimularPartida(N_RONDAS, 0.1)

# Imprimir las primeras jugadas
print(historique[:10])