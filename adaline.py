import numpy as np
import pylab
import matplotlib.pyplot as plt
import math

TAXA_APRENDIZAGEM = 0.3

# função de passo
def step(x):
    if (x > 0):
        return 1
    else:
        return -1;   

# Dados de entrada, incluindo bias
ENTRADAS = np.array([[-1,-1,1],
                   [-1,1,1],
                   [1,-1,1],
                   [1,1,1] ])
				   
# Dados de saída, que imprimem -1 se ambas entradas forem -1
SAIDAS = np.array([[-1,1,1,1]]).T

# Semear número "aleatório"
np.random.seed(1)

# Inicializar pesos
PESOS = 2 * np.random.random((3,1)) - 1
print ("Pesos antes do treinamento", PESOS)

# Lista de erros
erros = []

# Treinamento
for iterador in range(100):

    for itemEntrada, desejado in zip(ENTRADAS, SAIDAS):
               	
		# Semear a entrada para frente e calcular o Adaline
        SAIDA_ADALINE = (itemEntrada[0] * PESOS[0]) + (itemEntrada[1] * PESOS[1]) + (itemEntrada[2] * PESOS[2])

        # Executar um passo
        SAIDA_ADALINE = step(SAIDA_ADALINE)

        # Obter o erro e guardá-lo
        ERRO = desejado - SAIDA_ADALINE                
        erros.append(ERRO)
                
		# Calcular os pesos de acordo com a regra delta
        PESOS[0] = PESOS[0] + TAXA_APRENDIZAGEM * ERRO * itemEntrada[0]
        PESOS[1] = PESOS[1] + TAXA_APRENDIZAGEM * ERRO * itemEntrada[1]
        PESOS[2] = PESOS[2] + TAXA_APRENDIZAGEM * ERRO * itemEntrada[2]

print ("Pesos depois do treinamento", PESOS)

for itemEntrada, desejado in zip(ENTRADAS, SAIDAS):    
    
	SAIDA_ADALINE = (itemEntrada[0] * PESOS[0]) + (itemEntrada[1] * PESOS[1]) + (itemEntrada[2] * PESOS[2])   
    
	SAIDA_ADALINE = step(SAIDA_ADALINE)

	print ("Atual ", SAIDA_ADALINE, "Desejado ", desejado)

# Plotar erros
plt.plot(erros, c = '#aaaaff', label = 'Erros de treinamento')
plt.title("Erros Adaline (2,-2)")
plt.legend()
pylab.xlabel('Erro')
pylab.ylabel('Valor')
plt.show()

os.system("Pause")
