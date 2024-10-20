import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import progressbar  # pip install progressbar2

# Definindo a estrutura da rede usando Keras
modelo = keras.Sequential([
    layers.Input(shape=(2,)),  # Usando um objeto Input
    layers.Dense(2, activation='sigmoid'),  # Primeira camada: 2 entradas, 2 saídas
    layers.Dense(1, activation='sigmoid')    # Segunda camada: 2 entradas, 1 saída                    # Segunda camada: 2 entradas, 1 saída
])

# Compilando o modelo
modelo.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2), loss='mean_squared_error')

# Dados de entrada e saída (AND)
X_int = np.array([[0, 0], 
                  [0, 1], 
                  [1, 0], 
                  [1, 1]], dtype=np.float32)
y_int = np.array([[0], 
                  [0], 
                  [0], 
                  [1]], dtype=np.float32)

# Treinar o modelo
epochs = 5000
batch_size=100
#modelo.fit(X_int, y_int, epochs=epochs, verbose=0)
with progressbar.ProgressBar(max_value=int(epochs)) as bar:
    for epoch in range(0,epochs,batch_size):
        modelo.fit(X_int, y_int, epochs=batch_size, batch_size=4, verbose=0)  # Treinando por uma época
        bar.update(epoch)

# Testar o modelo
for x in X_int:
    y_pred = modelo.predict(np.array([x]),verbose=0)
    y = y_pred[0][0]
    resultado = 1 if y >= 0.5 else 0
    print(f"Entrada: {x} -> {y} Saída Prevista: {resultado}")
