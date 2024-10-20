import torch
import torch.nn as nn
import torch.optim as optim
import progressbar  #pip install progressbar2

# Definindo a estrutura da rede usando nn.Sequential
rede = nn.Sequential(
    nn.Linear(2, 2),  # Primeira camada: 2 entradas, 2 saídas
    nn.Sigmoid(),     # Função de ativação
    nn.Linear(2, 1),  # Segunda camada: 2 entradas, 1 saída
    nn.Sigmoid()      # Função de ativação na saída
)

# Definindo o critério de perda e o otimizador
criterio = nn.MSELoss()
otimizador = optim.SGD(rede.parameters(), lr=0.1)

# Simulação de dados (AND)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)

# Treinar a rede
epochs = 5000
with progressbar.ProgressBar(max_value=epochs) as bar:
    for epoch in range(epochs):
        bar.update(epoch)
        for x, target in zip(X, y):
            # Zerar os gradientes do otimizador
            otimizador.zero_grad()

            # Forward pass
            y_pred = rede(x)

            # Cálculo da perda
            perda = criterio(y_pred, target)

            # Backward pass
            perda.backward()

            # Atualizar os pesos
            otimizador.step()

# Testar a rede
with torch.no_grad():
    for x in X:
        print(f"Entrada: {x.numpy()} -> Saída Prevista: {rede(x).numpy()}")
