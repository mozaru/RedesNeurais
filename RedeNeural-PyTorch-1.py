import torch
import torch.nn as nn
import torch.optim as optim
import progressbar  #pip install progressbar2
#from tqdm import tqdm #pip install tqdm

# Definindo o modelo da rede neural
class RedeNeural(nn.Module):
    def __init__(self, estrutura):
        super(RedeNeural, self).__init__()
        # Criando as camadas
        camadas = []
        for i in range(len(estrutura) - 1):
            camadas.append(nn.Linear(estrutura[i], estrutura[i + 1]))
            if i < len(estrutura) - 2:  # Não adiciona ativação na última camada
                camadas.append(nn.Sigmoid())  # Usando Sigmoid como ativação
        self.modelo = nn.Sequential(*camadas)

    def forward(self, x):
        return self.modelo(x)

# Estrutura da rede: 2 entradas, 2 neuronios na camada oculta, 1 saida
estrutura = [2, 2, 1]
rede = RedeNeural(estrutura)

# Definindo o critério de perda e o otimizador
criterio = nn.MSELoss()
otimizador = optim.SGD(rede.parameters(), lr=0.1)

# Simulacao de dados (AND)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)

# Treinar a rede
epochs = 5000
bar = progressbar.ProgressBar(max_value=epochs)
#for epoch in tqdm(range(epochs)):
for epoch in range(epochs):
    bar.update(epoch)
    for x, target in zip(X, y):
        # Zerar os gradientes do otimizador
        otimizador.zero_grad()

        # Forward pass
        y_pred = rede(x)

        # Calculo da perda
        perda = criterio(y_pred, target)

        # Backward pass
        perda.backward()

        # Atualizar os pesos
        otimizador.step()
bar.finish()
# Testar a rede
with torch.no_grad():
    for x in X:
        print(f"Entrada: {x.numpy()} -> Saida Prevista: {rede(x).numpy()}")
