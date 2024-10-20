# https://www.deeplearningbook.com.br/algoritmo-backpropagation-parte-2-treinamento-de-redes-neurais/
# https://www.youtube.com/watch?v=eagmQz-4OI0
# https://medium.com/ensina-ai/redes-neurais-perceptron-multicamadas-e-o-algoritmo-backpropagation-eaf89778f5b8

# https://www.youtube.com/watch?v=nineTA2uYKA

import numpy as np
import progressbar  #pip install progressbar2

#um neuronio tem os pesos e para as entradas aplica o soma ponderada das entradas, o que vai ativar a saida
#O bias eh um parametro adicional em um neuronio que permite a funcao de ativacao deslocar a funcao de ativacao para a esquerda ou para a direita.
#Ajuda a ajustar a saida do neuronio independentemente das entradas, permitindo que o modelo se ajuste melhor aos dados.
#para a ativacao podemos usar varios metodos como sigmoid, tangente hiperbolica, "Rectified Linear Unit", softmax e etc
class Neuronio:
    def __init__(self, num_entradas):
        self.pesos = np.random.rand(num_entradas)
        self.bias = np.random.rand(1)[0]

        #guardamos as entradas e saidas devido ao treinamento
        self.entradas = []
        self.saida = 0

    #inicio dos metodos de ativacao
    def __sigmoide(self,x):
        # Sigmoide  - retorna um valor real de 0 a 1
        return 1 / (1 + np.exp(-x))  

    def __tanh(self,x):
        # Tangente Hiperbolica - retorna um valor real de -1 a 1
        return (np.exp(x)-np.exp(-x))/ (np.exp(x)+np.exp(-x))

    def __ReLU(self,x):
        # Rectified Linear Unit - retorna um valor real de >=0
        return np.max(0,x)

    def __softmax(self,x):
        pass
    #fim dos metodos de ativacao

    def ativacao(self, x):
        return self.__sigmoide(x)

    def ativacao_derivada(self, x):
        return x * (1 - x)  # Derivada da sigmoide

    def forward(self, entradas):
        self.entradas = entradas
        self.saida = self.ativacao(np.dot(entradas, self.pesos) + self.bias)
        return self.saida

    def backward(self, erro, learning_rate):
        # Atualiza os pesos e o bias
        derivada = self.ativacao_derivada(self.saida)
        pesosT = self.pesos[:, np.newaxis]
        sigma = erro * derivada
        self.pesos += learning_rate * sigma * self.entradas
        self.bias += learning_rate * sigma
        print("erro:",erro)
        print("derivada:",derivada)
        print("pesos:",self.pesos)
        print("saida:",self.saida)
        print(pesosT)
        return np.dot(erro * derivada, pesosT)

#uma camada so fala com a proxima e recebe dados da anterior
#a primeira recebe os dados de entrada
#a ultima gera a saida
class Camada:
    def __init__(self, num_neuronios, num_entradas):
        #aqui vamos criar os neuronios, cada um recebendo num_entradas
        self.neuronios = [Neuronio(num_entradas) for _ in range(num_neuronios)]

    def forward(self, entradas):
        return np.array([neuronio.forward(entradas) for neuronio in self.neuronios])
    
    def backward(self, erro, learning_rate):
        for neuronio in self.neuronios:
            erro = neuronio.backward(erro, learning_rate)

#a rede neural eh um conjunto de camadas, onde cada camada tem x neuronios
class RedeNeural:
    #recebe um vetor, 
    #o primeiro elemento eh a quantidade de entradas
    #o ultimo elemento eh a quantidade de saidas
    #o tamanho do vetor - 1, eh a quantidade de camadas
    #cada camada tem que ter como entrada a quantidade de neuronios da camada anterior
    #assim um vetor [10,4,4,3] cria uma rede neural com 3 camadas
    # a primeira receve a entrada do usuario que seriam 10 infos e passa para o 4 neuronios gerando 4 saida
    # a segunda recebe as 4 saidas da primeira com entrada, passa para os 4 neuronios e gera 4 saidas
    # a terceira recebe as 4 saidas da segunda com entrada, passa para os 3 neuronios e gera 3 saidas
    def __init__(self, estrutura):
        self.camadas = []
        for i in range(len(estrutura) - 1):
            self.camadas.append(Camada(estrutura[i + 1], estrutura[i]))

    def prever(self, inputs):
        for camada in self.camadas:
            inputs = camada.forward(inputs)
        return inputs

    def treinar(self, inputs, outputs, learning_rate=0.1, epochs=1000):
        with progressbar.ProgressBar(max_value=epochs) as bar:
            for epoch in range(epochs):
                bar.update(epoch)
                for x, y in zip(inputs, outputs):
                    # Forward pass
                    y_pred = self.prever(x)

                    # Calculo do erro
                    erro = y - y_pred
                    print("Erro ",erro)
                    # Backward pass
                    for camada in reversed(self.camadas):
                        erro = camada.backward(erro, learning_rate)



# Estrutura da rede: 2 entradas, 2 neuronios na camada oculta, 1 saida
estrutura = [2, 2, 1]
rede = RedeNeural(estrutura)

# Simulacao de dados (AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Treinar a rede
rede.treinar(X, y)

# Testar a rede
for x in X:
    print(f"Entrada: {x} -> Saida Prevista: {rede.prever(x)}")