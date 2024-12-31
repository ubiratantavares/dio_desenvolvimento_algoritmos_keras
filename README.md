# DIO - Desenvolvimento de Algoritmos no Keras

## Introdução ao Keras

Keras é uma biblioteca de código aberto de alto nível para construção e treinamento de redes neurais, desenvolvida em Python. 

Seu principal objetivo é permitir que desenvolvedores e pesquisadores criem e experimentem modelos de aprendizado profundo de forma simples e intuitiva. 

Keras foi projetado para ser modular, minimalista e extensível, facilitando tanto para iniciantes quanto para especialistas no campo da Inteligência Artificial.

Inicialmente, Keras foi construído para rodar sobre diferentes backends como TensorFlow, Theano e Microsoft Cognitive Toolkit (CNTK). 

No entanto, desde 2019, Keras tornou-se parte integrante do TensorFlow, sendo a interface oficial de alto nível desta biblioteca. 

Sua facilidade de uso está relacionada à abstração de operações complexas em funções simples e reutilizáveis, como a criação de camadas, a definição de otimizadores e a compilação de modelos.

## Keras na Prática

Keras fornece uma API amigável para construir modelos sequenciais e funcionais. 

O modelo sequencial é ideal para fluxos de dados lineares, enquanto o modelo funcional é mais versátil e permite a criação de arquiteturas de rede neurais complexas, 
como redes com múltiplas entradas ou saídas.


**Exemplo de Modelo Sequencial em Keras:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Criando o modelo sequencial
model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(10, activation='softmax')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

Além de criar modelos, Keras oferece ferramentas para visualização, monitoramento de desempenho durante o treinamento, 
e suporte nativo para treinamento em GPU, permitindo rápidas iterações em experimentos.


## Transfer Learning com Keras

Transfer learning (ou aprendizado por transferência) é uma técnica em que um modelo pré-treinado em uma tarefa 
é reutilizado como ponto de partida para resolver outra tarefa. 

Essa abordagem é amplamente utilizada quando se trabalha com conjuntos de dados pequenos ou problemas específicos em que é difícil treinar um modelo do zero.

Keras oferece acesso a vários modelos pré-treinados, como ResNet, VGG, Inception e MobileNet, 
disponíveis no módulo `tensorflow.keras.applications`. 

Esses modelos podem ser usados diretamente para extração de características ou ajustados (fine-tuning) para o problema em questão.

**Exemplo de Transfer Learning com ResNet:**

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Carregando o modelo ResNet50 pré-treinado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelando as camadas do modelo base
base_model.trainable = False

# Adicionando camadas personalizadas
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Criando o modelo final
model = Model(inputs=base_model.input, outputs=output)

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Transfer learning é especialmente útil em tarefas de visão computacional e processamento de linguagem natural, 
onde os modelos pré-treinados podem capturar representações úteis de dados genéricos. 

Isso acelera o treinamento e melhora a precisão, mesmo com recursos limitados.
