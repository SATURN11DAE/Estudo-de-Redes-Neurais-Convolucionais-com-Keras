# Estudos de Keras

Este repositório reúne tudo o que eu for fazendo enquanto estudo redes neurais com Keras: exercícios, tutoriais adaptados, experimentos e projetos pequenos.

A ideia é ter um lugar só para acompanhar a minha evolução, desde modelos simples até arquiteturas mais elaboradas, com o código comentado e explicado para eu (e quem mais quiser) conseguir entender o que está acontecendo em cada etapa.

> ## Sobre a origem dos projetos
> Muitos dos projetos aqui são baseados em tutoriais e materiais de estudo (como o GeeksforGeeks) e foram adaptados por mim para fins de aprendizado.
> Sempre que isso acontecer, vou indicar a fonte original no README do projeto.
> Este repositório é um caderno de estudos, e não uma coleção de soluções originais ou prontas para produção.

---

## Projetos

| # | Projeto | Descrição | Tecnologias | Fonte |
|---|---------|-----------|-------------|-------|
| 01 | [Classificação de placas de trânsito](#-projeto-01--classificação-de-placas-de-trânsito-gtsrb) | CNN que reconhece 43 tipos de placas do dataset GTSRB | Keras, TensorFlow, PyTorch (só para baixar o dataset) | Adaptado do GeeksforGeeks |

_A tabela será atualizada conforme eu for adicionando novos estudos._

---

## Organização do repositório

Cada projeto fica na sua própria pasta, com o código, o modelo treinado e os gráficos gerados:

```
.
├── README.md
├── 01-placas-transito/
│   ├── trafficsigns.py              # Baixa o dataset, treina o modelo e gera os gráficos
│   ├── testando.py                  # Testa o modelo treinado com imagens aleatórias
│   ├── modelo_placas_gtsrb.keras    # Modelo treinado (gerado pelo script)
│   └── grafico_treinamento.png      # Gráficos de acurácia e perda (gerado pelo script)
├── 02-...
└── ...
```

> Os scripts usam caminhos relativos (`./gtsrb_data/...`), então execute-os de dentro da pasta do projeto.

---

## Requisitos gerais

- Python 3.9+
- Bibliotecas mais usadas nos estudos:

```bash
pip install tensorflow keras numpy pandas matplotlib pillow
```

Alguns projetos precisam de dependências extras, que são listadas na seção de cada um (por exemplo, o projeto 01 usa `torch` e `torchvision`).

---

# Projeto 01 — Classificação de placas de trânsito (GTSRB)

Rede neural convolucional (CNN) que reconhece placas de trânsito alemãs usando o dataset **GTSRB** (German Traffic Sign Recognition Benchmark).

> **Origem:** baseado em um tutorial do **GeeksforGeeks** sobre reconhecimento de placas de trânsito, adaptado por mim para estudar redes neurais.
> 🔗 Tutorial original: _<cole aqui o link do artigo do GeeksforGeeks>_

## Como funciona

### 1. Treinamento (`trafficsigns.py`)

**a) Obtenção do dataset**
O script verifica se a pasta `./gtsrb_data/gtsrb/GTSRB/Training` já existe. Se não existir, ele baixa o GTSRB usando o `torchvision`.

**b) Conversão de imagens**
O GTSRB original vem em formato `.ppm`. O script converte todas as imagens para `.png` (e remove os `.ppm`), porque é o formato que o carregador de imagens do Keras lida melhor.

**c) Carregamento e divisão dos dados**
Com `image_dataset_from_directory`, as imagens são carregadas direto das pastas (cada pasta = uma classe), redimensionadas para **64×64 pixels** e divididas em:
- **80%** treino
- **20%** validação

O dataset tem **43 classes** de placas.

**d) Arquitetura da rede (CNN)**

| Etapa | Camada | Função |
|---|---|---|
| Entrada | `Input(64, 64, 3)` | Imagem RGB 64×64 |
| Aumento de dados | `RandomRotation(0.1)`, `RandomZoom(0.1)` | Gera pequenas variações nas imagens durante o treino para reduzir overfitting (só ativo no treino) |
| Normalização | `Rescaling(1/255)` | Converte pixels de 0–255 para 0–1 |
| Convolução 1 | `Conv2D(32, 3×3)` + `MaxPooling` | Detecta padrões simples (bordas, cantos) |
| Convolução 2 | `Conv2D(64, 3×3)` + `MaxPooling` | Detecta formas mais complexas |
| Convolução 3 | `Conv2D(128, 3×3)` + `MaxPooling` | Detecta padrões de alto nível (símbolos, números) |
| Classificação | `Flatten` → `Dense(128)` → `Dropout(0.3)` → `Dense(43, softmax)` | Transforma os mapas de características em uma probabilidade para cada classe |

**e) Treinamento**
- Otimizador: **Adam** (`learning_rate=0.001`)
- Função de perda: `sparse_categorical_crossentropy` (os rótulos são números inteiros)
- Até **15 épocas**, com **EarlyStopping** (`patience=3`): se a perda de validação parar de melhorar por 3 épocas seguidas, o treino para e os melhores pesos são restaurados.

**f) Saídas**
- `modelo_placas_gtsrb.keras`: o modelo treinado
- `grafico_treinamento.png`: curvas de acurácia e perda (treino vs. validação)

### 2. Teste (`testando.py`)

1. Baixa o **conjunto de teste** oficial do GTSRB (via `torchvision`).
2. Carrega o modelo `modelo_placas_gtsrb.keras`.
3. Sorteia **100 imagens** aleatórias do conjunto de teste.
4. Para cada imagem: redimensiona para 64×64, converte em array, adiciona a dimensão de batch e pede a previsão ao modelo.
5. Compara a classe prevista (`argmax`) com a classe real e imprime uma tabela com `ACERTOU` / `ERROU`, além da taxa de acerto final.

> A normalização (`Rescaling`) faz parte do modelo, então as imagens de teste **não** precisam ser divididas por 255 manualmente.

## Dependências extras

```bash
pip install torch torchvision
```

> O `torchvision` é usado **apenas** para baixar o dataset GTSRB com facilidade. O modelo em si é todo em Keras.

## Como executar

```bash
cd 01-placas-transito

# 1. Treinar o modelo
python trafficsigns.py

# 2. Testar o modelo
python testando.py
```

## Resultados

_Preencha aqui com os seus resultados depois de treinar:_

- Acurácia de validação: `XX%`
- Acurácia no teste (100 imagens aleatórias): `XX%`

![Gráfico de treinamento](01-placas-transito/grafico_treinamento.png)

## Observações

- Depois da conversão para `.png`, as imagens `.ppm` do treino são **apagadas**. Se quiser reutilizar o loader do `torchvision` para o split de treino, apague a pasta `gtsrb_data` e baixe de novo.
- O teste com 100 imagens aleatórias dá uma estimativa rápida, mas varia a cada execução. Para uma avaliação mais confiável, use o conjunto de teste completo.

## Ideias para evoluir

- Avaliar no conjunto de teste inteiro e gerar uma matriz de confusão
- Testar outras arquiteturas (mais camadas, `BatchNormalization`, transfer learning)
- Ajustar hiperparâmetros (tamanho da imagem, taxa de aprendizado, dropout)
- Criar uma interface simples para enviar uma foto e receber a previsão

---

## Próximos estudos

_Lista de coisas que quero explorar (edite à vontade):_

- [ ] Classificação de imagens com outro dataset (MNIST / CIFAR-10)
- [ ] Redes neurais densas para dados tabulares
- [ ] Transfer learning com modelos pré-treinados
- [ ] Redes recorrentes (LSTM) para texto ou séries temporais

---

## Créditos

- Tutorial base do projeto 01: **GeeksforGeeks** (adaptado para estudo)
- Dataset do projeto 01: [GTSRB – German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html)
