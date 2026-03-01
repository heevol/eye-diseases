# 🔬 Eye Diseases Classification Project

Um sistema de classificação de imagens médicas para diagnóstico de doenças oculares utilizando Redes Neurais Convolucionais (CNN).

## 📋 Descrição do Projeto

Este projeto implementa um modelo de deep learning capaz de classificar imagens de olhos em quatro categorias:
- **Cataract** (Catarata)
- **Diabetic Retinopathy** (Retinopatia Diabética)
- **Glaucoma** (Glaucoma)
- **Normal** (Saudável)

Base de imagens encontrada no Kaggle:

https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

## 🏗️ Estrutura do Projeto
```bash
EYE-DISEASES/
├── dataset/                    # Base de dados de treinamento
│ ├── cataract/                     ⮕ 800 imagens de catarata
│ ├── diabetic_retinopathy/         ⮕ 800 imagens de retinopatia diabética
│ ├── glaucoma/                     ⮕ 800 imagens de glaucoma
│ └── normal/                       ⮕ 800 imagens de olhos saudáveis
├── inputs/                     # Pasta para imagens de teste
│ ├── cataract/                     ⮕ 238 imagens de catarata
│ ├── diabetic_retinopathy/         ⮕ 298 imagens de retinopatia diabética
│ ├── glaucoma/                     ⮕ 207 imagens de glaucoma
│ └── normal/                       ⮕ 274 imagens de olhos saudáveis
├── train_model.py              # Script principal de treinamento
├── predict.py                  # Script para fazer predições
├── pre_process.py              # Script de pré-processamento e verificação
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

## 📊 Estrutura do Dataset

O dataset está organizado de forma balanceada com:
- **4 classes** distintas
- **1000 imagens** por classe
- **Total: 4000 imagens**
- Divisão automática: 80% treino, 20% validação

## 🛠️ Pré-requisitos

### Instalação das Dependências

```bash
# Instalar TensorFlow e outras bibliotecas
pip install tensorflow matplotlib numpy pillow scikit-learn

# Ou usando o requirements.txt
pip install -r requirements.txt
```
## Dependências Principais
- **tensorflow** - Framework de deep learning
- **matplotlib** - Visualização de gráficos
- **numpy** - Processamento numérico
- **Pillow** - Processamento de imagens
- **scikit-learn** - Métricas de avaliação

## 🚀 Como Usar

### 1. Verificação da Estrutura dos Dados
```bash
python pre_process.py
```
Saída esperada:

```text
Estrutura do dataset encontrada:
  - cataract: 1000 imagens
  - diabetic_retinopathy: 1000 imagens
  - glaucoma: 1000 imagens
  - normal: 1000 imagens
Found 3200 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
Found 4000 images belonging to 4 classes.

Dataset carregado com sucesso!
Classes: {'cataract': 0, 'diabetic_retinopathy': 1, 'glaucoma': 2, 'normal': 3}
Número de imagens de treino: 3200
Número de imagens de validação: 800
```

### 2. Treinamento do Modelo
```bash
python train_model.py
```
#### O que acontece durante o treinamento:

✅ Carregamento e verificação do dataset

✅ Pré-processamento das imagens (redimensionamento, normalização)

✅ Data augmentation (rotação, zoom, flip horizontal)

✅ Criação da arquitetura CNN

✅ Treinamento por 20 épocas

✅ Salvamento automático do melhor modelo

✅ Geração de gráficos de acurácia e loss

#### Arquivos gerados:

- **eye_disease_model.h5** - Melhor modelo durante treinamento
- **eye_disease_model_final.h5** - Modelo final
- **class_indices.txt** - Mapeamento das classes
- **training_history.png** - Gráfico do histórico de treinamento

### 3. Realizar Predições
```bash
python predict.py
```
#### Funcionamento:

🔍 Busca automaticamente todas as imagens na pasta ```inputs/```

📊 Processa cada imagem individualmente

🎯 Retorna probabilidades para todas as classes

📋 Gera um resumo final das predições

##### Exemplo de saída:

```text
==================================================
Imagem: teste_olho.jpg
Predição: glaucoma
Probabilidade: 87.45%

Probabilidades para todas as classes:
  - cataract: 5.23%
  - diabetic_retinopathy: 3.12%
  - glaucoma: 87.45%
  - normal: 4.20%
==================================================
```

## 🧠 Arquitetura do Modelo
### O projeto utiliza uma Rede Neural Convolucional (CNN) personalizada com:

#### Camadas da CNN:
- **Conv2D (32 filtros) + MaxPooling**
- **Conv2D (64 filtros) + MaxPooling**
- **Conv2D (128 filtros) + MaxPooling**
- **Conv2D (256 filtros) + MaxPooling**
- **Camadas Densas (512 → 256 neurônios)**
- **Dropout (50% e 30%) para evitar overfitting**
- **Saída Softmax com 4 neurônios (uma por classe)**

#### Técnicas Aplicadas:
- **Data Augmentation:** Aumento artificial do dataset
- **Dropout:** Regularização para generalização
- **Early Stopping:** Salva o melhor modelo automaticamente
- **Normalização:** Pixels escalonados para [0, 1]

## 📈 Métricas de Performance
### Durante o treinamento, o modelo é avaliado com:

- **Acurácia (Accuracy)**
- **Loss (Perda)**
- **Validação cruzada (80/20 split)**

Os gráficos gerados em ```training_history.png``` mostram a evolução do treinamento.

## 🎯 Como Adicionar Novas Imagens para Teste
Coloque as imagens na pasta ```inputs/```

Formatos suportados: ```JPG, JPEG, PNG, BMP```

Execute: ```python predict.py```

Se deseja testar imagem unica pode testar removendo o comentário no final do arquivo ```predict.py```
```
if __name__ == "__main__":
    # Predição para todas as imagens na pasta inputs
    predict_all_images_in_folder() <- COMENTE ESSA LINHA
    
    # Exemplo para predição de uma imagem específica
    # predict_single_image('inputs/cataract/2135_left.jpg') <- REMOVA O COMENTÁRIO DESSA
```

## 🔧 Personalização
### Modificar Parâmetros de Treinamento
Edite ```train_model.py```

```
# No arquivo train_model.py, modifique:
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Tamanho das imagens
BATCH_SIZE = 32                   # Tamanho do lote
EPOCHS = 20                       # Número de épocas
```

### Adicionar Novas Doenças
1. Adicione nova pasta em ```dataset/```
2. Cole imagens da nova classe
3. Retreine o modelo: ```python train_model.py```

# ⚠️ Considerações Importantes
## 🩺 Aviso Médico
 Este projeto é para fins educacionais e de pesquisa. Não substitui diagnóstico médico profissional. Sempre consulte um oftalmologista para diagnóstico preciso.

## 📷 Requisitos das Imagens
- **Formato: JPG, PNG, JPEG, BMP**
- **Tamanho: Qualquer (será redimensionado automaticamente)**
- **Cor: Colorida (RGB)**
- **Qualidade: Quanto maior, melhor**

## 🐛 Solução de Problemas
#### Erro: "No module named 'tensorflow'"
```bash
pip install tensorflow
```
#### Erro: "Dataset folder not found"
Verifique se a pasta ```dataset/``` existe com as subpastas das classes.

#### Erro: "No images found in inputs folder"
Adicione imagens na pasta ```inputs/``` antes de executar ```predict.py```.

## 📄 Licença
#### Este projeto é para fins educacionais. O dataset pertence aos respectivos criadores no Kaggle.

