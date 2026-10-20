import os
import glob
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory

warnings.filterwarnings('ignore')

dataset_dir = './gtsrb_data/gtsrb/GTSRB/Training'

# Baixa dataset se não tiver
if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
    print(f"Dataset encontrado na pasta '{dataset_dir}'.")
else:
    print("Dataset não encontrado. Baixar e extrair do Pytorch")

    import torchvision.datasets as datasets
    _ = datasets.GTSRB(root='./gtsrb_data', split='train', download=True)

# Converte pra png se precisar
ppm_files = glob.glob(os.path.join(dataset_dir, '**', '*.ppm'), recursive=True)

if ppm_files:
    print(f"\nDetectadas {len(ppm_files)} imagens no formato .ppm.")
    
    for ppm_path in ppm_files:
        png_path = ppm_path.replace('.ppm', '.png')
        try:
            with Image.open(ppm_path) as img:
                img.save(png_path)
            os.remove(ppm_path)
        except Exception as e:
            print(f"Erro ao converter {ppm_path}: {e}")
            
else:
    print("\nNenhuma imagem .ppm para converter")

print("\nCarregando e dividindo as imagens")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

num_classes = len(train_ds.class_names)
print(f"Número de classes: {num_classes}")

data_augmentation = Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

model = Sequential([
    # Camada de Entrada
    layers.Input(shape=(64, 64, 3)),
    
    # Aumento de Dados + Normalização dos Pixels (0 a 1)
    data_augmentation,
    layers.Rescaling(1./255),
    
    # Camadas Convolucionais
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Camadas Densas (Classificação)
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n Começando o treinamento")

my_callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=my_callbacks
)

model.save("modelo_placas_gtsrb.keras")
print("\nModelo salvo")

print("Gráficos")
plt.figure(figsize=(12, 4))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')

# Gráfico de Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda (Loss) por Época')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig("grafico_treinamento.png")
print("Gráficos salvos")

plt.show()
