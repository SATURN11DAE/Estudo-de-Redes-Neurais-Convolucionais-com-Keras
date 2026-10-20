import os
import random
import numpy as np
import keras
from keras.utils import img_to_array
import torchvision.datasets as datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("Baixando conjunto de teste")
test_set = datasets.GTSRB(root='./gtsrb_data', split='test', download=True)

# Carregar o modelo treinado
model_path = "modelo_placas_gtsrb.keras"

try:
    model = keras.models.load_model(model_path)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# Escolher aleatoriamente 100 imagens de teste
num_testes = 100
indices_aleatorios = random.sample(range(len(test_set)), num_testes)

acertos = 0

print(f"{'ID IMAGEM':<10} | {'CLASSE REAL':<12} | {'PREVISÃO':<12} | {'STATUS'}")
print("-" * 55)

for idx in indices_aleatorios:
    # O PyTorch entrega-nos a imagem original e o número da classe correta
    img_original, classe_real_idx = test_set[idx]
    
    # Redimensionar para 64x64 e converter para array
    img_redimensionada = img_original.resize((64, 64))
    img_array = img_to_array(img_redimensionada)
    
    # O modelo espera lotes (batches), por isso adicionar uma dimensão extra (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    previsoes = model.predict(img_array, verbose=0)
    classe_prevista_idx = np.argmax(previsoes)
    
    if classe_real_idx == classe_prevista_idx:
        status = "ACERTOU"
        acertos += 1
    else:
        status = "ERROU"
        
    print(f"Img {idx:<6} | Classe {classe_real_idx:<5} | Classe {classe_prevista_idx:<5} | {status}")

print("-" * 55)
taxa_acerto = (acertos / num_testes) * 100
print(f"Total de acertos: {acertos} de {num_testes} ({taxa_acerto:.1f}%)\n")
