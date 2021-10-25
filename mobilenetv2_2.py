import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow import keras
tf.keras.applications.mobilenet_v2.preprocess_input
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import requests
from io import BytesIO
from PIL import Image

TARGET_X = 224
TARGET_Y = 224
BATCH_SIZE = 32
NUM_CLASSES = 120
SEED = 2021

#Gera batches de dados de imagem de tensor com augmentation data em tempo real.
#Os dados serão repetidos (em batches).
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

train_datagen = datagen.flow_from_directory(
    directory=r"Train//",
    target_size=(TARGET_X, TARGET_Y),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

validation_datagen = datagen.flow_from_directory(
    directory=r"Validation//",
    target_size=(TARGET_X, TARGET_Y),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

test_datagen = datagen.flow_from_directory(
    directory=r"Test",
    target_size=(TARGET_X, TARGET_Y),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed = SEED
)

#Retorno de chamada para salvar o modelo Keras ou pesos do modelo com alguma frequência.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_modelMobileNetV2.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#Para o treino do modelo quando uma métrica monitorizada parar de melhorar.
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode='auto'
)

#Retorno de chamada que transmite resultados de época para um ficheiro CSV.
csvlogger = CSVLogger(
    filename='training_csv.log',
    separator=',',
    append=False
)

#Reduz a taxa de aprendizagem quando uma métrica para de melhorar.
reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1,
    mode='auto'
)


'''
def get_model():
  model = keras.Sequential()

  mobilev2 = keras.applications.MobileNetV2(include_top=False, input_shape=(TARGET_X, TARGET_Y, 3))

  #Ajustando o modelo base
  mobilev2.trainable = True
  for layer in mobilev2.layers:
    if layer.name.find('Conv_1') or layer.name.find('block_14') or layer.name.find('block_15') or layer.name.find(
            'block_16') or layer.name == 'out_relu':
      layer.trainable = True
    else:
      layer.trainable = False

  #Usado a MobileNetV2
  model.add(mobilev2)

  #Achatada a convolução e colocada numa rede neuronal profunda
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dropout(0.4))

  #Usado redes neuronais profundas com  forma de saída 120, porque no total são 120 raças
  model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

  return model


model = get_model()
model.summary()


#Agrad foi usado porque tem uma convergência mais rápida
model.compile(optimizer='adagrad', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Feito o treino do modelo, com 40 épocas e 300 passos por época, usando as callbacks definidas
history = model.fit(train_datagen,
                steps_per_epoch=300,
                epochs=40,
                callbacks=[model_checkpoint_callback, earlystop, csvlogger, reduceLR]
                )

#Obtem a precisão e a perda para a validação
(eval_loss, eval_accuracy) = model.evaluate(validation_datagen, batch_size=BATCH_SIZE, verbose=1)
print('Validation Loss: ', eval_loss)
print('Validation Accuracy: ', eval_accuracy)

#guarda a variável history num dicionário, contém os pesos do modelo
file_pi = open('MobileNetV2HistoryDict', 'wb')
pickle.dump(history.history, file_pi)
file_pi.close()
'''

#Carrega o modelo pretendido, neste caso a MobileNetV2
model = load_model('best_modelMobileNetV2.h5')

#Mosta o sumário do modelo
model.summary()

#Função que determina a raça da imagem de um cão da Internet, para o modelo MobileNetV2
#Recebe como argumento a imagem selecionada
def racaPredictMobileNetV2(imagem):
    key_list = test_datagen.class_indices
    response = requests.get(imagem)
    img = Image.open(BytesIO(response.content))

    img = img.resize((224, 224))
    img = np.array(img)

    pred = np.argmax(model.predict(img.reshape(1, 224, 224, 3)), axis=1)
    num = int(pred.astype(int))
    val_list = list(key_list.keys())
    plt.title("Raça prevista pelo modelo MobileNetV2: " + val_list[num])
    plt.axis('off')
    plt.imshow(img)
    plt.savefig('predict_MobileNetV2.png', transparent= False, bbox_inches= 'tight', dpi= 900)
    plt.show()

#exemplo de uma imagem
img = 'https://img.joomcdn.net/09b34f0bef2e97eb9a59aecd7d3bf15787ccf0d0_original.jpeg'
racaPredictMobileNetV2(img)
