import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau


seed = 42
img_size = 224
batch_size = 32

#Gera batches de dados de imagem de tensor com augmentation data em tempo real.
#Os dados serão repetidos (em batches).
datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    r"Train//",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed
)

validation_generator = datagen.flow_from_directory(
    r"Validation//",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=seed
)

test_datagen = datagen.flow_from_directory(
    directory=r"Test",
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed = seed
)

#Retorno de chamada para salvar o modelo Keras ou pesos do modelo com alguma frequência.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_modelInceptionV3.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

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
# Cria o modelo Inception-v3
base_model = InceptionV3(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False)

#Congela todas as camadas do modelo Inception-v3, tornando-as não treináveis
for layer in base_model.layers:
    layer.trainable = False

#Adicionada uma camada de average pooling espacial global
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

#Adiconada uma fully connected layer com 120 neurónios, porque são 120 raças distintas
predictions = Dense(120, activation='softmax')(x)

#Este é o modelo que vai ser treinado
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(
  train_generator,
  validation_data=validation_generator,
  epochs=40,
  callbacks=[model_checkpoint_callback, earlystop, csvlogger, reduceLR]
)


#Obtem a precisão e a perda para a validação
(eval_loss, eval_accuracy) = model.evaluate(validation_generator, batch_size=batch_size, verbose=1)
print('Perda da validação: ', eval_loss)
print('Precisão da validação: ', eval_accuracy)

#guarda a variável history num dicionário, contém os pesos do modelo
file_pi = open('InceptionV3HistoryDict', 'wb')
pickle.dump(history.history, file_pi)
file_pi.close()
'''

#Carrega o modelo pretendido, neste caso a Inception-v3
model = load_model('best_modelInceptionV3.h5')

#Mosta o sumário do modelo
model.summary()


#Função que determina a raça da imagem de um cão da Internet, para o modelo Inception-v3
#Recebe como argumento a imagem selecionada
def racaPredictInceptionv3(imagem):
    key_list = test_datagen.class_indices
    response = requests.get(imagem)
    img = Image.open(BytesIO(response.content))

    img = img.resize((224, 224))
    img = np.array(img)/255

    pred = np.argmax(model.predict(img.reshape(1, 224, 224, 3)), axis=1)
    num = int(pred.astype(int))
    val_list = list(key_list.keys())
    plt.title("Raça prevista pelo modelo Inception-v3: " + val_list[num])
    plt.axis('off')
    plt.imshow(img)
    plt.savefig('predict_Inception-v3.png', transparent= False, bbox_inches= 'tight', dpi= 900)
    plt.show()

#exemplo de uma imagem
img = 'https://img.joomcdn.net/09b34f0bef2e97eb9a59aecd7d3bf15787ccf0d0_original.jpeg'
racaPredictInceptionv3(img)
