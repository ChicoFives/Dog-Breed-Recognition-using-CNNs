import os
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from skimage import transform

'''
#MobileNetV2 dados 
TARGET_X = 224
TARGET_Y = 224
BATCH_SIZE = 32
NUM_CLASSES = 120
SEED = 2021

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
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
#Carrega o modelo pretendido
model = load_model('best_modelMobileNetV2.h5')

#Mosta o sumário do modelo
model.summary()

#Descobrir a percentagem de top-1 do modelo
eval_accuracy= model.evaluate(test_datagen, batch_size=BATCH_SIZE, verbose=1)
print('Top-1 MobileNetV2: ', eval_accuracy)

#Função que transforma a imagem no tamanho predefinido
#Recebe como argumento o nome da imagem
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image
'''

#Inception-v3 dados
seed = 42
img_size = 224
batch_size = 32


datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
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

model = load_model('best_modelInceptionV3.h5')
model.summary()

#Descobrir a percentagem de top-1 do modelo
#eval_accuracy= model.evaluate(test_datagen, batch_size=batch_size, verbose=1)
#print('Top-1 Inception-v3: ', eval_accuracy)

#Função que transforma a imagem no tamanho predefinido
#Recebe como argumento o nome da imagem
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

#Lê os dados que estão na test_list e copia para a lista listaTeste
test_file = open('test_list.txt', 'r')
listaTeste = [line.split(',') for line in test_file.readlines()]

#Cria um dicionário com as 120 raças e atribui uma chave a cada raça
numeros = test_datagen.class_indices

#Função verifica a que raça pertente uma determinada chave
def checkKey(dict, key):
    if key in dict.keys():
        return dict[key]
    else:
        print("Não existe!\n")

base_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Test//'
total=0
top=0
for imagepath in listaTeste:
    breed_name = imagepath[0].split('/')[0].split('-',1)[1]
    imagepath_str = ''.join(imagepath).replace('\n','').replace('/','//').split()
    imagepath_str2 = ''.join(imagepath_str)
    imagepath_str3 = ''.join(imagepath_str2.split('-')[1:])
    file = os.path.join(base_dir,imagepath_str3)
    image = load(file)
    pred = model.predict(image)
    top_components = tf.reshape(tf.math.top_k(pred, k=5).indices, shape=[-1])
    num = checkKey(numeros,breed_name)
    total+=1
    if num in top_components:
        top+=1
print(top)
print(total)
x = (top/total)*100
print("A accuracy do Top-5 é: {:.3f}".format(x))

