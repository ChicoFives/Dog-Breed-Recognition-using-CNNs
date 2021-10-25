from scipy import io
import random
import xml.etree.ElementTree as ET
import shutil
import os
from PIL import Image

Images_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Images//'
Annotations_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Annotation//'
Preprocessed_images_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Preprocessed_images//'

print('Número de categorias (raças):', len(os.listdir(Images_dir)))
print('Número de diretorias com anotações:', len(os.listdir(Annotations_dir)))
print('\n')

print('Lista com o número de cães existente em cada diretoria:\n')
for dir in os.listdir(Images_dir):
  breed_name = dir.split('-',1)[1]
  print(f'Número de images de cães da raça {breed_name}:', len(os.listdir(Images_dir+dir)))

#teste para ver se está tudo bem na listagem
print('Na posição 2 da lista encontra-se a raça: '+os.listdir(Images_dir)[2].split('-',1)[1])
print('\n')

#função retirada do website : https://www.kaggle.com/devang/transfer-learning-with-keras-and-efficientnets
def save_cropped_img(path, annotation, newpath):
  tree = ET.parse(annotation)
  xmin = int(tree.getroot().findall('.//xmin')[0].text)
  xmax = int(tree.getroot().findall('.//xmax')[0].text)
  ymin = int(tree.getroot().findall('.//ymin')[0].text)
  ymax = int(tree.getroot().findall('.//ymax')[0].text)
  image = Image.open(path)
  image = image.crop((xmin, ymin, xmax, ymax))
  image = image.convert('RGB')
  image.save(newpath)


# corta as imagens consoante as anotações fornecidas com as imagens
def crop_images():
  breeds = os.listdir(Images_dir)
  annotations = os.listdir(Annotations_dir)

  print('Raças: ', len(breeds), 'Anotações: ', len(annotations))

  total_images = 0

  for breed in breeds:
    dir_list = os.listdir(Images_dir + breed)
    ANNOTATIONS_DIR_list = os.listdir(Annotations_dir + breed)
    img_list = [Images_dir + breed + '/' + i for i in dir_list]
    os.makedirs(Preprocessed_images_dir + breed)

    for file in img_list:
      annotation_path = Annotations_dir + breed + '/' + os.path.basename(file[:-4])
      newpath = Preprocessed_images_dir + breed + '/' + os.path.basename(file)
      save_cropped_img(file, annotation_path, newpath)
      total_images += 1

  print("Total de imagens cortadas ", total_images)

#crop images foi executado só uma vez para criar as "imagens anotadas"
#crop_images

#Converte os ficheiros matlab para um array

train_list = io.loadmat('C://Users//Francisco Pereira//PycharmProjects//pythonProject//train_list.mat')['file_list']
test_list = io.loadmat('C://Users//Francisco Pereira//PycharmProjects//pythonProject//test_list.mat')['file_list']

train_samples = []
test_samples = []

train_labels = []
test_labels = []

for train_sample in train_list:
  train_labels.append(train_sample[0][0].split('-', 1)[1].split('/', 1)[0])
  train_samples.append(train_sample[0][0])

for test_sample in test_list:
  test_labels.append(test_sample[0][0].split('-', 1)[1].split('/', 1)[0])
  test_samples.append(test_sample[0][0])

# verificação
print("Número de amostras de treino :", len(train_samples))
print("Número de amostras de teste :", len(test_samples))

print("Número de rótulos de treino (raças) :", len(set(train_labels)))
print("Número de rótulos de teste (raças) :", len(set(test_labels)))

#Escreve a lista para um ficheiro (train_list.txt)
#Recebe como argumentos uma lista e o nome do ficheiro onde queremos guardar
def listaParaFicheiro(lista, nomeficheiro):
    with open(nomeficheiro, 'w') as filehandle:
        filehandle.writelines("%s\n" % sample for sample in lista)

#Executado uma única vez
'''
listaParaFicheiro(train_samples, 'train_list.txt')
listaParaFicheiro(test_samples, 'test_list.txt')
'''

#Copiar do ficheiro train_list.txt para a lista train_list_list com os nomes das imagens de treino (Total 12000)
with open('train_list.txt', 'r') as f:
  temp = [line.strip() for line in f]
  train_list_list = [s.strip("[']") for s in temp]

#Função que seleciona 80% dos dados aleatóriamente para serem colocados na diretoria de treino
#Recebe como argumento a lista neste caso a train_list_list
def genTrain(lista):
  quantidade = int(len(lista) * 0.8)
  train_items = random.sample(lista, k=quantidade)
  validation_items = [x for x in lista if x not in train_items]
  return train_items

#Função que seleciona os restanto dados que não foram colocados na diretoria de treino, portanto 20% da lista
#Recebe como argumento a lista neste caso a train_list_list
def genValidation(lista):
  s = set(lista)
  temp = [x for x in train_list_list if x not in s]
  return temp

#Função que gera as diretorias de treino, validação e teste consoante o argumento que é passado no sample.
#Recebe como argumentos a diretoria de origem, a de chegada e a lista
def genTrainValidationTest_Folders(src_dir, dst_dir, sample):
  imageNames = sample
  for imageName in imageNames:
    breed = imageName.split('/')[0].split('-', 1)[1]
    dst_conc = os.path.join(dst_dir,breed)
    if not os.path.exists(dst_conc):
      os.makedirs(dst_conc)
      shutil.copy(os.path.join(src_dir, imageName), dst_conc)
    else:
      # copy files into created directory
      shutil.copy(os.path.join(src_dir, imageName), dst_conc)
  print("All files moved!")


#gerar ficheiros de treino e validação 80% treino, 20% validação de 12000 imagens
#Executado uma única vez
'''
train = genTrain(train_list_list)
validation = genValidation(train)
test = test_samples
'''
'''
#Copia as imagens da diretoria Preprocessed_images para a diretoria Train dados os valores que estão na lista de treino
#Executado uma única vez
src_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Preprocessed_images'
train_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Train'
genTrainValidationTest_Folders(src_dir,train_dir,train)

#Copia as imagens da diretoria Preprocessed_images para a diretoria Validation dados os valores que estão na lista de validação
#Executado uma única vez
src_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Preprocessed_images'
validation_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Validation'
genTrainValidationTest_Folders(src_dir,validation_dir,validation)

#Copia as imagens da diretoria Preprocessed_images para a diretoria Test dados os valores que estão na lista de teste
#Executado uma única vez
src_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Preprocessed_images'
test_dir = 'C://Users//Francisco Pereira//PycharmProjects//pythonProject//Test'
genTrainValidationTest_Folders(src_dir,test_dir,test)
'''

print('Terminado o processo de processamento do dados do dataset!\n')