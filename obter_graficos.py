import pickle
import matplotlib.pyplot as plt

#carregar o modelo pretendido
#filename = 'MobileNetV2HistoryDict'
filename = 'InceptionV3HistoryDict'
loaded_history = pickle.load(open(filename, 'rb'))

#Gráfico da precisão do modelo escolhido
plt.subplot()
plt.title('Accuracy do modelo Inception-v3')
plt.plot(loaded_history['accuracy'])
plt.plot(loaded_history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Épocas')
plt.legend(['Accuracy do Treino','Accuracy da Validação'])
plt.savefig('baseline_acc_epochInception-v3.png', transparent= False, bbox_inches= 'tight', dpi= 900)
plt.show()

#Gráfico da perda do modelo escolhido
plt.title('Perda do modelo Inception-v3')
plt.plot(loaded_history['loss'])
plt.plot(loaded_history['val_loss'])
plt.ylabel('Perda')
plt.xlabel('Épocas')
plt.legend(['Perda do Treino','Perda da Validação'])
plt.savefig('baseline_loss_epochInception-v3.png', transparent= False, bbox_inches= 'tight', dpi= 900)
plt.show()

print(max(loaded_history['accuracy']))