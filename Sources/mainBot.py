# Imports de librerias para el Chatbot
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import json
import random
import pickle

# nltk.download('punkt')


print("-------------------- S t a r t ---------------------"+"\n")


#Cargo el json de tags.
with open("contenido.json") as archivo:
    datos = json.load(archivo)

# Cargo las variables del tensorflow.
palabras = []
tags = []
auxX = []
auxY = []

# Recorro el Json y extraigo patrones, palabras y tags.
for contenido in datos['contenido']:
    for patrones  in contenido['patrones']:
        auxPalabra = nltk.word_tokenize(patrones)
        palabras.extend(auxPalabra)
        auxX.append(auxPalabra)
        auxY.append(contenido["tag"])

        if contenido['tag'] not in tags:
            tags.append(contenido["tag"])

# Sorteo palabras y tags.
palabras = [stemmer.stem(w.lower()) for w in palabras if w !='?']
palabras = sorted(list(set(palabras)))
tags = sorted(tags)

# List para el entrenamiento como tal.
entrenamiento = []
salida = []

salidaVacia = [0 for _ in range(len(tags))]

# Genero lista de entrenamiento con sus salidas y sus indices 
# correspondientes. Esto es utilizado para el machine learning.
for x, documento in enumerate(auxX):
    cubeta = []
    auxPalabra = [stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
        if w in auxPalabra:
            cubeta.append(1)
        else:
             cubeta.append(0)
    filaSalida = salidaVacia[:]
    filaSalida [tags.index(auxY[x])] = 1
    entrenamiento.append(cubeta)
    salida.append(filaSalida)

# print(entrenamiento)
# print(salida)

# Cambio las listas a arreglos de la libreria numpy
entrenamiento = numpy.array(entrenamiento)
salida = numpy.array(salida)

# Coloco el espacio de trabajo de mi red neuronal en limpio
tensorflow.reset_default_graph()
# Genero la red de neuronas input a base de la cantidad de palabras entrenadas
# que tengo
red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
# Aca seteamos la cantidad de Hidden layers de neuronas
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
# Capa de neuronas de salida
red = tflearn.fully_connected(red,len(salida[0]), activation='softmax')
# Seteo las probabilidades de que tan eficaz es la clasificacion por tag.
red = tflearn.regression(red)
# Creo el modelo de aprendizaje a base de la red generada.
modelo = tflearn.DNN(red)
# Setteo los valores que van a ser neuronas de entrada, salida y la cantidad
# de Epochs que la red va a correr.
modelo.fit(entrenamiento,salida,n_epoch=500,batch_size=10, show_metric=True)
modelo.save("modelo.tflearn")

# Preparo file para poder loggear la conversacion con el usuario.
File_object_log = open(r"log.txt","a")


# Funcion principal para correr el chatbot:
def mainBot():
    while True:
        entrada = input("Tu: ")
        
        cubeta = [0 for _ in range(len(palabras))]
        # Quito los caracteres especiales de lo que Ingreso el usuario
        entradaProcesada = nltk.word_tokenize(entrada)
        # Proceso el input para que el chatbot pueda procesar lo ingresado
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        for palabraIndividual in entradaProcesada:
            for i,palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])
        # Preparo indices para dar respuestas.
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]
        # Elijo una de las respuestas para generar
        for tagAux in datos["contenido"]:
          if tagAux["tag"] ==  tag:
             respuesta = tagAux["respuestas"]
        respuesta_dada = random.choice(respuesta)
        print("Bot: ", respuesta_dada)
        # Printeo probabilidades de accuarcy
        print(resultados)
        # Pequeño flag para decidir
        print('Todavia estoy aprendiendo!, Te sirvio mi respuesta? --- (0:Si | 1:No)')
        likeable = input("Tu: ")
        # Loggeo la conversacion
        if(likeable == '1' ):
            File_object_log.write('El usuario pregunto: ' + entrada +'\n')
            File_object_log.write('El chatbot respondio: ' + respuesta_dada +'\n')
            File_object_log.write('El chatbot predijo que corresponde al TAG: ' + tag +'\n')
            File_object_log.write('----------------------------------------------' +'\n')
        print('--------------------------------------------------------------------')

mainBot()