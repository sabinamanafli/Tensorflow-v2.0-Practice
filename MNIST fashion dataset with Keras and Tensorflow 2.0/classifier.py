import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense
import numpy as np
import matplotlib.pyplot as plt


#get data
mnist=keras.datasets.fashion_mnist
#load data
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#define class names
class_names=['top','trousers','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']

#normalize data to between 0 1
x_train=x_train/255
x_test=x_train/255

#initialize model
model=Sequential()

#Input layer
model.add(Flatten(input_shape=(28,28)))

#1 hidden layer
model.add(Dense(128,activation='relu'))

#output layer
model.add(Dense(10,activation='softmax'))


#print(model.summary())

#Model Complation
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train model
model.fit(x_train,y_train, epochs=10)

#evaluate model

test_loss,test_accuracy=model.evaluate(x_test,y_test)



