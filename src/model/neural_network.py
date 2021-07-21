import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from numpy.random import seed
seed(1)
tensorflow.random.set_seed(1)

class NeuralNetwork:
    
    @staticmethod
    def create_model(neurons=(144, 70, 40, 30, 10),
                     optimizer=keras.optimizers.RMSprop(),
                     loss=keras.losses.SparseCategoricalCrossentropy(),
                     metrics=["accuracy"]):

        ll=[layers.Dense(units=neurons[0], activation='relu')]
        for n in neurons[1:-1]:
            ll.append(layers.Dense(n, activation='relu'))
        ll.append(layers.Dense(neurons[-1], activation='softmax'))

        model = keras.Sequential(ll)
        model.compile(optimizer=optimizer, 
                      loss=loss, 
                      metrics=metrics) 
        return model