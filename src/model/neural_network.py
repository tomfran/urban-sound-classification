import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from numpy.random import seed
seed(1)
tensorflow.random.set_seed(1)

class NeuralNetwork:
    
    def __init__(self, 
                 layers=[layers.Dense(units=132, activation='relu'),
                         layers.Dense(91, activation='relu'),
                         layers.Dense(50, activation='relu'),
                         layers.Dense(10, activation='softmax')], 
                 optimizer=keras.optimizers.RMSprop(),
                 loss=keras.losses.SparseCategoricalCrossentropy(),
                 metrics=[keras.metrics.SparseCategoricalAccuracy()]
                 ):
        """
        Initialize Neural Network object.
        It compiles the network.

        Args:
            layers (keras layer list, optional): list of layers to use in the neural network. 
            Defaults to [layers.Dense(units=132, activation='relu'), 
                         layers.Dense(91, activation='relu'), 
                         layers.Dense(50, activation='relu'), 
                         layers.Dense(10, activation='softmax')].
            
            optimizer (keras optimizer, optional): optimizer to use on the network. 
            Defaults to keras.optimizers.RMSprop().
            
            loss (keras loss, optional): loss to use on the network. 
            Defaults to keras.losses.SparseCategoricalCrossentropy().
            
            metrics (keras metrics, optional): metrics to use. 
            Defaults to [keras.metrics.SparseCategoricalAccuracy()].
        """
        
        self.model = keras.Sequential(layers)
        self.model.compile(optimizer=optimizer, 
                           loss=loss, 
                           metrics=metrics) 
    
    def fit(self, 
            x_train, 
            y_train, 
            validation_data=(),
            batch_size=128, 
            epochs=10, 
            verbose=1
            ):

        return self.model.fit(
            x_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=validation_data, 
            verbose=verbose
        )
        
        
    