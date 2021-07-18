from tensorflow import keras
from tensorflow.keras import layers

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
        
        self.model = keras.Sequential(layers)
        self.model.compile(optimizer=optimizer, 
                           loss=loss, 
                           metrics=metrics) 
    
    def fit(self, 
            x_train, 
            y_train, 
            validation_data=(),
            batch_size=500, 
            epochs=10, 
            ):
        
        return self.model.fit(
            x_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=validation_data, 
            verbose=2
        )
        
        
    