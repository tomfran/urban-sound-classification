import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from ..data import Dataset

from numpy.random import seed
seed(1)
tensorflow.random.set_seed(1)

class NeuralNetwork:
    
    @staticmethod
    def create_model(neurons=(144, 70, 40, 30, 10),
                    #  optimizer=keras.optimizers.RMSprop(),
                     learning_rate=0.01,
                     momentum=0.0,
                     loss=keras.losses.SparseCategoricalCrossentropy(),
                     metrics=["accuracy"]):

        ll=[layers.Dense(units=neurons[0], activation='relu')]
        for n in neurons[1:-1]:
            ll.append(layers.Dense(n, activation='relu'))
        ll.append(layers.Dense(neurons[-1], activation='softmax'))

        model = keras.Sequential(ll)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                      loss=loss, 
                      metrics=metrics) 
        return model
    
    @staticmethod
    def optimize_model(method="grid", 
                       param_grid={}, 
                       dataset_path="../data/processed/extended/train_scaled_extended.csv", 
                       iterations=10):

        model = KerasClassifier(build_fn=NeuralNetwork.create_model, verbose=0)
        
        d = Dataset(dataset_path, test_size=0)
        x_train, y_train = d.get_splits()
        
        if method == "grid":
            search = GridSearchCV(estimator=model, 
                                  param_grid=param_grid, 
                                  n_jobs=-1, 
                                  scoring="accuracy",
                                  error_score="raise")
            
        elif method == "random":
            search = RandomizedSearchCV(estimator=model, 
                                        param_distributions=param_grid, 
                                        n_iter=iterations, 
                                        scoring='accuracy', 
                                        n_jobs=-1, 
                                        error_score="raise", 
                                        verbose=2, 
                                        random_state=1)
        
        results = search.fit(x_train, y_train)
        return results