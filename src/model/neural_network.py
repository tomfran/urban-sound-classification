import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.utils import class_weight
from keras.wrappers.scikit_learn import KerasClassifier
from ..data import Dataset
from keras.callbacks import EarlyStopping
import numpy as np
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(1)

import warnings  
warnings.filterwarnings("ignore",category=FutureWarning)

class NeuralNetwork:
    
    @staticmethod
    def create_model(neurons=(144, 70, 40, 30, 10),
                     learning_rate=0.01,
                     momentum=0.0,
                     optimizer="sgd",
                     loss=keras.losses.SparseCategoricalCrossentropy(),
                     metrics=["accuracy"]):
        """Neural network creation function

        Args:
            neurons (tuple, optional): Number of neurons per layer. Defaults to (144, 70, 40, 30, 10).
            learning_rate (float, optional): Optimizer learning rate. Defaults to 0.01.
            momentum (float, optional): Optimizer momentum. Defaults to 0.0.
            loss ([type], optional): Loss function to use. Defaults to keras.losses.SparseCategoricalCrossentropy().
            metrics (list, optional): Metrics to compile. Defaults to ["accuracy"].

        Returns:
            Keras model: Compiled keras neural network
        """
        ll=[layers.Dense(units=neurons[0], activation='relu')]
        for n in neurons[1:-1]:
            ll.append(layers.Dense(n, activation='relu'))
        ll.append(layers.Dense(neurons[-1], activation='softmax'))

        model = keras.Sequential(ll)
        if optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer == "adam":
            opt = keras.optimizers.Adam()

        model.compile(optimizer=opt,
                      loss=loss, 
                      metrics=metrics) 
        return model
    
    @staticmethod
    def optimize_model(method="grid", 
                       param_grid={}, 
                       dataset_path="../data/processed/extended/train_scaled_extended.csv", 
                       iterations=10):
        """Optimize a model by performing random or grid search

        Args:
            method (str, optional): Random or grid search. Defaults to "grid".
            param_grid (dict, optional): Parameters grid. Defaults to {}.
            dataset_path (str, optional): Path to the dataset to load. Defaults to "../data/processed/extended/train_scaled_extended.csv".
            iterations (int, optional): Number of iteration for the random search. Defaults to 10.

        Returns:
            Scikitlearn search results: results of the parameter optimization
        """
        model = KerasClassifier(build_fn=NeuralNetwork.create_model, verbose=0)
        
        d = Dataset(dataset_path, test_size=0)
        x_train, y_train = d.get_splits()
        
        if method == "grid":
            search = GridSearchCV(estimator=model, 
                                  param_grid=param_grid, 
                                  n_jobs=-1, 
                                  scoring="accuracy",
                                  error_score="raise", 
                                  verbose=2, 
                                  cv=StratifiedKFold(n_splits=5))
            
        elif method == "random":
            search = RandomizedSearchCV(estimator=model, 
                                        param_distributions=param_grid, 
                                        n_iter=iterations, 
                                        scoring='accuracy', 
                                        n_jobs=-1, 
                                        error_score="raise", 
                                        verbose=2, 
                                        random_state=1, 
                                        cv=StratifiedKFold(n_splits=5))
            
        stopper = EarlyStopping(monitor='accuracy', patience=3, verbose=0)

        fit_params = dict(callbacks=[stopper])
        
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        weights_dict = dict(zip(np.unique(y_train), class_weights))
        
        results = search.fit(x_train, y_train, class_weight=weights_dict, **fit_params)
        return results