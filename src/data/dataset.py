import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    
    def __init__(self, 
                 dataset_path, 
                 test_size=0.1):
        """Initialize Dataset object

        Args:
            dataset_path (str): Dataset path to load
            test_size (float, optional): Default test split size. Defaults to 0.1.
        """
        self.dataset_path = dataset_path
        self.test_size = test_size
        
    def load_dataset(self):
        """Load the dataset from the dataset path

        Returns:
            Pandas dataframe: loaded dataset
        """
        return pd.read_csv(self.dataset_path)
    
    def get_splits(self):
        """
        Generate training and validation splits in the form of:
        x_train, x_validation, y_train, y_validation
            
        When a test size of 0 is passed in the initialization, 
        the return values are only: 
        x_train, y_train
        This is useful when loading test sets to perform evaluation.
        
        Returns:
            tuple: tuple with the splits discussed above
        """
        df = self.load_dataset()
        y = df["class"]
        x = df.drop("class", axis=1)
        
        if self.test_size == 0:
            return x, y
        else:
            return train_test_split(x, 
                                    y, 
                                    test_size=self.test_size, 
                                    random_state=0)