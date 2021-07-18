import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    
    def __init__(self, 
                 dataset_path, 
                 test_size=0.1):
        self.dataset_path = dataset_path
        self.test_size = test_size
        
    def load_dataset(self):
        return pd.read_csv(self.dataset_path)
    
    def get_splits(self):
        df = self.load_dataset()
        # get labels and features
        y = df["class"]
        x = df.drop("class", axis=1)
        
        if self.test_size == 0:
            return x, y
        else:
            return train_test_split(x, 
                                    y, 
                                    test_size=self.test_size, 
                                    random_state=0)