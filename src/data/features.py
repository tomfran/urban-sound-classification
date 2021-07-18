import librosa
import numpy as np
import pandas as pd
from glob import glob 
import warnings
from datetime import date

import dask
from dask import delayed
from dask.distributed import Client

from pickle import dump, load
from sklearn import preprocessing

"""
Ignore the warnings as librosa produce some when extracting
feature from certain audio files. They are related to short audios.
"""
warnings.filterwarnings("ignore")

class Features:
    
    def __init__(self, 
                 metadata_path="data/raw/metadata/UrbanSound8K.csv", 
                 audio_files_path="data/raw/audio", 
                 save_path="data/processed",
                 save_name="train",
                 folds=[1,2,3,4,6], 
                 workers=4):
        """
        Initialize Features object

        Args:
            metadata_path (str, optional): path to the metadata file. Defaults to "data/raw/metadata/UrbanSound8K.csv".
            audio_files_path (str, optional): path to the audio files. Defaults to "data/raw/audio".
            save_path (str, optional): path where to save the processed dataframe. Defaults to "data/processed".
            save_name (str, optional): dataframe name, it will be concatenated to the current date. Defaults to "train".
            folds (list, optional): folds to open. Defaults to [1,2,3,4,6].
            workers (int, optional): number of workers for Dask client. Defaults to 4.
        """
        self.metadata_path = metadata_path
        self.audio_files_path = audio_files_path
        self.folds = folds
        self.save_path = f"{save_path}/{save_name}_{date.today().strftime('%m_%d_%Y')}.csv"
        self.workers = workers

    def get_features(self, audio_file):
        """
        Extract features from an audio file

        Args:
            audio_file (str): path to the audio file
        
        Returns: 
            Numpy array: extracted features
        """
        def array_map(array):
            return [
                    np.min(array), 
                    np.max(array),
                    np.median(array), 
                    np.mean(array)
                ]
        
        def array_reduce(a, b):
            return a + b
        
        y, sr = librosa.load(audio_file, sr = None)
            
        mfcc = librosa.feature.mfcc(y, sr)
        chroma_stft = librosa.feature.chroma_stft(y, sr)
        rms = librosa.feature.rms(y, sr)
        
        total = np.concatenate((mfcc, chroma_stft, rms), axis=0)
        return np.apply_along_axis(array_map, 1, total).flatten()

    def get_dataframe(self):
        """
        Get the dataframe that represent the training dataset. 
        The structure is [class, feature_1, feature_2, ...]
        Dask is used to speed up the computation.
        
        Returns:
            Pandas Dataframe: training dataset
        """
        data = pd.read_csv(self.metadata_path)
        training_data = data[data["fold"].isin(self.folds)]
        values = training_data[["slice_file_name", "fold", "classID"]].values
        
        @delayed
        def m(x):
            audio_path = f"{self.audio_files_path}/fold{x[1]}/{x[0]}"
            return np.insert(self.get_features(audio_path), 0, int(x[2]))
        
        Client(n_workers = self.workers)
        
        feature_arrays = []
        for e in values:
            r = m(e)
            feature_arrays.append(r)
        
        feature_arrays = dask.compute(*feature_arrays)
        
        columns = ["class"] + [f"f_{i}" for i in range(len(feature_arrays[0]) -1)]
        return pd.DataFrame(feature_arrays, columns=columns)
    
    def scale_dataframe(self, 
                        dataframe,
                        save_path="../models/scalers/scaler_training.pkl", 
                        save_scaler=False):
        """
        Scale the dataframe and optionally save the scaler on file.

        Args:
            dataframe (Pandas dataframe): dataframe to scale
            save_path (str, optional): path where to save the scaler. Defaults to "../models/scalers/scaler_training.pkl".
            save_scaler (bool, optional): Choose wether to save the scaler on disk or not. Defaults to False.

        Returns:
            Pandas dataframe: scaled dataframe
        """
        x = dataframe.drop("class", axis=1)
        scaler = preprocessing.StandardScaler().fit(x)
        
        scaled_x = scaler.transform(x)
        scaled_df = pd.DataFrame(data=scaled_x, columns=dataframe.columns[1:])
        scaled_df.insert(0, "class", dataframe["class"])
        
        if save_scaler:
            dump(scaler, open(save_path, "wb"))
        
        return scaled_df
    
    def apply_scaling(self, dataframe, scaler_load_path):
        """
        Apply scaling to a dataframe, with a scaler loaded 
        from disk.

        Args:
            dataframe (Pandas dataframe): dataframe to scale
            scaler_load_path (str): path to the scaler on disk

        Returns:
            Pandas dataframe: scaled dataframe
        """
        scaler = load(open(scaler_load_path, "rb"))
        x = dataframe.drop("class", axis=1)
        scaled_x = scaler.transform(x)
        scaled_df = pd.DataFrame(data=scaled_x, columns=dataframe.columns[1:])
        scaled_df.insert(0, "class", dataframe["class"])
        return scaled_df
        
    
    def save_dataframe(self, dataframe):
        """
        Save the dataframe to disk

        Args:
            dataframe (Pandas Dataframe): dataframe to save on disk
        """
        dataframe.to_csv(self.save_path, index=False)  