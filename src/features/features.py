import librosa
import numpy as np
import pandas as pd
from glob import glob 
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings("ignore")

class Features:
    
    def __init__(self, 
                 metadata_path="data/raw/metadata/UrbanSound8K.csv", 
                 audio_files_path="data/raw/audio", 
                 folds=5):
        """
        Initialize Feature object

        Args:
            metadata_path (str, optional): csv dataset metadata. Defaults to "data/UrbanSound8K/metadata/UrbanSound8K.csv".
            audio_files_path (str, optional): folder with the audio files. Defaults to "data/UrbanSound8K/audio".
            folds (int, optional): first N folds to extract. Defaults to 5.
        """
        self.metadata_path = metadata_path
        self.audio_files_path = audio_files_path
        self.folds = folds

    def get_features(self, audio_file):
        """
        Extract features from an audio file using Librosa

        Args:
            audio_file (str): audio file path to load
        """
        def array_map(array):
            return [
                    np.min(array), 
                    np.max(array),
                    np.average(array)
                ]
        
        def array_reduce(a, b):
            return a + b
        
        y, sr = librosa.load(audio_file, sr = None)
            
        mfcc = librosa.feature.mfcc(y, sr)
        chroma_stft = librosa.feature.chroma_stft(y, sr)
        chroma_cqt = librosa.feature.chroma_cqt(y, sr)
        
        total = np.concatenate((chroma_stft, chroma_cqt), axis=0)
        return np.apply_along_axis(array_map, 1, total).flatten()

    def get_training_dataframe(self):
        """
        Get the dataframe that represent the training dataset. 
        The structure is [class, feature_1, feature_2, ...]

        Returns:
            Pandas Dataframe: training dataset
        """
        data = pd.read_csv(self.metadata_path)
        training_data = data[data["fold"] <= self.folds]
        values = training_data[["slice_file_name", "fold", "classID"]].values

        def m(x):
            audio_path = f"{self.audio_files_path}/fold{x[1]}/{x[0]}"
            return np.insert(self.get_features(audio_path), 0, int(x[2]))
        
        feature_arrays = []
        for i in tqdm(range(len(values))):
            feature_arrays.append(m(values[i]))
        
        columns = ["class"] + [f"f_{i}" for i in range(len(feature_arrays[0]) -1)]
        return pd.DataFrame(feature_arrays, columns=columns)