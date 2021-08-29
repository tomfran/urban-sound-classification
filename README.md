# Urban sound classification

The goal of the project is to build a neural network capable of classification on the [Urban Sound 8k dataset](https://urbansounddataset.weebly.com/urbansound8k.html).<br>
An in depth overview of the project is present on the [project report](https://github.com/tomfran/urban-sound-classification/blob/main/report/report.pdf).

## Project structure
The project folder is structured as follows:
- **data/** contains processed and raw data. To reproduce results using the dataset, 
put the folds folders inside **data/raw/audio**, then put the metadata file inside **data/raw/metadata**
- **models/** contains trained models, namely the scaler and pca used in the project
- **notebooks/** contains the Jupyter notebooks used execute the code
- **src/** contains **data**, **model** and **utils** sub-folders, with code regarding the different parts 
of the project
- **report/** contains the project report written in Latex

## Requirements
Libraries used in the project are the following: *pandas*, *numpy*, *matplotlib*, *tensorflow*, *librosa*, *dask*, *keras_nightly*, *keras*, *scikit_learn*
 
You can install them using the following command
```[shell]
pip install -r src/requirements.txt
```

## Methodology
The methodology followed in the project can be seen in the various jupyter notebooks.

### Feature extraction and dataset creation
In the [first notebook](https://github.com/tomfran/urban-sound-classification/blob/main/notebooks/01_dataset.ipynb) audio features are extracted using Librosa library and scaling 
is applied. <br>
In the [second notebook](https://github.com/tomfran/urban-sound-classification/blob/main/notebooks/02_dataset_extended.ipynb), more features are extracted and PCA feature selection is exploited to reduce the dataset dimensionality.

### Cross validation on the training sets
To understand what training set is best suited for the project, 
cross validation is performed on the initial, scaled, extended and pca dataset obtained at the previous step.
The results are presented in the [third notebook](https://github.com/tomfran/urban-sound-classification/blob/main/notebooks/03_cross_validation_mlp.ipynb).

### Hyperparameter tuning 
After selecting the best dataset from the cross validation results, 
a Random Search is performed to optimize the network hyperparameters, 
details about results as well as test set evaluation can be found on the [last notebook](https://github.com/tomfran/urban-sound-classification/blob/main/notebooks/04_hyperparameter_tuning_mlp.ipynb).
