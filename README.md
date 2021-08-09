# Urban sound classification

The goal of the project is to build a neural network capable of classification on the [Urban Sound 8k dataset](https://urbansounddataset.weebly.com/urbansound8k.html).
A detailed discussion about the methodology followed can be found on the [project report](https://github.com/tomfran/urban-sound-classification/blob/main/report/report.pdf).

## Requirements
Libraries used in the project are the following:
- pandas
- numpy
- matplotlib
- tensorflow
- librosa
- dask
- keras_nightly
- keras
- scikit_learn

You can install them using the following command
```[shell]
pip install -r src/requirements.txt
```

## Project structure

The project folder is structured as follows:
- **data/** contains processed and raw data. To reproduce results using the dataset, 
put the folds folders inside **data/raw/audio**, then put the metadata file inside **data/raw/metadata**
- **models/** contains trained models, namely the scaler and pca used in the project
- **notebooks/** contains the Jupyter notebooks used execute the code
- **src/** contains **data**, **model** and **utils** sub-folders, with code regarding the different parts 
of the project
- **report/** contains the project report written in Latex
