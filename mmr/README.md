## Move method refactoring task
Files related to the MMR task are located in [mmr](/mmr) directory.  
Training and evaluation of classifiers is present in [averaging.ipynb](/mmr/averaging.ipynb) (for classifiers applied to averaged lines) and [nns.ipynb](/mmr/nns.ipynb) (for classifiers operating on sequences) notebooks.  
Before that the dataset should be tokenized ([ds_preprocessing.py](/mmr/ds_preprocessing.py)) and vectorized ([ds_vec.py](/mmr/ds_vec.py)).
