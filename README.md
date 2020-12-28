# Bert on Source Code

In this project, we apply [CuBERT](https://github.com/google-research/google-research/tree/master/cubert) to the following ml4se tasks: 
* Move method refactoring
* Method name prediction

## Project structure

### Directories
* `MMR` - experiments on move method refactoring task
* `method-name-prediction` - experiments on method name prediction task for CodeSearchNet dataset
* `resources` - scripts for installing cubert dependency

### Files
* [bert_for_source_code_report.pdf](bert_for_source_code_report.pdf) - 
* `cubert_wrapper.py` - wrapper for cubert model to obtain contextualized embeddings directly 

## CuBERT dependency installation

```bash
cd resources
./install.sh
```

## Move method refactoring task
Files related to the MMR task are located in [mmr](/mmr) directory.  
Training and evaluation of classifiers is present in [averaging.ipynb](/mmr/averaging.ipynb) (for classifiers applied to averaged lines) and [nns.ipynb](/mmr/nns.ipynb) (for classifiers operating on sequences) notebooks.  
Before that the dataset should be tokenized ([ds_preprocessing.py](/mmr/ds_preprocessing.py)) and vectorized ([ds_vec.py](/mmr/ds_vec.py)). 
