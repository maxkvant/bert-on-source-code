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
* [bert_for_source_code_report.pdf](bert_for_source_code_report.pdf) - our method described
* `cubert_wrapper.py` - wrapper for cubert model to obtain contextualized embeddings directly 

## CuBERT dependency installation

```bash
cd resources
./install_cubert.sh
```

## Move method refactoring task
Files related to the MMR task are located in [mmr](/mmr) directory.  
Training and evaluation of classifiers is present in [averaging.ipynb](/mmr/averaging.ipynb) (for classifiers applied to averaged lines) and [nns.ipynb](/mmr/nns.ipynb) (for classifiers operating on sequences) notebooks.  
Before that the dataset should be tokenized ([ds_preprocessing.py](/mmr/ds_preprocessing.py)) and vectorized ([ds_vec.py](/mmr/ds_vec.py)). 

## Method name prediction

To run evaluation scripts you would need to manually install cubert package as described above, then proceed to install the rest of the packages via: `pip install -r method-name-prediction/requirements.txt`.

Git LFS pointers to gzip’ed training and evaluation data are stored in `method-name-prediction/data`. To proceed one would need to clone the actual files and manually gunzip them.

Run `evaluate_cubert.sh`, `evaluate_transformer.sh` or `evaluate_tfidf.sh` to obtain reported metrics for the corresponding models. For sequence-to-sequence models you can edit bash script to specify `--device` argument. With optional argument `--out-file` you can provide `.csv` file to save metrics obtained after evaluation.

Sequence-to-sequence transformer models are stored at huggingface and will be automatically downloaded during the execution of the scripts. Fitting tf-idf model doesn't take long, thus it is computed anew while executing `evaluate_tfidf.sh.

There are also notebooks with preprocessing and training, paths to data and vocabularies inside of them are specified according to their location on VM we worked on, so if somebody wishes to reproduce the training process they would need to change those variables accordingly.
