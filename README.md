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
* [bert_for_source_code_report.pdf](../blob/master/bert_for_source_code_report.pdf) - our method described
* `cubert_wrapper.py` - wrapper for cubert model to obtain contextualized embeddings directly 

## CuBERT dependency installation

```bash
cd resources
./install.sh
```



