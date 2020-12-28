## Method name prediction

To run evaluation scripts you would need to manually install cubert package as described above, then proceed to install the rest of the packages via: `pip install -r method-name-prediction/requirements.txt`.

Git LFS pointers to gzip’ed training and evaluation data are stored in `method-name-prediction/data`. To proceed one would need to clone the actual files and manually gunzip them.

Run `evaluate_cubert.sh`, `evaluate_transformer.sh` or `evaluate_tfidf.sh` to obtain reported metrics for the corresponding models. For sequence-to-sequence models you can edit bash script to specify `--device` argument. With optional argument `--out-file` you can provide `.csv` file to save metrics obtained after evaluation.

Sequence-to-sequence transformer models are stored at huggingface and will be automatically downloaded during the execution of the scripts. Fitting tf-idf model doesn't take long, thus it is computed anew while executing `evaluate_tfidf.sh.

There are also notebooks with preprocessing and training, paths to data and vocabularies inside of them are specified according to their location on VM we worked on, so if somebody wishes to reproduce the training process they would need to change those variables accordingly.
