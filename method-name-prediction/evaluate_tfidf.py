import argparse as ag
from pathlib import Path
import json

from utils import compute_metrics
from catalyst.utils import set_global_seed
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity


def parse_args():
    parser = ag.ArgumentParser()
    parser.add_argument('--out-file', type=str, help="Path to output .csv file to store mean metrics", required=False)
    return parser.parse_args()


def beam_search(src, model, bos_id, pad_id, end_id, device, max_len=10, k=5):
    src = src.view(1,-1).to(device)
    src_mask = (src != pad_id).to(device)
    
    memory = None
    
    input_seq = [bos_id]
    beam = [(input_seq, 0)] 
    for i in range(max_len):
        candidates = []
        candidates_proba = []
        for snt, snt_proba in beam:
            if snt[-1] == end_id:
                candidates.append(snt)
                candidates_proba.append(snt_proba)
            else:    
                snt_tensor = torch.tensor(snt).view(1, -1).long().to(device)
                
                if memory is None:
                    memory = model(
                        input_ids=src, 
                        attention_mask=src_mask,
                        decoder_input_ids=snt_tensor,
                        return_dict=False
                    )
                else:
                    memory = model(
                        input_ids=src, 
                        attention_mask=src_mask,
                        decoder_input_ids=snt_tensor,
                        encoder_outputs=(memory[1], memory[-1]),
                        return_dict=False
                    )
                    
                proba = memory[0].cpu()[0,-1, :]
                proba = torch.log_softmax(proba, dim=-1).numpy()
                best_k = np.argpartition(-proba, k - 1)[:k]

                for tok in best_k:
                    candidates.append(snt + [tok])
                    candidates_proba.append(snt_proba + proba[tok]) 
                    
        best_candidates = np.argpartition(-np.array(candidates_proba), k - 1)[:k]
        beam = [(candidates[j], candidates_proba[j]) for j in best_candidates]
        beam = sorted(beam, key=lambda x: -x[1])
        
    return beam


def read_jsonl(path):
    with open(path, 'r') as istream:
        return [json.loads(l) for l in istream]


def get_and_flatten(data, key):
    return [
        [
            tok for line in e[key]
            for tok in line
        ]
        for e in data
    ]


def get_bow(dictionary, corpus):
    return [dictionary.doc2bow(d) for d in corpus]


def evaluate_tfidf(index, tokenized_candidates, tfidf_corpus, tokenized_names):
    metrics = []
    for i, example in tqdm(enumerate(tfidf_corpus)):
        top_5_idx = np.argsort(
            index.get_similarities(example)
        )[-1:-5:-1]
        candidates = [tokenized_candidates[j] for j in top_5_idx]
        metrics.append(compute_metrics(tokenized_names[i], candidates))
    return pd.DataFrame(metrics)


if __name__ == "__main__":
    args = parse_args()

    set_global_seed(33)


    DATA_FOLDER = Path("data")
    train = read_jsonl(DATA_FOLDER / "train_preprocessed.jsonl")
    test = read_jsonl(DATA_FOLDER / "test_preprocessed.jsonl")

    body_key = 'function_body_tokenized'
    train_sentences = get_and_flatten(train, body_key)
    test_sentences = get_and_flatten(test, body_key)

    name_key = 'function_name_tokenized'
    train_names = get_and_flatten(train, name_key)
    test_names = get_and_flatten(test, name_key)
    EOS_TOKEN = '\\u\\u\\uNEWLINE\\u\\u\\u_'
    train_names = [name[:name.index(EOS_TOKEN)] for name in train_names]
    test_names = [name[:name.index(EOS_TOKEN)] for name in test_names]
    
    dictionary = Dictionary(train_sentences)
    
    bow_train = get_bow(dictionary, train_sentences)
    bow_test = get_bow(dictionary, test_sentences)

    tfidf_train = TfidfModel(bow_train)[bow_train]
    tfidf_test = TfidfModel(bow_test)[bow_test]

    index = SparseMatrixSimilarity(tfidf_train, num_features=len(dictionary))
    
    test_metrics = evaluate_tfidf(index, train_names, tfidf_test, test_names)
    print("Metrics for tf-idf model:")
    print(test_metrics.mean())
    
    if args.out_file is not None:
        test_metrics.mean().to_csv(args.out_file)
