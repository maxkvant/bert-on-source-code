import argparse as ag
from pathlib import Path
import json
from transformers import EncoderDecoderModel

from token_to_index import TokenToIndexConverter
from sequence_to_sequence_dataset import get_method_name_dataset
from utils import compute_metrics
from catalyst.utils import set_global_seed
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch


def parse_args():
    parser = ag.ArgumentParser()
    parser.add_argument('--model', type=str, help="Path to huggingface EncoderDecoderModel")
    parser.add_argument('--device', type=str, help="Device to run model on, possible values are: cpu, cuda")
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


if __name__ == "__main__":
    args = parse_args()

    token_to_index = TokenToIndexConverter(
        "vocab/github_python_minus_ethpy150open_deduplicated_vocabulary.txt"
    )


    set_global_seed(19)


    DATA_FOLDER = Path("data")
    train = read_jsonl(DATA_FOLDER / "train_preprocessed.jsonl")
    test = read_jsonl(DATA_FOLDER / "test_preprocessed.jsonl")

    model = EncoderDecoderModel.from_pretrained(args.model)

    train_dataset = get_method_name_dataset(train, token_to_index, token_to_index.pad_index, model.encoder.config.max_position_embeddings)
    test_dataset = get_method_name_dataset(test, token_to_index, token_to_index.pad_index, model.encoder.config.max_position_embeddings)

    DEVICE = torch.device(args.device)
    model.to(DEVICE).eval()

    metrics = []

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            src, ref = test_dataset[i]
            name = token_to_index.decode_list(ref)
            gen = beam_search(
                src,
                model,
                bos_id=token_to_index.bos_index,
                pad_id=token_to_index.pad_index,
                end_id=token_to_index.eos_index,
                device=DEVICE
            )
            generated = sorted(
                [{"cand": token_to_index.decode_list(t), "score": s} for t, s in gen],
                key=lambda e: e["score"],
                reverse=True
            )

            candidates = [g["cand"] for g in generated]
            metrics.append(compute_metrics(name, candidates))

    metrics = pd.DataFrame(metrics)
    print(f"Mean metrics for {args.model} are as follows:")
    print(metrics.mean())
    
    if args.out_file is not None:
        print(f"Saving metrics to {args.out_file}")
        metrics.mean().to_csv(args.out_file)
