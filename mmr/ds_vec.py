from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np

from cubert_wrapper import CuBertWrapper


def calc_vectors_dir(in_path: Path, out_path: Path, model: CuBertWrapper, max_len: int):
    for tokens_file_path in in_path.glob('*.json'):
        with tokens_file_path.open() as tokens_file:
            tokens = json.load(tokens_file)
        rel_path = tokens_file_path.relative_to(in_path)
        vecs_path = out_path / rel_path.with_suffix('.npy')
        if vecs_path.is_file() or len(tokens) > max_len:
            continue
        vecs_path.parent.mkdir(exist_ok=True, parents=True)
        token_vectors, line_vectors = model(tokens)
        with open(vecs_path, 'wb') as out_file:
            np.savez(out_file, np.stack(line_vectors))


def calc_vectors(in_path: Path, out_path: Path, model: CuBertWrapper, 
                 max_class_len: int, max_method_len: int):
    for project_path in in_path.iterdir():
        project_name = project_path.name
        project_vecs_path = out_path / project_name
        calc_vectors_dir(project_path / 'methods', project_vecs_path / 'methods', model, max_method_len)
        calc_vectors_dir(project_path / 'classes', project_vecs_path / 'classes', model, max_class_len)
        calc_vectors_dir(project_path / 'classes_without_methods', project_vecs_path / 'classes_without_methods', model, max_class_len)
                
                
if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('in_path', type=Path)
    arg_parser.add_argument('out_path', type=Path)
    arg_parser.add_argument('model_path', type=Path)
    arg_parser.add_argument('vocab_path', type=Path)
    arg_parser.add_argument('--max_class_len', type=int, default=1000)
    arg_parser.add_argument('--max_method_len', type=int, default=100)
    args = arg_parser.parse_args()
    
    model = CuBertWrapper(args.model_path, args.vocab_path)
    calc_vectors(args.in_path, args.out_path, model, args.max_class_len, args.max_method_len)
