from pathlib import Path
import argparse
import json

import numpy as np

from cubert_wrapper import CuBertWrapper


def calc_vectors_dir(in_path: Path, out_path: Path, model: CuBertWrapper, one_run: bool, *,
                     max_lines: int = -1):
    for tokens_file_path in in_path.glob('*.json'):
        with tokens_file_path.open() as tokens_file:
            tokens = json.load(tokens_file)
        rel_path = tokens_file_path.relative_to(in_path)
        vectors_path = out_path / rel_path.with_suffix('.npy')
        if vectors_path.is_file():
            continue
        if one_run:
            tokens = [sum(tokens, [])]
        else:
            if len(tokens) > max_lines:
                continue
        vectors_path.parent.mkdir(exist_ok=True, parents=True)
        token_vectors, line_vectors = model(tokens)
        with open(vectors_path, 'wb') as out_file:
            np.savez(out_file, np.stack(line_vectors))


def calc_line_vectors(arguments: argparse.Namespace, model: CuBertWrapper):
    for project_path in arguments.in_path.iterdir():
        project_name = project_path.name
        project_vectors_path = arguments.out_path / project_name
        calc_vectors_dir(project_path / 'methods', project_vectors_path / 'methods', model, False,
                         max_lines=arguments.max_method_len)
        calc_vectors_dir(project_path / 'classes', project_vectors_path / 'classes', model, False,
                         max_lines=arguments.max_class_len)
        calc_vectors_dir(project_path / 'classes_without_methods', project_vectors_path / 'classes_without_methods',
                         model, False, max_lines=arguments.max_class_len)


def calc_method_vectors(arguments: argparse.Namespace, model: CuBertWrapper):
    for project_path in arguments.in_path.iterdir():
        project_name = project_path.name
        project_vectors_path = arguments.out_path / project_name
        calc_vectors_dir(project_path / 'methods', project_vectors_path / 'methods', model, True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    subparsers = arg_parser.add_subparsers()

    per_line_parser = subparsers.add_parser('per_line')
    per_line_parser.add_argument('in_path', type=Path)
    per_line_parser.add_argument('out_path', type=Path)
    per_line_parser.add_argument('model_path', type=Path)
    per_line_parser.add_argument('vocab_path', type=Path)
    per_line_parser.add_argument('--max_class_len', type=int, default=1000)
    per_line_parser.add_argument('--max_method_len', type=int, default=100)
    per_line_parser.set_defaults(func=calc_line_vectors)

    whole_methods_parser = subparsers.add_parser('whole_methods')
    whole_methods_parser.add_argument('in_path', type=Path)
    whole_methods_parser.add_argument('out_path', type=Path)
    whole_methods_parser.add_argument('model_path', type=Path)
    whole_methods_parser.add_argument('vocab_path', type=Path)
    whole_methods_parser.set_defaults(func=calc_method_vectors)

    args = arg_parser.parse_args()
    model_wrapper = CuBertWrapper(args.model_path, args.vocab_path)
    args.func(args, model_wrapper)
