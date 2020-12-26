from pathlib import Path
import csv
import json
from multiprocessing import Pool
from typing import Callable, List

from absl import app, flags
from tensor2tensor.data_generators import text_encoder
import javalang
from cubert.cubert_tokenizer import CuBertTokenizer
from cubert.code_to_subtokenized_sentences import code_to_cubert_sentences


FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None,
                    'Path to the Python source code files directory.')
flags.DEFINE_string('output_dir', None,
                    'Path to the output directory sub_tokenized source code.')


def find_closing_brace_pos(tokens: List[javalang.tokenizer.JavaToken], open_pos: int) -> int:
    pos = open_pos
    balance = 1
    while balance:
        pos += 1
        if tokens[pos].value == '{':
            balance += 1
        elif tokens[pos].value == '}':
            balance -= 1
    return pos


def find_token_pos_in_source(source: str, token: javalang.tokenizer.JavaToken) -> int:
    pos, cur_line = 0, 1
    while cur_line != token.position.line:
        if source[pos] == '\n':
            cur_line += 1
        pos += 1
    return pos + token.position.column - 1


def find_token_on_top(tokens: List[javalang.tokenizer.JavaToken], start_pos: int,
                      predicate: Callable[[javalang.tokenizer.JavaToken], bool]) -> int:
    cur_token_index = start_pos
    balance = 0
    while not predicate(tokens[cur_token_index]) or balance:
        if isinstance(tokens[cur_token_index], javalang.tokenizer.Separator) and tokens[cur_token_index].value == '(':
            balance += 1
        elif isinstance(tokens[cur_token_index], javalang.tokenizer.Separator) and tokens[cur_token_index].value == ')':
            balance -= 1
        cur_token_index += 1
    return cur_token_index


def read_class(file_path: Path, offset: int) -> str:
    with file_path.open() as file:
        cropped_source = file.read()[offset:]
    tokens = list(javalang.tokenizer.tokenize(cropped_source))
    cur_token_index = find_token_on_top(tokens, 0, lambda token: (
            isinstance(token, javalang.tokenizer.Keyword) and token.value in ('class', 'enum')))
    cur_token_index = find_token_on_top(tokens, cur_token_index, lambda token: (
            isinstance(token, javalang.tokenizer.Separator) and token.value == '{'))
    end_token_index = find_closing_brace_pos(tokens, cur_token_index)
    end_pos = find_token_pos_in_source(cropped_source, tokens[end_token_index])
    return cropped_source[:end_pos + 1]


def read_method(file_path: Path, offset: int) -> str:
    with file_path.open() as file:
        cropped_source = file.read()[offset:]
    tokens = list(javalang.tokenizer.tokenize(cropped_source))
    open_token_index = find_token_on_top(tokens, 0, lambda token: (
            isinstance(token, javalang.tokenizer.Separator) and token.value == '{'))
    end_token_index = find_closing_brace_pos(tokens, open_token_index)
    end_pos = find_token_pos_in_source(cropped_source, tokens[end_token_index])
    return cropped_source[:end_pos + 1]


class MMRDatasetTokenizer:
    def __init__(self, input_dir_path: Path, output_dir_path: Path,
                 tokenizer: CuBertTokenizer, sub_word_tokenizer: text_encoder.SubwordTextEncoder):
        self.tokenizer = tokenizer
        self.sub_word_tokenizer = sub_word_tokenizer
        self.input_dir = input_dir_path
        self.output_dir = output_dir_path
        self.projects_list = [
            project_dir.name
            for project_dir in input_dir_path.iterdir()
            if project_dir.is_dir() and project_dir.name[0] != '.'
        ]

    def _tokenize_classes(self, project_dir: Path, project_out_dir: Path, project_name: str):
        classes_out_dir = project_out_dir / 'classes'
        classes_out_dir.mkdir(parents=True)
        cs_tokens = {}
        with open(project_dir / 'classes.csv') as classes_file:
            classes_reader = csv.reader(classes_file)
            next(classes_reader)
            for c_id, c_name, c_path, c_offset in classes_reader:
                c_body = read_class(project_dir / project_name / c_path, int(c_offset))
                c_tokens = code_to_cubert_sentences(
                    code=c_body, initial_tokenizer=self.tokenizer, subword_tokenizer=self.sub_word_tokenizer)
                cs_tokens[c_id] = c_tokens
                with open(classes_out_dir / f'{c_name}.json', 'w') as class_out_file:
                    json.dump(c_tokens, class_out_file)
        return cs_tokens

    def _tokenize_methods(self, project_dir: Path, project_out_dir: Path, project_name: str):
        methods_out_dir = project_out_dir / 'methods'
        methods_out_dir.mkdir(parents=True)
        ms_tokenized = []
        with open(project_dir / 'methods.csv') as methods_file:
            methods_reader = csv.reader(methods_file)
            next(methods_reader)
            for m_id, m_name, m_path, m_offset, m_src_class, _ in methods_reader:
                m_body = read_method(project_dir / project_name / m_path, int(m_offset))
                m_tokens = code_to_cubert_sentences(
                    code=m_body, initial_tokenizer=self.tokenizer, subword_tokenizer=self.sub_word_tokenizer)
                ms_tokenized.append((m_name, m_src_class, m_tokens))
                with (methods_out_dir / f'{m_name}.json').open('w') as method_out_file:
                    json.dump(m_tokens, method_out_file)
        return ms_tokenized

    @staticmethod
    def _method_found(m_tokens, c_tokens_cut):
        if len(m_tokens) == 1:
            if len(m_tokens[0]) > len(c_tokens_cut[0]):
                return False
            return any(m_tokens[0] == c_tokens_cut[0][i:i+len(m_tokens[0])]
                       for i in range(len(c_tokens_cut[0]) - len(m_tokens[0]) + 1))
        return c_tokens_cut[0][-len(m_tokens[0]):] == m_tokens[0] and \
            c_tokens_cut[1:-1] == m_tokens[1:-1] and \
            c_tokens_cut[-1][:len(m_tokens[-1]) - 1] == m_tokens[-1][:-1]

    @staticmethod
    def _remove_methods_from_classes(project_out_dir, cs_tokens, ms_tokenized):
        cwm_dir = project_out_dir / 'classes_without_methods'
        cwm_dir.mkdir(parents=True)
        for m_name, m_src_class, m_tokens in ms_tokenized:
            c_tokens = cs_tokens[m_src_class]
            c_pos = 0
            while not MMRDatasetTokenizer._method_found(m_tokens, c_tokens[c_pos:c_pos + len(m_tokens)]):
                c_pos += 1
                if c_pos > len(c_tokens) - len(m_tokens):
                    print('Failed to remove method from class:', project_out_dir, m_name)
                    raise AssertionError
            cwm_tokens = c_tokens[:c_pos] + c_tokens[c_pos + len(m_tokens):]
            with open(cwm_dir / f'{m_name}.json', 'w') as cwm_file:
                json.dump(cwm_tokens, cwm_file)

    def tokenize_project(self, project_name: str):
        project_dir = self.input_dir / project_name
        project_out_dir = self.output_dir / project_name
        if project_out_dir.is_dir():
            return
        try:
            cs_tokens = self._tokenize_classes(project_dir, project_out_dir, project_name)
            ms_tokens = self._tokenize_methods(project_dir, project_out_dir, project_name)
            self._remove_methods_from_classes(project_out_dir, cs_tokens, ms_tokens)
        except FileNotFoundError as e:
            print(f'Failed to tokenize project {project_name}: {e}')


def main(argv):
    tokenizer = FLAGS.tokenizer.value()
    sub_word_tokenizer = text_encoder.SubwordTextEncoder(FLAGS.vocabulary_filepath)

    input_dir = Path(FLAGS.input_dir)
    output_dir = Path(FLAGS.output_dir)

    dataset_tokenizer = MMRDatasetTokenizer(input_dir, output_dir, tokenizer, sub_word_tokenizer)
    with Pool() as pool:
        pool.map(dataset_tokenizer.tokenize_project, dataset_tokenizer.projects_list)


if __name__ == '__main__':
    flags.mark_flag_as_required('vocabulary_filepath')
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
