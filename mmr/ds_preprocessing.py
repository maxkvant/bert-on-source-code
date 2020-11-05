from pathlib import Path
import csv
import json
from enum import Enum
from multiprocessing import Pool

from absl import app, flags
from tensor2tensor.data_generators import text_encoder

from cubert.cubert_tokenizer import CuBertTokenizer
from cubert.code_to_subtokenized_sentences import code_to_cubert_sentences


FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None,
                    'Path to the Python source code files directory.')
flags.DEFINE_string('output_dir', None,
                    'Path to the output directory sub_tokenized source code.')


class ReadState(Enum):
    DEFAULT = 1
    SLASH = 2
    SINGLE_LINE_COMMENT = 3
    MULTILINE_COMMENT = 4
    MULTILINE_COMMENT_STAR = 5
    STRING_LITERAL = 6
    STRING_LITERAL_ESCAPED = 7
    CHAR_LITERAL = 8
    CHAR_LITERAL_ESCAPED = 9


def find_item_end(file_content: str, offset: int) -> int:
    pos = offset
    balance = 0
    state = ReadState.DEFAULT
    while True:
        c = file_content[pos]
        if state == ReadState.DEFAULT:
            if c == '{':
                balance += 1
            elif c == '}':
                balance -= 1
                if balance == 0:
                    break
            elif c == '/':
                state = ReadState.SLASH
            elif c == '"':
                state = ReadState.STRING_LITERAL
            elif c == "'":
                state = ReadState.CHAR_LITERAL
        elif state == ReadState.SLASH:
            if c == '/':
                state = ReadState.SINGLE_LINE_COMMENT
            elif c == '*':
                state = ReadState.MULTILINE_COMMENT
            else:
                state = ReadState.DEFAULT
        elif state == ReadState.SINGLE_LINE_COMMENT:
            if c == '\n':
                state = ReadState.DEFAULT
        elif state == ReadState.MULTILINE_COMMENT:
            if c == '*':
                state = ReadState.MULTILINE_COMMENT_STAR
        elif state == ReadState.MULTILINE_COMMENT_STAR:
            if c == '/':
                state = ReadState.DEFAULT
            elif c != '*':
                state = ReadState.MULTILINE_COMMENT
        elif state == ReadState.STRING_LITERAL:
            if c == '"':
                state = ReadState.DEFAULT
            elif c == '\\':
                state = ReadState.STRING_LITERAL_ESCAPED
        elif state == ReadState.STRING_LITERAL_ESCAPED:
            state = ReadState.STRING_LITERAL
        elif state == ReadState.CHAR_LITERAL:
            if c == "'":
                state = ReadState.DEFAULT
            elif c == '\\':
                state = ReadState.CHAR_LITERAL_ESCAPED
        elif state == ReadState.CHAR_LITERAL_ESCAPED:
            state = ReadState.CHAR_LITERAL
        pos += 1
    return pos + 1


def read_item(file_path: Path, offset: int) -> str:
    with file_path.open() as file:
        file_content = file.read()
    scope_end = find_item_end(file_content, offset)
    return file_content[offset:scope_end]


def read_class_without_method(file_path: Path, class_offset: int, method_offset: int) -> str:
    with file_path.open() as file:
        file_content = file.read()
    class_end = find_item_end(file_content, class_offset)
    method_end = find_item_end(file_content, method_offset)
    return file_content[class_offset:method_offset] + file_content[method_end:class_end]


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
        with open(project_dir / 'classes.csv') as classes_file:
            classes_reader = csv.reader(classes_file)
            next(classes_reader)
            for _, class_name, class_file_path, class_offset in classes_reader:
                class_body = read_item(project_dir / project_name / class_file_path, int(class_offset))
                class_subtokenized_sentences = code_to_cubert_sentences(
                    code=class_body, initial_tokenizer=self.tokenizer, subword_tokenizer=self.sub_word_tokenizer)
                with (classes_out_dir / f'{class_name}.json').open('w') as class_out_file:
                    json.dump(class_subtokenized_sentences, class_out_file)

    def _tokenize_methods(self, project_dir: Path, project_out_dir: Path, project_name: str):
        methods_out_dir = project_out_dir / 'methods'
        methods_out_dir.mkdir(parents=True)
        with open(project_dir / 'methods.csv') as methods_file:
            methods_reader = csv.reader(methods_file)
            next(methods_reader)
            for _, method_name, m_path, m_offset, _, _ in methods_reader:
                method_body = read_item(project_dir / project_name / m_path, int(m_offset))
                method_subtokenized_sentences = code_to_cubert_sentences(
                    code=method_body, initial_tokenizer=self.tokenizer, subword_tokenizer=self.sub_word_tokenizer)
                with (methods_out_dir / f'{method_name}.json').open('w') as method_out_file:
                    json.dump(method_subtokenized_sentences, method_out_file)

    def _tokenize_classes_without_methods(self, project_dir: Path, project_out_dir: Path, project_name: str):
        with open(project_dir / 'classes.csv') as classes_list_file:
            classes_reader = csv.reader(classes_list_file)
            next(classes_reader)
            class_locations = {int(c_id): (c_path, int(c_offset)) for c_id, _, c_path, c_offset in classes_reader}
        classes_without_methods_dir = project_out_dir / 'classes_without_methods'
        classes_without_methods_dir.mkdir(parents=True)
        with open(project_dir / 'methods.csv') as methods_file:
            methods_reader = csv.reader(methods_file)
            next(methods_reader)
            for _, m_name, m_path, m_offset, m_class_id, _ in methods_reader:
                c_path, c_offset = class_locations[int(m_class_id)]
                class_body_without_method = read_class_without_method(
                    project_dir / project_name / c_path, c_offset, int(m_offset))
                class_without_method_subtokenized_sentences = code_to_cubert_sentences(
                    code=class_body_without_method, initial_tokenizer=self.tokenizer,
                    subword_tokenizer=self.sub_word_tokenizer)
                with open(classes_without_methods_dir / f'{m_name}.json', 'w') as class_without_method_file:
                    json.dump(class_without_method_subtokenized_sentences, class_without_method_file)

    def tokenize_project(self, project_name: str):
        project_dir = self.input_dir / project_name
        project_out_dir = self.output_dir / project_name
        self._tokenize_classes(project_dir, project_out_dir, project_name)
        self._tokenize_classes_without_methods(project_dir, project_out_dir, project_name)
        self._tokenize_methods(project_dir, project_out_dir, project_name)


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
