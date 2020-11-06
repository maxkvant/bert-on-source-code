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
        cs_tokens = {}
        with open(project_dir / 'classes.csv') as classes_file:
            classes_reader = csv.reader(classes_file)
            next(classes_reader)
            for c_id, c_name, c_path, c_offset in classes_reader:
                class_body = read_item(project_dir / project_name / c_path, int(c_offset))
                class_subtokenized_sentences = code_to_cubert_sentences(
                    code=class_body, initial_tokenizer=self.tokenizer, subword_tokenizer=self.sub_word_tokenizer)
                cs_tokens[c_id] = class_subtokenized_sentences
                with open(classes_out_dir / f'{c_name}.json', 'w') as class_out_file:
                    json.dump(class_subtokenized_sentences, class_out_file)
        return cs_tokens

    def _tokenize_methods(self, project_dir: Path, project_out_dir: Path, project_name: str):
        methods_out_dir = project_out_dir / 'methods'
        methods_out_dir.mkdir(parents=True)
        ms_tokenized = []
        with open(project_dir / 'methods.csv') as methods_file:
            methods_reader = csv.reader(methods_file)
            next(methods_reader)
            for m_id, m_name, m_path, m_offset, m_src_class, _ in methods_reader:
                method_body = read_item(project_dir / project_name / m_path, int(m_offset))
                method_subtokenized_sentences = code_to_cubert_sentences(
                    code=method_body, initial_tokenizer=self.tokenizer, subword_tokenizer=self.sub_word_tokenizer)
                ms_tokenized.append((m_name, m_src_class, method_subtokenized_sentences))
                with (methods_out_dir / f'{m_name}.json').open('w') as method_out_file:
                    json.dump(method_subtokenized_sentences, method_out_file)
        return ms_tokenized

    @staticmethod
    def _remove_methods_from_classes(project_out_dir, cs_tokens, ms_tokenized):
        cwm_dir = project_out_dir / 'classes_without_methods'
        cwm_dir.mkdir(parents=True)
        for m_name, m_src_class, m_tokens in ms_tokenized:
            c_tokens = cs_tokens[m_src_class]
            c_pos = 0
            while c_tokens[c_pos:c_pos + len(m_tokens)] != m_tokens:
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
        cs_tokens = self._tokenize_classes(project_dir, project_out_dir, project_name)
        ms_tokens = self._tokenize_methods(project_dir, project_out_dir, project_name)
        self._remove_methods_from_classes(project_out_dir, cs_tokens, ms_tokens)


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
