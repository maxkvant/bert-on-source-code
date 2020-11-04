from pathlib import Path
import csv
import json
from enum import Enum

from absl import app, flags
from tensor2tensor.data_generators import text_encoder

from cubert.code_to_subtokenized_sentences import code_to_cubert_sentences


FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir_path', None,
                    'Path to the Python source code files directory.')
flags.DEFINE_string('output_dir_path', None,
                    'Path to the output directory subtokenized source code.')


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


def read_item(file_path: Path, offset: int):
    with file_path.open() as file:
        file_content = file.read()
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
    return file_content[offset:pos + 1]


def main(argv):
    tokenizer = FLAGS.tokenizer.value()
    subword_tokenizer = text_encoder.SubwordTextEncoder(FLAGS.vocabulary_filepath)

    input_dir_path = Path(FLAGS.input_dir_path)
    output_dir_path = Path(FLAGS.output_dir_path)

    for project_dir in input_dir_path.iterdir():
        project_name = project_dir.name
        if not project_dir.is_dir() or project_name[0] == '.':
            continue
        project_out_path = output_dir_path / project_name
        project_out_path.mkdir(parents=True)
        methods_out_dir = project_out_path / 'methods'
        classes_out_dir = project_out_path / 'classes'
        methods_out_dir.mkdir(parents=True)
        classes_out_dir.mkdir(parents=True)

        with (project_dir / 'methods.csv').open() as methods_file:
            methods_reader = csv.reader(methods_file)
            next(methods_reader)
            for _, method_name, method_file_path, method_offset, _, _ in methods_reader:
                try:
                    method_body = read_item(project_dir / project_name / method_file_path, int(method_offset))
                    method_subtokenized_sentences = code_to_cubert_sentences(
                        code=method_body, initial_tokenizer=tokenizer, subword_tokenizer=subword_tokenizer)
                    with (methods_out_dir / f'{method_name}.json').open('w') as method_out_file:
                        json.dump(method_subtokenized_sentences, method_out_file)
                except IndexError as e:
                    print(f'Failed to read {method_name}')
        with (project_dir / 'classes.csv').open() as classes_file:
            classes_reader = csv.reader(classes_file)
            next(classes_reader)
            for _, class_name, class_file_path, class_offset in classes_reader:
                try:
                    class_body = read_item(project_dir / project_name / class_file_path, int(class_offset))
                    class_subtokenized_sentences = code_to_cubert_sentences(
                        code=class_body, initial_tokenizer=tokenizer, subword_tokenizer=subword_tokenizer)
                    with (classes_out_dir / f'{class_name}.json').open('w') as class_out_file:
                        json.dump(class_subtokenized_sentences, class_out_file)
                except IndexError as e:
                    print(f'Failed to read {class_name}')


if __name__ == '__main__':
    flags.mark_flag_as_required('vocabulary_filepath')
    flags.mark_flag_as_required('input_dir_path')
    flags.mark_flag_as_required('output_dir_path')
    app.run(main)
