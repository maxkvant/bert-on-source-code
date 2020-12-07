from pathlib import Path
import csv
import json
from typing import Callable

import numpy as np


class TooManyLinesException(Exception):
    pass


class MMRDatasetProject:
    def __init__(self, meta_dir: Path, tokens_dir: Path,
                 vector_method: Callable[[list], np.ndarray], vector_class: Callable[[list], np.ndarray],
                 oversampling: bool, yield_names: bool = False,
                 class_max_lines: int = 1000, method_max_lines: int = 100):
        self.vector_class = vector_class
        self.vector_method = vector_method
        self.oversampling = oversampling
        with open(meta_dir / 'methods.csv') as methods_list_file:
            methods_reader = csv.reader(methods_list_file)
            next(methods_reader)
            self.methods = [
                (m_name, int(m_class), tuple(map(int, m_destinations.split())))
                for _, m_name, _, _, m_class, m_destinations in methods_reader
            ]
        with open(meta_dir / 'classes.csv') as classes_list_file:
            classes_reader = csv.reader(classes_list_file)
            next(classes_reader)
            self.classes = {int(c_id): c_name for c_id, c_name, _, _ in classes_reader}
        self.tokens_dir = tokens_dir
        self.yield_names = yield_names
        self.name = meta_dir.name
        self.class_max_lines = class_max_lines
        self.method_max_lines = method_max_lines

    def _get_class_vector(self, class_id: int) -> np.ndarray:
        class_name = self.classes[class_id]
        with open(self.tokens_dir / 'classes' / f'{class_name}.json') as c_tokens_file:
            c_tokens = json.load(c_tokens_file)
        if len(c_tokens) > self.class_max_lines:
            raise TooManyLinesException
        return self.vector_class(c_tokens)

    def _get_class_without_method_vector(self, method_name: str) -> np.ndarray:
        with open(self.tokens_dir / 'classes_without_methods' / f'{method_name}.json') as cwm_tokens_file:
            cwm_tokens = json.load(cwm_tokens_file)
        if len(cwm_tokens) > self.class_max_lines:
            raise TooManyLinesException
        return self.vector_method(cwm_tokens)

    def _get_method_vector(self, method_name: str) -> np.ndarray:
        with open(self.tokens_dir / 'methods' / f'{method_name}.json') as m_tokens_file:
            m_tokens = json.load(m_tokens_file)
        if len(m_tokens) > self.method_max_lines:
            raise TooManyLinesException
        return self.vector_class(m_tokens)

    def __iter__(self):
        for m_name, m_class, m_destinations in self.methods:
            try:
                m_vector = self._get_method_vector(m_name)
                n_pos_samples = len(m_destinations) if self.oversampling else 1
                mc_name = self.classes[m_class]
            except TooManyLinesException:
                continue

            for m_destination in m_destinations:
                try:
                    mwc_vector = self._get_class_without_method_vector(m_name)
                    for i in range(n_pos_samples):
                        if self.yield_names:
                            yield m_name, mc_name, m_vector, mwc_vector, True
                        else:
                            yield m_vector, mwc_vector, True
                except TooManyLinesException:
                    pass

                try:
                    md_vector = self._get_class_vector(m_destination)
                    destination_name = self.classes[m_destination]
                    if self.yield_names:
                        yield m_name, destination_name, m_vector, md_vector, False
                    else:
                        yield m_vector, md_vector, False
                except TooManyLinesException:
                    pass


class MMRDataset:
    def __init__(self, orig_root: Path, tokenized_root: Path,
                 vector_method: Callable[[list], np.ndarray], vector_class: Callable[[list], np.ndarray],
                 oversampling: bool, yield_names: bool):
        self.yield_names = yield_names
        project_names = [p.name for p in orig_root.iterdir() if p.is_dir() and p.name[0] != '.']
        self.projects = [
            MMRDatasetProject(orig_root / project_name, tokenized_root / project_name,
                              vector_method, vector_class, oversampling, yield_names)
            for project_name in project_names
        ]

    def __iter__(self):
        for project in self.projects:
            if self.yield_names:
                for method_data in project:
                    yield (project.name, *method_data)
            yield from project