from pathlib import Path
import csv
import json
from typing import Callable

import numpy as np


class TooManyLinesException(Exception):
    pass


def _create_vectors(folder: Path, name: str, to_vector: Callable, precalculated: bool, max_lines: int):
    if precalculated:
        path = folder / f'{name}.npy'
        if not path.is_file():
            raise TooManyLinesException
        class_line_vectors = np.load(str(path))['arr_0']
        return to_vector(class_line_vectors)
    else:
        path = folder / f'{name}.json'
        with open(path) as tokens_file:
            tokens = json.load(tokens_file)
        if 0 < max_lines < len(tokens):
            raise TooManyLinesException
        return to_vector(tokens)


class MMRDatasetProject:
    def __init__(self, meta_dir: Path, m_dir: Path, c_dir: Path, cwm_dir: Path,
                 vector_method: Callable[[list], np.ndarray], vector_class: Callable[[list], np.ndarray],
                 oversampling: bool, yield_names: bool = False,
                 class_max_lines: int = 1000, method_max_lines: int = 100,
                 precalculated: bool = False):
        self.precalculated = precalculated
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
        self.m_dir = m_dir
        self.c_dir = c_dir
        self.cwm_dir = cwm_dir
        self.yield_names = yield_names
        self.name = meta_dir.name
        self.class_max_lines = class_max_lines
        self.method_max_lines = method_max_lines

    def _get_class_vector(self, class_id: int) -> np.ndarray:
        class_name = self.classes[class_id]
        return _create_vectors(self.c_dir, class_name, self.vector_class, self.precalculated, self.class_max_lines)

    def _get_class_without_method_vector(self, method_name: str) -> np.ndarray:
        return _create_vectors(self.cwm_dir, method_name, self.vector_class, self.precalculated, self.class_max_lines)

    def _get_method_vector(self, method_name: str) -> np.ndarray:
        return _create_vectors(self.m_dir, method_name, self.vector_method, self.precalculated, self.method_max_lines)

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
                    cwm_vector = self._get_class_without_method_vector(m_name)
                    for i in range(n_pos_samples):
                        if self.yield_names:
                            yield m_name, mc_name, m_vector, cwm_vector, True
                        else:
                            yield m_vector, cwm_vector, True
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
    def __init__(self, orig_root: Path, methods_data_root: Path, classes_data_root: Path,
                 vector_method: Callable[[list], np.ndarray], vector_class: Callable[[list], np.ndarray],
                 oversampling: bool, yield_names: bool, *,
                 class_max_lines: int = 1000, method_max_lines: int = 100, precalculated: bool = False):
        self.yield_names = yield_names
        project_names = [p.name for p in orig_root.iterdir() if p.is_dir() and p.name[0] != '.']
        self.projects = [
            MMRDatasetProject(orig_root / project_name, methods_data_root / project_name / 'methods',
                              classes_data_root / project_name / 'classes',
                              classes_data_root / project_name / 'classes_without_methods',
                              vector_method, vector_class, oversampling, yield_names,
                              class_max_lines, method_max_lines, precalculated)
            for project_name in project_names
        ]

    def __iter__(self):
        for project in self.projects:
            if self.yield_names:
                for method_data in project:
                    yield (project.name, *method_data)
            else:
                yield from project
