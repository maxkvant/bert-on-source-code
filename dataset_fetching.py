from pathlib import Path
import sys
import json
import subprocess
import shutil
from typing import Set, Optional, List


class SavedSet:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            with path.open() as f:
                self.items = {item.rstrip() for item in f}
        else:
            self.items = set()

    def add(self, item):
        self.items.add(item)
        with self.path.open('a') as f:
            f.write(item + '\n')

    def __contains__(self, item):
        return item in self.items

    def __len__(self):
        return len(self.items)


class DatasetCloner:
    LIST_FILE_NAME = 'repos_files.json'
    REPOS_DIR_NAME = 'repos'
    FINISHED_FILE_NAME = 'finished.txt'
    FAILED_FILE_NAME = 'failed.txt'

    def __init__(self, root: Path):
        self.root = root
        files_list_path = root / DatasetCloner.LIST_FILE_NAME
        with files_list_path.open() as files_list_file:
            self.repos_file_lists = json.load(files_list_file)
        self.repos_path = root / DatasetCloner.REPOS_DIR_NAME
        self.finished_repos = SavedSet(root / DatasetCloner.FINISHED_FILE_NAME)
        self.failed_repos = SavedSet(root / DatasetCloner.FAILED_FILE_NAME)

    def _declare_finished(self, repo: str):
        self.finished_repos.add(repo)
        print(f'Finished {len(self.finished_repos)} of {len(self.repos_file_lists)} : {repo}')
            
    def _declare_failed(self, repo: str, exception: Exception):
        self.failed_repos.add(repo)
        print(f'Failed to clone {repo}: {exception}')

    @staticmethod
    def _clean_dir(path: Path, files_to_retain: Set[str], root_path: Optional[Path] = None) -> bool:
        if root_path is None:
            root_path = path
        if path.is_symlink():
            path.unlink()
            return False
        if path.is_file():
            relative_path = path.relative_to(root_path)
            if str(relative_path) in files_to_retain:
                return True
            path.unlink()
            return False
        children_retained = [
            DatasetCloner._clean_dir(child, files_to_retain, root_path)
            for child in path.iterdir()
        ]
        if any(children_retained):
            return True
        path.rmdir()
        return False

    def _clone(self, repo: str):
        print(f'Cloning {repo}')
        url = 'https://:@github.com/' + repo + '.git'
        target_path = self.repos_path / repo
        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True)
        clone_res = subprocess.run(['git', 'clone', url, str(target_path), '--depth=1', '-q'],
                                   capture_output=True, check=True, input='')
        clone_res.check_returncode()

    def _checkout_required(self, repo: str, files_list: List[str]):
        print(f'Checking out from {repo}')
        url = 'https://:@github.com/' + repo + '.git'
        target_path = self.repos_path / repo
        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True)
        clone_res = subprocess.run(['git', 'init', str(target_path)], capture_output=True, check=True)
        clone_res.check_returncode()
        clone_res = subprocess.run(['git', 'remote', 'add', 'origin', url],
                                   cwd=str(target_path), capture_output=True, check=True)
        clone_res.check_returncode()
        clone_res = subprocess.run(['git', 'config', 'core.sparseCheckout', 'true'], cwd=str(target_path),
                                   capture_output=True, check=True)
        clone_res.check_returncode()
        (target_path / '.git/info/sparse-checkout').open('w').write('\n'.join(files_list))
        clone_res = subprocess.run(['git', 'pull', '--depth=1', 'origin', 'master'], cwd=str(target_path),
                                   capture_output=True, check=True)
        clone_res.check_returncode()
        shutil.rmtree(target_path / '.git')

    def collect(self):
        for repo, files_list in self.repos_file_lists.items():
            if repo in self.finished_repos or repo in self.failed_repos:
                continue
            try:
                # self._checkout_required(repo, files_list)
                self._clone(repo)
                self._clean_dir(self.repos_path / repo, set(files_list))
                self._declare_finished(repo)
            except subprocess.CalledProcessError as e:
                self._declare_failed(repo, e)


if __name__ == '__main__':
    dataset_root = Path(sys.argv[1])
    DatasetCloner(dataset_root).collect()
