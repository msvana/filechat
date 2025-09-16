import os

import pytest

from filechat import get_index
from filechat.config import Config
from filechat.embedder import Embedder


@pytest.fixture
def embedder(config: Config):
    return Embedder(config.embedding_model, config.embedding_model_path, config.embedding_model_url)


def test_index_files(test_directory, config: Config, embedder: Embedder):
    index, _ = get_index(test_directory, config, embedder)

    indexed_files = [file.path() for file in index._files]
    assert len(indexed_files) == len(os.listdir(test_directory))

    for file in os.listdir(test_directory):
        assert file in indexed_files

    assert len(index._files) == index._vector_index.ntotal
    assert len(index._files) == len(set(f.hash for f in index._files))


def test_new_file(test_directory, config: Config, embedder: Embedder):
    index, _ = get_index(test_directory, config, embedder)

    new_file = "new_file.txt"
    with open(os.path.join(test_directory, new_file), "w") as f:
        f.write("This is the content of the new file")

    num_updates = 0
    for file in os.listdir(test_directory):
        num_updates += index.add_file(file)
    assert num_updates == 1

    indexed_files = [file.path() for file in index._files]
    assert new_file in indexed_files

    assert len(index._files) == index._vector_index.ntotal
    assert len(index._files) == len(set(f.hash for f in index._files))


def test_file_change(test_directory, config: Config, embedder: Embedder):
    index, _ = get_index(test_directory, config, embedder)
    num_files_before = len(index._files)

    filename = "test.txt"
    with open(os.path.join(test_directory, filename), "w") as f:
        f.write("This is the content of {filename}. There is some new stuff to it")

    num_updates = 0
    for file in os.listdir(test_directory):
        num_updates += index.add_file(file)
    assert num_updates == 1

    indexed_files = [file.path() for file in index._files]
    assert filename in indexed_files

    assert len(index._files) == index._vector_index.ntotal
    assert len(index._files) == num_files_before


def test_delete_file(test_directory, config: Config, embedder: Embedder):
    index, _ = get_index(test_directory, config, embedder)
    os.remove(os.path.join(test_directory, "test.md"))
    os.remove(os.path.join(test_directory, "test.json"))
    index, _ = get_index(test_directory, config, embedder)

    indexed_files = [file.path() for file in index._files]
    assert "test.md" not in indexed_files
    assert "test.json" not in indexed_files

    assert len(indexed_files) == len(os.listdir(test_directory))
    assert len(index._files) == index._vector_index.ntotal


def test_delete_file_ignored_directory(test_directory, config: Config, embedder: Embedder):
    ignored_dir = "temp_ignored"
    os.makedirs(os.path.join(test_directory, ignored_dir))
    ignored_file = os.path.join(ignored_dir, "ignored_file.txt")
    with open(os.path.join(test_directory, ignored_file), "w") as f:
        f.write("This file should be ignored")

    index, _ = get_index(test_directory, config, embedder)
    initial_count = len(index._files)

    indexed_files = [file.path() for file in index._files]
    assert ignored_file in indexed_files

    config.ignored_dirs.append(ignored_dir)

    index.clean_old_files(config)

    indexed_files = [file.path() for file in index._files]
    assert ignored_file not in indexed_files
    assert len(index._files) == initial_count - 1
    assert index._vector_index.ntotal == initial_count - 1


def test_delete_file_suffix(test_directory, config: Config, embedder: Embedder):
    disallowed_file = "test.md"
    index, _ = get_index(test_directory, config, embedder)
    initial_count = len(index._files)

    indexed_files = [file.path() for file in index._files]
    assert disallowed_file in indexed_files

    config.allowed_suffixes = [s for s in config.allowed_suffixes if s != ".md"]
    index.clean_old_files(config)

    indexed_files = [file.path() for file in index._files]
    assert disallowed_file not in indexed_files
    assert len(index._files) == initial_count - 1
    assert index._vector_index.ntotal == initial_count - 1


def test_rebuild(test_directory, config: Config, embedder: Embedder):
    get_index(test_directory, config, embedder)
    _, num_indexed = get_index(test_directory, config, embedder)
    assert num_indexed == 0
    _, num_indexed = get_index(test_directory, config, embedder, True)
    assert num_indexed == len(os.listdir(test_directory))
