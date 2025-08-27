import os

import pytest
from sentence_transformers import SentenceTransformer

from filechat import get_index
from filechat.config import Config


@pytest.fixture
def embedding_model(config: Config):
    return SentenceTransformer(config.embedding_model, trust_remote_code=True)


def test_index_files(test_directory, config: Config, embedding_model: SentenceTransformer):
    index, _ = get_index(test_directory, config, embedding_model)

    indexed_files = [file.path() for file in index._files]
    assert len(indexed_files) == len(os.listdir(test_directory))

    for file in os.listdir(test_directory):
        assert file in indexed_files

    assert len(index._files) == index._vector_index.ntotal
    assert len(index._files) == len(set(f.hash for f in index._files))


def test_new_file(test_directory, config: Config, embedding_model: SentenceTransformer):
    index, _ = get_index(test_directory, config, embedding_model)

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


def test_file_change(test_directory, config: Config, embedding_model: SentenceTransformer):
    index, _ = get_index(test_directory, config, embedding_model)
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


def test_delete_file(test_directory, config: Config, embedding_model: SentenceTransformer):
    index, _ = get_index(test_directory, config, embedding_model)
    os.remove(os.path.join(test_directory, "test.md"))
    os.remove(os.path.join(test_directory, "test.json"))
    index, _ = get_index(test_directory, config, embedding_model)

    indexed_files = [file.path() for file in index._files]
    assert "test.md" not in indexed_files
    assert "test.json" not in indexed_files

    assert len(indexed_files) == len(os.listdir(test_directory))
    assert len(index._files) == index._vector_index.ntotal


def test_rebuild(test_directory, config: Config, embedding_model: SentenceTransformer):
    get_index(test_directory, config, embedding_model)
    _, num_indexed = get_index(test_directory, config, embedding_model)
    assert num_indexed == 0
    _, num_indexed = get_index(test_directory, config, embedding_model, True)
    assert num_indexed == len(os.listdir(test_directory))
