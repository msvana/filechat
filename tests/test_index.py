import os
import shutil
import tempfile

import pytest
from sentence_transformers import SentenceTransformer

from filechat import get_index
from filechat.config import Config


@pytest.fixture
def test_directory():
    test_dir = tempfile.mkdtemp()
    test_files = ["test.txt", "test.json", "test.py", "test.toml", "test.html", "test.md"]

    for file in test_files:
        with open(os.path.join(test_dir, file), "w") as f:
            f.write(f"This is the content of {file}")

    yield test_dir
    shutil.rmtree(test_dir)


def test_index_files(test_directory):
    config = Config()
    embedding_model = SentenceTransformer(config.get_embedding_model())
    index = get_index(test_directory, config, embedding_model)

    indexed_files = [file.path() for file in index._files]
    assert len(indexed_files) == len(os.listdir(test_directory))

    for file in os.listdir(test_directory):
        assert file in indexed_files

    assert len(index._files) == index._vector_index.ntotal
    assert len(index._files) == len(set(f.hash for f in index._files))
