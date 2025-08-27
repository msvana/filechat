import shutil
import tempfile

import pytest
import os
from filechat.index import IndexStore
from filechat.config import Config


@pytest.fixture
def config():
    index_dir = tempfile.mkdtemp()
    yield Config(index_store_path=index_dir)
    shutil.rmtree(index_dir)


@pytest.fixture(scope="function")
def test_directory(config: Config):
    test_dir = tempfile.mkdtemp()
    test_files = ["test.txt", "test.json", "test.py", "test.toml", "test.html", "test.md"]

    for file in test_files:
        with open(os.path.join(test_dir, file), "w") as f:
            f.write(f"This is the content of {file}")

    yield test_dir
    shutil.rmtree(test_dir)
    store = IndexStore(config.index_store_path)
    store.remove(test_dir)
