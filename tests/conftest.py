import shutil
import tempfile

import pytest
import os
from filechat.index import IndexStore
from filechat.config import Config, ModelConfig


@pytest.fixture
def config():
    index_dir = tempfile.mkdtemp()
    model_config = ModelConfig(
        provider="mistral",
        model="mistral-medium-2508",
        api_key=os.environ.get("MISTRAL_API_KEY", ""),
    )
    yield Config(index_store_path=index_dir, model=model_config)
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
