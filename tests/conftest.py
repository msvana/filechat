import shutil
import tempfile
from openai import OpenAI
from mistralai import Mistral
import pytest
import os
from filechat.index import IndexStore
from filechat.config import Config, ModelConfig


TEST_PROVIDERS = os.environ.get("TEST_PROVIDERS", "openai,mistral")
PROVIDERS = [p.strip() for p in TEST_PROVIDERS.split(",") if p.strip()]


@pytest.fixture(params=PROVIDERS)
def config(request):
    provider = request.param

    if provider == "openai":
        api_env = "OPENAI_API_KEY"
        model = "gpt-5-mini"
    elif provider == "mistral":
        api_env = "MISTRAL_API_KEY"
        model = "mistral-medium-2508"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    index_dir = tempfile.mkdtemp()
    model_config = ModelConfig(provider=provider, model=model, api_key=os.environ[api_env])
    config = Config(index_store_path=index_dir, model=model_config)
    os.makedirs(config.log_dir, exist_ok=True)
    yield config
    shutil.rmtree(index_dir)


@pytest.fixture
def client(config: Config):
    if config.model.provider == "openai":
        client = OpenAI(api_key=config.model.api_key)
    elif config.model.provider == "openai-selfhosted":
        client = OpenAI(api_key=config.model.api_key, base_url=config.model.base_url)
    elif config.model.provider == "mistral":
        client = Mistral(api_key=config.model.api_key)
    return client


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
