import json
import os
from pathlib import Path

from pydantic import BaseModel, ValidationError
from rich import print as rprint

HOME_DIR = os.path.expanduser("~")
CONFIG_PATH_DEFAULT = os.path.join(HOME_DIR, ".config", "filechat.json")

MODEL_CHOICES = {
    "openai": [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-codex",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ],
    "mistral": [
        "mistral-medium-2508",
        "magistral-medium-2509",
        "codestral-2508",
    ],
}


class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key: str


class Config(BaseModel):
    max_file_size_kb: int = 25
    ignored_dirs: list[str] = [
        ".git",
        "__pycache__",
        ".venv",
        ".pytest_cache",
        "node_modules",
        "dist",
    ]
    allowed_suffixes: list[str] = [
        # Text files
        ".md",
        ".txt",
        # Config files
        ".json",
        ".toml",
        # HTML/CSS
        ".html",
        ".css",
        ".sass",
        ".scss",
        # Python
        ".py",
        # Basic Javascript files
        ".js",
        ".ts",
        ".vue",
        ".tsx",
        ".jsx",
        # Golang
        ".go",
        ".mod",
        ".work",
        ".gotmpl",
        # C
        ".c",
        ".h",
        ".i",
        ".mk",
        ".cmake",
        ".config",
        ".pc",
        ".dox",
        ".ld",
    ]
    index_store_path: str = os.path.join(HOME_DIR, ".cache", "filechat")
    model: ModelConfig
    api_key: str | None = os.environ.get("MISTRAL_API_KEY")

    @property
    def embedding_model(self) -> str:
        return "nomic-ai/nomic-embed-text-v1.5"

    @property
    def index_batch_size(self) -> int:
        return 1

    @property
    def log_dir(self) -> str:
        return os.path.join(self.index_store_path, "logs")

    @property
    def embedding_model_url(self) -> str:
        return "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_q4.onnx?download=true"

    @property
    def embedding_model_path(self) -> Path:
        return Path(self.index_store_path) / "models" / "embedding.onnx"


def load_config(path: str = CONFIG_PATH_DEFAULT, discard: bool=False) -> Config:
    if os.path.exists(path) and not discard:
        with open(path, "r") as config_file:
            config_json = json.load(config_file)

        try:
            config = Config.model_validate(config_json)
            return config
        except ValidationError:
            pass

    config = setup_config()
    config_json = config.model_dump_json(indent=4)
    config_dir = os.path.dirname(path)
    if config_dir != "":
        os.makedirs(config_dir, exist_ok=True)
    with open(path, "w") as config_file:
        config_file.write(config_json)
        rprint(f"[blue]Config file stored at {path}[/blue]")
        print("Press ENTER to start FileChat")
        input()

    return config


def setup_config() -> Config:
    rprint("[blue]Hi. Looks like you don't have a valid config file yet. Let's create one.[/blue]")

    print("Choose your LLM provider (mistral, openai):")
    while True:
        provider = input().strip().lower()
        if provider in ["mistral", "openai"]:
            break
        print("You entered an incorrect choice, try again (mistral, openai):")

    api_key_env = f"{provider.upper()}_API_KEY"
    api_key_env_key = os.environ.get(api_key_env)
    if api_key_env_key:
        api_key_env_message = f"or presss ENTER to use ${api_key_env}"
    else:
        api_key_env_message = ""

    print(f"Enter your API key {api_key_env_message}:")

    while True:
        api_key = input().strip()
        if api_key != "":
            break

        if api_key_env_key:
            api_key = api_key_env_key
            break

        print("You didn't enter any API key, try again:")

    print("Choose a model. You have the following options:")
    
    for model in MODEL_CHOICES[provider]:
        print(f"- {model}")

    print("Your choice:")

    while True:
        model = input().strip().lower()
        if model in MODEL_CHOICES[provider]:
            break
        print("This model isn't in the list, try again:")

    model_config = ModelConfig(provider=provider, model=model, api_key=api_key)
    config = Config(model=model_config)
    return config
