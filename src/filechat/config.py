import json
import os

import torch
from pydantic import BaseModel

CONFIG_PATH_DEFAULT = os.path.join(os.environ["HOME"], ".config", "filechat.json")


class Config(BaseModel):
    max_file_size_kb: int = 25
    ignored_dirs: list[str] = [".git", "__pycache__", ".venv", ".pytest_cache", "node_modules"]
    allowed_suffixes: list[str] = [
        ".txt",
        ".json",
        ".py",
        ".toml",
        ".html",
        ".md",
        ".js",
        ".ts",
        ".vue",
    ]
    index_store_path: str = os.path.join(os.environ["HOME"], ".cache", "filechat")
    model: str = "mistral-medium-2508"
    api_key: str | None = os.environ.get("MISTRAL_API_KEY")

    @property
    def embedding_model(self) -> str:
        return "nomic-ai/nomic-embed-text-v1.5"

    @property
    def index_batch_size(self) -> int:
        return 10

    @property
    def device(self):
        if torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


def load_config(path: str = CONFIG_PATH_DEFAULT) -> Config:
    if os.path.exists(path):
        with open(path, "r") as config_file:
            config_json = json.load(config_file)
            config = Config.model_validate(config_json)
    else:
        config = Config()
        if path == CONFIG_PATH_DEFAULT:
            config_json = config.model_dump_json(indent=4)
            os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
            with open(path, "w") as config_file:
                config_file.write(config_json)
        else:
            raise FileNotFoundError(f"Config file {path} not found")

    return config
