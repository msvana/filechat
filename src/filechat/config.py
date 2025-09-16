import json
import os
from pathlib import Path

from pydantic import BaseModel

HOME_DIR = os.path.expanduser("~")
CONFIG_PATH_DEFAULT = os.path.join(HOME_DIR, ".config", "filechat.json")


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
    model: str = "mistral-medium-2508"
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


def load_config(path: str = CONFIG_PATH_DEFAULT) -> Config:
    if os.path.exists(path):
        with open(path, "r") as config_file:
            config_json = json.load(config_file)
            config = Config.model_validate(config_json)
    else:
        config = Config()
        if path == CONFIG_PATH_DEFAULT:
            config_json = config.model_dump_json(indent=4)
            config_dir = os.path.dirname(path)
            os.makedirs(config_dir, exist_ok=True)
            with open(path, "w") as config_file:
                config_file.write(config_json)
        else:
            raise FileNotFoundError(f"Config file {path} not found")

    return config
