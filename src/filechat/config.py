import os


class Config:

    def get_allowed_suffixes(self):
        return [".txt", ".json", ".py", ".toml", ".html", ".md"]

    def get_ignored_directories(self):
        return [".git", "__pycache__", ".venv"]

    def get_max_file_size(self):
        return 25 * 1024

    def get_embedding_model(self):
        return "nomic-ai/nomic-embed-text-v1.5"

    def get_index_store_path(self):
        home = os.environ["HOME"]
        return os.path.join(home, ".cache", "filechat")
