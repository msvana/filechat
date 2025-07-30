class Config:

    def get_allowed_suffixes(self):
        return [".txt", ".json", ".py", ".toml", ".html", ".md"]

    def get_ignored_directories(self):
        return [".git", "__pycache__", ".venv"]

    def get_max_file_size(self):
        return 25 * 1024
