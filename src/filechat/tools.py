from pathlib import Path
from filechat.config import Config


def list_directory(project_path: Path, directory: str, config: Config) -> list[dict]:
    path = Path(directory).resolve()

    if project_path.resolve() not in list(path.parents) and project_path.resolve() != path:
        raise ValueError("Looks like you want to access a directory that's not in the project")

    if not path.exists():
        raise FileNotFoundError("This directory doesn't exist")

    if not path.is_dir():
        raise FileNotFoundError("This path is not a directory")

    directory_contents = []

    for item in path.iterdir():
        if item.is_file() and any(item.name.endswith(s) for s in config.allowed_suffixes):
            directory_contents.append({"name": item.name, "type": "file"})

        if item.is_dir() and all(p not in config.ignored_dirs for p in item.parts):
            directory_contents.append({"name": item.name, "type": "directory"})

    return directory_contents
