import os
from pathlib import Path

from filechat.config import Config

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists files and directories in a given directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to project. Do not explore paths outside the project directory",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a file inside the project. Returns file name, relative path and file content. Access outside the project is not allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file relative to project root. Do not access files outside the project directory.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


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

def read_file(project_path: Path, file_path: str, config: Config) -> dict:
    candidate = Path(file_path)
    if not candidate.is_absolute():
        path = (project_path / candidate).resolve()
    else:
        path = candidate.resolve()

    if project_path.resolve() not in list(path.parents):
        raise ValueError("Looks like you want to access a file that's not in the project")

    if not path.exists():
        raise FileNotFoundError("This file doesn't exist")

    if not path.is_file():
        raise FileNotFoundError("This path is not a file")

    if not any(path.name.endswith(s) for s in config.allowed_suffixes):
        raise FileNotFoundError("File type not allowed")

    if any(p in config.ignored_dirs for p in path.parts):
        raise FileNotFoundError("File is in an ignored directory")

    file_size = os.path.getsize(path)
    if file_size > config.max_file_size_kb * 1024:
        raise ValueError("File is too large to read")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    relative = os.path.relpath(path, project_path.resolve())
    return {"name": path.name, "path": relative, "content": content}
