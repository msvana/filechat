from filechat.config import Config
from filechat.tools import list_directory
from pathlib import Path
from pytest import raises


def test_list_directory(config: Config):
    directory_contents = list_directory(Path("."), ".", config)

    assert {"name": "README.md", "type": "file"} in directory_contents
    assert {"name": "pyproject.toml", "type": "file"} in directory_contents
    assert {"name": "src", "type": "directory"} in directory_contents

    assert {"name": "uv.lock", "type": "file"} not in directory_contents
    assert {"name": ".git", "type": "directory"} not in directory_contents


def test_list_directory_is_file(config: Config):
    with raises(FileNotFoundError):
        list_directory(Path("."), "pyproject.toml", config)


def test_list_directory_not_exists(config: Config):
    with raises(FileNotFoundError):
        list_directory(Path("."), "something_that_is_not_here", config)


def test_list_directory_outside_project(config: Config):
    with raises(ValueError):
        list_directory(Path("."), "/home/", config)

    with raises(ValueError):
        list_directory(Path("."), "..", config)
