import os
from argparse import ArgumentParser

from filechat.config import Config

arg_parser = ArgumentParser(description="Index files in a directory")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")

config = Config()


def index_files():
    args = arg_parser.parse_args()
    directory = args.directory
    allowed_suffixes = config.get_allowed_suffixes()
    ignored_directories = config.get_ignored_directories()

    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

    indexed_files = []
    for root, _, files in os.walk(directory):
        if any(ignored in root for ignored in ignored_directories):
            continue

        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)

            if any(file.endswith(suffix) for suffix in allowed_suffixes):
                get_size = os.path.getsize(full_path)
                if get_size < config.get_max_file_size():
                    print(relative_path)
                    indexed_files.append(relative_path)
