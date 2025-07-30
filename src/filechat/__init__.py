import os
from argparse import ArgumentParser
from textwrap import dedent

from sentence_transformers import SentenceTransformer

from filechat.config import Config

arg_parser = ArgumentParser(description="Index files in a directory")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")

config = Config()

sentence_transformer = SentenceTransformer(config.get_embedding_model())


def index_files():
    args = arg_parser.parse_args()
    directory = args.directory
    allowed_suffixes = config.get_allowed_suffixes()
    ignored_directories = config.get_ignored_directories()

    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

    for root, _, files in os.walk(directory):
        if any(ignored in root for ignored in ignored_directories):
            continue

        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)

            if any(file.endswith(suffix) for suffix in allowed_suffixes):
                get_size = os.path.getsize(full_path)
                if get_size < config.get_max_file_size():
                    store_file(relative_path, full_path)


def store_file(relative_path, full_path):
    with open(full_path) as f:
        content = f.read()

    text_to_embed_template = dedent("""\
    <filename>{relative_path}</filename>

    <content>
    {content}
    </content>""")

    text_to_embed = text_to_embed_template.format(relative_path=relative_path, content=content)
    text_embedding = sentence_transformer.encode(text_to_embed)
    
