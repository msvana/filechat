import logging
import os
from argparse import ArgumentParser

from sentence_transformers import SentenceTransformer

from filechat.chat import Chat
from filechat.config import Config
from filechat.index import FileIndex, IndexStore
from filechat.watcher import FileWatcher

logging.basicConfig(level=logging.INFO)

arg_parser = ArgumentParser(description="Index files in a directory")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")
arg_parser.add_argument(
    "-r", "--rebuild", action="store_true", help="Ignore cache, rebuild index from scratch"
)

config = Config()


def main():
    args = arg_parser.parse_args()
    sentence_transformer = SentenceTransformer(
        config.get_embedding_model(), trust_remote_code=True, device=config.get_device()
    )

    index, _ = get_index(args.directory, config, sentence_transformer, args.rebuild)
    watcher = FileWatcher(index, config)
    watcher.start()
    chat = Chat()

    try:
        while True:
            user_message = input(">>> ")
            if user_message == "/exit":
                break
            if user_message.strip() == "":
                continue
            files = index.query(user_message)
            chat.user_message(user_message, files)

            print("---------")
            print("Files in context: ", end="")
            print(", ".join(f.path() for f in files))
            print("---------")
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()


def get_index(
    directory: str, config: Config, embedding_model: SentenceTransformer, rebuild: bool = False
) -> tuple[FileIndex, int]:
    allowed_suffixes = config.get_allowed_suffixes()
    ignored_directories = config.get_ignored_directories()
    index_store = IndexStore(config.get_index_store_path())

    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

    if rebuild:
        logging.info("Rebuilding index from scratch")
        index = FileIndex(embedding_model, directory, 768)
    else:
        try:
            index = index_store.load(directory, embedding_model)
            index.clean_old_files()
        except FileNotFoundError:
            logging.info("Index file not found. Creating new index from scratch")
            index = FileIndex(embedding_model, directory, 768)

    num_indexed = 0
    batch = []
    for root, _, files in os.walk(directory):
        if any(ignored in root for ignored in ignored_directories):
            continue

        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)

            if all(not file.endswith(suffix) for suffix in allowed_suffixes):
                continue

            file_size = os.path.getsize(full_path)
            if file_size > config.get_max_file_size():
                continue

            batch.append(relative_path)

            if len(batch) >= config.get_index_batch_size():
                num_indexed += index.add_files(batch)
                batch = []

    if batch:
        num_indexed += index.add_files(batch)

    index_store.store(index)
    return index, num_indexed
