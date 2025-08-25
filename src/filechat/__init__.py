import logging
from argparse import ArgumentParser

from sentence_transformers import SentenceTransformer

from filechat.chat import Chat
from filechat.config import Config
from filechat.index import get_index
from filechat.tui import FilechatApp
from filechat.watcher import FileWatcher

logging.basicConfig(level=logging.CRITICAL)

arg_parser = ArgumentParser(description="Index files in a directory")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")
arg_parser.add_argument(
    "-r", "--rebuild", action="store_true", help="Ignore cache, rebuild index from scratch"
)


def main():
    config = Config()
    args = arg_parser.parse_args()

    sentence_transformer = SentenceTransformer(
        config.get_embedding_model(), trust_remote_code=True, device=config.get_device()
    )

    index, _ = get_index(args.directory, config, sentence_transformer, args.rebuild)
    watcher = FileWatcher(index, config)
    watcher.start()

    chat = Chat()

    app = FilechatApp(chat, index)
    app.run()

    watcher.stop()


if __name__ == "__main__":
    main()
