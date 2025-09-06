import datetime
import logging
from argparse import ArgumentParser

from sentence_transformers import SentenceTransformer

from filechat.chat import Chat, ChatStore
from filechat.config import load_config, CONFIG_PATH_DEFAULT
from filechat.index import get_index
from filechat.tui import FilechatApp
from filechat.watcher import FileWatcher
import os

arg_parser = ArgumentParser(description="Chat with an LLM about your local project")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")
arg_parser.add_argument(
    "-r", "--rebuild", action="store_true", help="Ignore cache, rebuild index from scratch"
)
arg_parser.add_argument(
    "-c", "--config", type=str, help="Path to a config file", default=CONFIG_PATH_DEFAULT
)


def main():
    args = arg_parser.parse_args()
    config = load_config(args.config)

    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(
        config.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    )
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_file)])

    print(f"Using device {config.device}")
    sentence_transformer = SentenceTransformer(
        config.embedding_model, trust_remote_code=True, device=config.device
    )

    index, _ = get_index(args.directory, config, sentence_transformer, args.rebuild)
    watcher = FileWatcher(index, config)
    watcher.start()

    chat = Chat(config.model, config.api_key)
    chat_store = ChatStore(args.directory, config)

    app = FilechatApp(chat, index, chat_store)
    app.run()

    watcher.stop()


if __name__ == "__main__":
    main()
