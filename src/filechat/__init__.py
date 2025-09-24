import datetime
import logging
import os
from argparse import ArgumentParser

from mistralai import Mistral
from openai import OpenAI

from filechat.chat import Chat, ChatStore
from filechat.config import CONFIG_PATH_DEFAULT, load_config
from filechat.embedder import Embedder
from filechat.index import get_index
from filechat.tui import FilechatApp
from filechat.watcher import FileWatcher

arg_parser = ArgumentParser(description="Chat with an LLM about your local project")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")
arg_parser.add_argument(
    "-r", "--rebuild", action="store_true", help="Ignore cache, rebuild index from scratch"
)
arg_parser.add_argument(
    "-c", "--config", type=str, help="Path to a config file", default=CONFIG_PATH_DEFAULT
)
arg_parser.add_argument(
    "-s", "--setup", action="store_true", help="Discard config and run LLM provider setup"
)


def main():
    args = arg_parser.parse_args()
    config = load_config(args.config, args.setup)

    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(
        config.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    )
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_file)])

    embedder = Embedder(
        config.embedding_model, config.embedding_model_path, config.embedding_model_url
    )

    index, _ = get_index(args.directory, config, embedder, args.rebuild)
    watcher = FileWatcher(index, config)
    watcher.start()

    if config.model.provider == "openai":
        client = OpenAI(api_key=config.model.api_key)
    elif config.model.provider == "openai-selfhosted":
        client = OpenAI(api_key=config.model.api_key, base_url=config.model.base_url)
    elif config.model.provider == "mistral":
        client = Mistral(api_key=config.model.api_key)

    chat = Chat(client, config.model.model)
    chat_store = ChatStore(args.directory, config, client)

    app = FilechatApp(chat, index, chat_store)
    app.run()

    watcher.stop()


if __name__ == "__main__":
    main()
