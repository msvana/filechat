import logging
import os
from argparse import ArgumentParser
from textwrap import dedent

from mistralai import Mistral
from sentence_transformers import SentenceTransformer

from filechat.config import Config
from filechat.index import FileIndex, IndexedFile, IndexStore

logging.basicConfig(level=logging.INFO)

arg_parser = ArgumentParser(description="Index files in a directory")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")

config = Config()

sentence_transformer = SentenceTransformer(config.get_embedding_model())


def index_files():
    args = arg_parser.parse_args()
    directory = args.directory
    allowed_suffixes = config.get_allowed_suffixes()
    ignored_directories = config.get_ignored_directories()

    index_store = IndexStore(config.get_index_store_path())

    try:
        index = index_store.load(directory, sentence_transformer)
    except FileNotFoundError:
        logging.info("Index file not found. Creating new index from scratch")
        index = FileIndex(sentence_transformer, directory, 1024)

    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

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

            index.add_file(relative_path)

    index_store.store(index)
    chat = Chat()

    while True:
        user_message = input(">>> ")
        if user_message == "/exit":
            break
        files = index.query(user_message)
        chat.user_message(user_message, files)


class Chat:
    MODEL = "codestral-2508"
    SYSTEM_MESSAGE = dedent("""\
    You are a local project assistant. Your task is to assist the user with various projects. 
    They can ask you question to understand the project or for suggestions on how to improve the projects.

    Besides the user's query. You will be also provided with the contents of 
    the files potentially most relevant to this query. If needed, you can use the content of these file
    to create a better response.
    """)

    def __init__(self):
        self._message_history = [{"role": "system", "content": self.SYSTEM_MESSAGE}]
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def user_message(self, message: str, files: list[IndexedFile]):
        user_message = {"role": "user", "content": message}
        context_message = self._get_context_message(files)
        self._message_history.append(user_message)
        response = self._client.chat.stream(
            model=self.MODEL,
            messages=self._message_history + [context_message],  # type: ignore
        )

        response_str = ""

        for chunk in response:
            chunk_content = chunk.data.choices[0].delta.content
            response_str += str(chunk_content)
            print(chunk_content, end="")

        self._message_history.append({"role": "assistant", "content": response_str})
        print()

    def _get_context_message(self, files: list[IndexedFile]) -> dict:
        message = "<context>"

        for file in files:
            message += "<file>"
            message += file.content_for_embedding()
            message += "</file>"

        message += "</context>"
        return {"role": "user", "content": message}
