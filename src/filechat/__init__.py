import logging
import os
from argparse import ArgumentParser
from hashlib import sha256
from textwrap import dedent

import faiss
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

from filechat.config import Config

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

    chat = Chat()

    while True:
        user_message = input(">>> ")
        if user_message == "/exit":
            break
        files = index.query(user_message)
        chat.user_message(user_message, files)


class IndexedFile:
    EMBEDDING_TEMPLATE = dedent("""\
        <filename>{relative_path}</filename>
        <content>
        {content}
        </content>""")

    def __init__(self, directory: str, relative_path: str):
        self._relative_path = relative_path
        self._full_path = os.path.join(directory, relative_path)
        self._load_content()

    def __repr__(self):
        return f"IndexedFile('{self._relative_path}')"

    def content(self):
        return self._content

    def content_for_embedding(self) -> str:
        embedding_text = self.EMBEDDING_TEMPLATE.format(
            relative_path=self._relative_path, content=self._content
        )
        return embedding_text

    def _load_content(self):
        with open(self._full_path) as f:
            self._content: str = f.read()
        self._sha_hash = sha256(self._content.encode())


class FileIndex:
    def __init__(self, embedding_model: SentenceTransformer, directory: str, dimensions: int):
        self._model = embedding_model
        self._directory = os.path.abspath(directory)
        self._dimensions = dimensions
        self._vector_index = faiss.IndexFlatL2(self._dimensions)
        self._files: list[IndexedFile] = []

    def add_file(self, relative_path: str):
        logging.info(f"Indexing `{relative_path}`")
        indexed_file = IndexedFile(self._directory, relative_path)
        embedding = self._model.encode(indexed_file.content_for_embedding())
        self._vector_index.add(embedding.reshape(1, -1))
        self._files.append(indexed_file)

    def query(self, query: str, top_k: int = 5) -> list[IndexedFile]:
        logging.info(f"Querying: `{query}`")
        query_embedding = self._model.encode(query)
        _, indices = self._vector_index.search(query_embedding.reshape(1, -1), k=top_k)
        matching_files = []
        for idx in indices[0]:
            matching_files.append(self._files[idx])
        return matching_files


class Chat:
    MODEL = "devstral-medium-2507"
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
