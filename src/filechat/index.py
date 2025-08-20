import logging
import os
import pickle
from hashlib import sha256
from textwrap import dedent

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


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

    def path(self) -> str:
        return self._relative_path

    def hash(self) -> str:
        return self._sha_hash

    def _load_content(self):
        with open(self._full_path) as f:
            self._content: str = f.read()
        self._sha_hash = sha256(self._content.encode()).hexdigest()


class FileIndex:
    def __init__(self, embedding_model: SentenceTransformer, directory: str, dimensions: int):
        self._directory = os.path.abspath(directory)
        self._dimensions = dimensions
        self._vector_index = faiss.IndexFlatL2(self._dimensions)
        self._files: list[IndexedFile] = []
        self.set_model(embedding_model)

    def set_model(self, embedding_model: SentenceTransformer | None):
        self._model = embedding_model

    def model(self) -> SentenceTransformer | None:
        return self._model

    def add_file(self, relative_path: str) -> bool:
        logging.info(f"Indexing `{relative_path}`")
        indexed_file = IndexedFile(self._directory, relative_path)
        idx, needs_update = self._file_needs_update(indexed_file)

        if not needs_update:
            logging.info(f"File {relative_path} is already up to date")
            return False

        if idx:
            self._delete_file(idx)

        assert self._model is not None
        embedding = self._model.encode(indexed_file.content_for_embedding())
        self._vector_index.add(embedding.reshape(1, -1))

        self._files.append(indexed_file)
        return True

    def clean_old_files(self):
        for file in self._files:
            full_path = os.path.join(self._directory, file.path())
            if not os.path.exists(full_path):
                idx, _ = self._file_needs_update(file)
                if idx is not None:
                    logging.info(f"Removing deleted file {file.path()}")
                    self._delete_file(idx)

    def query(self, query: str, top_k: int = 10) -> list[IndexedFile]:
        logging.info(f"Querying: `{query}`")
        assert self._model is not None
        query_embedding = self._model.encode(query)
        _, indices = self._vector_index.search(query_embedding.reshape(1, -1), k=top_k)
        matching_files = []
        for idx in indices[0]:
            matching_files.append(self._files[idx])
        return matching_files

    def directory(self) -> str:
        return self._directory

    def _file_needs_update(self, indexed_file: IndexedFile):
        for i, f in enumerate(self._files):
            if indexed_file.path() == f.path():
                return (i, True) if indexed_file.hash() != f.hash() else (i, False)
        return None, True

    def _delete_file(self, idx: int):
        self._files.pop(idx)
        self._vector_index.remove_ids(np.array([idx]))


class IndexStore:

    def __init__(self, directory: str):
        self._directory = directory
        os.makedirs(self._directory, exist_ok=True)

    def store(self, file_index: FileIndex):
        logging.info(f"Storing index for {file_index.directory()}")
        model = file_index.model()
        file_index.set_model(None)
        file_path = self._get_file_path(file_index.directory())
        with open(file_path, "wb") as f:
            pickle.dump(file_index, f)
        file_index.set_model(model)
        logging.info("Index stored")

    def load(self, directory: str, embedding_model: SentenceTransformer) -> FileIndex:
        directory_abs_path = os.path.abspath(directory)
        logging.info(f"Trying to load cached index for {directory_abs_path}")
        file_path = self._get_file_path(directory_abs_path)
        with open(file_path, "rb") as f:
            file_index = pickle.load(f)
        file_index.set_model(embedding_model)
        logging.info("Index loaded")
        return file_index

    def remove(self, directory: str):
        directory_abs_path = os.path.abspath(directory)
        file_path = self._get_file_path(directory_abs_path)
        if os.path.exists(file_path):
            os.remove(file_path)

    def _get_file_path(self, directory: str) -> str:
        file_hash = sha256(directory.encode()).hexdigest()
        file_name = f"{file_hash}.pickle"
        file_path = os.path.join(self._directory, file_name)
        return file_path
