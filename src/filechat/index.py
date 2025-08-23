import logging
import os
import pickle
from hashlib import sha256
from textwrap import dedent

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from filechat.config import Config


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
        return self.add_files([relative_path]) > 0

    def add_files(self, relative_paths: list[str]) -> int:
        logging.info(f"Indexing batch of {len(relative_paths)} files")
        indexed_files = [self._prepare_for_indexing(r) for r in relative_paths]
        indexed_files = [f for f in indexed_files if f is not None]
        if not indexed_files:
            return 0
        
        texts = [f"search document: {f.content_for_embedding()}" for f in indexed_files]
        assert self._model is not None
        logging.info("Creating embeddings")
        embeddings = self._model.encode(texts)
        logging.info("Adding to vector index")
        self._vector_index.add(embeddings)

        for f in indexed_files:
            self._files.append(f)
            logging.info(f"Indexed file {f.path()}")

        return len(indexed_files)

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
        query_embedding = self._model.encode(f"search_query: {query}")
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

    def _prepare_for_indexing(self, relative_path: str) -> IndexedFile | None:
        indexed_file = IndexedFile(self._directory, relative_path)
        idx, needs_update = self._file_needs_update(indexed_file)

        if not needs_update:
            logging.info(f"File {relative_path} is already up to date")
            return None

        if idx:
            self._delete_file(idx)

        return indexed_file


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
