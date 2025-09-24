import logging
import os
import pickle
from hashlib import sha256
from textwrap import dedent
from threading import Lock

import faiss
import numpy as np

from filechat.config import Config
from filechat.embedder import Embedder


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
    def __init__(self, embedder: Embedder, directory: str, dimensions: int):
        self._file_lock = Lock()
        self._directory = os.path.abspath(directory)
        self._dimensions = dimensions
        self._vector_index = faiss.IndexFlatL2(self._dimensions)
        self._files: list[IndexedFile] = []
        self.set_embedder(embedder)

    def set_embedder(self, embedder: Embedder | None):
        self._embedder = embedder

    def embedder(self) -> Embedder | None:
        return self._embedder

    def add_file(self, relative_path: str) -> bool:
        return self.add_files([relative_path]) > 0

    def add_files(self, relative_paths: list[str]) -> int:
        with self._file_lock:
            logging.info(f"Indexing batch of {len(relative_paths)} files")
            indexed_files = [self._prepare_for_indexing(r) for r in relative_paths]
            indexed_files = [f for f in indexed_files if f is not None]
            if not indexed_files:
                return 0

            texts = [f"search document: {f.content_for_embedding()}" for f in indexed_files]
            assert self._embedder is not None
            logging.info("Creating embeddings")
            embeddings = self._embedder.embed(texts)
            logging.info("Adding to vector index")
            self._vector_index.add(embeddings)

            for f in indexed_files:
                self._files.append(f)
                logging.info(f"Indexed file {f.path()}")

        return len(indexed_files)

    def clean_old_files(self, config: Config):
        with self._file_lock:
            files_to_delete = []
            for i, file in enumerate(self._files):
                full_path = os.path.join(self._directory, file.path())

                if is_ignored(self._directory, full_path, config):
                    logging.info(f"Removing deleted file {file.path()}")
                    files_to_delete.append(i)

            for i in files_to_delete[::-1]:
                self._delete_file(i)

    def query(self, query: str, top_k: int = 10) -> list[IndexedFile]:
        filenames = [f.path() for f in self._files]
        assert len(filenames) == len(set(filenames))
        logging.info(f"Querying: `{query}`")
        assert self._embedder is not None
        query_embedding = self._embedder.embed([f"search_query: {query}"])
        _, indices = self._vector_index.search(query_embedding.reshape(1, -1), k=top_k)

        matching_files = []
        for idx in set(indices[0]):
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

        if idx is not None:
            self._delete_file(idx)

        return indexed_file


class IndexStore:

    def __init__(self, directory: str):
        self._directory = directory
        os.makedirs(self._directory, exist_ok=True)

    def store(self, file_index: FileIndex):
        logging.info(f"Storing index for {file_index.directory()}")
        model = file_index.embedder()
        file_index.set_embedder(None)
        file_index._file_lock = None
        file_path = self._get_file_path(file_index.directory())
        with open(file_path, "wb") as f:
            pickle.dump(file_index, f)
        file_index.set_embedder(model)
        file_index._file_lock = Lock()
        logging.info("Index stored")

    def load(self, directory: str, embedder: Embedder) -> FileIndex:
        directory_abs_path = os.path.abspath(directory)
        logging.info(f"Trying to load cached index for {directory_abs_path}")
        file_path = self._get_file_path(directory_abs_path)
        with open(file_path, "rb") as f:
            file_index = pickle.load(f)
        file_index.set_embedder(embedder)
        file_index._file_lock = Lock()
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
    directory: str, config: Config, embedder: Embedder, rebuild: bool = False
) -> tuple[FileIndex, int]:
    index_store = IndexStore(config.index_store_path)

    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

    if rebuild:
        logging.info("Rebuilding index from scratch")
        index = FileIndex(embedder, directory, 768)
    else:
        try:
            index = index_store.load(directory, embedder)
            index.clean_old_files(config)
        except FileNotFoundError:
            logging.info("Index file not found. Creating new index from scratch")
            index = FileIndex(embedder, directory, 768)

    num_indexed = 0
    batch = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            if not is_ignored(directory, full_path, config):
                batch.append(os.path.relpath(full_path, directory))

            if len(batch) >= config.index_batch_size:
                num_indexed += index.add_files(batch)
                batch = []

    if batch:
        num_indexed += index.add_files(batch)

    index_store.store(index)
    return index, num_indexed


def is_ignored(directory: str, full_path: str, config: Config) -> bool:
    relative_path = os.path.relpath(full_path, directory)
    directory_parts = relative_path.split(os.sep)[:-1]
    file_size = os.path.getsize(full_path)

    file_exists = os.path.exists(full_path)
    file_ignored = any(ign in directory_parts for ign in config.ignored_dirs)
    file_suffix_allowed = any(full_path.endswith(s) for s in config.allowed_suffixes)
    file_above_max_size = file_size > config.max_file_size_kb * 1024

    should_ignore = (
        not file_exists or file_ignored or not file_suffix_allowed or file_above_max_size
    )
    return should_ignore
