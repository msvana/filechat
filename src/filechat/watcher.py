import logging
import os

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from filechat.config import Config
from filechat.index import FileIndex


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, index: FileIndex, config: Config):
        self._index = index
        self._allowed_suffixes = config.allowed_suffixes
        self._ignored_directories = config.ignored_dirs

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return
        self._handle_file_deletion(event.src_path)

    def _handle_file_change(self, file_path: bytes | str):
        file_path = str(file_path)
        relative_path = os.path.relpath(file_path, self._index.directory())
        if any(ignored in relative_path for ignored in self._ignored_directories):
            return

        if all(not relative_path.endswith(suffix) for suffix in self._allowed_suffixes):
            return

        logging.info(f"File changed: {relative_path}")
        self._index.add_file(relative_path)

    def _handle_file_deletion(self, file_path: bytes | str):
        file_path = str(file_path)
        relative_path = os.path.relpath(file_path, self._index.directory())

        if any(ignored in relative_path for ignored in self._ignored_directories):
            return

        if all(not relative_path.endswith(suffix) for suffix in self._allowed_suffixes):
            return

        logging.info(f"File deleted: {relative_path}")
        self._index.clean_old_files()


class FileWatcher:
    def __init__(self, index: FileIndex, config: Config):
        self._index = index
        self._config = config
        self._observer = Observer()

    def start(self):
        event_handler = FileChangeHandler(self._index, self._config)
        self._observer.schedule(event_handler, self._index.directory(), recursive=True)
        self._observer.start()
        logging.info(f"Started watching directory: {self._index.directory()}")

    def stop(self):
        self._observer.stop()
        self._observer.join()
        logging.info(f"Stopped watching directory: {self._index.directory()}")
