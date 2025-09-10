import logging
import os

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from filechat.config import Config
from filechat.index import FileIndex, is_ignored


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, index: FileIndex, config: Config):
        super().__init__()
        self._index = index
        self._config = config

    def on_modified(self, event: FileSystemEvent):
        logging.info(event)
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def on_created(self, event: FileSystemEvent):
        logging.info(event)
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return
        self._handle_file_deletion(event.src_path)

    def on_moved(self, event: FileSystemEvent):
        self._handle_file_deletion(event.src_path)
        self._handle_file_change(event.dest_path)

    def _handle_file_change(self, file_path: bytes | str):
        try:
            file_path = str(file_path)
            relative_path = os.path.relpath(file_path, self._index.directory())
            logging.info(f"File changed: {relative_path}")

            if not is_ignored(self._index.directory(), file_path, self._config):
                self._index.add_file(relative_path)
        except Exception as e:
            logging.warning(type(e))
            logging.warning(e)

    def _handle_file_deletion(self, file_path: bytes | str):
        try:
            file_path = str(file_path)
            relative_path = os.path.relpath(file_path, self._index.directory())
            logging.info(f"File deleted: {relative_path}")

            if is_ignored(self._index.directory(), file_path, self._config):
                self._index.clean_old_files()
        except Exception as e:
            logging.warning(type(e))
            logging.warning(e)


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
