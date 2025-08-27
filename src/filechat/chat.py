import os
from hashlib import sha256
from textwrap import dedent

from mistralai import Mistral

from filechat.config import Config
from filechat.index import IndexedFile
import sqlite3


class Chat:
    SYSTEM_MESSAGE = dedent("""\
    You are a local project assistant. Your task is to assist the user with various projects. 
    They can ask you question to understand the project or for suggestions on how to improve the projects.

    Besides the user's query. You will be also provided with the contents of 
    the files potentially most relevant to this query. If needed, you can use the content of these file
    to create a better response.
    """)

    def __init__(self, model: str, api_key: str | None, chat_id: int | None = None):
        self._message_history = [{"role": "system", "content": self.SYSTEM_MESSAGE}]
        self._model = model
        assert api_key is not None, (
            "Please provide an API key, either in the config file or in the MISTRAL_API_KEY"
            " environment variable"
        )
        self._client = Mistral(api_key=api_key)
        self._id = chat_id

    def user_message(self, message: str, files: list[IndexedFile]):
        user_message = {"role": "user", "content": message}
        context_message = self._get_context_message(files)
        self._message_history.append(user_message)
        response = self._client.chat.stream(
            model=self._model,
            messages=self._message_history + [context_message],  # type: ignore
        )

        response_str = ""

        for chunk in response:
            chunk_content = chunk.data.choices[0].delta.content
            response_str += str(chunk_content)
            yield str(chunk_content)

        self._message_history.append({"role": "assistant", "content": response_str})

    @property
    def chat_id(self) -> int | None:
        return self._id

    @chat_id.setter
    def chat_id(self, chat_id: int):
        self._id = chat_id

    def _get_context_message(self, files: list[IndexedFile]) -> dict:
        message = "<context>"

        for file in files:
            message += "<file>"
            message += file.content_for_embedding()
            message += "</file>"

        message += "</context>"
        return {"role": "user", "content": message}


class ChatStore:
    VERSION_LATEST = 1

    def __init__(self, directory: str, config: Config):
        self._file_path = self._get_file_path(directory, config.index_store_path)
        if not os.path.exists(self._file_path):
            self._conn, self._cursor = self._create_database()
        else:
            self._conn = sqlite3.connect(self._file_path)
            self._cursor = self._conn.cursor()

    def _get_file_path(self, directory: str, store_directory: str) -> str:
        directory = os.path.abspath(directory)
        file_hash = sha256(directory.encode()).hexdigest()
        file_name = f"{file_hash}.sqlite"
        file_path = os.path.join(store_directory, file_name)
        return file_path

    def store(self, chat: Chat):
        if chat.chat_id is None:
            self._cursor.execute("INSERT INTO chats (title) VALUES ('New chat')")
            assert self._cursor.lastrowid is not None
            chat.chat_id = self._cursor.lastrowid

    def _create_database(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(self._file_path)
        cursor = conn.cursor()

        cursor.execute("CREATE TABLE version (version INTEGER)")

        cursor.execute("""
        CREATE TABLE chats
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT
        )         
        """)

        cursor.execute("""
        CREATE TABLE messages
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            files_used TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )         
        """)

        cursor.execute("INSERT INTO version (version) VALUES (1)")
        conn.commit()

        return conn, cursor
