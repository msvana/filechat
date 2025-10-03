import json
import os
import sqlite3
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

from mistralai import Mistral
from openai import OpenAI, omit

from filechat import tools
from filechat.config import Config
from filechat.index import IndexedFile
from filechat.utils import truncate_text


class Chat:
    TITLE_MAX_LENGTH = 30

    SYSTEM_MESSAGE = dedent("""\
    You are a local development assistant with access to project files. You help developers understand, debug, and improve their codebase.
    
    Key capabilities:
    - Code analysis and explanation
    - Bug identification and fixes  
    - Architecture and refactoring suggestions
    - Implementation guidance following project patterns
    - Documentation generation
    
    Context: You'll receive relevant file contents with each query. Use this context to:
    - Reference actual code patterns and structures
    - Suggest changes that fit the existing codebase
    - Identify inconsistencies or potential issues
    - Provide concrete, implementable solutions
    - This doesn't cover every file in the project, only the most relevant ones. You can use tools to get other files if you consider it useful

    You have programmatic tools to inspect the project (list_directory and read_file). 
    When you need any file contents or directory listing to answer correctly, prefer using those tools instead of guessing.
    If you call read_file, pass the exact relative path within the project.
    
    Respond with actionable advice. When suggesting code changes, show specific examples using the project's existing conventions.
    """)

    def __init__(
        self,
        client: Mistral | OpenAI,
        model: str,
        config: Config,
        project_directory: str,
        chat_id: int | None = None,
    ):
        self._message_history: list[dict] = [{"role": "system", "content": self.SYSTEM_MESSAGE}]
        self._model = model
        self._client = client
        self._config = config
        self._project_directory = Path(project_directory)
        self._id = chat_id

    def user_message(self, message: str | None, files: list[IndexedFile], use_tools: bool = True):
        if message:
            user_message = {"role": "user", "content": message}
            self._message_history.append(user_message)

        if isinstance(self._client, Mistral):
            response = self._client.chat.stream(
                model=self._model,
                messages=self._history_with_context(files),  # type: ignore
                tools=tools.TOOLS if use_tools else None,  # type: ignore
                tool_choice="auto" if use_tools else "none"
            )
        else:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._history_with_context(files),  # type: ignore
                tools=tools.TOOLS if use_tools else None,  # type: ignore
                stream=True,
                parallel_tool_calls=False if use_tools else omit,
            )

        response_str = ""
        tool_call_id = tool_call_name = tool_call_arguments = None

        for chunk in response:
            if hasattr(chunk, "data"):
                chunk = chunk.data

            chunk_delta = chunk.choices[0].delta  # type: ignore

            if not chunk_delta.content and not chunk_delta.tool_calls:
                continue

            if chunk_delta.tool_calls:
                if not tool_call_id:
                    tool_call_id = chunk_delta.tool_calls[0].id
                    tool_call_name = chunk_delta.tool_calls[0].function.name
                    tool_call_arguments = chunk_delta.tool_calls[0].function.arguments
                else:
                    assert isinstance(tool_call_arguments, str)
                    tool_call_arguments += str(chunk_delta.tool_calls[0].function.arguments)
                continue

            chunk_content = chunk_delta.content
            response_str += str(chunk_content)
            yield str(chunk_content)

        filenames = [f.path() for f in files]
        self._message_history.append({
            "role": "assistant",
            "content": response_str,
            "files_used": filenames,
        })

        if tool_call_id and tool_call_name and tool_call_arguments:
            yield self._call_tool(tool_call_id, tool_call_name, str(tool_call_arguments))

    @property
    def chat_id(self) -> int | None:
        return self._id

    @chat_id.setter
    def chat_id(self, chat_id: int):
        self._id = chat_id

    @property
    def messages(self):
        return self._message_history

    @messages.setter
    def messages(self, messages: list[dict]):
        self._message_history = messages

    @property
    def title(self):
        if len(self._message_history) < 2:
            return "New chat"

        first_user_message = self._message_history[1]["content"]
        return truncate_text(first_user_message, 50)

    def _history_with_context(self, files: list[IndexedFile]) -> list[dict]:
        message = (
            "Here are the most relevant files to user's query found using embedding search."
            "These are not the same as files returned via a tool call. Do no confuse the two."
            "If you think these files are not enough, feel free to call a tool."
        )
        message += "<context>"

        for file in files:
            message += "<file>"
            message += file.content_for_embedding()
            message += "</file>"

        message += "</context>"

        messages = self._message_history.copy()
        messages.insert(
            1,
            {
                "role": "system",
                "content": message,
            },
        )

        return messages

    def _call_tool(self, tool_call_id: str, tool_call_name: str, tool_call_arguments: str) -> dict:
        arguments_parsed: dict = json.loads(tool_call_arguments)
        if tool_call_name == "list_directory":
            try:
                result = tools.list_directory(
                    self._project_directory, arguments_parsed["path"], self._config
                )
            except Exception as e:
                result = e
        elif tool_call_name == "read_file":
            try:
                result = tools.read_file(
                    self._project_directory, arguments_parsed["path"], self._config
                )
            except Exception as e:
                result = e
        else:
            raise ValueError(f"Unknown tool '{tool_call_name}'")

        self._message_history.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call_name,
                        "arguments": tool_call_arguments,
                    },
                }
            ],
        })

        self._message_history.append({
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": tool_call_name,
            "content": str(result),
        })

        return self._message_history[-1]


class ChatStore:
    VERSION_LATEST = 1

    def __init__(self, directory: str, config: Config, client: Mistral | OpenAI):
        self._client = client
        self._project_directory = directory
        self._file_path = self._get_file_path(directory, config.index_store_path)
        self._config = config
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

    def new_chat(self) -> Chat:
        return Chat(self._client, self._config.model.model, self._config, self._project_directory)

    def store(self, chat: Chat):
        if chat.chat_id is None:
            title = chat.title
            self._cursor.execute("INSERT INTO chats (title) VALUES (?)", (title,))
            assert self._cursor.lastrowid is not None
            chat.chat_id = self._cursor.lastrowid

        self._cursor.execute("SELECT MAX(id) FROM messages WHERE chat_id = ?", (chat.chat_id,))
        messages_to_store = chat.messages
        max_id = self._cursor.fetchone()[0]
        start_id = 0 if max_id is None else max_id + 1
        if max_id is not None:
            messages_to_store = messages_to_store[start_id:]

        self._store_messages(chat.chat_id, messages_to_store, start_id)
        self._conn.commit()

    def chat_list(self) -> list[tuple]:
        self._cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
        chats = self._cursor.fetchall()
        return chats

    def load(self, chat_id: int) -> Chat | None:
        self._cursor.execute("SELECT * FROM chats WHERE id == ?", (chat_id,))
        chat = self._cursor.fetchone()

        if not chat:
            return None

        chat = Chat(
            self._client, self._config.model.model, self._config, self._project_directory, chat_id
        )
        self._cursor.execute("SELECT * FROM messages WHERE chat_id = ?", (chat_id,))
        messages_raw = self._cursor.fetchall()
        messages = []

        for message_raw in messages_raw:
            try:
                message_dict = json.loads(message_raw[3])
                if "tool_calls" in message_dict:
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": message_dict["tool_calls"],
                    })
                elif "tool_response" in message_dict:
                    messages.append(message_dict["tool_response"])
                continue
            except json.JSONDecodeError:
                pass
            message = {
                "role": message_raw[2],
                "content": message_raw[3],
            }
            if message_raw[4]:
                message["files_used"] = json.loads(message_raw[4])
            messages.append(message)

        chat.messages = messages
        return chat

    def delete(self, chat_id: int) -> int:
        self._cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        self._cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        self._conn.commit()
        return self._cursor.rowcount

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
            id INTEGER,
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

    def _store_messages(self, chat_id: int, messages: list[dict], start_id: int):
        query_template = "INSERT INTO messages VALUES (?, ?, ?, ?, ?)"
        for i, message in enumerate(messages):
            files_used = message.get("files_used")
            if isinstance(files_used, list):
                files_used = json.dumps(files_used)

            if "tool_calls" in message:
                message["content"] = json.dumps({"tool_calls": message["tool_calls"]})
            elif "tool_call_id" in message:
                message["content"] = json.dumps({"tool_response": message})

            self._cursor.execute(
                query_template,
                (
                    start_id + i,
                    chat_id,
                    message["role"],
                    message["content"],
                    files_used,
                ),
            )
        self._conn.commit()
