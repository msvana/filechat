import os
import sqlite3

from mistralai import Mistral
from openai import OpenAI

from filechat.chat import Chat, ChatStore
from filechat.config import Config
from filechat.index import IndexedFile


def test_chat_store_creation(test_directory: str, config: Config, client: OpenAI | Mistral):
    chat_store = ChatStore(test_directory, config, client)
    assert os.path.exists(chat_store._file_path)

    conn = sqlite3.connect(chat_store._file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM version")
    version = cursor.fetchone()
    assert version[0] == 1


def test_chat_store(test_directory: str, config: Config, client: OpenAI | Mistral):
    test_message = "This project seems to contain many test files"
    chat_store = ChatStore(test_directory, config, client)

    chats = chat_store.chat_list()
    assert len(chats) == 0

    chat1 = Chat(client, config.model.model, config, test_directory)
    for _ in chat1.user_message(test_message, [], False):
        pass

    chat_store.store(chat1)

    chat2 = Chat(client, config.model.model, config, test_directory)
    for _ in chat1.user_message(test_message, [], False):
        pass

    chat_store.store(chat2)

    chats = chat_store.chat_list()
    assert len(chats) == 2

    assert chats[0][0] == chat2.chat_id
    assert chats[1][0] == chat1.chat_id
    assert chats[1][2] == "This project seems to contain many test files"


def test_load_nonexistent(test_directory: str, config: Config, client: OpenAI | Mistral):
    chat_store = ChatStore(test_directory, config, client)
    chat = chat_store.load(999)
    assert chat is None


def test_load_existing(test_directory: str, config: Config, client: OpenAI | Mistral):
    chat_store = ChatStore(test_directory, config, client)
    test_file = IndexedFile(test_directory, "test.md")

    chat = Chat(client, config.model.model, config, test_directory)
    for _ in chat.user_message("This project seems to contain many test files", [test_file], False):
        pass

    chat_store.store(chat)

    assert chat.chat_id is not None
    chat_loaded = chat_store.load(chat.chat_id)

    assert chat_loaded is not None
    assert chat_loaded.chat_id == chat.chat_id
    print(chat_loaded.messages)
    assert len(chat_loaded.messages) == 3
    last_message = chat_loaded.messages[-1]
    assert last_message["role"] == "assistant"
    assert last_message["files_used"][0] == "test.md"

    for _ in chat.user_message("Thank you", []):
        pass
