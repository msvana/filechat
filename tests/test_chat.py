import os
import sqlite3

from filechat.chat import Chat, ChatStore
from filechat.config import Config


def test_chat_store_creation(test_directory: str, config: Config):
    chat_store = ChatStore(test_directory, config)
    assert os.path.exists(chat_store._file_path)

    conn = sqlite3.connect(chat_store._file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM version")
    version = cursor.fetchone()
    assert version[0] == 1


def test_chat_store(test_directory: str, config: Config):
    chat_store = ChatStore(test_directory, config)
    chat = Chat(config.model, config.api_key)

    for _ in chat.user_message("This project seems to contain many test files", []):
        pass

    chat_store.store(chat)
    assert chat.chat_id is not None

    chat_store._cursor.execute("SELECT * FROM chats WHERE id = ?", (chat.chat_id,))
    chats = chat_store._cursor.fetchall()
    assert len(chats) == 1
    assert chats[0][-1] == "New chat"
    assert chats[0][1] is not None

    chat_store._cursor.execute("SELECT * FROM messages WHERE chat_id = ?", (chat.chat_id,))
    messages = chat_store._cursor.fetchall()
    assert len(messages) == 3
    assert messages[1][2] == "user"
    assert messages[1][3] == "This project seems to contain many test files"
    assert messages[-1][-1] == "[]"
    assert messages[-1][0] == 2

    for _ in chat.user_message("I'd like to include a test.c file with a hello world example", []):
        pass

    chat_store.store(chat)
    chat_store._cursor.execute("SELECT * FROM chats WHERE id = ?", (chat.chat_id,))
    chats = chat_store._cursor.fetchall()
    assert len(chats) == 1
    
    chat_store._cursor.execute("SELECT * FROM messages WHERE chat_id = ?", (chat.chat_id,))
    messages = chat_store._cursor.fetchall()
    assert len(messages) == 5
    assert messages[-2][2] == "user"
    assert messages[-2][3] == "I'd like to include a test.c file with a hello world example"
    assert messages[-1][0] == 4
