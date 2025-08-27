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
    chat_store.store(chat)
    assert chat.chat_id is not None
    
    chat_store._cursor.execute("SELECT COUNT(*) FROM chats WHERE id = ?", (chat.chat_id,))
    count = chat_store._cursor.fetchone()[0]
    assert count == 1

    chat_store._cursor.execute("SELECT title FROM chats WHERE id = ?", (chat.chat_id,))
    title = chat_store._cursor.fetchone()[0]
    assert title == "New chat"

    chat_store._cursor.execute("SELECT created_at FROM chats WHERE id = ?", (chat.chat_id,))  
    created_at = chat_store._cursor.fetchone()[0]
    assert created_at is not None
