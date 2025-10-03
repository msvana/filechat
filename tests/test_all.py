from typing import cast

import pytest
from mistralai import Mistral
from openai import OpenAI
from textual.widgets import ListItem, Static
import asyncio
from filechat.chat import Chat, ChatStore
from filechat.config import Config
from filechat.embedder import Embedder
from filechat.index import get_index
from filechat.tui import FilechatApp, HistoryScreen


@pytest.mark.asyncio
async def test_basic_workflow(config: Config, client: OpenAI | Mistral):
    embedder = Embedder(
        config.embedding_model,
        config.embedding_model_path,
        config.embedding_model_url,
    )

    directory = "."
    index, _ = get_index(directory, config, embedder)
    chat = Chat(client, config.model.model, config, directory)
    chat_store = ChatStore(directory, config, client)
    app = FilechatApp(chat, index, chat_store)

    async with app.run_test(headless=True) as pilot:
        # First message
        message_1 = "What is the name of this project?"
        await pilot.press(*message_1)
        await pilot.press("enter")

        for _ in range(8):
            await asyncio.sleep(1)

            user_messages = cast(list[Static], list(pilot.app.query("Static.user")))
            llm_messages = cast(list[Static], list(pilot.app.query("Static.llm")))
            file_messages = cast(list[Static], list(pilot.app.query("Static.files")))

            if len(user_messages) != 1 or len(file_messages) != 1:
                continue
            if user_messages[0].content != message_1:
                continue
            if "filechat" not in str(llm_messages[0].content).lower():
                continue
            break
        else:
            raise AssertionError("User message, response or list of used files not visible")

        # New chat
        await pilot.press(*"/new")
        await pilot.press("enter")

        user_messages = cast(list[Static], list(pilot.app.query("Static.user")))
        llm_messages = cast(list[Static], list(pilot.app.query("Static.llm")))
        file_messages = cast(list[Static], list(pilot.app.query("Static.files")))

        assert len(user_messages) == 0
        assert len(llm_messages) == 0
        assert len(file_messages) == 0

        # Second message
        message_2 = "Which framework is used to implement the TUI"
        await pilot.press(*message_2)
        await pilot.press("enter")

        for _ in range(18):
            await asyncio.sleep(1)

            user_messages = cast(list[Static], list(pilot.app.query("Static.user")))
            llm_messages = cast(list[Static], list(pilot.app.query("Static.llm")))
            file_messages = cast(list[Static], list(pilot.app.query("Static.files")))

            if len(user_messages) != 1 or len(file_messages) != 1:
                continue
            if user_messages[0].content != message_2:
                continue
            if "textual" not in str(llm_messages[-1].content).lower():
                continue
            break
        else:
            raise AssertionError("User message, response or list of used files not visible")

        # History
        await pilot.press(*"/history")
        await pilot.press("enter")
        await asyncio.sleep(1)

        assert isinstance(pilot.app.screen, HistoryScreen)
        items = cast(list[ListItem], list(pilot.app.screen.query("ListItem")))
        first_chat_name: Static = cast(Static, items[0].children[0])
        assert len(items) == 2
        assert first_chat_name.content == message_2

        await pilot.press("down", "enter")
        await asyncio.sleep(1)

        user_messages = cast(list[Static], list(pilot.app.query("Static.user")))
        llm_messages = cast(list[Static], list(pilot.app.query("Static.llm")))
        file_messages = cast(list[Static], list(pilot.app.query("Static.files")))

        assert user_messages[0].content == message_1
        assert "filechat" in str(llm_messages[0].content).lower()
        assert len(file_messages) == 1


@pytest.mark.asyncio
async def test_tool_use(config: Config, client: OpenAI | Mistral):
    embedder = Embedder(
        config.embedding_model,
        config.embedding_model_path,
        config.embedding_model_url,
    )

    directory = "."
    index, _ = get_index(directory, config, embedder)
    chat = Chat(client, config.model.model, config, directory)
    chat_store = ChatStore(directory, config, client)
    app = FilechatApp(chat, index, chat_store)

    async with app.run_test(headless=True) as pilot:
        message_1 = "Show me the contents of the project's root directory"
        await pilot.press(*message_1)
        await pilot.press("enter")

        for _ in range(10):
            await asyncio.sleep(1)
            llm_messages = cast(list[Static], list(pilot.app.query("Static.llm")))
            if len(llm_messages) == 2 and "list_directory" in str(llm_messages[0].content):
                break
        else:
            raise AssertionError("Didn't see an assistant message with list_directory call")

        message_2 = (
            "Read the pyproject.toml file with the read_file tool, even if you "
            "already have it in your context. Then tell me the current version of the project"
        )
        await pilot.press(*message_2)
        await pilot.press("enter")

        for _ in range(10):
            await asyncio.sleep(1)
            llm_messages = cast(list[Static], list(pilot.app.query("Static.llm")))
            if len(llm_messages) == 4 and "read_file" in str(llm_messages[-2].content):
                break
        else:
            raise AssertionError("Didn't see an assistant message with read_file call")
