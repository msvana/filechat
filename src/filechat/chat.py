import os
from textwrap import dedent

from mistralai import Mistral

from filechat.index import IndexedFile


class Chat:
    SYSTEM_MESSAGE = dedent("""\
    You are a local project assistant. Your task is to assist the user with various projects. 
    They can ask you question to understand the project or for suggestions on how to improve the projects.

    Besides the user's query. You will be also provided with the contents of 
    the files potentially most relevant to this query. If needed, you can use the content of these file
    to create a better response.
    """)

    def __init__(self, model: str, api_key: str | None):
        self._message_history = [{"role": "system", "content": self.SYSTEM_MESSAGE}]
        self._model = model
        assert api_key is not None, (
            "Please provide an API key, either in the config file or in the MISTRAL_API_KEY"
            " environment variable"
        )
        self._client = Mistral(api_key=api_key)

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

    def _get_context_message(self, files: list[IndexedFile]) -> dict:
        message = "<context>"

        for file in files:
            message += "<file>"
            message += file.content_for_embedding()
            message += "</file>"

        message += "</context>"
        return {"role": "user", "content": message}
