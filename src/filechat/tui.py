from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from filechat.chat import Chat
from filechat.index import FileIndex


class FilechatApp(App):
    CSS = """
        Static {
            padding: 0 1;
            margin: 0;
        }

        Static.llm {
            border: solid green;
        }

        Static.user {
            border: solid blue;
        }

        Input {
            border: solid blue;
        }

        VerticalScroll {
            scrollbar-size: 0 0;
        }
    """

    def __init__(self, chat: Chat, index: FileIndex):
        super().__init__()
        self._chat = chat
        self._index = index
        self._chat_list = VerticalScroll()
        self._user_input = Input(placeholder="Enter chat message ...")

    def compose(self) -> ComposeResult:
        yield self._chat_list
        yield self._user_input

        self._user_input.focus()

    def on_input_submitted(self, event: Input.Submitted):
        self._user_input.value = ""
        self.send_message(event.value)

    @work(thread=True)
    def send_message(self, message: str):
        self.call_from_thread(self._user_input.set_loading, True)

        message_widget = Static(message, classes="user")
        self.call_from_thread(self._chat_list.mount, message_widget)

        output_widget = Static(classes="llm")
        self.call_from_thread(self._chat_list.mount, output_widget)

        files = self._index.query(message)
        output_text = ""

        for chunk in self._chat.user_message(message, files):
            output_text += chunk
            self.call_from_thread(output_widget.update, output_text)
            self.call_from_thread(self._chat_list.scroll_end)

        self.call_from_thread(self._user_input.set_loading, False)
