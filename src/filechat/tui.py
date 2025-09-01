from textual import work
from textual.app import App, ComposeResult
from textual.containers import Center, VerticalScroll, Vertical
from textual.widgets import Input, ListItem, ListView, Static
from textual.screen import ModalScreen

from filechat.chat import Chat, ChatStore
from filechat.index import FileIndex


class HistoryScreen(ModalScreen):

    def __init__(self, chat_store: ChatStore, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history_view = ListView()
        self._chat_store = chat_store

    def compose(self) -> ComposeResult:
        with Vertical():
            center = Center(self._history_view)
            center.border_title = "Chat History"
            yield center

    def on_mount(self):
        chats = self._chat_store.chat_list()
        for chat in chats:
            chat_item = ListItem(Static(chat[2]), Static(chat[1], classes="timestamp"))
            self._history_view.append(chat_item)

        if chats:
            self._history_view.index = 0
            self._history_view.focus()

    def key_escape(self):
        self.dismiss()


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

        Center {
            background: $surface;
            width: 60;
            height: 50%;
            min-height: 20;
            border: heavy $primary;
        }

        Vertical {
            align: center middle;
            width: 100%;
        }

        Static.timestamp {
            color: gray;
        }
    """

    def __init__(self, chat: Chat, index: FileIndex, chat_store: ChatStore):
        super().__init__()
        self._chat = chat
        self._index = index
        self._chat_store = chat_store
        self._chat_list = VerticalScroll()
        self._user_input = Input(placeholder="Enter chat message ... (or type /exit to quit)")

    def compose(self) -> ComposeResult:
        yield self._chat_list
        yield self._user_input

        self._user_input.focus()

    def on_input_submitted(self, event: Input.Submitted):
        if event.value.strip() == "/exit":
            self.exit()
        elif event.value.strip() == "/history":
            self._show_history_modal()
        elif event.value.strip() != "":
            self.send_message(event.value)
        self._user_input.value = ""

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

        self.call_from_thread(self._chat_store.store, self._chat)
        self.call_from_thread(self._user_input.set_loading, False)

    def _show_history_modal(self):
        self.push_screen(HistoryScreen(self._chat_store))
