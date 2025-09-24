from textual import work
from textual.app import App, ComposeResult
from textual.containers import Center, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView, Static

from filechat.chat import Chat, ChatStore
from filechat.index import FileIndex


class HistoryScreen(ModalScreen):

    def __init__(self, chat_store: ChatStore, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history_view = ListView()
        self._chat_store = chat_store
        self._chats = self._chat_store.chat_list()

    def compose(self) -> ComposeResult:
        with Vertical():
            center = Center(self._history_view)
            center.border_title = "Chat History (press 'd' to delete a chat)"
            yield center

    def on_mount(self):
        for chat in self._chats:
            chat_item = ListItem(Static(chat[2]), Static(chat[1], classes="timestamp"))
            self._history_view.append(chat_item)

        if self._chats:
            self._history_view.index = 0
            self._history_view.focus()

    def key_escape(self):
        self.dismiss()

    def key_enter(self):
        if self._history_view.highlighted_child:
            selected_index = self._history_view.index
            assert selected_index is not None
            self._close_with_selected_chat(selected_index)

    def key_d(self):
        if self._history_view.highlighted_child:
            selected_index = self._history_view.index
            assert selected_index is not None
            self._delete_selected_chat(selected_index)

    def on_list_view_selected(self, event: ListView.Selected):
        selected_index = event.index
        self._close_with_selected_chat(selected_index)

    def _close_with_selected_chat(self, selected_index: int):
        chat_id = self._chats[selected_index][0]
        chat = self._chat_store.load(chat_id)
        self.dismiss(chat)

    def _delete_selected_chat(self, selected_index: int):
        chat_id = self._chats.pop(selected_index)[0]
        self._chat_store.delete(chat_id)
        self._history_view.remove_items([selected_index])


class FilechatApp(App):
    CSS = """
        Static {
            padding: 0;
            margin: 0;
        }

        Static.llm {
            border-top: solid green;
            border-bottom: solid green;
        }

        Static.user {
            border-top: solid blue;
            border-bottom: solid blue;
        }
        
        Static.files {
            border-top: solid orange;
            border-bottom: solid orange;
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
        self._user_input = Input(
            placeholder=(
                "Enter chat message ... (type /exit to quit, /new to start a new chat, or /history"
                " to revisit previous chats)"
            )
        )

    def compose(self) -> ComposeResult:
        yield self._chat_list
        yield self._user_input

        self._user_input.focus()

    def on_input_submitted(self, event: Input.Submitted):
        user_message = event.value.strip()
        if user_message == "/exit":
            self.exit()
        elif user_message == "/history":
            self._show_history_modal()
        elif user_message == "/new":
            self._start_new_chat()
        elif user_message != "":
            self.send_message(event.value)
        self._user_input.value = ""

    @work(thread=True)
    def send_message(self, message: str):
        self.call_from_thread(self._user_input.set_loading, True)

        message_widget = Static(message, classes="user")
        message_widget.border_title = "User"
        self.call_from_thread(self._chat_list.mount, message_widget)

        output_widget = Static(classes="llm")
        output_widget.border_title = "Assistant"
        self.call_from_thread(self._chat_list.mount, output_widget)

        files = self._index.query(message)
        output_text = ""

        for chunk in self._chat.user_message(message, files):
            output_text += chunk
            self.call_from_thread(output_widget.update, output_text)
            self.call_from_thread(self._chat_list.scroll_end)

        self.call_from_thread(self._chat_store.store, self._chat)
        self.call_from_thread(self._user_input.set_loading, False)

        files_used = "; ".join(f.path() for f in files)
        files_widget = Static(files_used, classes="files")
        files_widget.border_title = "Files"
        self.call_from_thread(self._chat_list.mount, files_widget)
        self.call_from_thread(self._chat_list.scroll_end)

    def _show_history_modal(self):
        def handle_history_result(chat: Chat | None):
            if not chat:
                return
            self._load_chat(chat)

        self.push_screen(HistoryScreen(self._chat_store), callback=handle_history_result)

    def _load_chat(self, chat: Chat):
        self._chat = chat
        self._chat_list.remove_children()

        for message in self._chat.messages:
            if message["role"] == "system":
                continue

            message_widget = Static(
                message["content"], classes="user" if message["role"] == "user" else "llm"
            )
            message_widget.border_title = "User" if message["role"] == "user" else "Assistant"
            self._chat_list.mount(message_widget)

            if message["role"] == "assistant":
                files_widget = Static("; ".join(message["files_used"]), classes="files")
                files_widget.border_title = "Files"
                self._chat_list.mount(files_widget)

        self._chat_list.scroll_end()

    def _start_new_chat(self):
        self._chat = self._chat_store.new_chat()
        self._chat_list.remove_children()
