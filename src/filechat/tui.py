import logging
from argparse import ArgumentParser

from sentence_transformers import SentenceTransformer
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from filechat.chat import Chat
from filechat.config import Config
from filechat.index import FileIndex, get_index
from filechat.watcher import FileWatcher

logging.basicConfig(level=logging.CRITICAL)

arg_parser = ArgumentParser(description="Index files in a directory")
arg_parser.add_argument("directory", type=str, help="Directory to index files from")
arg_parser.add_argument(
    "-r", "--rebuild", action="store_true", help="Ignore cache, rebuild index from scratch"
)


class FilechatApp(App):
    CSS = """
        Static {
            border: solid green;
            padding: 0 1;
            margin: 0;
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
        output_widget = Static()
        self._chat_list.mount(output_widget)
        self._user_input.value = ""
        self.send_message(output_widget, event.value)

    @work(thread=True)
    def send_message(self, output_widget: Static, message: str):
        self.call_from_thread(self._user_input.set_loading, True)
        files = self._index.query(message)
        output_text = ""
        for chunk in self._chat.user_message(message, files):
            output_text += chunk
            self.call_from_thread(output_widget.update, output_text)
            self.call_from_thread(self._chat_list.scroll_end)
        self.call_from_thread(self._user_input.set_loading, False)


def main():
    config = Config()
    args = arg_parser.parse_args()

    sentence_transformer = SentenceTransformer(
        config.get_embedding_model(), trust_remote_code=True, device=config.get_device()
    )

    index, _ = get_index(args.directory, config, sentence_transformer, args.rebuild)
    watcher = FileWatcher(index, config)
    watcher.start()

    chat = Chat()

    app = FilechatApp(chat, index)
    app.run()

    watcher.stop()


if __name__ == "__main__":
    main()
