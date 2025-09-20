import logging
import os
import atexit
import subprocess
import tempfile
import time
from urllib.error import URLError
import urllib.request
import zipfile
from pathlib import Path
import re

from filechat.config import load_config
from filechat.utils import DownloadProgressBar

logging.basicConfig(level=logging.INFO)


class LocalLLM:
    LLAMA_CPP_DOWNLOAD_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b6524/llama-b6524-bin-ubuntu-x64.zip"

    def __init__(self, llama_cpp_path: Path, port: int = 18000):
        self._llama_cpp_path = llama_cpp_path
        self._llama_server_path = self._llama_cpp_path / "build" / "bin" / "llama-server"
        self._port = port
        self._server_url = f"http://127.0.0.1:{port}"
        self._server_process: subprocess.Popen | None = None
        self._ensure_llama_cpp()
        atexit.register(self.cleanup)

    def __del__(self):
        self.cleanup()

    def start_llama_cpp(self):
        logging.info("Starting Llama.cpp...")
        if self._server_process and self._server_process.poll():
            logging.info("Server process already running")
            return

        self.cleanup()

        cmd = [
            str(self._llama_server_path),
            "-hf",
            "unsloth/gemma-3-1b-it-GGUF",
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--n-gpu-layers",
            "0",
            "--ctx-size",
            "4096",
            "--cont-batching",
            "--threads",
            str(os.cpu_count() or 4),
        ]

        self._server_process = subprocess.Popen(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(1)
        assert self._server_process.stderr

        for line in self._server_process.stderr:
            if line.startswith("main: server is listening"):
                print()
                break
            if line.startswith("common_download_file_single: trying to"):
                logging.info("Llama.cpp needs to download the local model. This will take a while")
            if re.match(r"^\s*\d+\s+\d*M", line):
                print(f"\t{line.strip()}\r", end="")

        max_retries = 20
        for _ in range(max_retries):
            try:
                response = urllib.request.urlopen(f"{self._server_url}/health", timeout=1)
                if response.getcode() == 200:
                    break
            except (urllib.request.HTTPError, URLError) as e:
                logging.error(f"HTTP Error when talking to Llama.cpp: {e}")

            time.sleep(1)
        else:
            if self._server_process:
                self._server_process.terminate()

    def cleanup(self):
        if not self._server_process:
            return

        logging.info("Terminating Llama.cpp...")
        try:
            self._server_process.terminate()
            self._server_process.wait(5.0)
        except subprocess.TimeoutExpired:
            self._server_process.kill()

        self._server_process = None
        atexit.unregister(self.cleanup)

    def _ensure_llama_cpp(self):
        if self._llama_server_path.exists():
            return

        logging.info("Llama CPP not not found. Downloading...")
        os.makedirs(self._llama_cpp_path, exist_ok=True)
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1) as t:
            temp_fd, temp_filename = tempfile.mkstemp()
            urllib.request.urlretrieve(
                self.LLAMA_CPP_DOWNLOAD_URL, filename=temp_filename, reporthook=t.update_to
            )
            with zipfile.ZipFile(temp_filename, "r") as zip_fd:
                zip_fd.extractall(self._llama_cpp_path)
            os.close(temp_fd)
            os.remove(temp_filename)

        self._llama_server_path.chmod(0o755)


config = load_config()
local_llm = LocalLLM(config.llama_cpp_path)

try:
    local_llm.start_llama_cpp()
finally:
    local_llm.cleanup()
