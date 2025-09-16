import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tqdm
from tokenizers import Encoding, Tokenizer


class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class Embedder:

    def __init__(self, model_name: str, model_path: Path, model_url: str):
        self._model_name = model_name
        self._model_path = model_path
        self._model_url = model_url

        self._tokenizer: Tokenizer = Tokenizer.from_pretrained(model_name)

        self._ensure_downloaded()

        ort.preload_dlls(cuda=True, cudnn=True)

        self._session = ort.InferenceSession(
            self._model_path,
            providers=["CUDAExecutionProvider","OpenVINOExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{}, {"device_type": "AUTO:GPU,CPU"}, {}],
        )

    def _ensure_downloaded(self):
        if self._model_path.exists():
            return

        logging.info("Embedding model not found. Downloading ...")
        os.makedirs(self._model_path.parent, exist_ok=True)
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1) as t:
            temp_file = tempfile.mkstemp()[1]
            urllib.request.urlretrieve(self._model_url, filename=temp_file, reporthook=t.update_to)
            shutil.move(temp_file, self._model_path)

    def embed(self, texts: list[str]) -> np.ndarray:
        encoded: list[Encoding] = self._tokenizer.encode_batch(texts)
        embeddings = self._session.run(
            None,
            {
                "input_ids": np.array([e.ids for e in encoded]),
                "token_type_ids": np.array([e.type_ids for e in encoded]),
                "attention_mask": np.array([e.attention_mask for e in encoded]),
            },
        )
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], np.ndarray)
        embeddings = embeddings[0][:, -1, :]
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norm
        return embeddings
