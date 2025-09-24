# FileChat

FileChat is an AI assistant designed to help users understand and improve their local projects.
It allows you to chat about files in your local folder while maintaining full control over your code.

Here is a short video:

https://github.com/user-attachments/assets/dd3c6617-b141-47ab-926e-c62abcc7b4a6

## Features

- **Project Indexing**: Creates a searchable index of your project files
- **Contextual Chat**: Ask questions about your project with AI that understands your codebase
- **Real-time Updates**: Automatically detects and indexes file changes
- **Chat History**: ChatGPT-like chat history for each directory
- **Configurable**: Customize which files to index, and choose your own LLM provider. We currently support models from:
    - [Mistral AI](https://mistral.ai/)
    - [OpenAI](https://openai.com/)
    - Self-hosted servers with OpenAI-compatible API like [Ollama](https://ollama.com/) or [llama.cpp](https://github.com/ggml-org/llama.cpp).
      We recommend a context window of at least 16384.

## Installation

### Prerequisites

- Python 3.12 or higher
- An API key for the LLM provider you want to use or access to a self-hosted LLM server with an OpenAI-compatible API
- On Windows, you need [Visual C++ Redistributable](https://learn.microsoft.com/en-au/cpp/windows/latest-supported-vc-redist?view=msvc-170).
  It's very likely you have it already installed on your machine.

### Option 1: Install from PyPI

You can use any Package management tool you like. Here is an example for `pip`:

```bash
pip install filechat
```

And here is an example of installing FileChat as a UV tool:

```bash
uv tool install filechat
```

**On Linux, you should also specify the hardware accelerator as an optional dependency**.
This accelerator will be used to run the local embedding model.
We support `xpu` (Intel Arc), and `cuda`.
If you don't specify a hardware accelerator, the embedding model will run on a CPU.
Here is an example of installing FileChat with `xpu` support:

PIP:

```bash
pip install filechat[xpu]
```

UV Tool:

```bash
uv tool install filechat[xpu]
```

### Option 2: Clone the repository and use UV

1. Clone the repository:

```bash
git clone https://github.com/msvana/filechat
cd filechat
```

2. Install dependencies using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

3. (Optional) Install GPU support:

```bash
# CUDA (NVIDIA)
uv sync --extra cuda

# XPU (Intel Arc)
uv sync --extra xpu
```

## Usage

```bash
filechat /path/to/your/project
```

## Configuration

On the first run, FileChat guides you through an initial setup where you will choose your LLM provider, select a model, and set an API key.
These settings will be then stored at `~/.config/filechat.json`. Feel free to change the file as you need.

You can invoke the initial setup at any time by running FileChat with the `--setup` or `-s` flag.
You can make FileChat use a different config file path by using the `--config` or `-c` argument.

Here is an example of a valid config file:

```json
{
    "max_file_size_kb": 25,
    "ignored_dirs": [".git", "__pycache__", ".venv", ".pytest_cache", "node_modules", "dist"],
    "allowed_suffixes": [".md", ".txt", ".json", ".toml", ".html", ".css", ...],
    "index_store_path": "/home/milos/.cache/filechat",
    "model": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "api_key": "[VALID_OPENAI_API_KEY]",
        "base_url": null
    }
}
```
