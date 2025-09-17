# FileChat

FileChat is an AI assistant designed to help users understand and improve their local projects.
It allows you to chat about files in your local folder while maintaining full control over your code.

https://github.com/user-attachments/assets/dd3c6617-b141-47ab-926e-c62abcc7b4a6


## Features

- **Project Indexing**: Creates a searchable index of your project files
- **Contextual Chat**: Ask questions about your project with AI that understands your codebase
- **Real-time Updates**: Automatically detects and indexes file changes
- **Configurable**: Customize which files to index and how to process them
- **Chat History**: ChatGPT-like chat history for each directory

## Installation

### Prerequisites

- Python 3.12 or higher
- A [Mistral AI](https://mistral.ai/) API key stored in the `MISTRAL_API_KEY` environment variable

### Option 1: Install the pre-built wheel

You can use any Package management tool you like. Here is an example for `pip`:

```bash
pip install https://github.com/msvana/filechat/releases/download/latest/filechat-0.2.0-py3-none-any.whl
```

And here is an example of installing FileChat as a UV tool:

```bash
uv tool install https://github.com/msvana/filechat/releases/download/latest/filechat-0.2.0-py3-none-any.whl
```

**On Linux, you should also specify the hardware accelerator as an optional dependency**. We support `cpu`, `xpu` (Intel Arc), and `cuda`.
If you don't specify the accelerator, you'll get a version with CUDA support, which might be unnecessarily large. Here is an example of 
installing FileChat with `xpu` support:

PIP:

```bash
pip install "filechat[xpu] @ https://github.com/msvana/filechat/releases/download/latest/filechat-0.2.0-py3-none-any.whl"
```

UV Tool:

```bash
uv tool install https://github.com/msvana/filechat/releases/download/latest/filechat-0.2.0-py3-none-any.whl[xpu]
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

3. (Optional) Install GPU support if available:

```bash
# CUDA (NVIDIA)
uv sync --extra cuda

# XPU (Intel Arc)
uv sync --extra xpu
```

## Usage

```bash
uv run filechat /path/to/your/project
```

## Configuration

On the first run, FileChat creates a configuration file at `~/.config/filechat.json`. Feel free to change it as you need.
Here is a full example:

```json
{
    "max_file_size_kb": 30,
    "ignored_dirs": [".git", "__pycache__", ".venv", ".pytest_cache", "node_modules"],
    "allowed_suffixes": [".txt", ".json", ".py", ".toml", ".html", ".md", ".js", ".ts", ".vue"],
    "index_store_path": "/home/milos/.cache/filechat",
    "model": "mistral-medium-2508",
    "api_key": "[MISTRAL_API_KEY]"
}
```
## Roadmap

### Short term (weeks)

- Add support for other models
- Add tools for browsing the filesystem
- Improve inference for embedding models (switch from sentence-transformers to an ONNX runtime)
- Support CUDA on Windows
- Publish on PyPI

### Long term (months)

- Improve file retrieval (for example, via Graph RAG)
- Reimplement file indexing and querying in a compiled language
- Support important binary file types (images, PDFs)
- Add web search tools
