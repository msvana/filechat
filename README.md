# FileChat

FileChat is an AI assistant designed to help users with various local projects.
It allows you to chat about files in your local folder.

## Motivation

Most AI agents focus on directly modifying your code.

I don't like that. I want a tool that allows me to seamlessly chat
about my projects with an LLM, but that doesn't let the LLM to directly make changes.
I want to be in control of my code.

## Features

- **Indexing Files**: FileChat can index files in a directory, allowing it to understand the structure and content of the project.
- **Querying Files**: Users can ask questions about the project, and FileChat will provide relevant information based on the indexed files.
- **Improving Projects**: FileChat can suggest improvements to the project based on the content of the files and the user's queries.

## Installation

The project is in an early stage of development. For now you can install the project by cloning the repository and
replicating the environment using [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/msvana/filechat
uv sync
```

## Usage

FileChat uses [Mistral AI](https://mistral.ai/)'s LLMs.
You need to store a Mistral AI API key in the `MISTRAL_API_KEY` environment
variable. Then you can invoke FileChat, with the following command:

```bash
uv run index /path/to/your/project
```

## Limitations

FileChat was tested only on Fedora 42. It might not work on other distros or operating systems.
