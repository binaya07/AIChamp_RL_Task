# AI Champ RL Task

This project implements a reinforcement learning (RL) task for an AI assistant that answers digital forensics questions.

## Description

The RL task involves retrieving relevant context from a vector store using Chroma and HuggingFace embeddings. The retrieved context is combined with the question to form a prompt, which is then passed to Claude (via the Anthropic API). Claude uses tools to execute Python expressions and submit answers in a structured JSON format.

The response is graded based on its adherence to the required JSON structure, which must include:
- `question`: The original question (string)
- `answer`: The final answer (string)
- `sources`: A list of source objects, each with `title` and `authors` (strings)

The grading process also verifies that the titles and authors in the sources match those present in the retrieved context.

## Setup

The vector store data is available for download from the following dropbox url
`https://www.dropbox.com/scl/fi/z7nyvp1b3xdk67pwxqd03/chroma.sqlite3?rlkey=90z7n2yfyq4sh4ejejd8rvaf4&st=mh8s0yan&dl=0`

Download the vector store and place it inside the `vectorstore/` folder before running the project.

To run the task, execute `uv run main.py`.
