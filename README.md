# OmniDex

A lightweight Python CLI for running local agents.

## Documentation

- [Docs Index](docs/README.md)

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Environment Variables

This project uses dotenv files. Copy `.env.dist` to `.env` and update values.

## Initial Agent

The first shipped entry point is `orchestrator`, an interactive local-first
delegator that runs through `llama-cpp-python` and renders responses with Rich.

The orchestrator delegates to:
- `chat_agent` for general conversation, broad questions on any topic, and
  questions about the current session context or active artifact
- `research_assistant` for PDF analysis, research workflows, and artifact
  save/export follow-ups

Routing is model-proposed, then validated through a shared policy layer.
Handoffs are also model-proposed. The validator only enforces hard constraints
such as:
- explicit PDF workflows staying with `research_assistant`
- explicit save/export follow-ups staying with `research_assistant` when it can
  complete them

It now includes:
- short-term memory for recent conversation turns
- persistent long-term memory for durable user facts
- context retrieval that merges recent chat with relevant stored memory
- shared session artifact state across delegated agents
- bounded artifact history for follow-up requests such as `save the first
  insights`

CLI logging:
- `OMNIDEX_RENDER_MARKDOWN` (default `1`)
- `OMNIDEX_LLAMA_MODEL_PATH` (required)
- `OMNIDEX_DEVICE` (`cpu` by default, set to `gpu` for CUDA-backed loading)
- `OMNIDEX_LLAMA_TEMPERATURE` (optional)
- `OMNIDEX_LLAMA_MAX_TOKENS` (optional)
- `OMNIDEX_LLAMA_TOP_P` (optional)
- `OMNIDEX_LLAMA_CTX` (optional runtime context length)
- `OMNIDEX_LLAMA_THREADS` (optional CPU thread count)
- `OMNIDEX_LLAMA_GPU_LAYERS` (optional GPU offload override; defaults to all layers on `gpu`, `0` on `cpu`)
- `OMNIDEX_LLAMA_VERBOSE` (set to `1` to show llama.cpp logs)
- `OMNIDEX_STREAM` (set to `1` to stream tokens live in the terminal)
- `OMNIDEX_SHORT_TERM_LIMIT` (optional recent-message window, default `5`)
- `OMNIDEX_MEMORY_PATH` (optional persistent memory file path)

## Run OmniDex

```bash
export OMNIDEX_LLAMA_MODEL_PATH=/path/to/model.gguf
python -m omnidex
```

Slash commands:
- `/help`
- `/clear`
- `/exit`

For a one-shot prompt:

```bash
python -m omnidex --prompt "Explain what this project should do."
```
