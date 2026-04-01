# Agents

This directory documents the agents currently shipped with omnidex.

## Available Agents

- `orchestrator`

## Agent Overview

- `orchestrator`: Runs as an interactive local-first chat agent backed by a GGUF
  model loaded through `llama-cpp-python`.

## CLI Entry Points

- `python -m omnidex`: Start the interactive orchestrator.
- `python -m omnidex --prompt "..."`: Run a single prompt and exit.
