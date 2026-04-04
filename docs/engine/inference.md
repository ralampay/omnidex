# Inference

Inference in OmniDex is the process of turning a chat-style message list into a
model response through the local GGUF runtime. The two main components are
`LocalLLMSettings`, which defines the runtime requirements and model parameters,
and `LocalChatModel`, which loads the model and executes chat completion calls.

## LocalLLMSettings

`LocalLLMSettings` is the configuration contract for local inference. It is
defined in `omnidex/runtime.py` and is typically created with
`LocalLLMSettings.from_env(system_prompt=...)`.

### Responsibilities

- hold the required model path
- store the system prompt for the owning agent
- define inference parameters such as temperature, top-p, max tokens, and context size
- define device and execution settings such as thread count and GPU layer offload
- carry UI-adjacent defaults that higher layers use, such as Markdown rendering and streaming

### Required Inputs

Inference requires these conditions to be satisfied before a model call can work:

- `OMNIDEX_LLAMA_MODEL_PATH` must be set
- `OMNIDEX_LLAMA_MODEL_PATH` must point to an existing local GGUF file
- the caller must provide a `system_prompt`
- `OMNIDEX_DEVICE` must be either `cpu` or `gpu` if set

If `OMNIDEX_LLAMA_MODEL_PATH` is missing, `from_env(...)` raises a `ValueError`.
If `OMNIDEX_DEVICE` is set to an unsupported value, `from_env(...)` raises a
`ValueError`.

### Environment Mapping

`LocalLLMSettings.from_env(...)` resolves inference behavior from the environment.

- `OMNIDEX_LLAMA_MODEL_PATH`: required absolute or expandable path to the GGUF model
- `OMNIDEX_DEVICE`: runtime device, either `cpu` or `gpu`
- `OMNIDEX_LLAMA_TEMPERATURE`: sampling temperature
- `OMNIDEX_LLAMA_TOP_P`: top-p sampling value
- `OMNIDEX_LLAMA_MAX_TOKENS`: maximum generated tokens per completion
- `OMNIDEX_LLAMA_CTX`: llama.cpp context window size
- `OMNIDEX_LLAMA_THREADS`: CPU thread count override as an integer
- `OMNIDEX_LLAMA_GPU_LAYERS`: explicit GPU offload override
- `OMNIDEX_LLAMA_VERBOSE`: whether llama.cpp logs remain visible
- `OMNIDEX_RENDER_MARKDOWN`: higher-level CLI rendering default
- `OMNIDEX_STREAM`: default inference streaming mode

### Default Resolution Rules

- `temperature` defaults to `0.2`
- `top_p` defaults to `0.95`
- `max_tokens` defaults to `512`
- `ctx_size` defaults to `8192`
- `threads` defaults to dynamic selection only when `OMNIDEX_LLAMA_THREADS` is unset
- Unset CPU threading prefers physical cores when they can be detected, otherwise it uses a bounded heuristic from available CPUs
- Unset GPU threading uses a smaller CPU thread pool to avoid oversubscription while feeding GPU inference
- `gpu_layers` defaults to `-1` when `OMNIDEX_DEVICE=gpu` and no explicit override is set
- `gpu_layers` defaults to `0` on CPU when no explicit override is set
- `verbose` defaults to `False`
- `render_markdown` defaults to `True`
- `stream` defaults to `False`

### Why It Matters

`LocalLLMSettings` determines whether inference can start at all and how the
local model is loaded. Agents do not manually pass these low-level values each
time; they prepare this settings object once and give it to `LocalChatModel`.

## LocalChatModel

`LocalChatModel` is the execution wrapper around `llama_cpp.Llama`. It is also
defined in `omnidex/runtime.py`.

### Responsibilities

- validate that the configured model file exists
- initialize `llama_cpp.Llama` with the resolved inference settings
- suppress llama.cpp stderr noise unless verbose mode is enabled
- submit chat-format messages for completion
- expose both raw and normalized text inference APIs

### What It Does Not Do

`LocalChatModel` is not responsible for:

- routing requests to the correct agent
- building memory context
- choosing tools
- extracting PDF text
- formatting the final answer
- owning prompt design or business logic

Those responsibilities belong to the agent that owns the model instance.

### Initialization Requirements

Creating `LocalChatModel(settings)` requires:

- a valid `LocalLLMSettings` instance
- an existing `settings.model_path`

If the configured path does not exist, `LocalChatModel` raises a
`FileNotFoundError`.

During initialization it constructs:

```python
Llama(
    model_path=str(settings.model_path),
    n_ctx=settings.ctx_size,
    n_threads=settings.threads,
    n_gpu_layers=settings.gpu_layers,
    verbose=settings.verbose,
)
```

### Inference Call Contract

`LocalChatModel` exposes two inference methods:

```python
complete(messages, stream=None)
generate_text(messages, stream=None)
```

Use them this way:

- `complete(...)`: low-level access to the raw llama.cpp completion object or stream iterator
- `generate_text(...)`: high-level helper that returns final assistant text directly

The default application-facing method should be `generate_text(...)` when the
caller only needs content.

### Raw Completion API

The low-level method is:

```python
complete(messages, stream=None)
```

Requirements for calling `complete(...)`:

- `messages` must be an iterable of chat message dictionaries
- each message is expected to follow the llama chat format, typically with `role` and `content`
- the caller decides what goes into the system and user messages

`LocalChatModel` forwards the request to:

```python
self._llm.create_chat_completion(
    messages=list(messages),
    temperature=self.settings.temperature,
    top_p=self.settings.top_p,
    max_tokens=self.settings.max_tokens,
    stream=should_stream,
)
```

### Text Generation API

The high-level method is:

```python
generate_text(messages, stream=None)
```

`generate_text(...)` calls `complete(...)` internally and normalizes the output:

- non-streaming mode extracts `choices[0]["message"]["content"]`
- streaming mode collects `choices[0]["delta"]["content"]` chunks and joins them
- both paths return a final stripped `str`

### Streaming Behavior

- if `stream` is passed explicitly to either inference method, that value is used
- otherwise `settings.stream` is used

Return shape:

- non-streaming mode returns a completion dictionary
- streaming mode returns an iterator of event chunks

Typical raw non-stream extraction:

```python
text = completion["choices"][0]["message"]["content"].strip()
```

Typical raw streaming extraction:

```python
piece = event["choices"][0].get("delta", {}).get("content")
```

Typical text-first call:

```python
text = model.generate_text(messages, stream=False)
```

## How Inference Works End to End

The inference lifecycle is:

1. An agent chooses a system prompt for its role.
2. The agent creates `LocalLLMSettings.from_env(system_prompt=...)`.
3. The agent creates `LocalChatModel(settings)`.
4. The agent builds a `messages` list for the task.
5. The agent usually calls `model.generate_text(messages, stream=...)`.
6. `LocalChatModel` either extracts message content or collects streamed deltas.
7. The agent receives final text and decides what to do next.

This means the behavior of inference is shaped by the agent's prompt assembly,
not by `LocalChatModel` alone.

## How Agents Use These Components

The main delegated agents follow the same pattern:

```python
self.settings = LocalLLMSettings.from_env(system_prompt=DEFAULT_SYSTEM_PROMPT)
self.model = LocalChatModel(self.settings)
```

After that, the agent controls inference by changing the message list.

- `OrchestratorAgent` uses the local model for initial route proposals
- `ChatAgent` uses the local model for general chat responses and handoff
  proposals grounded in shared context
- `ResearchAssistant` uses the same local model for summarization, PDF question
  answering, research planning, and handoff proposals

The orchestrator is not the final chat responder. It uses inference to choose an
initial delegate, then validates that proposal through shared policy rules. The
same pattern applies to agent-to-agent handoffs.

So when `LocalChatModel` is passed into or attached to an agent, it should be
understood as the common inference engine, while the agent remains responsible
for prompt design and response handling.

## Atomic Example

This is the smallest useful example for getting a response from local inference:

```python
from omnidex.runtime import LocalChatModel, LocalLLMSettings

settings = LocalLLMSettings.from_env(
    system_prompt="You are a concise assistant."
)
model = LocalChatModel(settings)

messages = [
    {"role": "system", "content": settings.system_prompt},
    {"role": "user", "content": "Say hello in one sentence."},
]

text = model.generate_text(messages, stream=False)

print(text)
```

This example depends on `OMNIDEX_LLAMA_MODEL_PATH` already being set to a valid
local GGUF file.
