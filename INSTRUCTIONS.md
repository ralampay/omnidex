You are an expert python programmer that is helping build this `omnidex` software which is a python based cli program used to run various agents using gguf models and the llama-cpp-python library.

This serves as a repository for agents, tools where each agent is developed as a standalone class of sorts.

The agent's capabilities can be found in:

- docs/agents/chat_agent.md
- docs/agents/research_summarizer.md

## Coding Convention

Code heavy business logic using command patterns.

```python
class MyFunction:
    def __init__(self, param1:, param2:):
        self.param1 = param1
        self.param2 = param2

    def run(self):
        # Logic here
        pass

cmd = MyFunction(param1: "foo", param2: "bar")
cmd.run()
```

The CLI uses `rich` for spinners, styled logs, and chat thinking indicators.
