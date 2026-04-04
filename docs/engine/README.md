# Engine

This directory documents the runtime and inference components that power local
OmniDex execution.

## Table Of Contents

- [Inference](./inference.md): Documents how local inference works, including
  the roles of `LocalLLMSettings` and `LocalChatModel`.
- [Orchestration Routing](./orchestration_routing.md): Documents the
  orchestrator, direct fast paths, planner-based execution, and the current
  engine-level flow.
- [Planner](./planner.md): Documents the shared planner infrastructure, its
  contracts, and a minimal agent example using two tools.
- [Tool Routing](./tool_routing.md): Documents how `research_assistant`
  decides between deterministic direct flows and model-planned tool execution.
- [State Artifacts](./state_artifacts.md): Documents ephemeral session artifact
  state, including `last_artifact_content` and follow-up save behavior.
