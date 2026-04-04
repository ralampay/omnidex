"""Chat agent package."""

__all__ = ["ChatAgent"]


def __getattr__(name: str):
    """Lazily expose the chat agent class without import cycles."""
    if name == "ChatAgent":
        from .agent import ChatAgent

        return ChatAgent
    raise AttributeError(name)
