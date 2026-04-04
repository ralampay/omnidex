"""Research assistant agent package."""

__all__ = ["ResearchAssistant"]


def __getattr__(name: str):
    """Lazily expose the research assistant class without import cycles."""
    if name == "ResearchAssistant":
        from .agent import ResearchAssistant

        return ResearchAssistant
    raise AttributeError(name)
