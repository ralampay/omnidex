"""Runtime helpers for local OmniDex agents."""

from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import re
import struct
from tempfile import TemporaryFile
from typing import Iterable

from llama_cpp import Llama


def env_flag(name: str, default: bool) -> bool:
    """Parse a boolean-ish environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


def env_int(name: str, default: int | None = None) -> int | None:
    """Parse an optional integer environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def env_float(name: str, default: float | None = None) -> float | None:
    """Parse an optional float environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def env_str(name: str, default: str) -> str:
    """Parse a string environment variable with trimming."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip()
    return normalized or default


def available_cpu_count() -> int | None:
    """Return the CPU count available to this process."""
    try:
        affinity = os.sched_getaffinity(0)
    except AttributeError:
        affinity = None
    if affinity:
        return max(1, len(affinity))
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return None
    return max(1, cpu_count)


def physical_cpu_count() -> int | None:
    """Best-effort physical core count for Linux hosts."""
    topology_root = Path("/sys/devices/system/cpu")
    if topology_root.exists():
        core_keys: set[tuple[str, str]] = set()
        for cpu_dir in topology_root.glob("cpu[0-9]*"):
            package_file = cpu_dir / "topology" / "physical_package_id"
            core_file = cpu_dir / "topology" / "core_id"
            try:
                package_id = package_file.read_text(encoding="utf-8").strip()
                core_id = core_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if package_id and core_id:
                core_keys.add((package_id, core_id))
        if core_keys:
            return len(core_keys)
    return None


def default_thread_count(device: str = "cpu") -> int | None:
    """Dynamically choose a thread count suited to the current host."""
    available_cpus = available_cpu_count()
    if available_cpus is None:
        return None

    if device == "gpu":
        return max(1, min(8, available_cpus // 2 or 1))

    physical_cores = physical_cpu_count()
    if physical_cores is not None:
        return max(1, min(available_cpus, physical_cores))

    if available_cpus <= 4:
        return available_cpus
    if available_cpus <= 8:
        return max(1, available_cpus - 1)
    return max(1, min(12, available_cpus - 2))


def resolve_thread_count(device: str) -> int | None:
    """Resolve thread count from env, using dynamic selection only when unset."""
    raw_value = os.getenv("OMNIDEX_LLAMA_THREADS")
    if raw_value is None or not raw_value.strip():
        return default_thread_count(device)

    return int(raw_value.strip())


@dataclass(slots=True)
class LocalLLMSettings:
    """Settings for the local GGUF-backed runtime."""

    model_path: Path
    system_prompt: str
    device: str = "cpu"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    ctx_size: int = 8192
    threads: int | None = None
    gpu_layers: int = 0
    verbose: bool = False
    render_markdown: bool = True
    stream: bool = False

    @classmethod
    def from_env(cls, system_prompt: str) -> "LocalLLMSettings":
        """Build runtime settings from the OmniDex environment."""
        model_path = os.getenv("OMNIDEX_LLAMA_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "OMNIDEX_LLAMA_MODEL_PATH is required. Copy .env.dist to .env and "
                "set it to an absolute GGUF model path."
            )

        device = env_str("OMNIDEX_DEVICE", "cpu").casefold()
        if device not in {"cpu", "gpu"}:
            raise ValueError(
                "OMNIDEX_DEVICE must be either 'cpu' or 'gpu'."
            )

        return cls(
            model_path=Path(model_path).expanduser(),
            system_prompt=system_prompt,
            device=device,
            temperature=env_float("OMNIDEX_LLAMA_TEMPERATURE", 0.2) or 0.2,
            top_p=env_float("OMNIDEX_LLAMA_TOP_P", 0.95) or 0.95,
            max_tokens=env_int("OMNIDEX_LLAMA_MAX_TOKENS", 512) or 512,
            ctx_size=env_int("OMNIDEX_LLAMA_CTX", 8192) or 8192,
            threads=resolve_thread_count(device),
            gpu_layers=resolve_gpu_layers(device),
            verbose=env_flag("OMNIDEX_LLAMA_VERBOSE", False),
            render_markdown=env_flag("OMNIDEX_RENDER_MARKDOWN", True),
            stream=env_flag("OMNIDEX_STREAM", False),
        )


def resolve_gpu_layers(device: str) -> int:
    """Resolve llama.cpp GPU layer offload from device-oriented config."""
    configured_layers = env_int("OMNIDEX_LLAMA_GPU_LAYERS")
    if configured_layers is not None:
        return configured_layers
    if device == "gpu":
        return -1
    return 0


def installed_llama_cpp_version() -> str | None:
    """Return the installed llama-cpp-python version when available."""
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        return None


def read_gguf_architecture(model_path: Path) -> str | None:
    """Read the GGUF ``general.architecture`` metadata when present."""

    def read_exact(handle, size: int) -> bytes:
        chunk = handle.read(size)
        if len(chunk) != size:
            raise ValueError("Unexpected EOF while reading GGUF metadata.")
        return chunk

    def read_u32(handle) -> int:
        return struct.unpack("<I", read_exact(handle, 4))[0]

    def read_u64(handle) -> int:
        return struct.unpack("<Q", read_exact(handle, 8))[0]

    def read_string(handle) -> str:
        size = read_u64(handle)
        return read_exact(handle, size).decode("utf-8")

    def skip_value(handle, value_type: int) -> None:
        fixed_sizes = {
            0: 1,   # uint8
            1: 1,   # int8
            2: 2,   # uint16
            3: 2,   # int16
            4: 4,   # uint32
            5: 4,   # int32
            6: 4,   # float32
            7: 1,   # bool
            10: 8,  # uint64
            11: 8,  # int64
            12: 8,  # float64
        }
        if value_type == 8:  # string
            read_string(handle)
            return
        if value_type == 9:  # array
            element_type = read_u32(handle)
            count = read_u64(handle)
            for _ in range(count):
                skip_value(handle, element_type)
            return
        size = fixed_sizes.get(value_type)
        if size is None:
            raise ValueError(f"Unsupported GGUF metadata value type: {value_type}")
        read_exact(handle, size)

    with model_path.open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            return None

        version = read_u32(handle)
        if version not in {2, 3}:
            return None

        _tensor_count = read_u64(handle)
        metadata_count = read_u64(handle)

        for _ in range(metadata_count):
            key = read_string(handle)
            value_type = read_u32(handle)
            if key == "general.architecture" and value_type == 8:
                return read_string(handle)
            skip_value(handle, value_type)

    return None


class LocalChatModel:
    """Thin wrapper around llama-cpp chat completion."""

    def __init__(self, settings: LocalLLMSettings):
        self.settings = settings
        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"Configured model does not exist: {settings.model_path}"
            )
        self._preflight_model_compatibility()
        self._llm = self._load_llm()

    def complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        stream: bool | None = None,
    ):
        """Create a chat completion with the configured local model."""
        should_stream = self.settings.stream if stream is None else stream
        return self._llm.create_chat_completion(
            messages=list(messages),
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            max_tokens=self.settings.max_tokens,
            stream=should_stream,
        )

    def generate_text(
        self,
        messages: Iterable[dict[str, str]],
        *,
        stream: bool | None = None,
    ) -> str:
        """Return normalized text content from a chat completion."""
        completion = self.complete(messages, stream=stream)
        should_stream = self.settings.stream if stream is None else stream
        if should_stream:
            return self._collect_stream_text(completion)
        return self._extract_message_content(completion)

    def _extract_message_content(self, completion: dict) -> str:
        """Extract assistant text from a non-streaming llama.cpp response."""
        return completion["choices"][0]["message"]["content"].strip()

    def _collect_stream_text(self, events: Iterable[dict]) -> str:
        """Collect streamed llama.cpp delta events into one final string."""
        chunks: list[str] = []
        for event in events:
            delta = event["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                chunks.append(piece)
        return "".join(chunks).strip()

    def _load_llm(self) -> Llama:
        """Initialize llama.cpp and preserve loader diagnostics on failure."""
        if self.settings.verbose:
            return self._build_llm()

        with TemporaryFile(mode="w+") as sink, redirect_stderr(sink):
            try:
                return self._build_llm()
            except Exception as exc:
                sink.seek(0)
                diagnostics = sink.read().strip()
                raise self._model_load_error(exc, diagnostics) from exc

    def _build_llm(self) -> Llama:
        """Construct the llama.cpp model instance."""
        return Llama(
            model_path=str(self.settings.model_path),
            n_ctx=self.settings.ctx_size,
            n_threads=self.settings.threads,
            n_gpu_layers=self.settings.gpu_layers,
            verbose=self.settings.verbose,
        )

    def _model_load_error(self, exc: Exception, diagnostics: str) -> ValueError:
        """Convert llama.cpp initialization failures into actionable errors."""
        architecture_match = re.search(
            r"unknown model architecture: '([^']+)'",
            diagnostics,
        )
        if architecture_match:
            architecture = architecture_match.group(1)
            return ValueError(
                "Failed to load GGUF model "
                f"{self.settings.model_path}: llama.cpp reported unsupported "
                f"model architecture '{architecture}' in the installed "
                "llama-cpp-python build. Upgrade llama-cpp-python to a build "
                "that supports this architecture or switch to a supported GGUF."
            )

        if diagnostics:
            return ValueError(
                "Failed to load GGUF model "
                f"{self.settings.model_path}. llama.cpp diagnostics:\n"
                f"{diagnostics}"
            )

        return ValueError(
            f"Failed to load GGUF model {self.settings.model_path}: {exc}"
        )

    def _preflight_model_compatibility(self) -> None:
        """Reject known-incompatible GGUF/runtime combinations before loading."""
        architecture = read_gguf_architecture(self.settings.model_path)
        runtime_version = installed_llama_cpp_version()
        if architecture == "gemma4" and runtime_version == "0.3.19":
            raise ValueError(
                "Failed to load GGUF model "
                f"{self.settings.model_path}: GGUF architecture 'gemma4' is not "
                f"supported by the installed llama-cpp-python {runtime_version} "
                "in this environment. Upgrade llama-cpp-python / llama.cpp or "
                "switch to a supported GGUF."
            )
