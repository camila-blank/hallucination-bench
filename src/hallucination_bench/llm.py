"""One structured-JSON judge client with two routes, shared by every stage of a process.

Ported from WorkspaceBench ``src/wsbench/llm.py`` @ 92d763e (MIT), with these changes:

* **Keyless Anthropic auth.** ``claude-*`` models go to the Anthropic SDK. When
  ``ANTHROPIC_API_KEY`` is unset, :func:`api_key` returns ``None`` (the source raised) and the
  client is built as ``AsyncAnthropic()`` with no ``api_key``, so the SDK picks up the ``ant``
  CLI OAuth profile. Everything else goes to OpenRouter (``OPENROUTER_API_KEY``).
* **No temperature on the Anthropic route** (newer Claude models reject it), as in the source.
* **Thinking.** Claude Opus 5.5 rejects ``thinking: {type: "disabled"}`` (400), so the request
  omits ``thinking`` and sets ``output_config.effort`` (default ``"low"``, the closest available
  setting to "thinking off"). Structured output uses ``output_config.format`` (json_schema),
  exactly as the source.
* **One client per route per process**, one asyncio semaphore (``concurrency``) and the
  process-wide RPM pacer: several arms and stages judged concurrently never multiply the rate.
* **Per-call usage** (tokens, USD) is returned with each result so callers can store it in the
  cache and tally spend per stage.

Transient failures (408/409/429/5xx, timeouts, connection errors, garbled bodies) are retried
with jittered backoff; any other per-call failure yields ``None`` (an unjudged call, never a
clean one). Problems that would fail every call raise :class:`JudgeConfigError`.
"""

import asyncio
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

OPENROUTER = "https://openrouter.ai/api/v1"
_TRANSIENT = (408, 409, 429, 500, 502, 503, 504, 529)
_FATAL = ("AuthenticationError", "PermissionDeniedError", "NotFoundError")
_ATTEMPTS = 12
_DEFAULT_MAX_TOKENS = {"openrouter": 8000, "anthropic": 16000}
DEFAULT_EFFORT = "low"

# USD per million tokens: (input, output, cache write 5m, cache read). First-party list prices.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 2.5, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.10),
}

Route = Literal["anthropic", "openrouter"]


class JudgeConfigError(RuntimeError):
    """A configuration problem that would fail every judge call (raised, never degraded)."""


@dataclass
class Usage:
    """Tally for one call or many. ``calls`` counts HTTP 200 responses (including garbled bodies
    that were then retried); ``errors``/``refusals`` count calls left unjudged."""

    usd: float = 0.0
    calls: int = 0
    retries: int = 0
    errors: int = 0
    refusals: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, other: "Usage | dict[str, Any] | None") -> None:
        if other is None:
            return
        d = other if isinstance(other, dict) else asdict(other)
        for k in asdict(self):
            setattr(self, k, getattr(self, k) + (d.get(k) or 0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def report(self) -> str:
        return (
            f"calls={self.calls} retries={self.retries} errors={self.errors} "
            f"refusals={self.refusals} spend=${self.usd:.4f} in_tok={self.input_tokens} "
            f"out_tok={self.output_tokens} cache_w={self.cache_write_tokens} "
            f"cache_r={self.cache_read_tokens}"
        )


def route(model: str) -> Route:
    return "anthropic" if model.startswith("claude-") else "openrouter"


def api_key(model: str) -> str | None:
    """The key for ``model``'s route. Anthropic: ``ANTHROPIC_API_KEY`` or ``None`` (keyless: the
    SDK falls back to the ``ant`` CLI profile). OpenRouter: required."""
    if route(model) == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY") or None
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key.startswith("sk-or-"):
        raise JudgeConfigError("OPENROUTER_API_KEY is missing or not an OpenRouter key (sk-or-…)")
    return key


def schema_block(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": properties,
        },
    }


def anthropic_usd(model: str, u: Usage) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    return (
        u.input_tokens * p[0]
        + u.output_tokens * p[1]
        + u.cache_write_tokens * p[2]
        + u.cache_read_tokens * p[3]
    ) / 1e6


# ---------------------------------------------------------------- pacing / backoff seams


def _now() -> float:
    return time.monotonic()


class _Pacer:
    """Process-wide next-slot time behind a lock: shared by every loop, thread and stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0

    def reset(self) -> None:
        with self._lock:
            self._next = 0.0

    def reserve(self, rpm: float) -> float:
        """Reserve the next slot; return how long the caller must wait for it (seconds)."""
        with self._lock:
            now = _now()
            slot = max(now, self._next)
            self._next = slot + 60.0 / rpm
            return slot - now


_PACER = _Pacer()

_sleep = asyncio.sleep  # seam: tests patch this instead of the global asyncio.sleep


async def _pace(rpm: float) -> None:
    delay = _PACER.reserve(rpm)
    if delay > 0:
        await _sleep(delay)


def _backoff(attempt: int, status: int | None) -> float:
    cap = 90.0 if status == 429 else 30.0
    return min(cap, 2.0 * 2**attempt) * (0.5 + random.random())


def _make_client(route_: str, key: str | None) -> Any:
    """The ONLY place SDK clients are built (tests monkeypatch this to inject fakes)."""
    if route_ == "anthropic":
        from anthropic import AsyncAnthropic

        if key is None:  # keyless: the SDK resolves the `ant` CLI OAuth profile
            return AsyncAnthropic(max_retries=0)
        return AsyncAnthropic(api_key=key, max_retries=0)
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=key, base_url=OPENROUTER, max_retries=0)


# ---------------------------------------------------------------- one call


def _parse_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    data = json.loads(content)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("not a JSON object", content, 0)
    return data


async def _openrouter_once(
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    max_tokens: int,
    timeout: float,
    usage: Usage,
) -> dict[str, Any] | None:
    resp = await client.chat.completions.create(
        timeout=timeout,
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": schema},
        extra_body={"usage": {"include": True}, "provider": {"require_parameters": True}},
    )
    usage.calls += 1
    if not getattr(resp, "choices", None):  # an error delivered in a 200 body
        raise ValueError("response has no choices")
    u = getattr(resp, "usage", None)
    cost = getattr(u, "cost", None) if u is not None else None
    if cost is None and u is not None:
        cost = (getattr(u, "model_extra", None) or {}).get("cost")
    usage.usd += float(cost or 0.0)
    if u is not None:
        usage.input_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
        usage.output_tokens += int(getattr(u, "completion_tokens", 0) or 0)
    return _parse_object(resp.choices[0].message.content or "")


async def _anthropic_once(
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    effort: str | None,
    max_tokens: int,
    timeout: float,
    usage: Usage,
) -> dict[str, Any] | None:
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema["schema"]}}
    if effort is not None:
        output_config["effort"] = effort
    resp = await client.messages.create(
        timeout=timeout,
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config=output_config,
    )
    usage.calls += 1
    u = getattr(resp, "usage", None)
    if u is not None:
        call = Usage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        )
        call.usd = anthropic_usd(model, call)
        usage.add(call)
    stop = getattr(resp, "stop_reason", None)
    if stop == "refusal":
        usage.refusals += 1
        print(f"  llm refusal: {model}", flush=True)
        return None
    if stop in ("max_tokens", "model_context_window_exceeded"):
        usage.errors += 1
        print(f"  llm error: {model} stop_reason={stop}", flush=True)
        return None
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if text is None:
        raise ValueError("response has no text block")
    return _parse_object(text)


@dataclass
class CallResult:
    result: dict[str, Any] | None
    usage: Usage = field(default_factory=Usage)


class LLM:
    """The process's judge client: lazily built SDK clients (one per route), one semaphore over
    every call, the shared RPM pacer, and a preflight per model."""

    def __init__(
        self,
        *,
        rpm: float = 1500.0,
        concurrency: int = 64,
        effort: str | None = DEFAULT_EFFORT,
        timeout: float = 300.0,
        max_tokens: int | None = None,
    ) -> None:
        if concurrency < 1 or rpm <= 0:
            raise JudgeConfigError(
                f"concurrency must be >= 1 and rpm > 0 (got {concurrency}, {rpm})"
            )
        self.rpm = rpm
        self.effort = effort
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(concurrency)
        self._clients: dict[str, Any] = {}
        self._preflighted: set[str] = set()
        self._pf_lock = asyncio.Lock()

    def _client(self, model: str) -> Any:
        r = route(model)
        if r not in self._clients:
            self._clients[r] = _make_client(r, api_key(model))
        return self._clients[r]

    async def call(
        self, *, model: str, system: str, user: str, schema: dict[str, Any]
    ) -> CallResult:
        """One structured call -> parsed object (or ``None``) plus its usage. Raises
        :class:`JudgeConfigError` on a fatal configuration error."""
        r = route(model)
        client = self._client(model)
        max_tokens = self.max_tokens or _DEFAULT_MAX_TOKENS[r]
        # detokenised lens readouts can contain lone surrogates, which both APIs reject every time
        system = system.encode("utf-8", "replace").decode("utf-8")
        user = user.encode("utf-8", "replace").decode("utf-8")
        usage = Usage()
        async with self._sem:
            for attempt in range(_ATTEMPTS):
                await _pace(self.rpm)
                try:
                    if r == "anthropic":
                        res = await _anthropic_once(
                            client,
                            system,
                            user,
                            schema=schema,
                            model=model,
                            effort=self.effort,
                            max_tokens=max_tokens,
                            timeout=self.timeout,
                            usage=usage,
                        )
                    else:
                        res = await _openrouter_once(
                            client,
                            system,
                            user,
                            schema=schema,
                            model=model,
                            max_tokens=max_tokens,
                            timeout=self.timeout,
                            usage=usage,
                        )
                    return CallResult(res, usage)
                except Exception as e:
                    name = type(e).__name__
                    status = getattr(e, "status_code", None)
                    if name in _FATAL or (status in (400, 401, 402, 403) and _is_config_400(e)):
                        raise JudgeConfigError(f"{name}: {str(e)[:300]}") from e
                    transient = (
                        status in _TRANSIENT
                        or "Timeout" in name
                        or "Connection" in name
                        or "Deadline" in name
                        or isinstance(e, json.JSONDecodeError | ValueError)
                    )
                    if transient and attempt < _ATTEMPTS - 1:
                        usage.retries += 1
                        await _sleep(_backoff(attempt, status))
                        continue
                    usage.errors += 1
                    print(f"  llm error: {name}: {str(e)[:200]}", flush=True)
                    return CallResult(None, usage)
        return CallResult(None, usage)

    async def preflight(self, model: str) -> None:
        """One tiny structured call per model, so a bad key/auth/model/structured-output setup
        fails loudly before a long run starts."""
        async with self._pf_lock:
            if model in self._preflighted:
                return
            schema = schema_block("t", {"a": {"type": "integer"}}, ["a"])
            out = await self.call(
                model=model, system="Reply JSON.", user='Return {"a":1}', schema=schema
            )
            if out.result != {"a": 1}:
                raise JudgeConfigError(f"preflight failed for {model}: {out.result}")
            self._preflighted.add(model)

    async def aclose(self) -> None:
        for c in self._clients.values():
            close = getattr(c, "close", None)
            if close is not None:
                await close()
        self._clients.clear()


def _is_config_400(e: Exception) -> bool:
    """401/402/403 are always fatal (bad key, no credits, no access). A 400 is fatal only when it
    rejects a request parameter (it would fail every call), not a single prompt's content."""
    status = getattr(e, "status_code", None)
    if status != 400:
        return True
    msg = str(e).lower()
    return any(s in msg for s in ("not supported", "unsupported", "extra inputs", "schema"))
