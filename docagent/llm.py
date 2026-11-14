"""Minimal client for a local model served by Ollama (POST /api/chat), plus a test double."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int = 0       # prompt tokens actually evaluated (cached prefix may be excluded)
    completion_tokens: int = 0
    latency_s: float = 0.0

    def json(self) -> dict:
        return json.loads(self.text)


def encode_image(path) -> str:
    """Ollama expects images as base64 strings attached to a message."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 num_ctx: int = 16384, seed: int = 0, timeout: float = 600.0):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.seed = seed
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, messages: list[dict], schema: dict | None = None,
             temperature: float = 0.0, max_tokens: int | None = None) -> LLMReply:
        """One non-streamed chat call.

        messages: [{"role": "system" | "user" | "assistant", "content": str,
                    "images": [path, ...] (optional)}]
        schema:   JSON schema; if given, decoding is constrained to valid instances.
        """

        # 1. Copy messages so that we don't modify the caller's data.
        prepared_messages = []

        for message in messages:
            msg = dict(message)

            # 2. Convert image paths into Base64 strings.
            images = msg.get("images")

            if images:
                msg["images"] = [encode_image(path) for path in images]
            else:
                # Ollama doesn't need an empty images field.
                msg.pop("images", None)

            prepared_messages.append(msg)

        # 3. Configure generation options.
        options = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
        }

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # 4. Build the HTTP request.
        payload = {
            "model": self.model,
            "messages": prepared_messages,
            "stream": False,
            "options": options,
        }

        # 5. Add structured output constraint if requested.
        if schema is not None:
            payload["format"] = schema

        # 6. Send the request and measure wall-clock latency.
        start = time.perf_counter()

        response = self.session.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )

        latency = time.perf_counter() - start

        # 7. Raise an exception if the HTTP request failed.
        response.raise_for_status()

        # 8. Decode Ollama's response.
        result = response.json()

        text = result["message"]["content"]

        # 9. Return a standardized response object.
        return LLMReply(
            text=text,
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            latency_s=latency,
        )


    def vram_bytes(self) -> int:
        """VRAM used by the loaded model, read from GET /api/ps (0 if it is not loaded)."""
        r = self.session.get(f"{self.host}/api/ps", timeout=10)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if self.model in (m.get("name"), m.get("model")):
                return int(m.get("size_vram", 0))
        return 0


class ScriptedLLM:
    """Test double: returns pre-written replies in order and records every call.

    Lets you test the agent loop deterministically, without a GPU.
    """

    def __init__(self, replies: list):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls: list[dict] = []

    def chat(self, messages, schema=None, temperature=0.0, max_tokens=None) -> LLMReply:
        self.calls.append({"messages": [dict(m) for m in messages], "schema": schema,
                           "temperature": temperature})
        if not self.replies:
            raise RuntimeError("ScriptedLLM: no reply left")
        text = self.replies.pop(0)
        prompt = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return LLMReply(text=text, prompt_tokens=prompt, completion_tokens=len(text) // 4)
