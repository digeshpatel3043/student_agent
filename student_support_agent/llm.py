"""Groq client adapter.

Groq's API is OpenAI-compatible (chat completions + function calling). The agent loop in agent.py
speaks a small neutral format (content blocks: text / tool_use / tool_result); this adapter
translates that format to/from the OpenAI wire format, so the agent logic is provider-independent.

Groq-specific handling:
  * the model sometimes emits a malformed tool call -> HTTP 400 "tool_use_failed": retried up to 2 times
  * rate limits (429) are retried with back-off by the SDK
  * tool arguments may arrive as "", "null" or non-objects: normalised to {}
"""
import json
import os
from types import SimpleNamespace as NS

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_api_key():
    return os.environ.get("GROQ_API_KEY")


def to_openai_tools(tool_schemas):
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tool_schemas
    ]


def to_openai_messages(system, messages):
    out = [{"role": "system", "content": system}]
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
        elif m["role"] == "assistant":
            text = "".join(b.text for b in c if b.type == "text")
            calls = [{"id": b.id, "type": "function",
                      "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                     for b in c if b.type == "tool_use"]
            msg = {"role": "assistant", "content": text or ""}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn made of tool results
            for r in c:
                out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": r["content"]})
    return out


def _args(raw):
    try:
        val = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return val if isinstance(val, dict) else {}


def from_openai_response(resp):
    msg = resp.choices[0].message
    blocks = []
    if msg.content:
        blocks.append(NS(type="text", text=msg.content))
    for tc in (msg.tool_calls or []):
        blocks.append(NS(type="tool_use", id=tc.id, name=tc.function.name, input=_args(tc.function.arguments)))
    has_tools = any(b.type == "tool_use" for b in blocks)
    u = resp.usage
    return NS(
        content=blocks,
        stop_reason="tool_use" if has_tools else "end_turn",
        usage=NS(input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                 output_tokens=getattr(u, "completion_tokens", 0) or 0),
    )


class GroqClient:
    """Exposes .messages.create(...) so the agent loop stays provider-neutral."""

    def __init__(self, api_key=None, base_url=None, sdk=None):
        if sdk is None:
            from openai import OpenAI  # imported lazily so the offline self-test needs no SDK
            sdk = OpenAI(api_key=api_key, base_url=base_url or os.environ.get("GROQ_BASE_URL", GROQ_BASE_URL),
                         timeout=60.0, max_retries=2)
        self.sdk = sdk
        self.messages = self

    def create(self, model, system, messages, tools, max_tokens=1200, temperature=None, **_):
        kw = dict(model=model, messages=to_openai_messages(system, messages),
                  tools=to_openai_tools(tools), max_tokens=max_tokens)
        if temperature is not None:
            kw["temperature"] = temperature
        last = None
        for attempt in range(3):
            try:
                return from_openai_response(self.sdk.chat.completions.create(**kw))
            except Exception as e:
                if getattr(e, "status_code", None) == 400 and "tool_use_failed" in str(e) and attempt < 2:
                    last = e
                    kw["temperature"] = 0.3  # tiny variation so the retry is not an identical failure
                    continue
                raise
        raise last

    def list_models(self):
        return sorted(m.id for m in self.sdk.models.list().data)
