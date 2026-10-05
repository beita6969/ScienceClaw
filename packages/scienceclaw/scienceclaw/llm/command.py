"""A chat model backed by a local command.

``backend: command`` runs a program once per completion: the conversation is written to its standard input and the text it
prints is the answer. This connects any command-line model front end (a local runtime, a vendor CLI, a wrapper script)
without HTTP credentials. The command line comes from ``llm.command`` or ``$SCIENCECLAW_LLM_COMMAND`` and is split like a shell
line; the placeholder ``{system}`` is replaced by the system prompt as one argument (without it the system prompt is placed at
the top of standard input) and ``{model}`` by the role's model name. Caching, usage accounting, role settings and concurrency
are those of :class:`~scienceclaw.llm.client.LLMClient`; token counts are estimated from text length.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import replace
from typing import Any

from scienceclaw.config import LLMConfig, ModelRole
from scienceclaw.llm.client import LLMClient, LLMError, empty_usage

JSON_HINT = "Reply with a single JSON object and nothing else."
CHARS_PER_TOKEN = 4


def render_transcript(messages: list[dict]) -> str:
    """The non-system turns as text: a single user turn verbatim, otherwise role-labelled blocks."""
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1 and turns[0].get("role") == "user":
        return str(turns[0]["content"])
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in turns)


class CommandChatModel(LLMClient):
    """``ChatModel`` whose completions are produced by an external command."""

    def __init__(self, cfg: LLMConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        raw = (getattr(cfg, "command", "") or os.environ.get("SCIENCECLAW_LLM_COMMAND", "")).strip()
        if not raw:
            raise LLMError("backend 'command' needs llm.command or SCIENCECLAW_LLM_COMMAND")
        self._argv = shlex.split(raw)

    def role_config(self, role: str) -> ModelRole:
        try:
            return super().role_config(role)
        except LLMError:
            return replace(getattr(self.cfg, role), model="command")

    def _complete(self, body: dict, role: str) -> tuple[str, dict, str | None, dict]:
        messages = body["messages"]
        system = "\n\n".join(str(m["content"]) for m in messages if m.get("role") == "system")
        if "response_format" in body:
            system = f"{system}\n\n{JSON_HINT}".strip()
        prompt = render_transcript(messages)
        inline = "{system}" in self._argv
        argv = [system if a == "{system}" else a.replace("{model}", str(body.get("model", ""))) for a in self._argv]
        stdin = prompt if inline else f"{system}\n\n{prompt}".strip()

        last = ""
        retries = max(0, min(int(self.cfg.max_retries), 3))
        for attempt in range(retries + 1):
            try:
                done = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=float(self.cfg.timeout_s))
            except subprocess.TimeoutExpired:
                last = f"command timed out after {self.cfg.timeout_s:.0f} s"
            except OSError as ex:
                raise LLMError(f"cannot run the model command {argv[0]!r}: {ex}") from ex
            else:
                text = done.stdout.strip()
                if done.returncode == 0 and text:
                    usage = empty_usage()
                    usage.update(calls=1, prompt_tokens=(len(stdin) + len(system if inline else "")) // CHARS_PER_TOKEN,
                                 completion_tokens=max(1, len(text) // CHARS_PER_TOKEN))
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    return text, usage, "stop", {"retries": attempt, "usage_estimated": 1}
                last = f"exit status {done.returncode}: {(done.stderr or done.stdout).strip()[:300]}"
            if attempt < retries:
                self._sleep(self._backoff_base * (2 ** attempt))
        raise LLMError(f"model command failed after {retries + 1} attempt(s): {last}")
