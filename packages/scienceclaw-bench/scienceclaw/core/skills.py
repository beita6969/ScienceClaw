"""Skills 𝒮_r: strategy-level instructions for decomposition, workflow construction and recovery."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Skill:
    id: str
    version: int
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    created: str = ""

    @property
    def ref(self) -> str:
        return f"skill:{self.id}"

    @property
    def version_id(self) -> str:
        """ω: globally unique id of this version."""
        return f"skill:{self.id}@v{self.version}"

    def render(self) -> str:
        tags = f" (tags: {', '.join(self.tags)})" if self.tags else ""
        return f"### {self.ref} — {self.title}{tags}\n{self.body.strip()}"

    def search_text(self) -> str:
        return " ".join([self.title, self.body, " ".join(self.tags)])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(**{k: d[k] for k in ("id", "version", "title", "body") if k in d},
                   tags=list(d.get("tags", [])), provenance=dict(d.get("provenance", {})), created=d.get("created", ""))
