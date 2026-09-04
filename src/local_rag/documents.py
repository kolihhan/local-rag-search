from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    path: str
    collection: str
    context: str = ""


def stable_doc_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").strip("/")
    stem = normalized.rsplit(".", 1)[0].replace("/", ":")
    return stem


def load_markdown_corpus(root: str | Path, contexts: dict[str, str] | None = None) -> list[Document]:
    root = Path(root)
    contexts = contexts or {}
    docs: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        collection = rel.split("/", 1)[0] if "/" in rel else "default"
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ").replace("_", " ").title()
        docs.append(Document(stable_doc_id(rel), title, text, rel, collection, contexts.get(collection, "")))
    return docs
