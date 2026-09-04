from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    original: str
    intent: str = ""
    lex: tuple[str, ...] = ()
    vec: tuple[str, ...] = ()
    hyde: tuple[str, ...] = ()

    @classmethod
    def hybrid(cls, query: str) -> "QueryPlan":
        return cls(original=query, lex=(query,), vec=(query,))

    @classmethod
    def lexical(cls, query: str) -> "QueryPlan":
        return cls(original=query, lex=(query,))

    @classmethod
    def vector(cls, query: str) -> "QueryPlan":
        return cls(original=query, vec=(query,))
