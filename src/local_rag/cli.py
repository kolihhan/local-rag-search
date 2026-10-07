from __future__ import annotations

import argparse
from pathlib import Path

from .index import build_search_service
from .query import QueryPlan


def _print_results(results, *, mode: str, explain: bool) -> None:
    print(f"mode={mode}")
    for index, row in enumerate(results, start=1):
        print(f"{index}. {row.doc_id}  score={row.score:.4f}  {row.title}")
        print(f"   {row.snippet}")
        if explain:
            print(f"   BM25={row.bm25_rank} Dense={row.dense_rank} RRF={row.rrf_rank} Rerank={row.reranker_score} signals={','.join(row.matched_signals)}")
            if row.context:
                print(f"   Context: {row.context}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-search")
    parser.add_argument("--corpus", default="demo_docs")
    parser.add_argument("--cache-dir", default=".rag-cache")
    parser.add_argument("--embedding", choices=("simple", "ollama"), default="simple")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index")
    for name in ("search", "vsearch"):
        cmd = sub.add_parser(name)
        cmd.add_argument("query")
        cmd.add_argument("--limit", type=int, default=10)
        cmd.add_argument("--rerank", action="store_true")
        cmd.add_argument("--explain", action="store_true")

    query = sub.add_parser("query")
    query.add_argument("query")
    query.add_argument("--intent", default="")
    query.add_argument("--lex", action="append", default=[])
    query.add_argument("--vec", action="append", default=[])
    query.add_argument("--hyde", action="append", default=[])
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--rerank", action="store_true")
    query.add_argument("--explain", action="store_true")

    get = sub.add_parser("get")
    get.add_argument("doc_id")
    args = parser.parse_args(argv)

    service = build_search_service(args.corpus, cache_dir=args.cache_dir, embedding=args.embedding)
    if args.command == "index":
        print(f"indexed {len(service.documents)} documents")
        return 0
    if args.command == "get":
        doc = service.get(args.doc_id)
        print(f"# {doc.title}\n\n{doc.text}")
        return 0
    if args.command == "vsearch":
        results = service.search(args.query, mode="vec", limit=args.limit, rerank=args.rerank)
        _print_results(results, mode="vec", explain=args.explain)
        return 0
    if args.command == "search":
        results = service.search(args.query, mode="lex", limit=args.limit, rerank=args.rerank)
        _print_results(results, mode="lex", explain=args.explain)
        return 0

    if args.lex or args.vec or args.hyde:
        plan = QueryPlan(args.query, args.intent, tuple(args.lex), tuple(args.vec), tuple(args.hyde))
    else:
        plan = QueryPlan.hybrid(args.query)
    results = service.query(plan, limit=args.limit, rerank=args.rerank)
    _print_results(results, mode="typed/hybrid", explain=args.explain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
