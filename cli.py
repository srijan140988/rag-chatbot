"""python cli.py ingest data/sample/*   |   python cli.py ask "question"   |   python cli.py sources"""
import argparse

from rag.pipeline import RAGPipeline
from rag.store import list_sources


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sources")
    ing = sub.add_parser("ingest")
    ing.add_argument("paths", nargs="+")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    args = p.parse_args()

    rag = RAGPipeline()
    if args.cmd == "ingest":
        for path in args.paths:
            print(rag.ingest_file(path))
    elif args.cmd == "sources":
        print(list_sources(rag.store))
    else:
        a = rag.ask(args.question)
        print(a.answer, f"\n[refused: {a.reason}]" if a.refused else "")
        for c in a.citations:
            print(f"  [{c.n}] {c.label}  (sim={c.score})\n      {c.text[:160]!r}")


if __name__ == "__main__":
    main()
