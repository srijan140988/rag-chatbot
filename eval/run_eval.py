"""Evaluation harness.

  python -m eval.run_eval --mode retrieval   # cheap: embeddings only; calibrates the threshold
  python -m eval.run_eval --mode full        # end-to-end: also calls the LLM

Metrics
  Retrieval (answerable Qs):  hit@k, MRR  (a hit = right file AND chunk contains the expected fact)
  Refusal  (unanswerable Qs): correct-refusal rate  (higher = fewer hallucinations)
  Refusal  (answerable Qs):   false-refusal rate    (lower  = still useful)
  Full mode:                  answer accuracy, citation validity, citation grounding
"""
import argparse
import json
from pathlib import Path

from rag.pipeline import RAGPipeline
from rag.retriever import retrieve

EVAL_SET = Path(__file__).with_name("eval_set.json")
SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def is_relevant(hit, item) -> bool:
    return (
        hit.metadata["source"] == item["expected_source"]
        and item["expected_substring"].lower() in hit.text.lower()
    )


def retrieval_report(rag: RAGPipeline, items: list[dict]) -> dict:
    ans = [i for i in items if i["answerable"]]
    unans = [i for i in items if not i["answerable"]]
    k = rag.cfg.top_k

    ranks, top1_ans, top1_unans = [], [], []
    for it in ans:
        hits = retrieve(rag.store, it["question"], k)
        rank = next((r for r, h in enumerate(hits, 1) if is_relevant(h, it)), None)
        ranks.append(rank)
        top1_ans.append(hits[0].score if hits else 0.0)
    for it in unans:
        hits = retrieve(rag.store, it["question"], k)
        top1_unans.append(hits[0].score if hits else 0.0)

    sweep = []
    for t in SWEEP:
        sweep.append(
            {
                "threshold": t,
                "correct_refusal_rate": sum(s < t for s in top1_unans) / max(1, len(unans)),
                "false_refusal_rate": sum(s < t for s in top1_ans) / max(1, len(ans)),
            }
        )
    return {
        "k": k,
        "hit_at_k": sum(r is not None for r in ranks) / max(1, len(ans)),
        "mrr": sum(1 / r for r in ranks if r) / max(1, len(ans)),
        "top1_score_answerable_min": min(top1_ans, default=None),
        "top1_score_unanswerable_max": max(top1_unans, default=None),
        "threshold_sweep": sweep,
    }


def full_report(rag: RAGPipeline, items: list[dict]) -> dict:
    ans = [i for i in items if i["answerable"]]
    unans = [i for i in items if not i["answerable"]]
    rows = []
    for it in items:
        a = rag.ask(it["question"])
        row = {"question": it["question"], "answerable": it["answerable"], "refused": a.refused, "reason": a.reason, "answer": a.answer}
        if it["answerable"] and not a.refused:
            row["correct"] = it["answer_contains"].lower() in a.answer.lower()
            row["citations_valid"] = not a.invalid_citations
            row["cited_chunk_grounded"] = any(
                c.source == it["expected_source"] and it["expected_substring"].lower() in c.text.lower()
                for c in a.citations
            )
        rows.append(row)

    answered = [r for r in rows if r["answerable"] and not r["refused"]]
    return {
        "threshold": rag.cfg.min_relevance,
        "answer_accuracy": sum(r["correct"] for r in answered) / max(1, len(ans)),
        "false_refusal_rate": sum(r["refused"] for r in rows if r["answerable"]) / max(1, len(ans)),
        "correct_refusal_rate": sum(r["refused"] for r in rows if not r["answerable"]) / max(1, len(unans)),
        "hallucination_rate_on_unanswerable": sum(not r["refused"] for r in rows if not r["answerable"]) / max(1, len(unans)),
        "citation_validity": sum(r["citations_valid"] for r in answered) / max(1, len(answered)),
        "citation_grounding": sum(r["cited_chunk_grounded"] for r in answered) / max(1, len(answered)),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    rag = RAGPipeline()
    if args.threshold is not None:
        rag = rag.with_threshold(args.threshold)
    items = json.loads(EVAL_SET.read_text())

    if args.mode == "retrieval":
        rep = retrieval_report(rag, items)
        print(f"hit@{rep['k']}: {rep['hit_at_k']:.2%}   MRR: {rep['mrr']:.3f}")
        print(f"lowest top-1 sim on answerable Qs:   {rep['top1_score_answerable_min']}")
        print(f"highest top-1 sim on unanswerable Qs: {rep['top1_score_unanswerable_max']}")
        print("\nthreshold | correct refusal | false refusal")
        for r in rep["threshold_sweep"]:
            print(f"   {r['threshold']:.2f}   |     {r['correct_refusal_rate']:6.1%}      |    {r['false_refusal_rate']:6.1%}")
        print("\nPick the threshold with high correct-refusal and ~0% false-refusal, then set MIN_RELEVANCE.")
    else:
        rep = full_report(rag, items)
        for k, v in rep.items():
            if k != "rows":
                print(f"{k}: {v:.2%}" if isinstance(v, float) and k != "threshold" else f"{k}: {v}")
    Path(__file__).with_name("results.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
