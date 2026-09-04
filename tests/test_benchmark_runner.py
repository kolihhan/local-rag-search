import json

from evaluation.evaluate import BenchmarkReport, FrozenDevSet, ArmMetrics, PerQueryMetrics, RankTransition, decide_verdict, load_frozen_dev_set, run_benchmark


APPROVED = [
    "qst_0004", "qst_0009", "qst_0016", "qst_0019", "qst_0023", "qst_0024",
    "qst_0036", "qst_0037", "qst_0039", "qst_0045", "qst_0057", "qst_0081",
]


def test_loader_requires_exact_approved_order_and_nonempty_qrels(tmp_path):
    manifest = tmp_path / "manifest.json"
    questions = tmp_path / "questions.jsonl"
    manifest.write_text(json.dumps({"selected_question_ids": APPROVED}), encoding="utf-8")
    questions.write_text("\n".join(json.dumps({"question_id": qid, "question": qid, "expected_doc_ids": [qid]}) for qid in APPROVED), encoding="utf-8")

    dataset = load_frozen_dev_set(manifest_path=manifest, questions_path=questions)
    assert dataset.question_ids == tuple(APPROVED)
    assert list(dataset.questions) == APPROVED

    manifest.write_text(json.dumps({"selected_question_ids": APPROVED[:-1]}), encoding="utf-8")
    try:
        load_frozen_dev_set(manifest_path=manifest, questions_path=questions)
    except ValueError:
        pass
    else:
        raise AssertionError("loader accepted a non-approved selection")

    manifest.write_text(json.dumps({"selected_question_ids": APPROVED}), encoding="utf-8")
    with __import__("pytest").raises(ValueError, match="absent from corpus"):
        load_frozen_dev_set(manifest_path=manifest, questions_path=questions, corpus_doc_ids=set())


def _dataset(*, count=2):
    qids = tuple(f"qst_{index + 4:04d}" for index in range(count))
    return FrozenDevSet(
        question_ids=qids,
        questions={qid: f"question {index}" for index, qid in enumerate(qids)},
        qrels={qid: frozenset({f"doc-{index}"}) for index, qid in enumerate(qids)},
    )


def test_runner_calls_three_arms_once_and_rejects_duplicate_rank_ids():
    dataset = _dataset()
    calls = {name: 0 for name in ("bm25", "dense", "rrf")}

    def make_arm(name):
        def arm(question):
            calls[name] += 1
            index = int(question.rsplit(" ", 1)[1])
            return ["doc-0", "doc-0"] if index == 0 else ["noise", "doc-1", "doc-1"]
        return arm

    with __import__("pytest").raises(ValueError, match="unique"):
        run_benchmark(dataset=dataset, arms={name: make_arm(name) for name in calls})


def test_runner_rotates_timed_arms_without_untimed_warmup_and_reports_latency():
    dataset = _dataset(count=3)
    names = ("bm25", "dense", "rrf")
    calls = []
    durations = {"bm25": 1, "dense": 2, "rrf": 3}
    ticks = iter([value for index in range(3) for name in names[index % 3:] + names[:index % 3] for value in (0, durations[name] * 1_000_000)])

    def clock():
        return next(ticks)

    def make_arm(name):
        def arm(question):
            calls.append(name)
            return ["doc-0"]
        return arm

    report = run_benchmark(dataset=dataset, arms={name: make_arm(name) for name in names}, measure_latency=True, clock_ns=clock)
    assert calls == ["bm25", "dense", "rrf", "dense", "rrf", "bm25", "rrf", "bm25", "dense"]
    assert report.arms["bm25"].mean_latency_ms == 1.0
    assert report.arms["dense"].mean_latency_ms == 2.0
    assert report.arms["rrf"].mean_latency_ms == 3.0


def test_runner_rejects_duplicate_arm_ids_and_transitions_use_hit_at_10():
    dataset = FrozenDevSet(
        ("qst_0004",), {"qst_0004": "question"}, {"qst_0004": frozenset({"gold"})}
    )
    with __import__("pytest").raises(ValueError, match="unique"):
        run_benchmark(dataset=dataset, arms={name: (lambda question: ["gold", "gold"]) for name in ("bm25", "dense", "rrf")})

    report = run_benchmark(dataset=dataset, arms={
        "bm25": lambda question: [f"noise-{i}" for i in range(15)] + ["gold"],
        "dense": lambda question: ["gold"],
        "rrf": lambda question: [f"noise-{i}" for i in range(15)] + ["gold"],
    })
    assert report.arms["bm25"].per_query["qst_0004"].first_relevant_rank_at_20 == 16
    assert report.arms["bm25"].recall_at_10 == 0.0
    assert report.arms["bm25"].mrr_at_20 == 1.0 / 16.0
    assert report.transitions["bm25->dense"]["qst_0004"].category == "source_miss_target_hit"
    assert report.transitions["bm25->rrf"]["qst_0004"].category == "both_miss"


def _metric(recall, mrr, hit):
    value = PerQueryMetrics("qst_0004", {}, None, recall, recall, recall, mrr, hit, None)
    return ArmMetrics(recall, recall, recall, mrr, float(hit), None, None, None, {"qst_0004": value})


def test_verdict_rule_distinguishes_all_frozen_outcomes():
    def report(bm25, dense, rrf, transitions=None):
        return BenchmarkReport(("qst_0004",), {"bm25": bm25, "dense": dense, "rrf": rrf}, transitions or {
            "bm25->dense": {"qst_0004": RankTransition(None, None, "both_miss")},
            "bm25->rrf": {"qst_0004": RankTransition(None, None, "both_miss")},
        })
    assert decide_verdict(report(_metric(.5, .5, True), _metric(.5, .5, True), _metric(.5, .5, True))) == "KEEP BM25"
    assert decide_verdict(report(_metric(.5, .5, True), _metric(1, 1, True), _metric(.5, .5, True))) == "KEEP DENSE"
    assert decide_verdict(report(_metric(.5, .5, True), _metric(.5, .5, True), _metric(1, 1, True))) == "KEEP HYBRID"
    assert decide_verdict(report(_metric(.5, .5, True), _metric(1, 1, True), _metric(1, 1, True))) == "INSUFFICIENT"


def test_verdict_is_insufficient_when_bm25_does_not_weakly_dominate_conflicting_metrics():
    qid = "qst_0004"
    transitions = {
        "bm25->dense": {qid: RankTransition(1, 1, "both_hit")},
        "bm25->rrf": {qid: RankTransition(1, 1, "both_hit")},
    }
    report = BenchmarkReport(
        (qid,),
        {
            "bm25": _metric(.75, .50, True),
            "dense": _metric(.50, .75, True),
            "rrf": _metric(.50, .50, True),
        },
        transitions,
    )
    assert decide_verdict(report) == "INSUFFICIENT"


def test_verdict_ties_go_to_bm25_and_any_regression_is_insufficient():
    dataset = _dataset(count=2)
    same = lambda question: ["doc-0"]
    tied = run_benchmark(dataset=dataset, arms={name: same for name in ("bm25", "dense", "rrf")})
    assert decide_verdict(tied) == "KEEP BM25"

    regressed = run_benchmark(dataset=dataset, arms={
        "bm25": lambda question: ["doc-0"],
        "dense": lambda question: ["noise"] if question.endswith("0") else ["doc-1"],
        "rrf": lambda question: ["doc-0"],
    })
    assert decide_verdict(regressed) == "INSUFFICIENT"
