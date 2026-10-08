import json
from pathlib import Path

from evaluation.agreement_gate import diagnose

ROOT = Path(__file__).parents[1]


def test_agreement_gate_dev_to_core_diagnostic():
    result = diagnose(
        ROOT / "runs" / "enterprise-rag-qwen-dev-v2" / "report.json",
        ROOT / "runs" / "enterprise-rag-qwen-core-v1" / "report.json",
    )
    assert False, "AGREEMENT_GATE_DIAGNOSTIC=" + json.dumps(result, sort_keys=True)
