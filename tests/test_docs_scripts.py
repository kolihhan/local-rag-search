from pathlib import Path


def test_readme_is_portfolio_first_and_reports_eval_scope_honestly():
    root = Path(__file__).parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    headings = [
        "## Key result",
        "## Architecture",
        "## What the system exposes",
        "## Quickstart",
        "## Examples",
        "## Evaluation",
        "## Design decisions",
        "## Limitations",
    ]
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions)

    # Resume-facing frozen comparison stays explicit about result and scope.
    assert "64 official Confluence-compatible queries" in text
    assert "5,189 documents" in text
    assert "0.7630" in text and "0.7418" in text
    assert "not the full EnterpriseRAG-Bench leaderboard benchmark" in text

    # The deterministic demo remains clearly separated from research evidence.
    assert "83.3%" in text and "77.4%" in text
    assert "6-query" in text.lower()
    assert "product sanity check" in text.lower()
    assert "not research evidence" in text.lower()

    # Keep provenance transparent.
    assert "inspired by qmd" in text.lower()
    assert "clone" in text.lower()


def test_windows_launchers_use_local_temp_and_no_global_policy_change():
    root = Path(__file__).parents[1]
    for name in ("run-demo.ps1", "run-api.ps1"):
        text = (root / name).read_text(encoding="utf-8")
        assert ".run" in text and "TEMP" in text and "TMP" in text
        assert "Set-ExecutionPolicy" not in text
        assert "$Pid" not in text and "$PID" not in text
    for name in ("run-demo.cmd", "run-api.cmd"):
        text = (root / name).read_text(encoding="utf-8")
        assert "ExecutionPolicy Bypass" in text
