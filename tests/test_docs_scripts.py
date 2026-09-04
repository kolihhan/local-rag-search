from pathlib import Path


def test_readme_is_portfolio_first_and_reports_tiny_eval_honestly():
    root = Path(__file__).parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    headings = ["## The problem", "## How it works", "## Quickstart", "## Example", "## Results", "## Design decisions", "## Limitations"]
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions)
    assert "83.3%" in text and "77.4%" in text
    assert "six-query" in text.lower() or "6-query" in text.lower()
    assert "research benchmark" in text.lower()
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
