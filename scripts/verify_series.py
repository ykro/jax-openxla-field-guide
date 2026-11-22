#!/usr/bin/env python3
"""Checks the PyTorch to JAX and OpenXLA Field Guide and the unit repos cloned next to it.

For the index it checks its three diagrams. For each unit repo cloned in the same parent folder
it checks the standard layout (judge, results JSON, terminal log, at least three diagrams, each
HTML with its SVG and PNG) and collects the judge metrics in one JSON. Units you have not cloned
are reported as skipped and do not count as a failure. It does not rerun the experiments: run
`make all` inside each unit for that.

    git clone https://github.com/ykro/jax-for-pytorch-devs.git ../jax-for-pytorch-devs
    python3 scripts/verify_series.py
"""

from __future__ import annotations

import json
import pathlib
import sys

INDEX_DIR = pathlib.Path(__file__).resolve().parents[1]
SERIES_ROOT = INDEX_DIR.parent
INDEX_SLUG = "jax-openxla-field-guide"
PROJECTS = [
    ("jax-for-pytorch-devs", "verify_speedup.py"),
    ("flax-nnx-transformer-from-scratch", "verify_transformer.py"),
    ("keras3-multibackend-arena", "verify_arena.py"),
    ("gemma-tunix-spanish-posttraining", "verify_posttraining.py"),
    ("torchtpu-playbook", "verify_torchtpu_playbook.py"),
]


def count_diagrams(repo_dir: pathlib.Path) -> dict[str, int]:
    diag_dir = repo_dir / "diagrams"
    return {ext: len(list(diag_dir.glob(f"*.{ext}"))) for ext in ("html", "svg", "png")}


def check_project(repo_dir: pathlib.Path, judge_script: str) -> dict[str, object]:
    """Layout and metrics of one cloned unit repo."""
    results_file = repo_dir / "benchmarks" / "latest_results.json"
    terminal_file = repo_dir / "benchmarks" / "terminal_output.txt"
    diagrams = count_diagrams(repo_dir)
    status: dict[str, object] = {
        "cloned": True,
        "judge_script": (repo_dir / judge_script).exists(),
        "results_file": results_file.exists(),
        "terminal_output_file": terminal_file.exists(),
        "diagrams": diagrams,
    }
    passed = (
        bool(status["judge_script"]) and bool(status["results_file"])
        and bool(status["terminal_output_file"])
        and diagrams["html"] >= 3
        and diagrams["svg"] == diagrams["html"] == diagrams["png"]
    )
    if results_file.exists():
        try:
            status["metrics"] = json.loads(results_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            status["error"] = str(exc)
            passed = False
    status["passed"] = passed
    return status


def describe(status: dict[str, object]) -> str:
    """One short line per unit: what is there, or what is missing."""
    diagrams = status["diagrams"]
    if status["passed"]:
        return f"judge, results, log and {diagrams['html']} diagrams | PASS"
    missing = [name for key, name in (("judge_script", "judge"), ("results_file", "results"),
                                      ("terminal_output_file", "log")) if not status[key]]
    if diagrams["html"] < 3 or not diagrams["svg"] == diagrams["html"] == diagrams["png"]:
        missing.append(f"diagrams {diagrams}")
    if "error" in status:
        missing.append("readable results")
    return f"missing {', '.join(missing)} | FAIL"


def main() -> int:
    index_diagrams = count_diagrams(INDEX_DIR)
    index_ok = all(n == 3 for n in index_diagrams.values())
    lines = [
        "=" * 72,
        "PyTorch to JAX and OpenXLA Field Guide · index and unit repos",
        "=" * 72,
        f"{INDEX_SLUG} (index): {index_diagrams['html']} diagrams | {'PASS' if index_ok else 'FAIL'}",
    ]
    projects: dict[str, object] = {}
    checked, failed = 0, 0
    for slug, judge_script in PROJECTS:
        repo_dir = SERIES_ROOT / slug
        if not repo_dir.is_dir():
            projects[slug] = {"cloned": False, "passed": None}
            lines.append(f"{slug}: skipped, not cloned")
            continue
        status = check_project(repo_dir, judge_script)
        projects[slug] = status
        checked += 1
        failed += not status["passed"]
        lines.append(f"{slug}: {describe(status)}")

    all_passed = index_ok and checked > 0 and failed == 0
    lines.append("-" * 72)
    lines.append(f"{checked} checked, {len(PROJECTS) - checked} skipped, {failed} failed.")
    if checked == 0:
        first = PROJECTS[0][0]
        lines.append("No unit of this path is cloned next to this repo, so there is nothing to check.")
        lines.append("Clone at least the first one:")
        lines.append(f"  git clone https://github.com/ykro/{first}.git ../{first}")
    else:
        lines.append("ALL CHECKED UNITS PASSED." if all_passed else "SOME UNITS FAILED.")
    lines.append("=" * 72)
    report = "\n".join(lines) + "\n"

    summary = {
        "index_repo": {"slug": INDEX_SLUG, "diagrams": index_diagrams, "passed": index_ok},
        "checked": checked,
        "skipped": len(PROJECTS) - checked,
        "projects": projects,
        "all_passed": all_passed,
    }
    bench_dir = INDEX_DIR / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (bench_dir / "series_summary.json").write_text(payload, encoding="utf-8")
    (bench_dir / "latest_results.json").write_text(payload, encoding="utf-8")
    (bench_dir / "terminal_output.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
