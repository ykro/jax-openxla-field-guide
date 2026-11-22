"""The index check skips units that are not cloned and says what to clone."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_index(parent: Path) -> Path:
    repo = parent / "jax-openxla-field-guide"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "verify_series.py", repo / "scripts")
    shutil.copytree(ROOT / "diagrams", repo / "diagrams")
    return repo


def verify(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "scripts/verify_series.py"], cwd=repo,
                          capture_output=True, text=True, timeout=60, check=False)


def test_nothing_cloned_says_what_to_clone(tmp_path):
    run = verify(make_index(tmp_path))
    assert run.returncode == 1
    assert run.stdout.count("skipped, not cloned") == 5
    assert "0 checked, 5 skipped, 0 failed." in run.stdout
    assert "SOME UNITS FAILED." not in run.stdout
    assert ("git clone https://github.com/ykro/jax-for-pytorch-devs.git "
            "../jax-for-pytorch-devs") in run.stdout


def test_one_unit_cloned_passes_and_the_rest_are_skipped(tmp_path):
    repo = make_index(tmp_path)
    unit = tmp_path / "jax-for-pytorch-devs"
    (unit / "benchmarks").mkdir(parents=True)
    (unit / "diagrams").mkdir()
    (unit / "verify_speedup.py").write_text("")
    (unit / "benchmarks" / "latest_results.json").write_text('{"all_passed": true}')
    (unit / "benchmarks" / "terminal_output.txt").write_text("")
    for i in range(3):
        for ext in ("html", "svg", "png"):
            (unit / "diagrams" / f"0{i}.{ext}").write_text("")
    run = verify(repo)
    assert run.returncode == 0, run.stdout
    assert "jax-for-pytorch-devs: judge, results, log and 3 diagrams | PASS" in run.stdout
    assert run.stdout.count("skipped, not cloned") == 4
    assert "1 checked, 4 skipped, 0 failed." in run.stdout
    assert "ALL CHECKED UNITS PASSED." in run.stdout


def test_incomplete_unit_fails(tmp_path):
    repo = make_index(tmp_path)
    (tmp_path / "jax-for-pytorch-devs").mkdir()
    run = verify(repo)
    assert run.returncode == 1
    assert "1 checked, 4 skipped, 1 failed." in run.stdout
    assert "jax-for-pytorch-devs: missing judge, results, log, diagrams" in run.stdout
    assert "SOME UNITS FAILED." in run.stdout
