"""Each snippet in practica/ runs on CPU and prints what the practice video shows."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRACTICA = ROOT / "practica"


def run(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PRACTICA / f"{name}.py")], cwd=ROOT,
                          capture_output=True, text=True, timeout=300, check=False)


def value_after(line: str, label: str) -> object:
    return ast.literal_eval(line.split(label, 1)[1].strip())


def test_lado_a_lado_same_loss_in_torch_and_jax():
    r = run("01_lado_a_lado")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 5
    assert lines[0] == "paso 0  torch 6.00242  jax 6.00242"
    assert lines[4] == "paso 4  torch 1.06494  jax 1.06494"
    for line in lines:
        parts = line.split()
        assert parts[3] == parts[5]


def test_capas_from_jaxpr_to_fused_hlo():
    r = run("02_capas")
    assert r.returncode == 0, r.stderr
    jaxpr, stablehlo, hlo = r.stdout.splitlines()
    assert value_after(jaxpr, "jaxpr:") == [
        "dot_general", "broadcast_in_dim", "add", "tanh", "mul", "add"]
    ops = value_after(stablehlo, "StableHLO:")
    assert len(ops) == 11
    assert ops[0] == "dot_general" and ops.count("broadcast_in_dim") == 4
    fusions = value_after(hlo, "HLO compilado, fusiones:")
    # On this laptop's CPU (arm64): ['ynn_fusion', 'multiply_add_fusion']. Names change with the
    # backend; what holds is that eleven StableHLO ops become a couple of fused kernels.
    assert 1 <= len(fusions) <= 3


def test_compilado_roto_rejects_another_shape():
    r = run("03_compilado_roto")
    assert r.returncode != 0
    assert r.stdout.startswith("compilar: ") and "ms por llamada" in r.stdout
    assert "TypeError: Argument types differ from the types for which this computation was " \
           "compiled" in r.stderr
    assert "Argument 'x' compiled with float32[32,64] and called with float32[16,64]" in r.stderr


def test_compilar_one_executable_per_shape():
    r = run("03_compilar")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert [ln.split(":")[0] for ln in lines[:4]] == [
        "batch de 32", "batch de 32", "batch de 16", "batch de 32"]
    assert lines[4] == "llamadas: 4  compilaciones: 2"
    ms = [float(ln.split(":")[1].split()[0]) for ln in lines[:4]]
    assert ms[0] > ms[1]  # the first call compiles, the second reuses the executable


def test_dispositivos_shards_rows_and_adds_one_all_reduce():
    r = run("04_dispositivos")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "cpu:0 tiene las filas 0 a 1",
        "cpu:1 tiene las filas 2 a 3",
        "cpu:2 tiene las filas 4 a 5",
        "cpu:3 tiene las filas 6 a 7",
        "en un dispositivo: pérdida 1615.5, all-reduce 0",
        "repartido: pérdida 1615.5, all-reduce 1",
    ]
