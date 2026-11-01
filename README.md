# Field Guide: De PyTorch a JAX y OpenXLA en Google Cloud

**Por qué es relevante:** El rendimiento real en GPUs modernas y Cloud TPUs depende del compilador (`OpenXLA` y `StableHLO`), no solo de la sintaxis en Python. Esta serie demuestra con código verificable cómo pasar de loops imperativos en PyTorch a grafos fusionados, particionamiento distribuido y post-entrenamiento de modelos abiertos.
**Para quién es:** Desarrolladores e ingenieros de Machine Learning que dominan PyTorch y buscan adoptar `JAX`, `Flax NNX`, `Keras 3`, `Gemma + Tunix` o `PyTorch/XLA` con garantías de rendimiento.
**Quién debería verlo:** Equipos de ingeniería de IA, arquitectos de inferencia/entrenamiento y líderes técnicos que operan cargas sobre GPUs (`NVIDIA L4` / `A100`) y `Cloud TPU v5e/v6e` en Google Cloud.
**Serie paralela para Software Devs (sin experiencia previa en PyTorch):** [JAX & ML for Software Devs Field Guide](https://github.com/ykro/jax-ml-for-software-devs-field-guide).

---

## Arquitectura General del Ecosistema

### 1. Jerarquía desde Python hasta el Silicio (`OpenXLA`)

![Jerarquía de compilación OpenXLA](diagrams/01-openxla-layer-stack.png)

### 2. Árbol de Decisión: ¿Por qué repositorio empezar?

![Árbol de decisión del ecosistema OpenXLA](diagrams/02-decision-flowchart.png)

### 3. Cómputo On-Demand en Google Cloud (`ai-experiments`)

![Arquitectura de ejecución on-demand en GCP](diagrams/03-gcp-ephemeral-arch.png)

---

## Mapa de Repositorios y Resultados Verificados (Outside Judges)

Cada repositorio contiene **5 experimentos ejecutables (`01` a `05`)**, suite de tests con `pytest`, un script de verificación determinista (`verify_*.py`) y runners on-demand con auto-teardown para Google Cloud (`Vertex AI GPU L4`, `Cloud TPU v5e` y `Cloud Run` con escala a cero).

### Serie A — De PyTorch a JAX y OpenXLA

| # | Repositorio | Qué construye y demuestra | Outside Judge (Evidencia real medida) |
|---|---|---|---|
| **1** | [jax-for-pytorch-devs](https://github.com/ykro/jax-for-pytorch-devs) | Transición de loops imperativos de PyTorch a funciones puras, PyTrees, `@jax.jit`, `jax.vmap`, `jax.lax.scan` y *static bucket padding*. | **8.85x** speedup con `@jax.jit` (325 líneas StableHLO), **138.78x** en gradientes per-sample con `vmap`, **12.01x** en rollouts con `lax.scan` y **53.76x** eliminando recompilaciones dinámicas (`max_diff <= 3.34e-06`). |
| **2** | [flax-nnx-transformer-from-scratch](https://github.com/ykro/flax-nnx-transformer-from-scratch) | Construcción de un Transformer causal con RoPE, RMSNorm y SwiGLU en `Flax NNX` (`nnx.Module`, `nnx.split`/`nnx.merge`, `@nnx.jit`). | **136.2x** speedup tras compilación inicial (`449.39 ms` paso 1 vs `3.300 ms` promedio en pasos 2..50), **0 recompilaciones** (57 ops `dot_general`, 281 fusiones HLO) y caída monotónica de pérdida (`5.3140 -> 0.0105`). |
| **3** | [keras3-multibackend-arena](https://github.com/ykro/keras3-multibackend-arena) | Red multimodal Vision+Tabular (`657,418` parámetros) escrita una sola vez con `keras.ops` y evaluada en `KERAS_BACKEND=jax`, `torch` y `numpy`, más endpoint `Cloud Run`. | Paridad exacta de pesos entre backends (`max_diff = 2.98e-07`), **2.00x mayor throughput en entrenamiento** (`24,794.8 samples/s` en JAX vs `12,401.3 samples/s` en PyTorch) y **1.82x en inferencia** (`55,482.9 samples/s`). |
| **4** | [gemma-tunix-spanish-posttraining](https://github.com/ykro/gemma-tunix-spanish-posttraining) | Post-entrenamiento de arquitectura `Gemma` (`4.2M` params base en demo local / escalable a 2B+) con mesh 2D (`data`, `model`), LoRA quirúrgico y alineamiento **GRPO** estilo `Tunix`. | **97.90% reducción** de parámetros entrenables con LoRA (`0.0` drift en pesos base), reducción de **2.0x** en memoria por chip en mesh `2x2`, y mejora de recompensa GRPO de **`-0.0625` a `+1.4875`** (**75%** cumplimiento de formato XML en español con muestreo autorregresivo `G=4`). |
| **5** | [pytorch-xla-tpu-playbook](https://github.com/ykro/pytorch-xla-tpu-playbook) | Migración y optimización de modelos PyTorch en `Cloud TPU` con `PyTorch/XLA`, diagnóstico de *graph breaks*, sharding GSPMD y puente zero-copy `PyTorch <-> JAX` via DLPack. | Reducción de recompilaciones post-warmup de `11` a **`0`**, *host barrier stalls* de `12` a **`0`** y *fallbacks* `aten::nonzero` de `12` a **`0`** (`max_abs_diff = 2.98e-08`), más paridad exacta en `autograd.Function` llamando a `@jax.jit` (`grad_diff = 2.91e-11`). |

---

## Onboarding y Cómo Recorrer la Serie

Sigue los 5 niveles en orden progresivo desde tu terminal (o empieza por la [Serie B para Software Devs](https://github.com/ykro/jax-ml-for-software-devs-field-guide) si vienes de Python/NumPy sin PyTorch).

### 1. Orden progresivo de ejecución

1. **Nivel 1 — Fundamentos del compilador ([jax-for-pytorch-devs](https://github.com/ykro/jax-for-pytorch-devs))** para dominar `@jax.jit`, `Jaxpr`, `StableHLO`, PyTrees puros, `jax.vmap`, `jax.lax.scan` y *static bucket padding*.
2. **Nivel 2 — Ergonomía con estado en Python ([flax-nnx-transformer-from-scratch](https://github.com/ykro/flax-nnx-transformer-from-scratch))** para construir un Transformer causal con `nnx.Module`, `nnx.split` y `nnx.merge`.
3. **Nivel 3 — Portabilidad multi-backend ([keras3-multibackend-arena](https://github.com/ykro/keras3-multibackend-arena))** para comparar `KERAS_BACKEND=jax` contra `torch` y desplegar en Cloud Run con escala a cero.
4. **Nivel 4 — Post-entrenamiento distribuido ([gemma-tunix-spanish-posttraining](https://github.com/ykro/gemma-tunix-spanish-posttraining))** para particionar modelos en meshes 2D `('data', 'model')`, inyectar adaptadores `nnx.LoRA` y alinear con GRPO.
5. **Nivel 5 — Interoperabilidad PyTorch + TPU ([pytorch-xla-tpu-playbook](https://github.com/ykro/pytorch-xla-tpu-playbook))** para llevar código `torch.nn.Module` a Cloud TPU con `LazyTensor`, GSPMD y puentes DLPack hacia `@jax.jit`.

### 2. Validar el índice maestro localmente

```bash
git clone https://github.com/ykro/jax-openxla-field-guide.git
cd jax-openxla-field-guide
python3 scripts/verify_series.py
```

El reporte consolidado se almacena automáticamente en `benchmarks/series_summary.json`.
