"""jax.jit guarda un ejecutable por forma y compila solo cuando ve una nueva."""
import time

import jax
import jax.numpy as jnp

compilaciones = 0


def capa(x, w, b):
    global compilaciones
    compilaciones += 1  # este Python solo corre cuando JAX traza para compilar
    return jnp.tanh(x @ w + b) * 0.5 + 0.5


w, b = jnp.ones((64, 128)), jnp.ones(128)
rapida = jax.jit(capa)
formas = [32, 32, 16, 32]
for filas in formas:
    t0 = time.perf_counter()
    rapida(jnp.ones((filas, 64)), w, b).block_until_ready()
    print(f"batch de {filas}: {(time.perf_counter() - t0) * 1e3:7.2f} ms")
print(f"llamadas: {len(formas)}  compilaciones: {compilaciones}")
