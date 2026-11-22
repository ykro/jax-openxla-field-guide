"""Compilar una vez y ejecutar muchas: cuánto cuesta cada parte."""
import time

import jax
import jax.numpy as jnp


def capa(x, w, b):
    return jnp.tanh(x @ w + b) * 0.5 + 0.5


x, w, b = jnp.ones((32, 64)), jnp.ones((64, 128)), jnp.ones(128)
t0 = time.perf_counter()
compilado = jax.jit(capa).lower(x, w, b).compile()
t1 = time.perf_counter()
for _ in range(100):
    y = compilado(x, w, b)
y.block_until_ready()
t2 = time.perf_counter()
print(f"compilar: {(t1 - t0) * 1e3:.1f} ms   ejecutar: {(t2 - t1) / 100 * 1e3:.3f} ms por llamada")

otro_batch = jnp.ones((16, 64))
print(compilado(otro_batch, w, b).shape)
