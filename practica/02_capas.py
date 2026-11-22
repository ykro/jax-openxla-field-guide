"""Las capas que ve el compilador: jaxpr, StableHLO y el HLO ya optimizado."""
import re

import jax
import jax.numpy as jnp


def capa(x, w, b):
    return jnp.tanh(x @ w + b) * 0.5 + 0.5


x, w, b = jnp.ones((32, 64)), jnp.ones((64, 128)), jnp.ones(128)
jaxpr = jax.make_jaxpr(capa)(x, w, b)
print("jaxpr:    ", [eq.primitive.name for eq in jaxpr.eqns])

bajado = jax.jit(capa).lower(x, w, b)
print("StableHLO:", re.findall(r"= stablehlo\.(\w+)", bajado.as_text()))

hlo = bajado.compile().as_text()
entrada = hlo[hlo.index("ENTRY"):]
print("HLO compilado, fusiones:", re.findall(r"%(\w+) = \S+ fusion\(", entrada))
