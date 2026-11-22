"""Cuatro dispositivos simulados en el CPU: XLA reparte el batch y suma entre ellos."""
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_num_cpu_devices", 4)  # antes de que JAX use el CPU por primera vez
malla = jax.make_mesh((4,), ("data",))
w = jnp.ones((3, 2))
x = jnp.arange(8 * 3.0).reshape(8, 3)
x_repartido = jax.device_put(x, NamedSharding(malla, P("data")))
for pedazo in x_repartido.addressable_shards:
    print(pedazo.device, "tiene las filas", pedazo.index[0].start, "a", pedazo.index[0].stop - 1)


@jax.jit
def perdida(w, x):
    return jnp.mean((x @ w) ** 2)


for nombre, datos in [("en un dispositivo", x), ("repartido", x_repartido)]:
    hlo = perdida.lower(w, datos).compile().as_text()
    print(f"{nombre}: pérdida {perdida(w, datos)}, all-reduce {hlo.count('all-reduce(')}")
