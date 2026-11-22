"""El mismo paso de entrenamiento en PyTorch y en JAX, con los mismos datos."""
import jax
import jax.numpy as jnp
import numpy as np
import torch

X = np.random.default_rng(0).normal(size=(64, 3)).astype("float32")
y = X @ np.array([2.0, -1.0, 0.5], dtype="float32") + 1.0

modelo = torch.nn.Linear(3, 1)  # PyTorch: los pesos viven dentro del objeto
torch.nn.init.zeros_(modelo.weight)
torch.nn.init.zeros_(modelo.bias)
opt = torch.optim.SGD(modelo.parameters(), lr=0.1)

params = {"w": jnp.zeros(3), "b": jnp.zeros(())}  # JAX: un dict que tú pasas


def perdida(params, X, y):
    return jnp.mean((X @ params["w"] + params["b"] - y) ** 2)


@jax.jit
def paso(params, X, y):
    valor, grads = jax.value_and_grad(perdida)(params, X, y)
    return jax.tree.map(lambda p, g: p - 0.1 * g, params, grads), valor


for i in range(5):
    opt.zero_grad()
    loss = torch.mean((modelo(torch.from_numpy(X))[:, 0] - torch.from_numpy(y)) ** 2)
    loss.backward()
    opt.step()
    params, valor = paso(params, X, y)
    print(f"paso {i}  torch {loss.item():.5f}  jax {float(valor):.5f}")
