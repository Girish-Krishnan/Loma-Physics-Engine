#!/usr/bin/env python3
"""
Symplectic–Euler Hamiltonian dynamics with JAX autodiff.

Key change: the integrator no longer takes `n_steps` as a run-time argument;
`STEPS` is a global constant, avoiding the non-hashable static-arg error.
"""

import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax

# ---------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------
DIM   = 4          # DOF
M     = jnp.ones(DIM)
H     = 1e-2       # time step
STEPS = 10_000     # integration steps  (compile-time constant!)
BATCH = 1_024      # batch size

# ---------------------------------------------------------------------
# Potential energy and force
# ---------------------------------------------------------------------
def potential(q):
    """Quartic double-well in each dimension."""
    return jnp.sum(0.25 * q**4 - 0.5 * q**2, axis=-1)

force = jit(grad(lambda q: potential(q).sum()))  # −∇U

# ---------------------------------------------------------------------
# Symplectic Euler integrator (no n_steps arg)
# ---------------------------------------------------------------------
@jit
def symplectic_euler(q0, v0):
    """Integrate for STEPS steps and return (q, v)."""
    m_inv = 1.0 / M

    def body(_, state):
        q, v = state
        v = v - H * m_inv * force(q)
        q = q + H * v
        return q, v

    return lax.fori_loop(0, STEPS, body, (q0, v0))

# Batched (vmap) version; then JIT
symplectic_euler_batched = jit(vmap(symplectic_euler, in_axes=(0, 0)))

# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------
def main():
    key = jax.random.PRNGKey(0)
    q0  = jax.random.normal(key,   (BATCH, DIM))
    v0  = jax.random.normal(key+1, (BATCH, DIM))

    # Warm-up (compile)
    _ = symplectic_euler_batched(q0, v0)

    # Timed run
    t0 = time.perf_counter()
    qf, vf = symplectic_euler_batched(q0, v0)
    dt = time.perf_counter() - t0

    # Simple loss and gradient wrt first sample's q0
    def loss_fn(q_init):
        q_end, _ = symplectic_euler(q_init, v0[0])
        return jnp.sum(q_end**2)

    grad_q0 = grad(loss_fn)(q0[0])

    print(f"Batch size   : {BATCH}")
    print(f"Time steps   : {STEPS}")
    print(f"Elapsed (s)  : {dt:.3f}")
    print(f"‖∇_q0 loss‖₂ : {jnp.linalg.norm(grad_q0):.3e}")

if __name__ == "__main__":
    main()
