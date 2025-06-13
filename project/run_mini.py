#!/usr/bin/env python3
"""
Driver that runs our tiny Loma “mini_engine” and a JAX baseline side-by-side.
"""

import os
import sys
import ctypes
import numpy as np
import jax
import jax.numpy as jnp
import time

# ─── locate & import our compiler front-end ────────────────────────────
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
import compiler  # type: ignore

# ─── compile the Loma mini-engine ─────────────────────────────────────
src = os.path.join("loma_code", "mini_engine.py")
with open(src) as f:
    structs, lib = compiler.compile(
        f.read(),
        target="c",
        output_filename=os.path.join("_code", "mini_engine")
    )

step        = lib.step
d_potential = lib.d_potential

# ─── helpers to make ctypes float32 buffers (and keep them alive) ──────
def make_f32_buffer(arr):
    buf = np.array(arr, dtype=np.float32)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return buf, ptr

# ─── initial conditions & parameters ─────────────────────────────────
# positions [x0,y0,x1,y1], velocities likewise
q0_buf, q0_ptr = make_f32_buffer([0.0, 0.0, 1.0, 0.0])
v0_buf, v0_ptr = make_f32_buffer([0.0, 0.0, 0.0, 0.0])
q1_buf, q1_ptr = make_f32_buffer([0, 0, 0, 0])
v1_buf, v1_ptr = make_f32_buffer([0, 0, 0, 0])
dq_buf, dq_ptr = make_f32_buffer([0, 0, 0, 0])

# scalar params
m0 = np.float32(1.0)
m1 = np.float32(1.0)
k  = np.float32(2.0)
g  = np.float32(9.8)
h  = np.float32(0.01)

# Out-scalars for reverse mode
_dm0 = ctypes.c_float(0.0)
_dm1 = ctypes.c_float(0.0)
_dk  = ctypes.c_float(0.0)
_dg  = ctypes.c_float(0.0)

# ─── run one symplectic-Euler step via Loma ────────────────────────────
time_start = time.time()
step(
    q0_ptr, v0_ptr,
    m0, m1, k, g, h,
    q1_ptr, v1_ptr
)
print("LOMA — After one step:")
print("Simulation time:", time.time() - time_start, "seconds")
print(" q =", q1_buf)
print(" v =", v1_buf)

# ─── compute ∂U/∂q at q0 via reverse AD (Loma) ────────────────────────
d_potential(
    q0_ptr,         # q
    dq_ptr,         # _dq (Out)
    m0, ctypes.byref(_dm0),
    m1, ctypes.byref(_dm1),
    k,  ctypes.byref(_dk),
    g,  ctypes.byref(_dg),
    np.float32(1.0)
)
print("LOMA — ∂U/∂q at q0 =", dq_buf)
print("LOMA — ∂U/∂m0,∂m1,∂k,∂g =", _dm0.value, _dm1.value, _dk.value, _dg.value)

# ─── now do the same with JAX ─────────────────────────────────────────
def potential_jax(q, m0, m1, k, g):
    dx = q[0] - q[2]
    dy = q[1] - q[3]
    Uspring = 0.5 * k * (dx*dx + dy*dy)
    Ugrav   = m0*g*q[1] + m1*g*q[3]
    return Uspring + Ugrav

# gradient of potential wrt q
grad_U = jax.jit(jax.grad(potential_jax, argnums=0))

def step_jax(q, v, m0, m1, k, g, h):
    # force = -∇U
    f = -grad_U(q, m0, m1, k, g)
    # inverse‐mass per coordinate
    m_inv = jnp.array([1/m0, 1/m0, 1/m1, 1/m1], dtype=q.dtype)
    v1 = v + h * m_inv * f
    q1 = q + h * v1
    return q1, v1

# pack into DeviceArrays
q0_j = jnp.array(q0_buf)
v0_j = jnp.array(v0_buf)

time_start = time.time()
q1_j, v1_j = step_jax(q0_j, v0_j, float(m0), float(m1), float(k), float(g), float(h))
print("JAX — Simulation time:", time.time() - time_start, "seconds")

dq_j     = grad_U(q0_j, float(m0), float(m1), float(k), float(g))

print("\nJAX — After one step:")
print(" q =", np.array(q1_j))
print(" v =", np.array(v1_j))

print("JAX — ∂U/∂q at q0 =", np.array(dq_j))
print("JAX — ∂U/∂m0,∂m1,∂k,∂g =", jax.grad(potential_jax, argnums=(1, 2, 3, 4))(q0_j, float(m0), float(m1), float(k), float(g)))
