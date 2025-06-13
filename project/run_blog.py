#!/usr/bin/env python3
"""
Compare Loma vs. JAX for the two blog-post systems.
"""

import os
import sys
import ctypes
import numpy as np
import jax
from jax import grad
import jax.numpy as jnp
import time

# ─── locate & import our compiler front-end ────────────────────────────
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
import compiler  # type: ignore

# ─── compile Loma source ────────────────────────────────────────────
src_file = os.path.join("loma_code", "blog_engine.py")
structs, lib = compiler.compile(
    open(src_file).read(),
    target="c",
    output_filename=os.path.join("_code", "blog_engine")
)

step_simple     = lib.step_simple
step_pend       = lib.step_pend
torque_pend_ptr = lib.torque_pend          # useful for sign-check

# helper: keep NumPy buffer alive + return ctypes pointer
def buf_and_ptr(vals, dtype=np.float32):
    buf = np.array(vals, dtype=dtype)
    return buf, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# ─── parameters shared by both back-ends ────────────────────────────
g  = 9.8
dt = 0.1
steps = 25

# ===== 1) FREE-FALL ==================================================
print("simpleMain (Loma vs. JAX)")

m = 5.0
# initial config (positions, velocities)
q_np, q_ptr = buf_and_ptr([0., 0.])
v_np        = np.array([1., 3.], dtype=np.float32)
p_np        = m * v_np                          # canonical momentum
p_np, p_ptr = buf_and_ptr(p_np)

q_out, q_out_ptr = buf_and_ptr([0., 0.])
p_out, p_out_ptr = buf_and_ptr([0., 0.])

# run Loma
start = time.time()
for _ in range(steps):
    step_simple(q_ptr, p_ptr, ctypes.c_float(m), ctypes.c_float(g),
                ctypes.c_float(dt), q_out_ptr, p_out_ptr)
    q_np[:] = q_out; p_np[:] = p_out            # copy for next step
    q_ptr = q_out_ptr; p_ptr = p_out_ptr

end = time.time()

print("Loma final position:", q_np)
print("Loma time taken:", end - start, "seconds")

# run JAX counterpart
def simple_force(q):             # −∇U
    return jnp.array([0., -m*g])
def euler_simple(q, p):
    p = p + dt * simple_force(q)
    q = q + dt * p / m
    return q, p

q_jax = jnp.array([0., 0.]); p_jax = jnp.array([m*1., m*3.])
for _ in range(steps):
    q_jax, p_jax = euler_simple(q_jax, p_jax)

print("JAX  final position:", np.asarray(q_jax))
print("JAX  time taken:", time.time() - end, "seconds")
print()

# ===== 2) PENDULUM ===================================================
print("pendulumMain (Loma vs. JAX)")

start = time.time()

m, L = 5.0, 0.25
g_c  = ctypes.c_float(g)
m_c  = ctypes.c_float(m)
L_c  = ctypes.c_float(L)
dt_c = ctypes.c_float(dt)

theta_val = ctypes.c_float(0.0)                 # θ₀
p_val     = ctypes.c_float(m * L * L * 0.1)      # p₀ = m L² ω

for _ in range(steps):
    theta_out_c = ctypes.c_float()
    p_out_c     = ctypes.c_float()

    # pass INPUTS by value, OUTPUTS by pointer
    step_pend(theta_val, p_val,
              m_c, L_c, g_c, dt_c,
              ctypes.byref(theta_out_c), ctypes.byref(p_out_c))

    # promote outputs to next-step inputs
    theta_val, p_val = theta_out_c, p_out_c

end = time.time()

print("Loma final θ:", float(theta_val.value))
print("Loma time taken:", end - start, "seconds")

# --- JAX reference ---------------------------------------------------
start = time.time()
def U_pend(th): return m * g * L * (1.0 - jnp.cos(th))
dU_dθ = grad(U_pend)
# jit this
dU_dθ = jax.jit(dU_dθ)

th_jax = jnp.array(0.0)
p_jax  = jnp.array(float(m * L * L * 0.1))
for _ in range(steps):
    τ      = -dU_dθ(th_jax)
    p_jax  = p_jax + dt * τ
    th_jax = th_jax + dt * p_jax / (m * L * L)
end = time.time()
print("JAX  final θ:", float(th_jax))
print("JAX  time taken:", end - start, "seconds")
