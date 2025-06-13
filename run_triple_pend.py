#!/usr/bin/env python3
"""
Triple-pendulum benchmark: Loma vs. JAX.
"""

import os, sys, time, ctypes
import numpy as np
import jax, jax.numpy as jnp
from jax import grad

# ---- locate compiler -------------------------------------------------
root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(root))
import compiler       # type: ignore

# ---- compile Loma source --------------------------------------------
src = open("loma_code/triple_pend_engine.py").read()
_, lib = compiler.compile(src, target="c",
                          output_filename=os.path.join("_code",
                                                       "triple_pend"))

_step = lib.step
_step.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # q_in
    ctypes.POINTER(ctypes.c_float),   # p_in
    ctypes.POINTER(ctypes.c_float),   # mass
    ctypes.c_float,                   # dt
    ctypes.POINTER(ctypes.c_float),   # q_out
    ctypes.POINTER(ctypes.c_float) ]  # p_out

def buf(x):
    arr = np.array(x, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

N = 10
mass, mass_ptr = buf([1.0]*N)
dt  = 1e-2
steps = 10_000

q, q_ptr = buf([0.2, 0.3, -0.1] + [0.0]*(N-3))
v        = np.zeros(N, dtype=np.float32)
p, p_ptr = buf(v * mass)

traj = np.empty((steps+1, 3), dtype=np.float32)
traj[0] = q[:3]

t0 = time.perf_counter()
for s in range(1, steps+1):
    _step(q_ptr, p_ptr, mass_ptr, ctypes.c_float(dt), q_ptr, p_ptr)
    traj[s] = q[:3]
t_loma = time.perf_counter() - t0
print(f"Loma finished in {t_loma:.3f}s")

# ---- JAX reference ---------------------------------------------------
g, L, m = 9.8, 1.0, 1.0
def U(q):
    y1 = -L*jnp.cos(q[0])
    y2 = y1 - L*jnp.cos(q[1])
    y3 = y2 - L*jnp.cos(q[2])
    return m*g*(y1+y2+y3)
force = grad(U)

def step_jax(state, _):
    q, p = state
    p = p - dt*force(q)
    q = q + dt*(p/m)
    return (q, p), q

q_j = jnp.array(q[:3]); p_j = jnp.zeros(3)
(q_j, _), qs = jax.lax.scan(step_jax, (q_j, p_j), None, length=steps)
traj_jax = np.vstack((np.array(q[:3])[None], np.array(qs)))
print("JAX trajectory computed")

# ---- quick comparison ------------------------------------------------
print("Max |θ₃_Loma − θ₃_JAX|:",
      np.max(np.abs(traj[:,2] - traj_jax[:,2])))

# ---- plot θ₃ ---------------------------------------------------------
import matplotlib.pyplot as plt
t = np.arange(steps+1)*dt
plt.plot(t, traj[:,2], label="Loma θ₃")
plt.plot(t, traj_jax[:,2], "--", label="JAX θ₃")
plt.xlabel("time (s)"); plt.ylabel("angle (rad)"); plt.legend()
plt.tight_layout(); plt.show()
