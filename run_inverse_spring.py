#!/usr/bin/env python3
"""
Recover an unknown spring constant k⋆ from noisy positions using Loma
reverse-mode gradients.
"""

import os, sys, ctypes, time
import numpy as np
from scipy.optimize import minimize

# ---- compiler -------------------------------------------------------
root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(root))
import compiler        # type: ignore

code = open("loma_code/inverse_spring_engine.py").read()
_, lib = compiler.compile(code, target="c",
                          output_filename=os.path.join("_code",
                                                       "inverse_spring"))

_step  = lib.step
_force = lib.force
_dpot  = lib.d_pot

# ---- helpers --------------------------------------------------------
def buf(x):
    arr = np.array(x, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# ---- generate synthetic data ----------------------------------------
k_star = 4.2
dt, steps = 1e-2, 300
mass, mass_ptr = buf([1.0, 1.0, 0, 0])

q , q_ptr  = buf([ 0.5, -0.5, 0, 0])
p , p_ptr  = buf([ 0.0,  0.0, 0, 0])
traj = np.empty((steps+1, 2), np.float32); traj[0]=q[:2]

for _ in range(steps):
    _step(q_ptr, p_ptr, mass_ptr,
          ctypes.c_float(k_star), ctypes.c_float(dt),
          q_ptr, p_ptr)
    traj[_+1] = q[:2]

# add 1 % Gaussian noise
obs = traj + 0.01*np.random.randn(*traj.shape)

# ---- optimisation loop ---------------------------------------------
def loss_and_grad(k_val):
    k = np.float32(k_val)
    q_sim, _ = buf([0.5, -0.5, 0, 0])
    p_sim, _ = buf([0,0,0,0])
    m_ptr = mass_ptr
    tape_dk = ctypes.c_float(0)

    loss = 0.0
    grad_k = 0.0

    for t in range(steps):
        _step(q_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              p_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              m_ptr, k, ctypes.c_float(dt),
              q_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              p_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        diff0 = q_sim[0]-obs[t+1,0]
        diff1 = q_sim[1]-obs[t+1,1]
        loss += diff0*diff0 + diff1*diff1

        # accumulate d loss / d k  via chain rule
        dq = np.zeros(4, np.float32)
        _force(q_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               k, dq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        # gradient of loss wrt q = 2*(q - obs)
        grad_q0 = 2* diff0
        grad_q1 = 2* diff1
        grad_k += grad_q0*dq[0] + grad_q1*dq[1]

    return np.float64(loss), np.array([grad_k], dtype=np.float64)

res = minimize(lambda x: loss_and_grad(x)[0],
               x0=np.array([1.0]),
               jac=lambda x: loss_and_grad(x)[1],
               method="L-BFGS-B", options={"maxiter":200})

print(f"true k = {k_star:.3f}  recovered k = {res.x[0]:.6f}")
