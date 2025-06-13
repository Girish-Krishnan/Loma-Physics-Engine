import os, sys, ctypes, time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Setup compiler import
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
import compiler  # type: ignore

# Compile Loma version
SRC = os.path.join("loma_code", "generic_engine_pend.py")
_, lib = compiler.compile(open(SRC).read(), target="c", output_filename=os.path.join("_code", "generic_engine_pend"))

print("Loma engine compiled")

# C interface
_step = lib.step
_step.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

def make_buf(arr):
    buf = np.array(arr, dtype=np.float32)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return buf, ptr

# Simulation parameters
N = 10
dt = 0.01
steps = 1000
mass_b, mass_ptr = make_buf([1.0]*N)
q_b, q_ptr = make_buf([np.pi / 4] + [0.0]*(N-1))  # 45 degrees initial angle
v_b = np.zeros(N, dtype=np.float32)
p_b, p_ptr = make_buf(v_b * mass_b)
q_traj = np.zeros((steps+1, N), dtype=np.float32)
q_traj[0,:] = q_b

# Run Loma sim
start = time.time()
for s in range(1, steps+1):
    _step(q_ptr, p_ptr, mass_ptr, ctypes.c_float(dt), q_ptr, p_ptr)
    q_traj[s,:] = q_b
loma_time = time.time() - start
print(f"Loma sim took {loma_time:.4f}s")

# JAX potential
def potential_jax(q):
    g = 9.8
    L = 1.0
    m = 1.0
    theta = q[0]
    y = -L * jnp.cos(theta)
    return m * g * y

grad_U_jax = jax.grad(potential_jax)

def simulate_jax(q0, v0, mass, dt):
    steps = 1000
    q = q0
    p = mass * v0
    qs = [q]
    for _ in range(steps):
        f = -grad_U_jax(q)
        p = p + dt * f
        q = q + dt * (p / mass)
        qs.append(q)
    return jnp.stack(qs)

start = time.time()
qj = simulate_jax(jnp.array(q_b), jnp.array(v_b), jnp.array(mass_b), dt)
jax_time = time.time() - start
print(f"JAX sim took {jax_time:.4f}s")

# Plot angle over time (JAX only)
t_axis = np.arange(steps+1) * dt
plt.plot(t_axis, qj[:,0], label="θ(t) (Pendulum)")
plt.xlabel("time (s)")
plt.ylabel("angle θ (rad)")
plt.title("Pendulum Oscillation Over Time (JAX)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
