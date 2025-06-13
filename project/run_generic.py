import os, sys, ctypes, time
import numpy as np
import jax
import jax.numpy as jnp

# Locate & import compiler
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
import compiler  # type: ignore

# Compile Loma engine
SRC = os.path.join("loma_code", "generic_engine.py")
_, lib = compiler.compile(
    open(SRC).read(),
    target="c",
    output_filename=os.path.join("_code", "generic_engine")
)

print("Loma engine compiled successfully")

# Setup C interface
_step = lib.step
_step.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
_step.restype = None

def make_buf(arr):
    buf = np.array(arr, dtype=np.float32)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return buf, ptr

# Params
N = 10
dt = 0.05
steps = 200
mass_b, mass_ptr = make_buf([1.0]*N)

# Initial state: two masses stretched
q_b, q_ptr = make_buf([1.5, 3.0] + [0.0]*(N-2))
v_b = np.zeros(N, dtype=np.float32)
p_b, p_ptr = make_buf(v_b * mass_b)

q_traj = np.zeros((steps+1, N), dtype=np.float32)
q_traj[0,:] = q_b

# Run Loma sim
time_start = time.time()
for s in range(1, steps+1):
    _step(q_ptr, p_ptr, mass_ptr, ctypes.c_float(dt), q_ptr, p_ptr)
    q_traj[s,:] = q_b
    print(f"Step {s}/{steps} done", end="\r")

print("Loma simulation time:", time.time() - time_start, "seconds")

# JAX potential
def potential_jax(q):
    k = 1.0
    spring0 = 0.5 * k * (q[0] - 0.0)**2
    spring1 = 0.5 * k * (q[1] - q[0])**2
    return spring0 + spring1

grad_U_jax = jax.grad(potential_jax)
print("JAX potential function ready")

def simulate_jax(q0, v0, mass, dt):
    steps = 200
    q = q0
    p = mass * v0
    qs = [q]
    for _ in range(steps):
        f = -grad_U_jax(q)
        p = p + dt * f
        q = q + dt * (p / mass)
        qs.append(q)
    return qs

print("JAX simulation function ready")
simulate_jax = jax.jit(simulate_jax)
time_start = time.time()
qj = simulate_jax(jnp.array(q_b), jnp.array(v_b), jnp.array(mass_b), dt)
print("JAX simulation time:", time.time() - time_start, "seconds")
qj = jnp.stack(qj)
print("JAX simulation done")

# Compare
import matplotlib.pyplot as plt
t_axis = np.arange(steps+1) * dt
plt.plot(t_axis, qj[:,0], "--", label="q₀")
plt.plot(t_axis, qj[:,1], "--", label="q₁")
plt.xlabel("time")
plt.ylabel("position")
plt.title("2-DOF Spring-Mass Simulation")
plt.legend()
plt.tight_layout()
plt.show()
