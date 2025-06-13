#######################################################################
#  generic_engine.py  –– fixed-size (10 DOF) Hamiltonian sandbox
#
#  • User edits only `coords` and `potential`
#  • Everything else (force, step, simulate) is derived automatically
#######################################################################

# --------------------------------------------------------------------
# === USER SECTION ===================================================
# --------------------------------------------------------------------
# Just replace these two functions if you want a different system,
# but keep the Array sizes at [float, 10].
# --------------------------------------------------------------------

def coords(q : In[Array[float, 10]], out : Out[Array[float, 10]]):
    i : int
    i = 0
    while (i < 10, max_iter := 10):
        out[i] = q[i]
        i = i + 1

def potential(q : In[Array[float, 10]]) -> float:
    k : float = 1.0
    spring0 : float
    spring1 : float
    spring0 = 0.5 * k * (q[0] - 0.0)*(q[0] - 0.0)
    spring1 = 0.5 * k * (q[1] - q[0])*(q[1] - q[0])
    return spring0 + spring1

# --------------------------------------------------------------------
# === AUTODIFF / FORCE =================================================
# --------------------------------------------------------------------
d_potential = rev_diff(potential)

def grad_U(q : In[Array[float, 10]], out : Out[Array[float, 10]]):
    # ∂U/∂q
    d_potential(q, out, 1.0)

def force(q : In[Array[float, 10]], f : Out[Array[float, 10]]):
    g_arr : Array[float, 10]
    i     : int

    # compute ∇U
    grad_U(q, g_arr)

    # f = −∇U
    i = 0
    while (i < 10, max_iter := 10):
        f[i] = -g_arr[i]
        i = i + 1

# --------------------------------------------------------------------
# === ONE SYMPLECTIC‐EULER STEP =======================================
# --------------------------------------------------------------------
def step(q_in   : In[Array[float, 10]],
         p_in   : In[Array[float, 10]],
         mass   : In[Array[float, 10]],
         dt     : In[float],
         q_out  : Out[Array[float, 10]],
         p_out  : Out[Array[float, 10]]):
    f_arr : Array[float, 10]
    i     : int
    p_new : float
    q_new : float

    # compute forces
    force(q_in, f_arr)

    # update each DOF
    i = 0
    while (i < 10, max_iter := 10):
        force(q_in, f_arr)
        p_new = p_in[i] + dt * f_arr[i]
        q_new = q_in[i] + dt * (p_new / mass[i])
        p_out[i] = p_new
        q_out[i] = q_new
        i = i + 1

# --------------------------------------------------------------------
# === RUN MANY STEPS ==================================================
# --------------------------------------------------------------------
def simulate(q0      : In[Array[float, 10]],
             v0      : In[Array[float, 10]],
             mass    : In[Array[float, 10]],
             dt      : In[float],
             steps   : In[int],
             q_final : Out[Array[float, 10]],
             p_final : Out[Array[float, 10]]):
    q_arr : Array[float, 10]
    p_arr : Array[float, 10]
    i     : int
    s     : int

    # initialize state: p = m*v
    i = 0
    while (i < 10, max_iter := 10):
        q_arr[i] = q0[i]
        p_arr[i] = mass[i] * v0[i]
        i = i + 1

    # symplectic‐Euler loop
    s = 0
    while (s < steps, max_iter := 100000):
        step(q_arr, p_arr, mass, dt, q_arr, p_arr)
        s = s + 1

    # write back outputs
    i = 0
    while (i < 10, max_iter := 10):
        q_final[i] = q_arr[i]
        p_final[i] = p_arr[i]
        i = i + 1

