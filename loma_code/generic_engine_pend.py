# generic_engine.py -- Pendulum System in Loma (10 DOF template)

def coords(q : In[Array[float, 10]], out : Out[Array[float, 10]]):
    i : int = 0
    while (i < 10, max_iter := 10):
        out[i] = q[i]
        i = i + 1

def potential(q : In[Array[float, 10]]) -> float:
    g : float = 9.8
    L : float = 1.0
    m : float = 1.0
    theta : float = q[0]  # interpret q[0] as pendulum angle (in radians)
    y : float = -L * cos(theta)
    return m * g * y  # gravitational potential energy

# ========== Autodiff + Force ==========
d_potential = rev_diff(potential)

def grad_U(q : In[Array[float, 10]], out : Out[Array[float, 10]]):
    d_potential(q, out, 1.0)

def force(q : In[Array[float, 10]], f : Out[Array[float, 10]]):
    g_arr : Array[float, 10]
    i : int = 0
    grad_U(q, g_arr)
    while (i < 10, max_iter := 10):
        f[i] = -g_arr[i]
        i = i + 1

# ========== Symplectic Euler ==========
def step(q_in : In[Array[float, 10]],
         p_in : In[Array[float, 10]],
         mass  : In[Array[float, 10]],
         dt    : In[float],
         q_out : Out[Array[float, 10]],
         p_out : Out[Array[float, 10]]):
    f_arr : Array[float, 10]
    i     : int = 0
    force(q_in, f_arr)
    p_new : float
    q_new : float
    while (i < 10, max_iter := 10):
        p_new = p_in[i] + dt * f_arr[i]
        q_new = q_in[i] + dt * (p_new / mass[i])
        p_out[i] = p_new
        q_out[i] = q_new
        i = i + 1

# ========== Simulate Loop =============
def simulate(q0 : In[Array[float, 10]],
             v0 : In[Array[float, 10]],
             mass : In[Array[float, 10]],
             dt : In[float],
             steps : In[int],
             q_final : Out[Array[float, 10]],
             p_final : Out[Array[float, 10]]):
    q_arr : Array[float, 10]
    p_arr : Array[float, 10]
    i : int = 0
    s : int = 0
    while (i < 10, max_iter := 10):
        q_arr[i] = q0[i]
        p_arr[i] = mass[i] * v0[i]
        i = i + 1
    while (s < steps, max_iter := 100000):
        step(q_arr, p_arr, mass, dt, q_arr, p_arr)
        s = s + 1
    i = 0
    while (i < 10, max_iter := 10):
        q_final[i] = q_arr[i]
        p_final[i] = p_arr[i]
        i = i + 1
