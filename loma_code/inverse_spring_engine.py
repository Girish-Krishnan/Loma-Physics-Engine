#######################################################################
# inverse_spring_engine.py  –– 2-mass spring with *k* as an input
#
#   q[0], q[1] = positions  of mass 0 and 1 on a line
#   param k    = spring constant (to be learned)
#######################################################################

def potential(q : In[Array[float, 4]],   # pad to 4 for alignment
              k : In[float]) -> float:
    x0 : float = q[0]
    x1 : float = q[1]
    dx : float = x0 - x1
    return 0.5 * k * dx * dx

d_pot = rev_diff(potential)

def grad_U(q : In[Array[float,4]], k : In[float],
           d_q : Out[Array[float,4]], d_k : Out[float]):
    d_pot(q, d_q, k, d_k, 1.0)

# -------- wrap generic integrate boilerplate -------------------------
def force(q, k, f):
    g_arr : Array[float,4]
    dk : float
    grad_U(q, k, g_arr, dk)
    i : int = 0
    while (i < 4, max_iter := 4):
        f[i] = -g_arr[i]
        i = i + 1

def step(q_in, p_in, mass, k, dt, q_out, p_out):
    f_arr : Array[float,4]
    force(q_in, k, f_arr)
    i : int = 0
    p_new : float
    q_new : float
    while (i < 4, max_iter := 4):
        p_new = p_in[i] + dt * f_arr[i]
        q_new = q_in[i] + dt * (p_new / mass[i])
        p_out[i] = p_new
        q_out[i] = q_new
        i = i + 1
