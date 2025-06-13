#####################################################################
#  blog_engine.py  –  two small Hamiltonian systems in Loma
#
#  • simple 2-D point mass under gravity         (q∈R²,  p∈R²)
#  • planar pendulum with fixed rod length L     (θ  ,  pθ)
#
#  Each system exposes:
#     potential_*     – scalar U(q)
#     d_potential_*   – reverse-mode gradient
#     step_*          – one explicit Euler step in phase space
#####################################################################

# ------------------------------------------------------------------
# ===  SIMPLE FREE-FALL  =================================================
# ------------------------------------------------------------------

def potential_simple(q : In[Array[float, 2]],
                     m : In[float],
                     g : In[float]) -> float:
    # U = m g y               ( y = q[1] )
    return m * g * q[1]

d_potential_simple = rev_diff(potential_simple)

def force_simple(q : In[Array[float, 2]],
                 m : In[float],
                 g : In[float],
                 f : Out[Array[float, 2]]):
    # f = −∇U
    dU : Array[float, 2]
    dm : float
    dg : float
    d_potential_simple(q, dU, m, dm, g, dg, 1.0)
    f[0] = -dU[0]
    f[1] = -dU[1]

def step_simple(q_in  : In[Array[float, 2]],
                p_in  : In[Array[float, 2]],
                m     : In[float],
                g     : In[float],
                dt    : In[float],
                q_out : Out[Array[float, 2]],
                p_out : Out[Array[float, 2]]):
    # momentum update
    f : Array[float, 2]
    force_simple(q_in, m, g, f)

    p0 : float = p_in[0] + dt * f[0]
    p1 : float = p_in[1] + dt * f[1]

    # position update: dq/dt = p / m
    q0 : float = q_in[0] + dt * (p0 / m)
    q1 : float = q_in[1] + dt * (p1 / m)

    q_out[0] = q0
    q_out[1] = q1
    p_out[0] = p0
    p_out[1] = p1


# ------------------------------------------------------------------
# ===  PLANAR PENDULUM  ================================================
#      θ ∈ R  ,  p = m L² ω
#      Hamiltonian  H = p²/(2 m L²)  +  m g L (1 − cos θ)
# ------------------------------------------------------------------

def potential_pend(theta : In[float],
                   m     : In[float],
                   L     : In[float],
                   g     : In[float]) -> float:
    return m * g * L * (1.0 - cos(theta))

d_potential_pend = rev_diff(potential_pend)

def torque_pend(theta : In[float],
                m     : In[float],
                L     : In[float],
                g     : In[float]) -> float:
    # τ = −dU/dθ  (scalar)
    dθ : float
    dm : float
    dL : float
    dg : float
    d_potential_pend(theta, dθ, m, dm, L, dL, g, dg, 1.0)
    return -dθ

def step_pend(theta_in : In[float],
              p_in     : In[float],
              m        : In[float],
              L        : In[float],
              g        : In[float],
              dt       : In[float],
              theta_out: Out[float],
              p_out    : Out[float]):
    # momentum update (τ is “force” for generalized coord θ)
    τ : float = torque_pend(theta_in, m, L, g)
    p_new : float = p_in + dt * τ

    # position update  dθ/dt = p / (m L²)
    theta_new : float = theta_in + dt * (p_new / (m * L * L))

    theta_out = theta_new
    p_out     = p_new
