# ------------------------------------------------------------------
# physics_engine.py  (minimal 2-mass demo)
# ------------------------------------------------------------------

# (no structs, no arrays in Params—just plain scalars)
# q = [x0,y0, x1,y1]
# U(q) = ½ k‖p0–p1‖² + m0·g·y0 + m1·g·y1

def potential(q   : In[Array[float,4]],
              m0  : In[float],
              m1  : In[float],
              k   : In[float],
              g   : In[float]) -> float:
    dx : float = q[0] - q[2]
    dy : float = q[1] - q[3]
    U  : float = 0.5 * k * (dx*dx + dy*dy)
    U  = U + m0 * g * q[1]
    U  = U + m1 * g * q[3]
    return U

d_potential = rev_diff(potential)


# one symplectic-Euler step
def step(qi : In[Array[float,4]],
         vi : In[Array[float,4]],
         m0 : In[float],
         m1 : In[float],
         k  : In[float],
         g  : In[float],
         h  : In[float],
         qo : Out[Array[float,4]],
         vo : Out[Array[float,4]]):
    # spring force on mass 0
    fx : float = -k * (qi[0] - qi[2])
    fy : float = -k * (qi[1] - qi[3]) - m0 * g

    vx0 : float = vi[0] + h*(fx/m0)
    vy0 : float = vi[1] + h*(fy/m0)
    qo[0] = qi[0] + h*vx0
    qo[1] = qi[1] + h*vy0
    vo[0] = vx0
    vo[1] = vy0

    # equal and opposite on mass 1
    fx1 : float = -fx
    fy1 : float = -(-m0*g) - m1*g  # gravity on mass1
    vx1 : float = vi[2] + h*(fx1/m1)
    vy1 : float = vi[3] + h*(fy1/m1)
    qo[2] = qi[2] + h*vx1
    qo[3] = qi[3] + h*vy1
    vo[2] = vx1
    vo[3] = vy1
