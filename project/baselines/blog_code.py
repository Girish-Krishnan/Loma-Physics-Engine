#!/usr/bin/env python3
import jax.numpy as jnp
from jax import jacobian, grad, hessian
from dataclasses import dataclass
from typing import Callable, List, Tuple

@dataclass
class Config:
    positions: jnp.ndarray  # Rⁿ
    velocities: jnp.ndarray # Rⁿ

@dataclass
class Phase:
    positions: jnp.ndarray  # Rⁿ
    momenta: jnp.ndarray    # Rⁿ

@dataclass
class System:
    inertia: jnp.ndarray                 # length-m array
    coords: Callable[[jnp.ndarray], jnp.ndarray]  # Rⁿ→Rᵐ
    potential: Callable[[jnp.ndarray], float]     # U(q)

    def jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        # m×n Jacobian matrix ∂coordsᵢ/∂qⱼ
        return jacobian(self.coords)(q)

    def hessians_by_j(self, q: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Returns a Python list of length n, each entry an m×n matrix H_j
        with (H_j)₍ᵢₖ₎ = ∂² coordsᵢ / ∂qⱼ ∂qₖ.
        """
        m = self.inertia.shape[0]
        # build full Hessian tensor shape (m, n, n)
        H_full = jnp.stack([
            hessian(lambda qq, i=i: self.coords(qq)[i])(q)
            for i in range(m)
        ], axis=0)  # shape m×n×n

        n = q.shape[0]
        # slice out the j–th fiber to get H_j of shape m×n
        return [H_full[:, j, :] for j in range(n)]

    def potential_grad(self, q: jnp.ndarray) -> jnp.ndarray:
        # n-vector ∇U(q)
        return grad(self.potential)(q)


def momenta(system: System, config: Config) -> jnp.ndarray:
    q, v = config.positions, config.velocities
    J = system.jacobian(q)             # m×n
    M = jnp.diag(system.inertia)       # m×m
    return J.T @ (M @ (J @ v))         # Rⁿ


def to_phase(system: System, config: Config) -> Phase:
    return Phase(config.positions, momenta(system, config))


def hamiltonian_eqns(system: System, phase: Phase
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q, p = phase.positions, phase.momenta
    J   = system.jacobian(q)           # m×n
    trJ = J.T                           # n×m
    M   = jnp.diag(system.inertia)     # m×m
    K   = trJ @ (M @ J)                # n×n
    Kinv = jnp.linalg.inv(K)           # n×n

    dqdt = Kinv @ p                    # Rⁿ

    H_by_j = system.hessians_by_j(q)   # List of n matrices m×n

    # replicate the Haskell's “bigUglyThing” formula per coordinate j
    def comp(Hj):
        v1 = Kinv @ p                  # Rⁿ
        v2 = Hj   @ v1                 # Rᵐ
        v3 = M    @ v2                 # Rᵐ
        v4 = trJ  @ v3                 # Rⁿ
        v5 = Kinv @ v4                 # Rⁿ
        return - p.dot(v5)             # scalar

    big_ugly = jnp.array([comp(Hj) for Hj in H_by_j])  # Rⁿ
    dpdt = big_ugly - system.potential_grad(q)        # Rⁿ

    return dqdt, dpdt


def step_euler(system: System, dt: float, phase: Phase) -> Phase:
    dq, dp = hamiltonian_eqns(system, phase)
    return Phase(phase.positions + dt*dq,
                 phase.momenta   + dt*dp)


def run_system(system: System, dt: float, init: Phase, steps: int
              ) -> List[Phase]:
    traj = [init]
    for _ in range(steps-1):
        traj.append(step_euler(system, dt, traj[-1]))
    return traj


# --- simple 2D free‐fall under gravity ---
def simple_coords(q):      return q
def simple_potential(q):   return 9.8 * q[1]

simple_system = System(
    inertia    = jnp.array([5.,5.]),
    coords     = simple_coords,
    potential  = simple_potential
)

simple_config0 = Config(
    positions  = jnp.array([0.,0.]),
    velocities = jnp.array([1.,3.])
)


# --- pendulum of length 0.25 in 2D under gravity ---
def pendulum_coords(q):
    θ = q[0]
    return jnp.array([-0.25*jnp.sin(θ), -0.25*jnp.cos(θ)])

def pendulum_potential(q):
    return 9.8 * pendulum_coords(q)[1]

pendulum_system = System(
    inertia    = jnp.array([5.,5.]),
    coords     = pendulum_coords,
    potential  = pendulum_potential
)

pendulum_config0 = Config(
    positions  = jnp.array([0.0]),
    velocities = jnp.array([0.1])
)


def simple_main():
    print("simpleMain")
    traj = run_system(simple_system, 0.1,
                      to_phase(simple_system, simple_config0), 25)
    for ph in traj:
        x,y = ph.positions
        print(f"{x:.2f}  {y:.2f}")


def pendulum_main():
    print("pendulumMain")
    traj = run_system(pendulum_system, 0.1,
                      to_phase(pendulum_system, pendulum_config0), 25)
    for ph in traj:
        (θ,) = ph.positions
        print(f"{θ:.3f}")


if __name__ == "__main__":
    simple_main()
    print()
    pendulum_main()
