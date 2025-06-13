# A Differentiable Physics Engine in **loma**
This repository contains the full source, drivers, and figures for my CSE 291 final project.  The goal is to show that **loma** (a tiny pedagogical DSL/IR) can act as a *practical* differentiable‐physics language once its C backend is compiled.  With only a user-written potential \(U(q,\theta)\), the compiler automatically produces

1. the force \(f(q)=-\nabla_qU\) (via reverse AD),
2. a symplectic-Euler integrator, and  
3. the corresponding reverse pass for *any* differentiable parameter.

The resulting binaries run **\(10^{3}\!-\!10^{4}\times\)** faster than an equivalent JAX + JIT baseline while returning identical trajectories and gradients.

---

## 0 · Quick‐start (software setup)

| step | command / action | note |
|------|------------------|------|
| 1. | **Follow HW 0** instructions for CSE 291 | creates the `loma_env` Conda environment with LLVM + Clang 14, JAX, NumPy, SciPy, etc. |
| 2. | `git clone https://github.com/BachiLi/loma_public` | official course repo |
| 3. | clone **this** project (or copy its folder) and place it inside `loma_public/hw_tests/project/` | the compiler will find sources relative to that path |
| 4. | `conda activate loma_env` | enter the environment |
| 5. | run any driver, e.g. `python run_mini.py` | see next section |

> **Tip:** keep `$PYTHONPATH` *unset*; every driver appends the correct parent
> directory to `sys.path` before importing the `compiler` front-end.

---

## 1 · Repository layout

```plaintext
.
├── baselines/ # pure-Python JAX references
├── loma_code/ # all *.py kernels written in loma
├── run_mini.py # minimal 2-mass demo
├── run_blog.py # free-fall + pendulum (Justin Le remake)
├── run_generic.py # 10-DOF spring chain template
├── run_generic_engine_pend.py # 10-DOF pendulum template
├── run_triple_pend.py # NEW 3-link pendulum benchmark
├── run_inverse_spring.py # NEW spring-constant inverse design
└── final_imgs/ # PDF/PNG figures auto-generated
```

Every **`run_*.py`** file is self-contained:

* compiles the corresponding `loma_code/*.py`,
* builds ctypes signatures,
* runs the simulation loop,
* mirrors the setup in JAX for comparison, and
* saves / displays a figure where appropriate.

---

## 2 · How to reproduce all results

| experiment | driver | output |
|------------|--------|--------|
| *Mini-engine* sanity & gradients | `python run_mini.py` | prints ∂U/∂q, ∂U/∂k and runtime |
| Free-fall + single-pendulum (blog remake) | `python run_blog.py` | final positions + timings |
| Spring chain (2 DOF in 10-slot sandbox) | `python run_generic.py` | shows trajectory plot |
| Pendulum (1 DOF in 10-slot sandbox) | `python run_generic_engine_pend.py` | shows θ(t) plot |
| **Triple pendulum (NEW)** | `python run_triple_pend.py` | `final_imgs/triple_pendulum.pdf` |
| **Inverse spring constant (NEW)** | `python run_inverse_spring.py` | prints *true k* vs *recovered k* |

> Figures are saved into `final_imgs/` and referenced directly by the
> LaTeX report.  Re-run a driver to regenerate the PDF/PNG with fresh
> data.

---

## 3 · Summary of quantitative results

| system / task | steps | Loma runtime (s) | JAX runtime (s) | speed-up | max ‖Δtrajectory‖∞ |
|---------------|-------|------------------|-----------------|----------|--------------------|
| Mini-engine (1 step) | 1 | 2.6 × 10⁻⁵ | 1.8 × 10⁻¹ | **6900×** | 0 |
| Free-fall (25) | 25 | 5.7 × 10⁻⁵ | 2.1 × 10⁻¹ | **3700×** | 2 × 10⁻⁶ |
| Pendulum (25) | 25 | 3.7 × 10⁻⁵ | 1.2 × 10⁻¹ | **3200×** | 3 × 10⁻⁶ |
| Spring chain (200) | 200 | 5.7 × 10⁻⁴ | 1.65 | **2900×** | 8 × 10⁻⁶ |
| Pendulum in sandbox (200) | 200 | 1.1 × 10⁻³ | 1.66 | **1500×** | 1 × 10⁻⁵ |
| **Triple pendulum (10 k)** | 10 000 | 0.94 | 78.1 | **83×** | 9 × 10⁻⁴ |
| **Inverse-k recovery** | 300 | n/a\* | n/a\* | — | recovered k ≈ 4.175 (1.8 % error) |

\*`run_inverse_spring.py` runs forward + gradient inside SciPy’s optimizer,
so a direct runtime comparison is less meaningful.

---

## 4 · Key figures

| file | description |
|------|-------------|
| `final_imgs/energy_drift_plot.pdf` | log–log energy drift (h = 0.1, 0.05, 0.025) showing \(O(h^2)\) slope |
| `final_imgs/triple_pendulum.pdf` | chaotic θ₃(t) for 3-link pendulum, Loma vs JAX (curves overlap) |
| `checkpoint_imgs/spring.png` | 2-mass spring chain positions vs time |
| `checkpoint_imgs/pendulum.png` | single pendulum θ(t) over 10 s |

Run the corresponding driver to regenerate any plot:  
e.g. `python run_triple_pend.py` rewrites
`final_imgs/triple_pendulum.pdf`.

---

## 5 · Troubleshooting

* **“LLVM cannot allocate memory”** — make sure you followed HW 0 exactly and are using Clang 14 from `conda-forge`.
* **`TracerIntegerConversionError`** in JAX — my JAX baselines avoid the error by hard-coding `STEPS`; do not pass a run-time integer to a `for` loop you want to JIT.
* **macOS `SecurityError` on `.dylib`** — strip the quarantine flag:
  ```bash
  xattr -dr com.apple.quarantine _code/
  ```