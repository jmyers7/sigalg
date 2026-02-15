---
title: Home
hide:
  - navigation
  - toc
---

<style>
    .md-content__inner > h1:first-child {
        display: none !important;
    }

    .md-content__inner {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    .hero-header h1 {
        font-size: 5.5rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: -0.5rem;
        font-style: italic;
        line-height: 1;
        color: var(--md-default-fg-color);
    }
  
    .hero-header .accent {
        color: #ff0080;
    }

    .hero-header {
        margin-bottom: 2rem;
        border-left: 4px solid #ff0080;
        padding-left: 2rem;
    }

    .hero-header p {
        color: var(--md-default-fg-color);
        font-size: 0.8rem;
        font-style: italic;
    }
</style>

<div class="grid cards" markdown>

<div markdown>

<header class="hero-header">
  <h1>sig<span class="accent">alg</span></h1>
  <p>Measure-Theoretic Probability in Python</p>
</header>

SigAlg is a Python library for measure-theoretic probability: build probability spaces from sample spaces, $\sigma$-algebras, and probability measures, define random variables and stochastic processes, and compute derived objects (e.g., conditional expectations, checks for martingales).

Unlike most probabilistic computing libraries that treat probabilities primarily as arrays of numbers, SigAlg exposes the richer structures of measure-theoretic probability as manipulable, inspectable objects. The goal is to reduce friction when translating from mathematics to working code.

**Key Features:**

- **Core probabilistic objects** — Sample spaces, $\sigma$-algebras, and probability measures modeled close to their definitions.
- **Filtrations of $\sigma$-algebras** — Support for time-evolving information structures used in stochastic processes.
- **Random variables and vectors** — Algebraic operations and transformations, including conditional expectation and variance.
- **$L^2$ spaces of random variables** — Inner products, norms, orthogonal projections, Fourier expansions, measure-theoretic regression.
- **Stochastic processes** — Adapted and predictable processes, stopping times and stopped processes, discrete Itô integrals, and a growing library of built-in processes.
- **Exact and Monte Carlo** — Support for Monte Carlo simulation, discrete approximations to continuous-time objects, and exhaustive exact enumeration.
- **Integration with scientific Python** — NumPy/Pandas interoperability; visualization via Matplotlib/Plotly; probability distributions via SciPy.

All the above is implemented according to SigAlg's core design philosophy of a focus on mathematical fidelity and accuracy, not just black-box simulation and speed. SigAlg is meant to be a different kind of library—an interface between abstract mathematics and concrete code that complements the rest of the Python ecosystem. [Get Started →](getting_started/index.md)

</div>

<div markdown>

=== "random_walk.py"
    ```python
    --8<-- "random_walk.py"
    ```

    1. Create a time index $T = \{1,2,3,4\}$.
    2. Create a discrete-time IID Bernoulli process $B = \{B_t\}_{t=1}^4$, and enumerate all $2^4 = 16$ possible trajectories. A value of $B_t=1$ means step right, and $B_t=0$ means step left.
    3. Create the process $Y=2B-1$, which will serve as the increments of our random walk. A value of $Y_t=1$ means step right, and $Y_t=-1$ means step left.
    4. Create the random walk process $X$ by taking the cumulative sum of the increment process $Y$, so that $X_t = \sum_{s=1}^t Y_s$.
    5. Add the initial state $X_0=0$ to the process $X$.
    6. Get the natural filtration $\mathcal{F}$ of $X$, where $\mathcal{F}_t = \sigma(X_1,X_2,\ldots,X_t)$.
    7. Compute the conditional expectation $E(X_4 \mid \mathcal{F}_3)$.
    8. Print the trajectories of $X$, along with the computed conditional expectation.
    9. Our random walk has positive drift, so it should be a submartingale.

=== "Output"
    ```
    --8<-- "random_walk_output.txt"
    ```

</div>

</div>











