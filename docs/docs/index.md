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
=== "Output"
    ```
    --8<-- "random_walk_output.txt"
    ```

</div>

</div>










