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

SigAlg is a Python library for **measure-theoretic probability**: build probability spaces from sample spaces, $\sigma$-algebras, and probability measures, define random variables and stochastic processes, and compute derived objects (e.g., **conditional expectations**, **checks for martingales**).

Unlike most probabilistic computing libraries that treat probabilities primarily as arrays of numbers, SigAlg exposes the richer structures of measure-theoretic probability as manipulable, inspectable objects. The goal is to reduce friction when translating **from mathematics to working code**.

**Key Features:**

- **Core probabilistic objects** — Sample spaces, $\sigma$-algebras, and probability measures modeled close to their definitions.
- **Filtrations of $\sigma$-algebras** — Support for time-evolving information structures used in stochastic processes.
- **Random variables and vectors** — Algebraic operations and transformations, including conditional expectation and variance.
- **$L^2$ spaces** — Inner products, norms, orthogonal projections, Fourier expansions, measure-theoretic regression.
- **Stochastic processes** — A growing library of processes with an emphasis on experimentation, not just black-box simulation.
- **Integrations** — NumPy/Pandas interoperability; visualization via Matplotlib/Plotly; distribution support via SciPy.

[Get Started →](getting_started/index.md)

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










