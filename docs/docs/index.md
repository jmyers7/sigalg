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

SigAlg is a Python library for measure-theoretic probability theory and stochastic processes. It provides an API that closely mirrors the underlying mathematics, with the goal of making probability spaces, random variables, σ-algebras, filtrations, and stochastic processes programmable in a way that feels as natural as writing definitions and equations on a whiteboard.

**Key Features:**

- Construct and manipulate σ-algebras with set operations
- Define probability measures on measurable spaces
- Work with random variables, filtrations, and stochastic processes
- Verify measurability conditions programmatically
- Visualize information flow through filtrations

**Design Philosophy:**

Built by a mathematician, SigAlg prioritizes mathematical fidelity over computational efficiency. Objects in SigAlg correspond directly to their mathematical counterparts, making it particularly well suited for:

- Students and instructors learning or teaching measure-theoretic probability and stochastic processes
- Researchers who work with abstract probabilistic constructions and want a computational sandbox for experimenting with ideas that are usually confined to paper

SigAlg is not a replacement for production-grade Monte Carlo simulation libraries or high-performance statistical tools. Instead, it complements them by prioritizing clarity, inspectability, and conceptual alignment with the theory.

</div>

<div markdown>

```python
from scipy.stats import bernoulli
from sigalg.core import Time
from sigalg.processes import IIDProcess

# Create an IID process of coin flips
T = Time.discrete(start=1, length=2)
X = IIDProcess(
    distribution=bernoulli(p=0.7),
    support=[0, 1],
    time=T,
).from_enumeration()

# Access the probability measure
P_X = X.probability_measure
P_X((0, 1, 1))  # 0.147

# Get the natural filtration
F = X.natural_filtration
```

```python
from sigalg.processes import RandomWalk

# Create a random walk with drift
T = Time.discrete(length=100)
X = RandomWalk(p=0.7, time=T).from_simulation(
    n_trajectories=10,
    random_state=42,
)

# Plot trajectories
X.plot_trajectories()
```

</div>

</div>