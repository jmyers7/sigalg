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

SigAlg is a Python library for rigorous measure-theoretic probability theory. It provides intuitive tools for working with σ-algebras, measurable spaces, and probability measures while maintaining mathematical precision.

**Key Features:**

- Construct and manipulate σ-algebras with set operations
- Define probability measures on measurable spaces
- Work with random variables and their distributions
- Verify measurability conditions programmatically

Perfect for researchers, educators, and students working in probability theory, stochastic processes, or mathematical statistics.

</div>

<div markdown>

```python
from sigalg import SigmaAlgebra, ProbabilitySpace

# Create a σ-algebra on a finite set
Ω = {1, 2, 3, 4, 5, 6}
F = SigmaAlgebra.generated_by(
    [{1, 2}, {3, 4}, {5, 6}], 
    base_set=Ω
)

# Define a probability measure
P = ProbabilitySpace(Ω, F)
P.set_measure({1, 2}, 1/3)
P.set_measure({3, 4}, 1/3)

# Check measurability
P.is_measurable({1, 2, 3})  # True
P.measure({1, 2, 3})  # 2/3
```
```python
# Working with random variables
X = RandomVariable(
    lambda ω: ω**2, 
    domain=P
)

# Compute expectations
E_X = X.expectation()  # 15.167
```

</div>

</div>