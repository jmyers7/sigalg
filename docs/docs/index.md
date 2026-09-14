---
title: home
hide:
  - navigation
  - toc
---

<style>
  .md-header__button.md-logo,
  .md-header__title {
    visibility: hidden;
  }

  .md-grid {
    margin-top: 0 !important;
    }
</style>

<div class="grid cards" markdown>

<div markdown>

<header class="hero-header">
  <img src="assets/sigalg-light-no-logo.svg" class="hero-logo hero-logo--light" alt="SigAlg">
  <img src="assets/sigalg-dark-no-logo.svg" class="hero-logo hero-logo--dark" alt="SigAlg">
</header>

SigAlg is a Python library for measure-theoretic probability theory, designed as a general-purpose tool for researchers, instructors and students that work in the intersection of rigorous mathematics and computation. The central design philosophy of SigAlg is a high fidelity between mathematics and code, resulting in an API that closely mirrors the equations, formulas and expressions that a researcher would write on a sheet of paper or whiteboard. Many of the abstractions in mathematical probability theory — from primitive objects like sample spaces and probability measures, to richer structures like martingales and Radon-Nikodym derivatives — appear in SigAlg as first-class objects. *SigAlg is probability theory made computable.*

Named after a *$\sigma$-algebra*, a type of foundational object in modern probability theory, SigAlg naturally complements and builds on the existing Python scientific computing stack while filling its own niche. It sits in the research/prototyping layer of a workflow, where a user is first beginning to translate an abstract mathematical model to something computable in a machine. SigAlg does not blackbox or hide the underlying mathematics; it is all completely exposed, open to manipulation and fine-tuning, allowing a researcher to experiment with novel probabilistic architectures before translating to a production-grade system.

SigAlg was first conceived as a tool for experiments in quantitative finance, but its scope has since grown, and its core is now general enough for use in any field that intersects rigorous probability theory. Current development is focused along the following axes:

- Blah.

[Get Started →](getting_started/index.md)

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











