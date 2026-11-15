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

Named after a *$\sigma$-algebra*, a foundational type of object in modern probability theory, SigAlg is a Python library designed as a tool for theoreticians and practioners that work in the intersection of rigorous probability theory and computation. It was first conceived as a tool for experiments in quantitative finance, but its scope has since grown to include abstractions of core probabilistic objects ranging from probability spaces and random variables, to Radon-Nikodym derivatives and conditional KL divergences.

The core design philosophy of SigAlg centers on mathematical fidelity and accuracy, not just black-box simulation and speed. The user manipulates random variables and other objects in SigAlg code as easily as they would with pen and paper, facilitating experimentation and reducing the friction between abstract mathematical concepts and concrete executable code.

</div>

<div markdown>

## Installation

Open a virtual environment running Python 3.13 or later and install SigAlg using either `uv` or `pip`:

=== "uv"
    ```bash
    uv add sigalg
    ```
=== "pip"
    ```bash
    pip install sigalg
    ```

## Getting started

SigAlg is under active development. The core was completely rewritten between `v0.1.11` and `v0.1.12`, which made the older introductory tutorials obsolete. They are being rewritten. In the meantime, several new tutorials cover SigAlg's financial applications:

- [The basics of binomial trees](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/binomial-pricing-model-1.ipynb){target="_blank"}
- [Contingent claims](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/binomial-pricing-model-2.ipynb){target="_blank"}
- [Risk-neutral valuation using binomial trees](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/binomial-pricing-model-3.ipynb){target="_blank"}
- [Standard and arithmetic Brownian motions](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/brownian-motion-1.ipynb){target="_blank"}

These tutorials are pitched at a higher level than a first introduction, which will follow soon. Until then, the API reference contains many code examples.

</div>

</div>











