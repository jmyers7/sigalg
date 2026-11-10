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

Open a virtual environment running Python 3.11 or later and install SigAlg using either `uv` or `pip`:

=== "uv"
    ```bash
    uv add sigalg
    ```
=== "pip"
    ```bash
    pip install sigalg
    ```

## Getting started

SigAlg is in active development and recently underwent a complete rewrite of its core (moving from `v0.1.11` to `v0.1.12`). This made several older tutorials and code examples outdated, which are in the process of being rewritten. However, a new tutorial written after the core refactoring focuses on SigAlg's financial roots and is available here:

- Risk-neutral valuation in SigAlg with binomial trees [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/binomial-pricing-model.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/binomial-pricing-model.ipynb){target="_blank"}

Check back for more tutorials. In the meantime, there are many (many!) code examples in the API reference.

</div>

</div>











