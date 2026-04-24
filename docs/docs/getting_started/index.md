---
title: Getting started
hide:
  - toc
  - navigation
---

# Getting Started

SigAlg sits at the intersection of two worlds: mathematics and code. Using it comfortably takes a bit of background in both, but we expect most readers will come in with more Python than mathematics. The sections below list a few prerequisites and references; if you're coming from a mathematics background, the coding prerequisites may feel a bit bare for now (what is a virtual environment? what is `uv`/`pip`?), and we hope to expand them in the future.

## Coding prerequisites

Open a virtual environment running Python 3.11 or later (though 3.14 has not yet been tested) and install SigAlg using either `uv` or `pip`:


=== "uv"
    ```bash
    uv add sigalg
    ```
=== "pip"
    ```bash
    pip install sigalg
    ```

## Mathematical prerequisites

We hope that SigAlg can be used as a learning tool for beginners in measure-theoretic probability, and much of the mathematics can be learned on the fly through Google and Wikipedia, but for those who prefer a more systematic and textbook-style introduction, we recommend the following resources. (And perhaps use SigAlg as a companion to these texts.)

For measure-theoretic probability, we recommend the following textbooks:

- Williams, D., *Probability with martingales*, Cambridge Mathematical Textbooks, Cambridge University Press, 1991.
- Billingsley, P., *Probability and measure*, third, Wiley Series in Probability and Mathematical Statistics, John Wiley & Sons, 1995.

Williams's book is by far the most useful reference for the mathematical concepts used in SigAlg. The first six chapters on "Measure Spaces," "Events," "Random Variables," "Independence," "Integration," and "Expectation" contain more than enough background for the core objects in SigAlg. Later chapters on "Conditional Expectation" and "Martingales" provide background on stochastic processes. Much of this same content is also covered in Billingsley's book, though his text is more verbose and the reader wishing to get up to speed quickly may find it less accessible than Williams's book.

Several texts in applied mathematics also give good introductions to measure-theoretic probability, either in the text itself or in appendices. We have in mind texts in mathematical finance, such as:

- Shreve, S. E., *Stochastic calculus for finance II: Continuous-time models*, Springer Finance, 2004.
- Björk, T., *Arbitrage theory in continuous time*, fourth, Oxford Finance, Oxford University Press, 2020.

The first two chapters in Shreve's text cover the basics of measure-theoretic probability, while the third and fourth chapters cover stochastic processes with a view toward stochastic calculus. Björk's text covers similar material in the first three sections of the Appendices.

SigAlg also models $L^2$ spaces, the general mathematical theory of which is a blend of measure theory and infinite-dimensional linear algebra (i.e., functional analysis). Fortunately, since all the sample spaces in SigAlg are finite, the type of linear algebra required is finite dimensional, the kind that many from computer science and engineering backgrounds are familiar with. Even so, some users might not have ever seen a Hilbert space, which in the context of SigAlg is a finite-dimensional inner product space. For this, we recommend the following text:

- Hansen, V. L., *Functional analysis: Entering Hilbert space*, World Scientific Publishing, 2006.

The first three chapters on "Basic Elements of Metric Topology," "New Types of Function Spaces," and "Theory of Hilbert Spaces" cover the necessary background for understanding the $L^2$ spaces in SigAlg.

[Tutorials →](../tutorials/index.md)