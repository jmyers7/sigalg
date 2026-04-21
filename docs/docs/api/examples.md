---
title: Examples & Dictionary
---

## `core` module 

| Mathematical object | SigAlg class/method | Notebook |
|:---------|:--------|:--------|
| Sample space, $\Omega$ | `SampleSpace`
| Event, $A$ | `Event`
| $\sigma$-algebra, $\mathcal{F}$ | `SigmaAlgebra`
| Probability measure, $P$ | `ProbabilityMeasure`
| Probability space, $(\Omega, \mathcal{F}, P)$ | `ProbabilitySpace`
| Random variables and vectors, $X: \Omega \to \mathbb{R}^d$ | `RandomVariable`, `RandomVector`
| Lebesgue integral, $\int_\Omega X \, dP$ | `Operators.integrate` | [`integrate.ipynb`](https://github.com/jmyers7/sigalg/blob/dev/docs/docs/api/notebooks/integrate.ipynb){target="_blank"} <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/docs/api/notebooks/integrate.ipynb){target="_blank"}
| Expectation, $E(X\mid \mathcal{G})$ | `Operators.expectation` | [`expectation.ipynb`](https://github.com/jmyers7/sigalg/blob/dev/docs/docs/api/notebooks/expectation.ipynb){target="_blank"} <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/docs/api/notebooks/expectation.ipynb){target="_blank"}

## `l2` module

| Mathematical Object | SigAlg Class | Notebook |
|:---------|:--------|:--------|
| $L^2$-space, $L^2(\Omega, \mathcal{F}, P)$ | `L2` |