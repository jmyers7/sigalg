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

Named after a *$\sigma$-algebra*, a foundational type of object in measure theory, SigAlg is a Python library for computational measure-theoretic probability theory and stochastic calculus. Originally developed as a self-study and plotting library for quantitative finance, it has since grown into a general toolkit for probability on finite spaces and the stochastic processes built on them, including finite approximations of continuous objects.

## Installation

Open a virtual environment running Python 3.13 or 3.14 (3.15 has not been tested yet) and install SigAlg using either `uv` or `pip`:

=== "uv"
    ```bash
    uv add sigalg
    ```
=== "pip"
    ```bash
    pip install sigalg
    ```

## Tutorials

In addition to the tutorials below, the API Reference contains many examples of SigAlg.

- [Thinking in $\sigma$-algebras: an information-theoretic introduction to SigAlg](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/info-theory.ipynb){target="_blank"}
- [The basics of binomial trees](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/binomial-pricing-model-1.ipynb){target="_blank"}
- [Contingent claims](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/binomial-pricing-model-2.ipynb){target="_blank"}
- [Risk-neutral valuation using binomial trees](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/binomial-pricing-model-3.ipynb){target="_blank"}
- [Standard and arithmetic Brownian motions](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/brownian-motion-1.ipynb){target="_blank"}
- [Stochastic calculus in SigAlg I: a preview of Itô's lemma](https://github.com/jmyers7/sigalg/blob/main/docs/notebooks/stochastic-calc-1.ipynb){target="_blank"}


</div>

<div markdown>

## Code snippets

=== "Itô's lemma"
    ```python
    --8<-- "ito-lemma.py"
    ```

    1. Let $f(t,x)$ be a twice-differentiable function and $X_t$ a solution to the stochastic differential equation (SDE) $dX_t = \mu \, dt + \sigma \, dW_t$ where $\mu$ and $\sigma>0$ are constants and $W_t$ is a standard Brownian motion. One simple version of *Itô's lemma* states that the stochastic process $f(t, X_t)$ is a solution of the SDE $df = \left( f_t + \mu f_x + \frac{\sigma^2}{2} f_{xx}\right)\, dt + \sigma f_x \, dW_t$. (For brevity we omit the arguments of all functions in the SDE.)
    2. Get the "time" stochastic process $t$ and the Brownian motion $W_t$ that drives $X_t$.
    3. The function in the demonstration is $f(t,x) = tx^2$. Compute and set the partial derivatives of this function.
    4. Integrate the differentials in the SDE in the statement of Itô's lemma.
    5. Plot the integrals on the two sides of Itô's lemma to check for their (approximate) equality.

=== "Fundamental Theorem of Calculus with Itô's correction"
    ```python
    --8<-- "ftc-with-ito.py"
    ```

    1. The Fundamental Theorem of Calculus with Itô's correction states that $f(X_t) - f(X_0) = \int_0^t f'(X_u) \, dX_u + \frac{1}{2} \int_0^t f''(X_u) \, d[X]_u$, where $X_t$ is a "nice" stochastic process (like a diffusion), $f$ is a twice-differentiable function, and $[X]_t$ is the quadratic variation of $X_t$. The first term on the right of this equation is the *classical term*, while the second is *Itô's correction*.

=== "Quadratic variation of Brownian motion"
    ```python
    --8<-- "quad-var.py"
    ```

    1. One of the foundational results in stochastic calculus states that $[W]_t = t$, where $W$ is a standard Brownian motion.

=== "Binomial pricing models"
    ```python
    --8<-- "binomial-pricing.py"
    ```

    1. The value process $V_t$ of the replicating portfolio must have two properties: First, at maturity $T$ the value $V_T$ must equal the payoff of the option. Second, the portfolio must be *self-financing*, in the sense that $V_{t+1} = (1+r) B_t + S_{t+1} \Delta_t$.
    2. The no-arbitrage condition guarnatees the existence of an *equivalent martingale measure* $Q$ (also called a *risk-neutral measure*) such that the discounted price process of the option is a martingale.

=== "Trinomial pricing models"
    ```python
    --8<-- "trinomial-pricing.py"
    ```

    1. Trinomial pricing models are examples of incomplete markets, which means that multiple equivalent martingale measures (EMMs) exist. In SigAlg, the the EMMs are parametrized by a real number $\theta \in (0,1)$. The discounted price process of the asset is a martingale with respect to every EMM $Q$.
    3. American option contracts allow early exercise, so their risk-neutral valuation turns into an optimal stopping problem. The price process of an American option with respect to an EMM is not a martingale, but rather a *super*martingale.
    3. SigAlg can compute the optimal stopping time $\tau$ for an early-exercise option.
    4. If we evaluate the discounted price process $(1+r)^{-t}X_t$ of the American option at the optimal stopping time $\tau$, we get a random variable $(1+r)^{-\tau}X_\tau$ whose expectation is the initial price of the option.

=== "Markov chain"
    ```python
    --8<-- "markov-chain.py"
    ```

    1. The joint distribution $P$ of a Markov chain factors as $P(X_0=x_0,\ldots,X_T=x_T) = P(X_0=x_0) \prod_{k=0}^{T-1} P(X_{k+1}=x_{k+1} \mid X_{k} = x_{k})$.

=== "Polynomial regression in an $L^2$-space"
    ```python
    --8<-- "polynomial-regression.py"
    ```

    1. Define the Hilbert space $\mathcal{H} = L^2(\Omega, \mathcal{F},P)$.
    2. Define a random variable $X$ on a sample space of cardinality $20$, and another random variable $Y = 2+3X-X^2 + \epsilon$ where $\epsilon$ is a random noise term.
    3. We will project $Y$ onto the subspace spanned by $1$, $X$, and $X^2$ in an $L^2$-space.
    4. Get the coefficients from the orthogonal projection, and plot the fit.

</div>

</div>











