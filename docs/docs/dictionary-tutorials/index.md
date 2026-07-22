---
title: dictionary & tutorials
---

## Dictionary

The dictionary below provides a quick reference for translating between mathematical concepts and their SigAlg implementations. Each entry maps a mathematical object from measure-theoretic probability to its corresponding class or method in SigAlg, including a link to the relevant API documentation and interactive notebooks demonstrating basic usage that may be run in GitHub or Google Colab. For more in-depth examples and end-to-end tutorials, see the [Tutorials](../tutorials/index.md#tutorials) section below.

=== "Sample spaces & events"

    | Mathematical object | SigAlg class/method | Notebook |
    |:---------|:--------|:--------:|
    | Sample space | [`SampleSpace`](../api/modules/core.md#sigalg.core.SampleSpace){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/sample_space.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/sample_space.ipynb){target="_blank"}
    | Event | [`Event`](../api/modules/core.md#sigalg.core.Event){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/event.ipynb){target="_blank"}  [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/event.ipynb){target="_blank"}
    | Event space | [`MeasurableSpace`](../api/modules/core.md#sigalg.core.MeasurableSpace){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/measurable_space.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/measurable_space.ipynb){target="_blank"}

=== "$\sigma$-algebras"

    | Mathematical object | SigAlg class/method | Notebook |
    |:---------|:--------|:--------:|
    | $\sigma$-algebra | [`SigmaAlgebra`](../api/modules/core.md#sigalg.core.SigmaAlgebra){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/sigma_algebra.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/sigma_algebra.ipynb){target="_blank"}
    | Filtered $\sigma$-algebra | [`Filtration`](../api/modules/core.md#sigalg.core.Filtration){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/filtration.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/filtration.ipynb){target="_blank"}

=== "Probability spaces"

    | Mathematical object | SigAlg class/method | Notebook |
    |:---------|:--------|:--------:|
    | Probability measure | [`ProbabilityMeasure`](../api/modules/core.md#sigalg.core.ProbabilityMeasure){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/prob_measure.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/prob_measure.ipynb){target="_blank"}
    | Probability space | [`ProbabilitySpace`](../api/modules/core.md#sigalg.core.ProbabilitySpace){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/prob_space.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/prob_space.ipynb){target="_blank"}

=== "Random variables & vectors"

    | Mathematical object | SigAlg class/method | Notebook |
    |:---------|:--------|:--------:|
    | Random variables and vectors | [`RandomVariable`](../api/modules/core.md#sigalg.core.RandomVariable){target="_blank"}, [`RandomVector`](../api/modules/core.md#sigalg.core.RandomVector){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/random_vector.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/random_vector.ipynb){target="_blank"}
    | Pushforward (or image) measure | [`Operators.pushforward`](../api/modules/core.md#sigalg.core.Operators.pushforward){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/pushforward.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/pushforward.ipynb){target="_blank"}
    | Lebesgue integral | [`Operators.integrate`](../api/modules/core.md#sigalg.core.Operators.integrate){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/integrate.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/integrate.ipynb){target="_blank"}
    | Expectation | [`Operators.expectation`](../api/modules/core.md#sigalg.core.Operators.expectation){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/expectation.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/expectation.ipynb){target="_blank"}
    | Variance | [`Operators.variance`](../api/modules/core.md#sigalg.core.Operators.variance){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/variance.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/variance.ipynb){target="_blank"}
    | Standard deviation | [`Operators.std`](../api/modules/core.md#sigalg.core.Operators.std){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/std.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/std.ipynb){target="_blank"}
    | Covariance | [`Operators.cov`](../api/modules/core.md#sigalg.core.Operators.cov){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/cov.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/cov.ipynb){target="_blank"}
    | Correlation | [`Operators.corr`](../api/modules/core.md#sigalg.core.Operators.corr){target="_blank"} | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/corr.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/corr.ipynb){target="_blank"}
    
=== "$L^2$-spaces"

    | Mathematical object | SigAlg class/method | Notebook |
    |:---------|:--------|:--------:|
    | $L^2$-space | [`L2`](../api/modules/l2.md#sigalg.l2.L2) | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/l2.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/l2.ipynb){target="_blank"}
    | Inner product | [`L2.inner`](../api/modules/l2.md#sigalg.l2.L2.inner) | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/inner.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/inner.ipynb){target="_blank"}
    | Norm | [`L2.norm`](../api/modules/l2.md#sigalg.l2.L2.norm) | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/norm.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/norm.ipynb){target="_blank"}
    | Metric | [`L2.metric`](../api/modules/l2.md#sigalg.l2.L2.metric) | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/metric.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/metric.ipynb){target="_blank"}
    | Orthogonal projection | [`L2.proj`](../api/modules/l2.md#sigalg.l2.L2.proj) | [:material-github:](https://github.com/jmyers7/sigalg/blob/dev/docs/notebooks/proj.ipynb){target="_blank"} [:simple-googlecolab:](https://colab.research.google.com/github/jmyers7/sigalg/blob/dev/docs/notebooks/proj.ipynb){target="_blank"}

=== "Stochastic processes"

=== "Finance"

## Tutorials

Placeholder text here.