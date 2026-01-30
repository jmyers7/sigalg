# SigAlg

[![codecov](https://codecov.io/gh/jmyers7/sigalg/branch/main/graph/badge.svg?token=DORWUC3G6R)](https://codecov.io/gh/jmyers7/sigalg)

A Python package for finite, measure-theoretic probability theory and stochastic processes. The library emphasizes mathematical fidelity while remaining practical for simulations and numerical experiments.

**This package is under active development. Extensive documentation is coming soon.**

## Installation

```bash
pip install sigalg
```

## Quick Examples

Build and simulate a random walk:

```python
import matplotlib.pyplot as plt

from sigalg.core import Time
from sigalg.processes import RandomWalk

# Create a random walk with 100 discrete time steps
T = Time.discrete(length=100)
X = RandomWalk(p=0.7, time=T)

# Simulate 10 trajectories
X.from_simulation(n_trajectories=10, random_state=42)

# Plot trajectories
_, ax = plt.subplots(figsize=(7, 4))
X.plot_trajectories(ax=ax)
plt.show()
```

Compute conditional expectation of a random variable with respect to a $\sigma$-algebra:

```python
from sigalg.core import (
    Operators,
    ProbabilityMeasure,
    RandomVariable,
    SampleSpace,
    SigmaAlgebra,
)

# Create a sample space with 4 outcomes, labeled 0, 1, 2, 3
Omega = SampleSpace().from_sequence(size=4)

# Define a probability measure by assigning probabilities to each outcome
P = ProbabilityMeasure(sample_space=Omega).from_dict(
    {
        0: 0.2,  # P(0) = 0.2
        1: 0.3,  # P(1) = 0.3
        2: 0.3,  # P(2) = 0.3
        3: 0.2,  # P(3) = 0.2
    }
)

# Define a random variable by assigning values to each outcome
X = RandomVariable(domain=Omega, name="X").from_dict(
    {
        0: 1,  # X(0) = 1
        1: 2,  # X(1) = 2
        2: 8,  # X(2) = 8
        3: 3,  # X(3) = 3
    }
)

# Sigma-algebras are defined by partitioning the sample space into atoms
F = SigmaAlgebra(sample_space=Omega, name="F").from_dict(
    {
        0: "A",  # outcome 0 is in atom A
        1: "A",  # outcome 1 is in atom A
        2: "B",  # outcome 2 is in atom B
        3: "B",  # outcome 3 is in atom B
    }
)

# Compute conditional expectation E(X|F), which is an instance of `RandomVariable`
E_X_F = Operators.expectation(rv=X, sigma_algebra=F, probability_measure=P)
print(E_X_F)
```

## Documentation

Comprehensive documentation with tutorials, API reference, and mathematical background is coming soon.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

[John Myers](https://www.johnmyers-phd.com)
