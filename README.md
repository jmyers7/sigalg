
# sigalg

A Python package for working with stochastic processes, probability theory, *etc*.

**⚠️ Work in Progress**: This package is actively under development. New classes and features are being added regularly. The API may change as the project evolves.

This package is primarily used for my research and writing at [johnmyers-phd.com](https://johnmyers-phd.com).

## Features

- **Discrete-Time Stochastic Processes**
  - Independent and Identically Distributed (IID) processes
  - First-order Markov chains
  - Second-order Markov chains
  - Process transformations (cumulative sums, differences, etc.)

- **Sample Space Generation**
  - Enumerate complete sample spaces for small trajectory lengths
  - Compute joint probabilities for sequences

- **Simulation**
  - Trajectory simulation

- **Visualization**
  - Built-in plotting for trajectories
  - Color gradient support for multiple trajectories

- **Conditional Expectations**
  - Compute E(Y | σ-algebra) for random variables $Y$ and conditioning sigma-algebras

## Installation

Currently, you will need to install from source:

```bash
git clone https://github.com/jmyers7/sigalg.git
cd sigalg
pip install -e .
```

## Quick Start

### IID Process

```python
import numpy as np
from sigalg.StochasticProcesses import DiscreteIID

# Create a fair coin flip process
prob = np.array([0.5, 0.5])  # P(tails), P(heads)
coin_flips = DiscreteIID(prob=prob, trajectory_length=10)

# Generate sample space (for small trajectory lengths)
coin_flips.generate_sample_space()
print(coin_flips.omega)

# Simulate trajectories
coin_flips.simulate(num_trajectories=5)
print(coin_flips.trajectories)
```

### First-Order Markov Chain

```python
import numpy as np
from sigalg.StochasticProcesses import FirstOrderMarkovChain

# Define transition matrix and initial probabilities
transition_matrix = np.array([
    [0.7, 0.3],  # P(state 0 -> 0), P(state 0 -> 1)
    [0.4, 0.6]   # P(state 1 -> 0), P(state 1 -> 1)
])
init_prob = np.array([0.5, 0.5])

# Create Markov chain
mc = FirstOrderMarkovChain(
    transition_matrix=transition_matrix,
    init_prob=init_prob,
    trajectory_length=20
)

# Simulate and plot
mc.simulate(num_trajectories=10)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
mc.plot_simulations(ax=ax)
plt.show()
```

### Conditional Expectations

```python
import numpy as np
from sigalg.StochasticProcesses import FirstOrderMarkovChain, conditional_exp

# Create a Markov chain modeling coin flips with momentum
init_prob = np.array([0.5, 0.5])
transition_matrix = np.array([[0.7, 0.3], [0.2, 0.8]])
mc = FirstOrderMarkovChain(transition_matrix, init_prob)

# Generate the complete sample space
mc.generate_sample_space(trajectory_length=4)

# Define a random variable S3 = X1 + X2 + X3
def S3(omega):
    return omega["X1"] + omega["X2"] + omega["X3"]

# Compute E(S3 | X1, X2) - the conditional expectation given first two flips
cond_exp = conditional_exp(
    omega=mc.omega,
    RV=S3,
    sigma_algebra=["X1", "X2"],
    name="E(S3 | X1, X2)"
)

print(cond_exp)
#    X1  X2  E(S3 | X1, X2)
# 0   0   0             0.3
# 1   0   1             1.8
# 2   1   0             1.3
# 3   1   1             2.8
```

### Process Transformations

```python
from sigalg.StochasticProcesses import transform_discrete_process

# Create cumulative sum process
cumulative_heads = transform_discrete_process(
    coin_flips,
    lambda df: df.cumsum(axis=1)
)

# Simulate the original process first
coin_flips.simulate(trajectory_length=50, num_trajectories=1)

# Then transform
cumulative_heads.simulate()
cumulative_heads.plot_simulations(ax=ax)
```

### Custom Column Naming

```python
# Use S for state variables instead of X
mc.simulate(
    trajectory_length=10,
    num_trajectories=5,
    initial_time=0,  # Start at S₀ instead of S₁
    column_prefix="S"
)
print(mc.trajectories)  # Columns: S0, S1, S2, ...
```

## API Reference

### Core Classes

- `StochasticProcess` - Abstract base class for all stochastic processes
- `DiscreteTimeStochasticProcess` - Base class for discrete-time processes
- `DiscreteTimeStochasticProcessWithProb` - Discrete-time processes with probability measures
- `DiscreteIID` - Independent and identically distributed processes
- `FirstOrderMarkovChain` - First-order Markov chains
- `SecondOrderMarkovChain` - Second-order Markov chains
- `TransformedDiscreteTimeStochasticProcess` - Transformed processes

### Factory Functions

- `transform_discrete_process(process, transform_func)` - Create a transformed process

### Utility Functions

- `conditional_exp(omega, RV, sigma_algebra, name)` - Compute conditional expectations

### Key Methods

All stochastic processes support:

- `simulate(trajectory_length, num_trajectories, initial_time, column_prefix)` - Generate trajectories
- `plot_simulations(ax, colors, simulation_kwargs, plot_kwargs)` - Visualize trajectories

Processes with probability measures also support:

- `generate_sample_space(trajectory_length, initial_time, column_prefix)` - Enumerate sample space
- `joint_prob(X)` - Compute joint probability of a sequence

## Contact

- Author: John Myers
- Website: [johnmyers-phd.com](https://johnmyers-phd.com)
