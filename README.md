# sigalg

*A Python framework for finite measure-theoretic probability.*

```python
import sigalg as sa
import pandas as pd

state_space = [0, 1]
time = sa.DiscreteTime.from_list([1, 2, 3])
sample_space = sa.SampleSpace.create_from_sequences(state_space, time)
print(sample_space)

# time      1  2  3
# sequence         
# omega1    0  0  0
# omega2    0  0  1
# omega3    0  1  0
# omega4    0  1  1
# omega5    1  0  0
# omega6    1  0  1
# omega7    1  1  0
# omega8    1  1  1
```

```python
sample_points = ["omega2", "omega4", "omega6"]
event = sa.Event(sample_space, sample_points)
print(event)

# time      1  2  3
# sequence         
# omega2    0  0  1
# omega4    0  1  1
# omega6    1  0  1
```


```python
def measure_function(event: sa.Event) -> float:
    total_heads = event.sum(axis=1)
    total_tails = len(event.columns) - total_heads
    probabilities = (0.75**total_heads) * (0.25**total_tails)
    return probabilities.sum()

prob_measure = sa.ProbabilityMeasure(sample_space, measure_function)
print("Probability of event:", prob_measure(event))
print("Probability of sample space:", prob_measure(sample_space))

# Probability of event: 0.328125
# Probability of sample space: 1.0
```


```python
probability_space = sample_space.add_probability_measure(prob_measure)
print(type(probability_space))

# <class 'sigalg.spaces.probability_space.ProbabilitySpace'>
```

```python
print("Probability of event:", probability_space.prob_measure(event))

# Probability of event: 0.328125
```

```python
print(pd.concat([probability_space, probability_space.probabilities], axis=1))

#           1  2  3  probability
# sequence                      
# omega1    0  0  0     0.015625
# omega2    0  0  1     0.046875
# omega3    0  1  0     0.046875
# omega4    0  1  1     0.140625
# omega5    1  0  0     0.046875
# omega6    1  0  1     0.140625
# omega7    1  1  0     0.140625
# omega8    1  1  1     0.421875
```

<!-- ```python
def X_function(row: pd.Series) -> float:
    numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
    return numerical_row.sum()
X = sa.RandomVariable(sample_space=probability_space,rv_function=X_function)

atom_labels = pd.Series(
    index=probability_space.index,
    data=[0, 0, 1, 1, 2, 2, 3, 3],
    name="atom_id",
)

F = sa.SigmaAlgebra(probability_space, atom_labels=atom_labels)

X.values.name = "X"

cond_exp = X.expectation(given=F).values
cond_exp.name = "E[X|F]"

print(
    pd.concat(
        [
            probability_space,
            probability_space.probabilities,
            F.atom_labels,
            X.values,
            cond_exp,
        ],
      axis=1,
)
# output:
#           0  1  2  probability  atom_id  X  E[X|F]
# sequence
# omega1    T  T  T     0.015625        0  0    0.75
# omega2    T  T  H     0.046875        0  1    0.75
# omega3    T  H  T     0.046875        1  1    1.75
# omega4    T  H  H     0.140625        1  2    1.75
# omega5    H  T  T     0.046875        2  1    1.75
# omega6    H  T  H     0.140625        2  2    1.75
# omega7    H  H  T     0.140625        3  2    2.75
# omega8    H  H  H     0.421875        3  3    2.75


``` -->

## Core abstractions

| Mathematical object | `sigalg` class | Meaning | Typical operations |
|-|-|-|-|
| Sample space $\Omega$ | `SampleSpace`| The set of all possible outcomes of a random event. | create from sequences, add probability measures |
| $\sigma$-algebra $\mathcal{F}$ | `SigmaAlgebra` | In the finite case: a partition of $\Omega$ into atoms. | join, meet, test inclusion|
| Event $A \in \mathcal{F}$ | `Event` | A measurable subset of $\Omega$. | union, intersection, complement, check measurability |
| Probability measure $P:\mathcal{F}\to [0,1]$ | `ProbabilityMeasure` | Assigns probabilities to events. | compute probabilities of events, integrate random variables (i.e., compute expectations) |
| Probability space $(\Omega, \mathcal{F}, P)$ | `ProbabilitySpace` | Combines a sample space, $\sigma$-algebra, and measure into a single object. | query event probabilities |
| Random variable $X : \Omega \to \mathbb{R}$ | `RandomVariable` | A measurable function defined on the sample space. Its induced $\sigma$-algebra can be computed explicitly. | compute values, compute (conditional) expectations, check measurability |
| Time $t$ | `Time` | Provides an ordered index for stochastic processes and filtrations. Comes in either discrete or continuous flavors. | slice trajectories, define adapted processes |
| Filtered $\sigma$-algebra $(\mathcal{F},\{\mathcal{F}_t\}_{t \in T})$ | `FilteredSigmaAlgebra` | A $\sigma$-algebra $\mathcal{F}$ equipped with a nested sequence of $\sigma$-algebras $\mathcal{F}_t$ indexed by time $t$ and increasing toward $\mathcal{F}$. | slice in time|
| Stochastic process $\{X_t\}_{t \in T}$ | `StochasticProcess` | A collection of random variables indexed by time. | simulate trajectories |
