---
title: Code Snippets
---

# Code Snippets

The following code snippets demonstrate how to use the basic objects and methods of SigAlg. This page is not yet comprehensive, so the user will need to inspect the [API reference](../api/index.md) for additional code examples. More code snippets will be added to this page as they are written.

It is also worth checking out the [extended introduction to SigAlg](https://johnmyers-phd.com/writings.html#category=SigAlg){target="_blank"}.

## Sample spaces

### Creating sample spaces

API References: [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}

=== "create_sample_space.py"
    ```python
    --8<-- "create_sample_space.py"
    ```

    1. Create a sample space $\Omega = \{H, T\}$ from a Python list.
    2. Create a sample space $\Omega = \{1, 2, 3, 4, 5, 6\}$ using the `from_sequence` method.
    3. Create a sample space $\Omega = \{\omega_0, \omega_1, \omega_2, \omega_3\}$ using the `from_sequence` method with a prefix.
    4. Create a sample space $\Omega = \{\text{red}, \text{green}, \text{blue}\}$ from a `pd.Index` with the `from_pandas` method.

=== "Output"
    ```
    --8<-- "create_sample_space_output.txt"
    ```

### Extracting events

API References: [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}

=== "extract_event.py"
    ```python
    --8<-- "extract_event.py"
    ```
    
    1. Create a sample space $\Omega = \{\omega_1, \omega_2, \omega_3, \omega_4, \omega_5\}$.
    2. Extract the event $A=\{\omega_1, \omega_2, \omega_3\}$ using the `get_event` method.
    3. Extract the event $B=\{\omega_3, \omega_4, \omega_5\}$ by (positional-based) slicing.
    4. Extract the event $C=\{\omega_1, \omega_4\}$ by (positional-based) indexing.

=== "Output"
    ```
    --8<-- "extract_event_output.txt"
    ```

### Creating probability spaces

API References: [`ProbabilityMeasure`](../api/core.md#sigalg.core.ProbabilityMeasure){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}

=== "create_prob_space.py"
    ```python
    --8<-- "create_prob_space.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3\}$.
    2. Create a $\sigma$-algebra $\mathcal{F}$ on $\Omega$ with atoms $A_0 = \{0,2\}$ and $A_1 = \{1,3\}$.
    3. Create a probability measure $P$ on $\Omega$ with $P(\{\omega\}) = \begin{cases} 0.1 & \text{if } \omega = 0 \\ 0.2 & \text{if } \omega = 1 \\ 0.4 & \text{if } \omega = 2 \\ 0.3 & \text{if } \omega = 3 \end{cases}$
    4. Create a probability space $(\Omega, \mathcal{F}, P)$.

=== "Output"
    ```
    --8<-- "create_prob_space_output.txt"
    ```

### Accessing underlying data

API References: [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}

=== "sample_space_data.py"
    ```python
    --8<-- "sample_space_data.py"
    ```

    1. Create a sample space $\Omega = \{s_0,s_1,s_2,s_3,s_4\}$.
    2. Access the underlying data of the sample space as a `pd.Index` using the `data` attribute.

=== "Output"
    ```
    --8<-- "sample_space_data_output.txt"
    ```

## Events

### Set operations

API References: [`Event`](../api/core.md#sigalg.core.Event){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}

=== "set_operations.py"
    ```python
    --8<-- "set_operations.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3,4\}$.

=== "Output"
    ```
    --8<-- "set_operations_output.txt"
    ```

### Order operations

API References: [`Event`](../api/core.md#sigalg.core.Event){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}

=== "order_operations.py"
    ```python
    --8<-- "order_operations.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3,4\}$.

=== "Output"
    ```
    --8<-- "order_operations_output.txt"
    ```


## Event spaces

### Creating event spaces

API References: [`EventSpace`](../api/core.md#sigalg.core.EventSpace){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}

=== "create_event_space.py"
    ```python
    --8<-- "create_event_space.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3\}$.
    2. Create a $\sigma$-algebra $\mathcal{F}$ on $\Omega$ with atoms $A_0 = \{0,1\}$, $A_1 = \{2\}, A_2 = \{3\}$.
    3. Create an event space $(\Omega, \mathcal{F})$.
    4. The sample space $\Omega$ and $\sigma$-algebra $\mathcal{F}$ are accessible as attributes of the event space.
    5. Define a new $\sigma$-algebra $\mathcal{G}$.
    6. The `sigma_algebra` attribute of the event space is settable, so we can replace $\mathcal{F}$ with $\mathcal{G}$.

=== "Output"
    ```
    --8<-- "create_event_space_output.txt"
    ```


### Event space inherited methods

API References: [`EventSpace`](../api/core.md#sigalg.core.EventSpace){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}

=== "event_space_methods.py"
    ```python
    --8<-- "event_space_methods.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3\}$.
    2. Create a $\sigma$-algebra $\mathcal{F}$ on $\Omega$ with atoms $A_0 = \{0,1\}$, $A_1 = \{2\}, A_2 = \{3\}$.
    3. Create an event space $(\Omega, \mathcal{F})$
    4. The `EventSpace` inherits the method `get_event` from `SampleSpace`.
    5. The `EventSpace` inherits the method `is_measurable` from `SigmaAlgebra`. The event $A$ is *not* measurable, since it is not a union of atoms, but the event $B$ is measurable, since it is a union of atoms.


=== "Output"
    ```
    --8<-- "event_space_methods_output.txt"
    ```

## Probability spaces

### Creating probability spaces

API References: [`ProbabilityMeasure`](../api/core.md#sigalg.core.ProbabilityMeasure){target="_blank"}, [`ProbabilitySpace`](../api/core.md#sigalg.core.ProbabilitySpace){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}

=== "create_probability_space.py"
    ```python
    --8<-- "create_probability_space.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3\}$.
    2. Create a $\sigma$-algebra $\mathcal{F}$ on $\Omega$ with atoms $A_0 = \{0,1\}$, $A_1 = \{2\}, A_2 = \{3\}$.
    3. Create a probability measure $P$ on $\Omega$ with $P(\{\omega\}) = \begin{cases} 0.1 & \text{if } \omega = 0 \\ 0.2 & \text{if } \omega = 1 \\ 0.4 & \text{if } \omega = 2 \\ 0.3 & \text{if } \omega = 3 \end{cases}$
    4. Create a probability space $(\Omega, \mathcal{F}, P)$.
    5. The sample space $\Omega$, $\sigma$-algebra $\mathcal{F}$, and probability measure $P$ are accessible as attributes of the probability space.
    6. Define a new $\sigma$-algebra $\mathcal{G}$.
    7. Define a new probability measure $Q$ on $\Omega$.
    8. The `sigma_algebra` and `probability_measure` attributes of the probability space are settable, so we can replace $\mathcal{F}$ with $\mathcal{G}$ and $P$ with $Q$.
    

=== "Output"
    ```
    --8<-- "create_probability_space_output.txt"
    ```

### Probability space inherited methods

API References: [`ProbabilitySpace`](../api/core.md#sigalg.core.ProbabilitySpace){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}, [`ProbabilityMeasure`](../api/core.md#sigalg.core.ProbabilityMeasure){target="_blank"}

=== "probability_space_methods.py"
    ```python
    --8<-- "probability_space_methods.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3\}$.
    2. Create a probability space $(\Omega, \mathcal{F}, P)$, with default $\sigma$-algebra $\mathcal{F}$, the power set of $\Omega$, and default probability measure $P$, the uniform distribution on $\Omega$.
    3. The `ProbabilitySpace` inherits the method `get_event` from `SampleSpace`.
    4. The `ProbabilitySpace` inherits the method `is_measurable` from `SigmaAlgebra`.
    5. The `ProbabilitySpace` inherits the method `P` from `ProbabilityMeasure`, which computes the probability of an event.

=== "Output"
    ```
    --8<-- "probability_space_methods_output.txt"
    ```

<!-- ## Time

### Creating time

API References: [`Time`](../api/core.md#sigalg.core.Time){target="_blank"}

=== "create_time.py"
    ```python
    --8<-- "create_time.py"
    ```

    1. Create a discrete time index $T = \{1, 2, 3, 4\}$ using the `discrete` class method, along with the `start` and `stop` parameters.
    2. Create a discrete time index $T = \{1, 2, 3, 4, 5, 6\}$ using the `discrete` class method, along with the `start` and `length` parameters. Bear in mind that the latter parameter is the duration of time spanned by the index, **not** the number of time points in the index.

=== "Output"
    ```
    --8<-- "create_time_output.txt"
    ``` -->

## $L^2$-spaces

### Creating $L^2$-spaces

API References: [`L2`](../api/l2.md#sigalg.l2.L2){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}, [`ProbabilityMeasure`](../api/core.md#sigalg.core.ProbabilityMeasure){target="_blank"}

=== "create_l2_space.py"
    ```python
    --8<-- "create_l2_space.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2,3\}$.
    2. Create a $\sigma$-algebra $\mathcal{F}$ on $\Omega$.
    3. Create a probability measure $P$ on $\Omega$.
    4. Create the space $H = L^2(\Omega, \mathcal{F}, P)$.
    5. Define two random variables $X,Y: \Omega \to \mathbb{R}$.
    6. The random variable $X$ is constant on the atoms of $\mathcal{F}$, therefore it is $\mathcal{F}$-measurable, so it is in $H$.
    7. The random variable $Y$ is not constant on the atoms of $\mathcal{F}$, therefore it is not $\mathcal{F}$-measurable, so it is not in $H$.

=== "Output"
    ```
    --8<-- "create_l2_space_output.txt"
    ```

### Bases of $L^2$-spaces

API References: [`L2`](../api/l2.md#sigalg.l2.L2){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}, [`ProbabilityMeasure`](../api/core.md#sigalg.core.ProbabilityMeasure){target="_blank"}

=== "l2_basis.py"
    ```python
    --8<-- "l2_basis.py"
    ```

    1. Create a sample space $\Omega = \{0,1,2\}$.
    2. Create a $\sigma$-algebra $\mathcal{F}$ on $\Omega$.
    3. Create a probability measure $P$ on $\Omega$.
    4. Create the space $H = L^2(\Omega, \mathcal{F}, P)$.
    5. The basis consists of normalized indicator functions of the atoms of $\mathcal{F}$. This is an orthonormal basis of $H$.
    6. Define a new probability measure $Q$ on $\Omega$ that assigns zero probability to one of the atoms of $\mathcal{F}$.
    7. Change the probability measure of the $L^2$-space to $Q$, so that now $H = L^2(\Omega, \mathcal{F}, Q)$.
    8. The basis is updated to reflect the change in the probability measure, so the indicator function of the atom with zero probability is removed from the basis. The $L^2$-space is only $1$-dimensional under $Q$.

=== "Output"
    ```
    --8<-- "l2_basis_output.txt"
    ```

### Polynomial regression with $L^2$-spaces

API References: [`L2`](../api/l2.md#sigalg.l2.L2){target="_blank"}, [`SampleSpace`](../api/core.md#sigalg.core.SampleSpace){target="_blank"}, [`SigmaAlgebra`](../api/core.md#sigalg.core.SigmaAlgebra){target="_blank"}, [`ProbabilityMeasure`](../api/core.md#sigalg.core.ProbabilityMeasure){target="_blank"}

=== "polynomial_regression.py"
    ```python
    --8<-- "polynomial_regression.py"
    ```

    1. The data consists of $159$ pairs $(x,y)$ of real numbers. We want to find a cubic polynomial that fits the data well.
    2. Load the data into a `RandomVector` object using the `from_numpy` method.
    3. The sample space $\Omega$ and probability meausure $P$ are automatically created; $\Omega$ consists of the numbers $0,1,\ldots,158$, and $P$ is the uniform distribution on $\Omega$.
    4. Extract the component random variables $X$ and $Y$ from the random vector $Z=(X,Y)$.
    5. Create the $L^2$-space $H = L^2(\Omega, \mathcal{F}, P)$, where $\mathcal{F}$ is the default $\sigma$-algebra on $\Omega$, the power set.
    6. Perform an orthogonal projection of $Y$ onto the subspace of cubic polynomials in $X$. The coefficients of the best-fit polynomial are stored in a `np.ndarray` object `u`.
    7. Extract the coefficients from `u` and create the best-fit polynomial.
    8. Plot the data and the fitted polynomial.


=== "Output"
    ![Regression Plot](./scripts/polynomial_regression.png){width=50%}