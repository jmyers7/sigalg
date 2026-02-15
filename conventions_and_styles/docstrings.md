# Docstring style and conventions

## Structure

- All docstrings are written in the NumPy style.

### Module docstrings

- Every module contains a docstring at the top of the file, with the following sections in order:

    1. A first sentence simply and briefly stating the content of the module.

    2. An extended description providing more details about the module, including the main classes and functions defined in the module.

    3. A "Classes" section listing the classes defined in the module, along with a one-line description of each class.

    4. A "Functions" section listing the functions defined in the module, along with a one-line description of each function.

- Module docstrings do not contain an "Examples" section.



### Class and function docstrings

- Every class and function contains a docstring, with the following sections in order:
    
    1. A one-line summary of the object or function.
    
    2. An optional extended description providing more details about the object or function.
    
    3. A "Parameters" section listing the parameters of a function, including their types and descriptions.
    
    4. An optional "Raises" section listing any exceptions that a function may raise, including their types and descriptions.
    
    5. A "Returns" section describing the return value of a function.
    
    6. An optional "Examples" section providing usage examples for the object or function.

## Style

### Typographical style

- Words in docstrings that refer to code should be wrapped in backticks (e.g. `SampleSpace` or `probability_measure`) to indicate that they are code references.

- The docstrings of objects that model key mathematical objects should include a mathematical definition of the object in the extended description. LaTeX is required for these types of descriptions, and definitions of technical words should be wrapped in italics, much as they would be in a mathematical text. **The one exception is that the first sentence of a docstring should not contain any LaTeX.**

### Parameter and return value style

- If a parameter is optional and has a default value of `None`, the type of the parameter should be listed as `type | None, default=None`.

- If a parameter is a list, it must be written as `list[type]` rather than `list of type`.

- If a function or method returns a value, the return value must be named, and its type must be specified in the "Returns" section. Return values *must* be named, even if this name does not appear in the implementation of the function.

## Examples

### Example of module docstring:

```python
"""Classes for modeling sample spaces in probability theory.

This module provides the `SampleSpace` class, which models the indices or labels of all possible outcomes in a random experiment. A mixin class is also provided for other classes that contain a `sample_space` attribute, allowing them to delegate sample space operations.

Classes
-------
SampleSpace
    Represents a sample space as a collection of outcomes.
SampleSpaceMethods
    Mixin providing sample space methods to other classes.
"""
```

### Example of class docstring:

```python
r"""A class representing a sample space.

    Mathematically, a *sample space* is simply a nonempty set $\Omega$. In probability theory, sample spaces are used to model the set of all possible outcomes of a random experiment. Each element $\omega$ of a sample space $\Omega$ is called a *sample point* or *outcome*.

    In SigAlg, an instance of `SampleSpace` is intended to contain the indices or labels of sample points. In particular, an instance of `SampleSpace` is *not* intended to contain data. Data should be represented as an instance of `RandomVariable` or `RandomVector`, which are defined on a sample space.

    Parameters
    ----------
    name : Hashable | None, default="Omega"
        Name identifier for the sample space.
    data_name : Hashable | None, default="sample"
        Name for the internal `pd.Index`.

    Examples
    --------
    >>> from sigalg.core import SampleSpace
    >>> import pandas as pd
    >>> # Construction with list
    >>> Omega_1 = SampleSpace(name="Omega_1").from_list(["red", "green", "blue"])
    >>> Omega_1 # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega_1':
    ['red', 'green', 'blue']
    >>> # Construction with pd.Index
    >>> data = pd.Index(["a", "b", "c"], name="sample")
    >>> Omega_2 = SampleSpace(name="Omega_2").from_pandas(data=data)
    >>> Omega_2 # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega_2':
    ['a', 'b', 'c']
    """
```