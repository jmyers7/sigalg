"""A class representing a random vector."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterator, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn
from .operators import OperatorsMethods

if TYPE_CHECKING:
    from ...processes.base.stochastic_process import StochasticProcess
    from ..base.event import Event
    from ..base.feature_vector import FeatureVector
    from ..base.index import Index
    from ..base.probability_space import ProbabilitySpace
    from ..base.sample_space import SampleSpace
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..random_objects.random_variable import RandomVariable
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class RandomVector(OperatorsMethods):
    r"""A class representing a random vector.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : SampleSpace | None, default=None
        The sample space of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space.
    index : Index | None, default=None
        The index of the random vector.
    name : Hashable, default="X"
        The name of the random vector.
    **kwargs
        Additional keyword arguments for subclass constructors.

    Raises
    ------
    TypeError
        If `index` is not an `Index` (if given), or if `name` is not a `Hashable`.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core import (
    ...     EventSpace,
    ...     ProbabilitySpace,
    ...     ProbabilityMeasure,
    ...     RandomVector,
    ...     SampleSpace,
    ...     SigmaAlgebra,
    ... )
    >>> # Generate a 2-dimensional random vector on a pre-existing sample space — the power-set sigma-algebra and uniform probability measure are automatically generated
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> X = RandomVector(Omega, name="X").from_dict(
    ...     {
    ...         0: (1, 1),
    ...         1: (1, 1),
    ...         2: (2, 2),
    ...     }
    ... )
    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
    Random vector 'X':
    X        X_0  X_1
    Omega
    0          1    1
    1          1    1
    2          2    2
    >>> print(X.sig_alg) # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'power_set':
          power_set
    Omega
    0             0
    1             1
    2             2
    >>> print(X.prob_measure) # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
            probability
    Omega
    0          0.333333
    1          0.333333
    2          0.333333
    >>> # Generate a random vector on a pre-existing event space — a uniform probability measure is automatically generated
    >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
    ...     {
    ...         0: 0,  # Atom A_0 = {0, 1}
    ...         1: 0,  # Atom A_0 = {0, 1}
    ...         2: 1,  # Atom A_1 = {2}
    ...     }
    ... )
    >>> event_space = EventSpace(Omega, F)
    >>> Y = RandomVector(*event_space, name="Y").from_dict(
    ...     {
    ...         0: (1, 1),  # <- Constant on atom A_0 = {0, 1}
    ...         1: (1, 1),  # <- Constant on atom A_0 = {0, 1}
    ...         2: (2, 2),  # <- Constant on atom A_1 = {2}
    ...     }
    ... )
    >>> print(Y.sig_alg) # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
                  F
    Omega
    0             0
    1             0
    2             1
    >>> print(Y.prob_measure) # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
            probability
    F
    0          0.666667
    1          0.333333
    >>> # Generate a random vector on a pre-existing probability space
    >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
    ...     {
    ...         0: 0.5,
    ...         1: 0.5,
    ...     }
    ... )
    >>> prob_space = ProbabilitySpace(Omega, F, P)
    >>> Z = RandomVector(*prob_space, name="Z").from_dict(
    ...     {
    ...         0: (1, 1),
    ...         1: (1, 1),
    ...         2: (2, 2),
    ...     }
    ... )
    >>> print(Z.sig_alg) # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
                  F
    Omega
    0             0
    1             0
    2             1
    >>> print(Z.prob_measure) # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
            probability
    F
    0               0.5
    1               0.5
    >>> # Attempt to define a random vector that is not F-measurable
    >>> W = RandomVector(*prob_space, name="W").from_dict(
    ...     {
    ...        0: (1, 2),  # <- Not constant on atom A_0 = {0, 1}
    ...        1: (3, 4),  # <- Not constant on atom A_0 = {0, 1}
    ...        2: (5, 6),
    ...     }
    ... ) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: Random vector W is not measurable.

    Notes
    -----
    Given a probability space $(\Omega,\mathcal{F},P)$, a *random vector* is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}^d$, where $d$ is the *dimension* of the vector and $\mathbb{R}^d$ is equipped with its Borel $\sigma$-algebra. The image $X(\omega)\in \mathbb{R}^d$ of a sample point $\omega \in \Omega$ is called a *feature vector*.

    If $\Omega$ is finite (as it always is, in SigAlg), so that $\mathcal{F}$ is determined by its atoms, then $X$ is $\mathcal{F}$-measurable if and only if $X$ is constant on the atoms of $\mathcal{F}$.
    """

    # --------------------- constructors --------------------- #

    _properties = [
        "_point_outputs",
        "_atom_outputs",
        "_data",
        "_atom_data",
        "_dimension",
        "_components",
        "_generated_sig_alg",
        "_range",
        "_is_identity",
    ]

    def __init__(
        self,
        domain: SampleSpace | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        index: Index | None = None,
        name: Hashable = "X",
        **kwargs,
    ) -> None:
        from ..base.index import Index
        from ..base.probability_space import ProbabilitySpace

        if index is not None and not isinstance(index, Index):
            raise TypeError("If given, index must be an Index.")
        if not isinstance(name, Hashable):
            raise TypeError("Name must be a Hashable.")

        self._prob_space = ProbabilitySpace(
            sample_space=domain,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
        )
        self._index = index
        self._name = name
        self._initialize_property_caches()

    def _initialize_property_caches(self, exceptions: set | None = None) -> None:
        if exceptions is None:
            exceptions = set()
        for property in set(self._properties) - exceptions:
            setattr(self, property, None)

    def from_dict(
        self,
        outputs: Mapping[Hashable, Hashable],
        type: str = "point",
        overwrite_domain: bool = False,
        overwrite_index: bool = False,
    ) -> RandomVector:
        """Create a `RandomVector` from either a dictionary mapping sample points to outputs, or a dictionary mapping atom identifiers to outputs.

        If the `type` parameter is set to `'point'`, the dictionary is interpreted as mapping sample points in the domain to their corresponding output vectors:

        * If the `domain` is provided at construction and the `overwrite_domain` flag is set to `False`, then the keys of the provided dictionary must match the `domain`.
        * If the `domain` is provided at construction and the `overwrite_domain` flag is set to `True`, then a new `domain` is automatically generated from the keys of the provided dictionary, the sigma-algebra is set to the power-set of the new domain, and the probability measure is set to the uniform measure.
        * If the `domain` is not provided at construction, then it is automatically generated from the keys of the provided dictionary, the sigma-algebra is set to the power-set of the new domain, and the probability measure is set to the uniform measure. In this case, the `overwrite_domain` flag is ignored.

        If the `type` parameter is set to `'atom'`, the dictionary is interpreted as mapping atom identifiers of the sigma-algebra to their corresponding output vectors. In this case, the `sig_alg` parameter must be provided at construction, the keys of the provided dictionary must match the atom IDs of the sigma-algebra, and the `overwrite_domain` flag is ignored.

        Additionally, there is an `overwrite_index` parameter. If the dimension of the random vector is 1, then this parameter is ignored. Otherwise:

        * If an `index` is provided at construction and `overwrite_index` is set to `False`, then the dimension of the random vector must match the length of the provided `index`.
        * If an `index` is provided at construction and `overwrite_index` is set to `True`, then `index` is reset to a sequential index using the name of the random vector (if provided).
        * If an `index` is not provided at construction, then `index` is set to a sequential index using the name of the random vector (if provided). In this case, the `overwrite_index` flag is ignored.

        Parameters
        ----------
        outputs : Mapping[Hashable, Hashable]
            A mapping from sample points or atom identifiers to their corresponding output vectors.
        type : str, default="point"
            A string indicating whether the provided dictionary maps sample points (`'point'`) or atom identifiers (`'atom'`) to outputs.
        overwrite_domain : bool, default=False
            A boolean flag indicating whether to overwrite the domain with a new sample space generated from the keys of the provided dictionary. See the description above.
        overwrite_index : bool, default=False
            A boolean flag indicating whether to overwrite an existing `index` provided at construction. See the description above.

        Raises
        ------
        ValueError
            If `type` is not `'point'` or `'atom'`, or if `type` is `'atom'` and `sig_alg` is not provided at construction, or if the provided outputs do not yield a measurable function in the `type='point'` case.

        Returns
        -------
        self : RandomVector
            The constructed `RandomVector` instance.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra().from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega, sig_alg=F).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ...     type="point",
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          3    4
        2          3    4
        >>> Y = RandomVector(domain=Omega, sig_alg=F, name="Y").from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...     },
        ...     type="atom",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        Y        Y_0  Y_1
        Omega
        0          1    2
        1          3    4
        2          3    4
        """
        from ..base.index import Index
        from ..base.probability_space import ProbabilitySpace
        from ..base.sample_space import SampleSpace
        from .random_variable import RandomVariable

        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a dict.")
        if type not in ["point", "atom"]:
            raise ValueError("type must be either 'point' or 'atom'.")
        if type == "atom" and self.sig_alg is None:
            raise ValueError(
                "The sig_alg parameter must be set during construction for the from_dict method with type='atom'."
            )

        self._initialize_property_caches()

        if type == "point" and overwrite_domain:
            self._prob_space = ProbabilitySpace()
        if overwrite_index:
            self._index = None

        v = SampleSpaceMappingIn(
            mapping=outputs,
            sample_space=self.domain if type == "point" else self.sig_alg.atom_space,
        )

        first_output = next(iter(v.mapping.values()))
        self._dimension = len(first_output) if isinstance(first_output, tuple) else 1

        if isinstance(self, RandomVariable) and self.dimension != 1:
            raise ValueError("A random variable must have dimension 1.")

        if type == "point" and self.domain is None:
            self.domain = SampleSpace().from_list(list(v.mapping.keys()))

        if self.dimension > 1:
            if self._index is None:
                self._index = Index(name=self.name).from_sequence(
                    size=self.dimension, prefix=self.name
                )
            if len(self._index) != self.dimension:
                raise ValueError(
                    "Length of index must match the dimension of the RandomVector."
                )
        else:
            self._index = None

        if type == "point":
            self._point_outputs = v.mapping
            if not self.is_measurable():
                raise ValueError(f"Random vector {self.name} is not measurable.")
        else:
            self._atom_outputs = v.mapping

        return self

    def from_pandas(
        self,
        data: pd.Series | pd.DataFrame,
        type: str = "point",
        overwrite_domain: bool = False,
        overwrite_index: bool = False,
    ) -> RandomVector:
        """Create a `RandomVector` from either a `pd.DataFrame` or `pd.Series` mapping sample points to outputs, or a pandas object mapping atom identifiers to outputs.

        If the `type` parameter is set to `'point'`, the pandas object is interpreted as mapping sample points in the domain to their corresponding output vectors:

        * If the `domain` is provided at construction and the `overwrite_domain` flag is set to `False`, then the index of the provided pandas object must match the `domain`.
        * If the `domain` is provided at construction and the `overwrite_domain` flag is set to `True`, then a new `domain` is automatically generated from the index of the provided pandas object, the sigma-algebra is set to the power-set of the new domain, and the probability measure is set to the uniform measure.
        * If the `domain` is not provided at construction, then it is automatically generated from the index of the provided pandas object, the sigma-algebra is set to the power-set of the new domain, and the probability measure is set to the uniform measure. In this case, the `overwrite_domain` flag is ignored.

        If the `type` parameter is set to `'atom'`, the pandas object is interpreted as mapping atom identifiers of the sigma-algebra to their corresponding output vectors. In this case, the `sig_alg` parameter must be provided at construction, the keys of the provided pandas object must match the atom IDs of the sigma-algebra, and the `overwrite_domain` flag is ignored.

        Additionally, there is an `overwrite_index` parameter. If the pandas object is a `pd.Series`, then this parameter is ignored. Otherwise, if the object is a `pd.DataFrame`, then:

        * If an `index` is provided at construction and `overwrite_index` is set to `False`, then the column index of the data frame must match the provided `index`.
        * If an `index` is provided at construction and `overwrite_index` is set to `True`, then `index` is reset to the column index of the data frame.
        * If an `index` is not provided at construction, then `index` is set to the column index of the data frame. In this case, the `overwrite_index` flag is ignored.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            A `pd.Series` (if 1-dimensional) or `pd.DataFrame` (if 2-dimensional or higher) mapping sample points or atom identifiers to their corresponding output vectors.
        type : str, default="point"
            A string indicating whether the provided pandas object maps sample points (`'point'`) or atom identifiers (`'atom'`) to outputs.
        overwrite_domain : bool, default=False
            A boolean flag indicating whether to overwrite the domain with a new sample space generated from the index of the provided pandas object. See the description above.
        overwrite_index : bool, default=False
            A boolean flag indicating whether to overwrite an existing `index` provided at construction. See the description above.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series` or `pd.DataFrame`.
        ValueError
            If `type` is not `'point'` or `'atom'`, or if `type` is `'atom'` and `sig_alg` is not provided at construction, or if the provided data does not yield a measurable function in the `type='point'` case, or if the index/columns of the provided data do not match the domain/index of the random vector (if provided at construction).

        Returns
        -------
        self : RandomVector
            The constructed `RandomVector` instance.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra().from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> data = pd.DataFrame([(1, 2), (3, 4), (3, 4)])
        >>> X = RandomVector(domain=Omega, sig_alg=F).from_pandas(data=data, type="point")
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
           0  1
        0  1  2
        1  3  4
        2  3  4
        >>> atom_data = pd.Series(data=[1, 2])
        >>> Y = RandomVector(domain=Omega, sig_alg=F, name="Y").from_pandas(
        ...     data=atom_data, type="atom"
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        Omega
        0       1
        1       2
        2       2
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from ..base.index import Index
        from ..base.probability_space import ProbabilitySpace
        from ..base.sample_space import SampleSpace
        from ..base.time import Time
        from .random_variable import RandomVariable

        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("data must be a pd.Series or pd.DataFrame.")
        if type not in ["point", "atom"]:
            raise ValueError("type must be either 'point' or 'atom'.")
        if type == "atom" and self.sig_alg is None:
            raise ValueError(
                "The sig_alg parameter must be set during construction for the from_dict method with type='atom'."
            )

        self._initialize_property_caches()

        if type == "point" and overwrite_domain:
            self._prob_space = ProbabilitySpace()
        if overwrite_index:
            self._index = None

        v = SampleSpaceMappingIn(
            mapping=data,
            sample_space=self.domain if type == "point" else self.sig_alg.atom_space,
        )

        self._dimension = 1 if isinstance(data, pd.Series) else data.shape[1]

        if isinstance(self, RandomVariable) and self.dimension != 1:
            raise ValueError("A random variable must have dimension 1.")

        if type == "point" and self.domain is None:
            self.domain = SampleSpace().from_pandas(v.mapping.index)

        if self.dimension > 1:
            if self._index is None:
                if isinstance(self, StochasticProcess):
                    self._index = Time().from_pandas(v.mapping.columns)
                else:
                    self._index = Index().from_pandas(v.mapping.columns)
            elif not v.mapping.columns.equals(self.index.data):
                raise ValueError(
                    "The existing index must match the column index of the data."
                )
        else:
            self._index = None

        if self.dimension == 1 and isinstance(data, pd.DataFrame):
            data = v.mapping.iloc[:, 0].copy()

        if type == "point":
            self._data = data.copy()
            if not self.is_measurable():
                raise ValueError(f"Random vector {self.name} is not measureable.")
        else:
            self._atom_data = data

        return self

    def from_numpy(self, array: np.ndarray) -> RandomVector:
        """Create a `RandomVector` from an `np.ndarray` object.

        If the `domain` sample space is not provided at construction, then it is automatically generated as a default sample space with indices `0, 1, ..., n-1`, where `n` is the number of rows in the provided `np.ndarray`. Similarly, if the `index` is not provided at construction and the random vector has dimension 2 or greater, a default feature index (i.e., an instance of `Index`) is also automatically generated.

        Parameters
        ----------
        array : np.ndarray
            Array where rows are feature vectors of sample points and columns are features.

        Returns
        -------
        self : RandomVector
            A random vector constructed from the array.

        Raises
        ------
        TypeError
            If `array` is not a NumPy ndarray.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Index, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_list(["s1", "s2", "s3"])
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({"s1": 0, "s2": 1, "s3": 1})
        >>> index = Index(name="feature_index").from_list(["A", "B"], variable_names=["feature"])
        >>> arr = np.array([[1, 2], [3, 4], [3, 4]])
        >>> X = RandomVector(domain=Omega, sig_alg=F, index=index).from_numpy(arr)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature   A  B
        Omega
        s1        1  2
        s2        3  4
        s3        3  4
        >>> Y = RandomVector(name="Y").from_numpy(arr)
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                 0  1
        0        1  2
        1        3  4
        2        3  4
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray.")
        data = pd.DataFrame(
            array,
            index=self.domain.data if self.domain else None,
            columns=self.index.data if self.index else None,
        )
        return self.from_pandas(data=data)

    def from_constant(self, constant: Hashable) -> RandomVector:
        """Create a `RandomVector` that maps every sample point in the domain to the same constant output vector.

        For this construction method, the `domain` must be provided at construction.

        Parameters
        ----------
        constant : Hashable
            The constant output vector that every sample point in the domain maps to.

        Returns
        -------
        self : RandomVector
            A random vector mapping every sample point in the domain to the same constant output vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_constant(constant=(1, 2))
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          1    2
        2          1    2
        >>> Y = RandomVector(domain=Omega, name="Y").from_constant(2)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        Omega
        0       2
        1       2
        2       2
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(constant, Hashable):
            raise TypeError("constant must be a Hashable.")
        if (
            self.index is not None
            and isinstance(constant, tuple)
            and len(constant) != len(self.index)
        ):
            raise ValueError(
                "Length of constant tuple must match the length of the index."
            )
        if isinstance(constant, tuple):
            outputs = dict.fromkeys(self.domain.data, constant)
            rv = self.from_dict(outputs=outputs)
            return rv
        elif self.index is not None:
            outputs = dict.fromkeys(self.domain.data, (constant,) * len(self.index))
            rv = self.from_dict(outputs=outputs)
            return rv
        else:
            outputs = dict.fromkeys(self.domain.data, constant)
            return self.from_dict(outputs=outputs)

    def from_identity(self) -> RandomVector:
        """Create a random vector that maps every sample point in the domain to itself.

        For this construction method, the domain must be provided at construction, and the sigma-algebra must be the power set.

        Raises
        ------
        ValueError
            If the domain is not provided at construction, or if the sigma-algebra is not the power set.

        Returns
        -------
        self : RandomVector
            A random vector mapping every sample point in the domain to itself.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_product(
        ...     indices1=[0, 1], indices2=[0, 1], variable_names=["X", "Y"]
        ... )
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         (0, 0): 0.15,
        ...         (0, 1): 0.05,
        ...         (1, 0): 0.2,
        ...         (1, 1): 0.6,
        ...     }
        ... )
        >>> X = RandomVector(Omega, F, P).from_identity()
        >>> print(X.range)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, power_set, P)
        =======================================
        <BLANKLINE>
        * Sample space 'Omega':
         X  Y
         0  0
         0  1
         1  0
         1  1
        <BLANKLINE>
        * Sigma algebra 'power_set':
        power_set  power_set_0  power_set_1
        X Y
        0 0                  0            0
          1                  0            1
        1 0                  1            0
          1                  1            1
        <BLANKLINE>
        * Probability measure 'P':
            probability
        X Y
        0 0         0.15
          1         0.05
        1 0         0.20
          1         0.60
        """
        from ..base.index import Index

        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not self.sig_alg.is_power_set:
            raise ValueError(
                "Sigma algebra must be the power set for the identity random vector."
            )
        self._data = self.domain.data.to_frame()
        if self._data.shape[1] == 1:
            self._data = self._data.squeeze(axis=1)
            self._index = None
            self._dimension = 1
        else:
            self._index = Index(name=self.name).from_pandas(self._data.columns)
            self._dimension = len(self._index)

        self._range = self.prob_space
        self._is_identity = True

        return self

    def from_randint(
        self,
        low: int,
        high: int,
        dim: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> RandomVector:
        """Generate a random vector with integer outputs uniformly sampled from the range [low, high).

        For this construction method, the `domain` must be provided at construction.

        Parameters
        ----------
        low : int
            The lower bound (inclusive) of the random integers.
        high : int
            The upper bound (exclusive) of the random integers.
        dim : int | None, default=None
            The dimension of the random vector. If `None`, then the index of the random vector must be provided at construction, and the dimension is inferred from the length of the index.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.

        Raises
        ------
        ValueError
            If the domain is not provided at construction, or if `dim` is `None` and the index is not provided at construction.
        TypeError
            If `low` or `high` are not integers, or if `dim` is not a positive integer or `None`, or if `random_state` is not an integer, Generator, or `None`.

        Returns
        -------
        self : RandomVector
            A random vector with integer outputs uniformly sampled from the range [low, high).

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_randint(low=0, high=5, dim=2, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
                0  1
        Omega
        0       0  3
        1       3  2
        2       2  4
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError("low and high must be integers.")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise TypeError("dim must be a positive integer or None.")
        if dim is None and self.index is None:
            raise ValueError("If dim is None, index must be provided at construction.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        if dim is None:
            dim = len(self.index)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        arr = rng.integers(low, high, size=(len(self.domain.data), dim))
        return self.from_numpy(array=arr)

    def from_randnorm(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        dim: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> RandomVector:
        """Generate a random vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        For this construction method, the `domain` must be provided at construction.

        Parameters
        ----------
        loc : float, default=0.0
            The mean of the normal distribution.
        scale : float, default=1.0
            The standard deviation of the normal distribution.
        dim : int | None, default=None
            The dimension of the random vector. If `None`, then the index of the random vector must be provided at construction, and the dimension is inferred from the length of the index.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.

        Raises
        ------
        ValueError
            If the domain is not provided at construction, or if `dim` is `None` and the index is not provided at construction.
        TypeError
            If `loc` or `scale` are not real numbers, or if `dim` is not a positive integer or `None`, or if `random_state` is not an integer, Generator, or `None`.

        Returns
        -------
        self : RandomVector
            A random vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_randnorm(dim=2, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
                    0         1
        Omega
        0       0.304717 -1.039984
        1       0.750451  0.940565
        2      -1.951035 -1.302180
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(loc, Real) or not isinstance(scale, Real):
            raise TypeError("loc and scale must be real numbers.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise TypeError("dim must be a positive integer or None.")
        if dim is None and self.index is None:
            raise ValueError("If dim is None, index must be provided at construction.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        if dim is None:
            dim = len(self.index)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        arr = rng.normal(loc, scale, size=(len(self.domain.data), dim))
        return self.from_numpy(array=arr)

    # TODO: write unit tests
    @classmethod
    def from_random_vectors(
        cls, rvs: list[RandomVector], name: Hashable
    ) -> RandomVector:
        """Create a random vector from a list of random vectors.

        Parameters
        ----------
        rvs : list[RandomVector]
            A list of random vectors to combine.
        name : Hashable
            The name of the resulting random vector.

        Raises
        ------
        TypeError
            If `rvs` is not a list of `RandomVector` instances or if `name` is not a `Hashable`.
        ValueError
            If `rvs` is an empty list or if the random vectors are not defined on the same probability space.

        Returns
        -------
        RandomVector
            A new random vector created by combining the input random vectors.
        """
        if not isinstance(rvs, list) or not all(
            isinstance(rv, RandomVector) for rv in rvs
        ):
            raise TypeError("rvs must be a list of RandomVector instances.")
        if len(rvs) == 0:
            raise ValueError("rvs must be a non-empty list.")
        if not all(rv.prob_space == rvs[0].prob_space for rv in rvs):
            raise ValueError(
                "All RandomVector instances must be defined on the same probability space."
            )
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")

        combined_data = pd.concat([rv.data for rv in rvs], axis=1)
        return cls(*rvs[0].prob_space, name=name).from_pandas(combined_data)

    def __or__(self, other: RandomVector) -> RandomVector:
        """Pass."""
        return RandomVector.from_random_vectors(
            [self, other], name=f"{self.name}{other.name}"
        )

    def product(self, other: RandomVector) -> RandomVector:
        """Pass."""
        from ..base.sample_space import SampleSpace

        self_data = self.data.copy()
        other_data = other.data.copy()

        self_domain_variable_names = self_data.index.names
        other_domain_variable_names = other_data.index.names
        self_index_variable_names = (
            self_data.columns if self.dimension > 1 else [self.name]
        )
        other_index_variable_names = (
            other_data.columns if other.dimension > 1 else [other.name]
        )

        self_domain_variable_names, other_domain_variable_names = self._merge(
            self_domain_variable_names,
            other_domain_variable_names,
        )
        self_domain_marked_variable_names = [
            f"{name}_domain" for name in self_domain_variable_names
        ]
        other_domain_marked_variable_names = [
            f"{name}_domain" for name in other_domain_variable_names
        ]
        self_index_variable_names, other_index_variable_names = self._merge(
            self_index_variable_names, other_index_variable_names
        )

        self_data.index = self_data.index.set_names(self_domain_marked_variable_names)
        other_data.index = other_data.index.set_names(
            other_domain_marked_variable_names
        )
        if self.dimension > 1:
            self_data.columns = self_index_variable_names
        else:
            self_data.name = self_index_variable_names[0]

        if other.dimension > 1:
            other_data.columns = other_index_variable_names
        else:
            other_data.name = other_index_variable_names[0]

        self_reset = self_data.reset_index()
        other_reset = other_data.reset_index()

        merged_data = pd.merge(
            self_reset,
            other_reset,
            how="cross",
        )

        merged_data.set_index(
            self_domain_marked_variable_names + other_domain_marked_variable_names,
            inplace=True,
        )
        merged_data.index = merged_data.index.set_names(
            self_domain_variable_names + other_domain_variable_names
        )
        merged_data.columns.name = f"{self.name}{other.name}"

        if self.is_identity and other.is_identity:
            domain = SampleSpace().from_pandas(merged_data.index)
            return RandomVector(
                domain=domain, name=f"{self.name}{other.name}"
            ).from_identity()
        else:
            return RandomVector(name=f"{self.name}{other.name}").from_pandas(
                merged_data
            )

    @staticmethod
    def _merge(list1, list2):
        def base(s):
            m = re.fullmatch(r"(.+)_(\d+)", s)
            return (s, None) if not m else (m.group(1), int(m.group(2)))

        list1_tuples = [base(s) for s in list1]
        list2_tuples = [base(s) for s in list2]
        list1_bases = [s[0] for s in list1_tuples]
        list2_bases = [s[0] for s in list2_tuples]
        common_bases = set(list1_bases) & set(list2_bases)

        for common_base in common_bases:
            idx = 0
            for s in list1_tuples:
                if s[0] == common_base:
                    list1_tuples[list1_tuples.index(s)] = (s[0], idx)
                    idx += 1
            for s in list2_tuples:
                if s[0] == common_base:
                    list2_tuples[list2_tuples.index(s)] = (s[0], idx)
                    idx += 1

        list1_return = [
            f"{s[0]}_{s[1]}" if s[1] is not None else s[0] for s in list1_tuples
        ]
        list2_return = [
            f"{s[0]}_{s[1]}" if s[1] is not None else s[0] for s in list2_tuples
        ]

        return list1_return, list2_return

    def __matmul__(self, other: RandomVector) -> RandomVector:
        """Pass."""
        return self.product(other)

    # TODO: write unit tests
    @classmethod
    def indicator_of(cls, event: Event, dim: int = 1) -> RandomVector:
        r"""Create the indicator random vector of a given event of a given dimension.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : Event
            The event for which the indicator random vector is to be created.
        dim : int, default=1
            The dimension of the indicator random vector.

        Raises
        ------
        TypeError
            If `event` is not an instance of `Event`, or if `dim` is not a positive integer.

        Returns
        -------
        indicator_rv : RandomVector
            The indicator random variable of the given event.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> print(Omega)
        Sample space 'Omega':
         Omega
             0
             1
             2
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 1])
        >>> I_A = RandomVector.indicator_of(event=A, dim=2)
        >>> print(I_A) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'I_A':
        feature  I_A_0  I_A_1
        Omega
        0            1      1
        1            1      1
        2            0      0

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space. Given an event $A\in \mathcal{F}$ and a dimension $d$, the *indicator random vector* is the random vector $I_A: \Omega \to \mathbb{R}^d$ such that

        $$
        I_A(\omega) = \begin{cases}
        (1, 1, \ldots, 1) & : \omega \in A,\\
        (0, 0, \ldots, 0) & : \omega \notin A.
        \end{cases}
        $$
        """
        from ..base.event import Event
        from ..base.index import Index

        if not isinstance(event, Event):
            raise TypeError("event must be an Event.")
        if not isinstance(dim, int) or dim <= 0:
            raise TypeError("dim must be a positive integer.")

        if dim == 1:
            return event.indicator
        data = pd.concat([event.indicator.data] * dim, axis=1)
        index = Index(name="index").from_sequence(
            size=dim, prefix=event.indicator.name, variable_name="feature"
        )
        data.columns = index.data

        return cls(
            domain=event.sample_space,
            sig_alg=event.indicator.sig_alg,
            name=event.indicator.name,
            index=index,
        ).from_pandas(data)

    # --------------------- properties --------------------- #

    # TODO: write unit tests
    @property
    def point_outputs(self) -> Mapping[Hashable, Hashable] | None:
        """Get the outputs of the random vector as a dictionary mapping sample points in the domain to their corresponding output vectors.

        Returns
        -------
        point_outputs : Mapping[Hashable, Hashable] | None
            A dictionary mapping sample points in the domain to their corresponding output vectors, or `None`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra().from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega, sig_alg=F).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ...     type="point",
        ... )
        >>> print(X.point_outputs)
        {0: (1, 2), 1: (3, 4), 2: (3, 4)}
        """
        if self._point_outputs is None:
            if self._data is not None:
                self._point_outputs = self._pandas_to_dict(self._data)
            elif self._atom_outputs is not None:
                self._point_outputs = self._atom_to_point_outputs(self._atom_outputs)
            elif self._atom_data is not None:
                self._atom_outputs = self._pandas_to_dict(self._atom_data)
                self._point_outputs = self._atom_to_point_outputs(self._atom_outputs)
        return self._point_outputs

    # TODO: write unit tests
    @property
    def atom_outputs(self) -> Mapping[Hashable, Hashable] | None:
        """Get the outputs of the random vector as a dictionary mapping atom identifiers to their corresponding output vectors.

        Returns
        -------
        atom_outputs : Mapping[Hashable, Hashable] | None
            A dictionary mapping atom identifiers to their corresponding output vectors, or `None`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra().from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega, sig_alg=F).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ...     type="point",
        ... )
        >>> print(X.atom_outputs)
        {0: (1, 2), 1: (3, 4)}
        """
        if self._atom_outputs is None:
            if self._atom_data is not None:
                self._atom_outputs = self._pandas_to_dict(self._atom_data)
            elif self._point_outputs is not None:
                self._atom_outputs = self._point_to_atom_outputs(self._point_outputs)
            elif self._data is not None:
                self._point_outputs = self._pandas_to_dict(self._data)
                self._atom_outputs = self._point_to_atom_outputs(self._point_outputs)
        return self._atom_outputs

    # TODO: write unit tests
    @property
    def data(self) -> pd.Series | pd.DataFrame | None:
        """Get the mapping of the random vector from sample points to outputs as a `pd.Series` (if 1-dimensional) or `pd.DataFrame` (if 2-dimensional or higher).

        Returns
        -------
        data : pd.Series | pd.DataFrame | None
            A `pd.Series` (if the random vector is 1-dimensional) or `pd.DataFrame` (if the random vector is 2-dimensional or higher) representing the mapping of the random vector from sample points to outputs, or `None`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra().from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega, sig_alg=F).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ...     type="point",
        ... )
        >>> print(X.data)  # doctest: +NORMALIZE_WHITESPACE
        X        X_0  X_1
        Omega
        0          1    2
        1          3    4
        2          3    4
        >>> Y = RandomVector(domain=Omega, sig_alg=F, name="Y").from_dict(
        ...     {
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ...     type="point",
        ... )
        >>> print(Y.data)  # doctest: +NORMALIZE_WHITESPACE
        Omega
        0    1
        1    2
        2    2
        Name: Y, dtype: int64
        """
        if self._data is None and self.point_outputs is not None:
            self._data = self._dict_to_pandas(
                dict_param=self._point_outputs,
                pandas_index=self.domain.data,
                pandas_columns=self.index.data if self.index is not None else None,
            )
        return self._data

    # TODO: write unit tests
    @property
    def atom_data(self) -> pd.Series | pd.DataFrame | None:
        """Get the mapping of the random vector from atom identifiers to outputs as a `pd.Series` (if 1-dimensional) or `pd.DataFrame` (if 2-dimensional or higher).

        Returns
        -------
        atom_data : pd.Series | pd.DataFrame | None
            A `pd.Series` (if the random vector is 1-dimensional) or `pd.DataFrame` (if the random vector is 2-dimensional or higher) representing the mapping of the random vector from atom identifiers to outputs, or `None`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra().from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega, sig_alg=F).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ...     type="point",
        ... )
        >>> print(X.atom_data)  # doctest: +NORMALIZE_WHITESPACE
        X        X_0  X_1
        F
        0          1    2
        1          3    4
        >>> Y = RandomVector(domain=Omega, sig_alg=F, name="Y").from_dict(
        ...     {
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ...     type="point",
        ... )
        >>> print(Y.atom_data)  # doctest: +NORMALIZE_WHITESPACE
        F
        0    1
        1    2
        Name: Y, dtype: int64
        """
        if self._atom_data is None and self.atom_outputs is not None:
            self._atom_data = self._dict_to_pandas(
                dict_param=self._atom_outputs,
                pandas_index=self.sig_alg.atom_space.data,
                pandas_columns=self.index.data if self.index is not None else None,
            )
        return self._atom_data

    def _pandas_to_dict(self, data: pd.Series | pd.DataFrame) -> dict:
        """Convert pandas data to dictionary."""
        if isinstance(data, pd.Series):
            return data.to_dict()
        else:
            return data.apply(tuple, axis=1).to_dict()

    def _dict_to_pandas(
        self,
        dict_param: dict,
        pandas_index: pd.Index | None = None,
        pandas_columns: pd.Index | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Convert dictionary to pandas data."""
        data = pd.DataFrame.from_dict(dict_param, orient="index")
        dimension = data.shape[1]
        if dimension == 1:
            data = data.iloc[:, 0]
            data.name = self.name
        else:
            data.columns = pandas_columns
        data.index = pandas_index

        return data

    def _point_to_atom_outputs(self, point_outputs: dict) -> dict:
        """Convert point outputs to atom outputs."""
        return {
            atom_id: point_outputs[sample_ids[0]]
            for atom_id, sample_ids in self.sig_alg.atom_id_to_sample_ids.items()
        }

    def _atom_to_point_outputs(self, atom_outputs: dict) -> dict:
        """Convert atom outputs to point outputs."""
        return {
            sample_id: atom_outputs[self.sig_alg.sample_id_to_atom_id[sample_id]]
            for sample_id in self.domain
        }

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the random vector.

        Returns
        -------
        dimension : int | None
            The dimension of the random vector, or `None` if it has not been set.
        """
        return self._dimension

    # TODO: write unit tests
    @property
    def components(self) -> list[RandomVariable] | None:
        r"""Get the component random variables of the random vector.

        See the Notes section below for the mathematical details.

        Raises
        ------
        ValueError
            If `self` has an empty `data` attribute.

        Returns
        -------
        components : list[RandomVariable] | None
            A list of the component random variables of the random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Get the components of a 2D random vector
        >>> X = RandomVector(domain=Omega).from_randint(low=0, high=3, dim=2, random_state=42)
        >>> for component in X.components:
        ...     print(component) # doctest: +NORMALIZE_WHITESPACE
        Random variable '0':
                0
        Omega
        0       0
        1       1
        2       1
        Random variable '1':
                1
        Omega
        0       2
        1       1
        2       2
        >>> # Get the component of a 1D random vector
        >>> Y = RandomVector(domain=Omega, name="Y").from_randint(low=0, high=3, dim=1, random_state=42)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        Omega
        0       0
        1       2
        2       1
        >>> for component in Y.components:
        ...     print(component) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        Omega
        0       0
        1       2
        2       1

        Notes
        -----
        If $X: \Omega \to \mathbb{R}^d$ is a random vector, then for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega),X_2(\omega),\ldots, X_d(\omega))
        $$

        where $X_j: \Omega \to \mathbb{R}$ is the *$j$-th component random variable* of $X$.

        If the dimension of `self` is $1$, then this method returns a list consisting of `self` itself.
        """
        from .random_variable import RandomVariable

        if self._components is None and self.data is not None:
            if self.dimension == 1:
                if isinstance(self, RandomVariable):
                    self._components = [self]
                else:
                    self._components = [self.to_random_variable()]
            else:
                self._components = [
                    self.get_component_rv(idx) for idx in self.index.data
                ]
        return self._components

    @property
    def name(self) -> Hashable:
        """Get the name of the random vector.

        Returns
        -------
        name : Hashable
            The name of the random vector.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        self._name = name
        if isinstance(self._data, pd.Series):
            self._data.name = name

    def with_name(self, name: Hashable, modify_index: bool = False) -> RandomVector:
        """Set the name of the random vector and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the random vector.
        modify_index : bool, default=False
            If `True` and the random vector has a feature index, also updates the feature index to reflect the new name of the random vector.

        Returns
        -------
        self : RandomVector
            Returns self to allow method chaining.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          3    4
        2          5    6
        >>> print(X.with_name("Y")) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        X        X_0  X_1
        Omega
        0          1    2
        1          3    4
        2          5    6
        >>> print(X.with_name("Y", modify_index=True)) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        Y        Y_0  Y_1
        Omega
        0          1    2
        1          3    4
        2          5    6
        """
        from ..base.index import Index

        self.name = name
        if modify_index and self.index is not None:
            prefix = name if isinstance(name, str) else None
            self._index = Index(name=self.name).from_sequence(
                size=self.dimension, prefix=prefix
            )
            self._data.columns = self._index.data
        return self

    @property
    def index(self) -> Index | None:
        """Get the index of the random vector.

        Returns
        -------
        index : Index | None
            The index of the random vector, or `None` if the random vector is 1-dimensional.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Print the index of a 2D random vector
        >>> outputs_2d = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_2d)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          3    4
        2          5    6
        >>> print(X.index)  # doctest: +NORMALIZE_WHITESPACE
        Index 'X':
        X
        X_0
        X_1
        >>> # Print the index of a 1D random vector
        >>> outputs_1d = dict(zip(Omega, [1, 2, 3]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_1d)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        Omega
        0       1
        1       2
        2       3
        >>> print(Y.index)
        None
        """
        return self._index

    # TODO: write unit tests
    @index.setter
    def index(self, index: Index) -> None:
        """Set the index of the random vector.

        Parameters
        ----------
        index : Index
            The new index for the random vector.

        Raises
        ------
        TypeError
            If `index` is not an instance of `Index`.
        ValueError
            If the random vector has a non-empty `data` attribute and the length of `index` does not match the dimension of the random vector.
        """
        from ..base.index import Index

        if not isinstance(index, Index):
            raise TypeError("index must be an Index.")

        if self.data is not None:
            if len(index) != self.dimension:
                raise ValueError(
                    "index size must match the dimension of the random vector."
                )
            self.data.columns = index.data

        self._index = index

    @property
    def generated_sig_alg(self) -> SigmaAlgebra | None:
        r"""Get the sigma-algebra generated by a random vector.

        See the Notes section below for the mathematical details.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra induced by the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega, sig_alg=F).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     }
        ... )
        >>> sig_X = X.generated_sig_alg
        >>> print(sig_X)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(X)':
        sigma(X)  sigma(X)_0  sigma(X)_1
        Omega
        0                  1           2
        1                  3           4
        2                  3           4
        3                  3           4
        >>> print(sig_X <= F)
        True

        Notes
        -----
        A random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F},P)$ generates a $\sigma$-algebra denoted $\sigma(X)$. On a finite sample space $\Omega$, this $\sigma$-algebra is determined by its atoms, which are the nonempty level sets

        $$
        X^{-1}(x) = \{ \omega \in \Omega : X(\omega) = x\},
        $$

        for $x\in \mathbb{R}^d$. The atom identifiers may thus be taken as the vectors $x\in \mathbb{R}^d$ in the range of $X$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self._generated_sig_alg is None and self.data is not None:
            self._generated_sig_alg = SigmaAlgebra.from_random_vector(self)
        return self._generated_sig_alg

    @property
    def prob_space(self) -> ProbabilitySpace | None:
        """Get the probability space on which the random vector is defined.

        Returns
        -------
        prob_space : ProbabilitySpace | None
            The probability space on which the random vector is defined.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.8,
        ...     }
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(*prob_space).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     }
        ... )
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         Omega
             0
             1
             2
        <BLANKLINE>
        * Sigma algebra 'F':
                      F
        Omega
        0             0
        1             1
        2             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        F
        0                0.2
        1                0.8
        """
        return self._prob_space

    # TODO: write unit tests
    @property
    def domain(self) -> SampleSpace | None:
        """Get the domain of the random vector.

        The `domain` property is settable. If the random vector is not defined on an empty probability space, the new domain must have the same number of sample points as the existing domain and the sample spaces of the sigma-algebra and probability measure are updated to the new sample space. If in addition the random vector is not empty (i.e., if it has outputs), then the outputs of the random vector are remapped to the new domain according to the order of sample points in the new domain. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the domain may be set freely, the sigma-algebra is updated to the power-set sigma-algebra on the new domain, and the probability measure is updated to the uniform measure on the new domain.

        Returns
        -------
        domain : SampleSpace | None
            The domain of the random vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.25,
        ...         1: 0.75,
        ...     }
        ... )
        >>> X = RandomVector(Omega, F, P).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     }
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          1    2
        2          3    4
        3          3    4
        >>> print(X.domain)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         Omega
             0
             1
             2
             3
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         Omega
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'F':
                      F
        Omega
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        F
        0               0.25
        1               0.75
        >>> Omega_new = SampleSpace(name="Omega_new").from_list(["a", "b", "c", "d"])
        >>> X.domain = Omega_new
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega_new
        a          1    2
        b          1    2
        c          3    4
        d          3    4
        >>> print(X.domain)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega_new':
         Omega_new
                 a
                 b
                 c
                 d
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega_new, F, P)
        ===================================
        <BLANKLINE>
        * Sample space 'Omega_new':
         Omega_new
                 a
                 b
                 c
                 d
        <BLANKLINE>
        * Sigma algebra 'F':
                      F
        Omega_new
        a             0
        b             0
        c             1
        d             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        F
        0               0.25
        1               0.75
        >>> empty_RV = RandomVector()
        >>> empty_RV.domain = Omega_new
        >>> print(empty_RV.domain)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega_new':
         Omega_new
                 a
                 b
                 c
                 d
        >>> print(empty_RV.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega_new, power_set, U)
        ===========================================
        <BLANKLINE>
        * Sample space 'Omega_new':
         Omega_new
                 a
                 b
                 c
                 d
        <BLANKLINE>
        * Sigma algebra 'power_set':
              power_set
        Omega_new
        a             a
        b             b
        c             c
        d             d
        <BLANKLINE>
        * Probability measure 'U':
                probability
        Omega_new
        a              0.25
        b              0.25
        c              0.25
        d              0.25
        """
        return self.prob_space.sample_space

    # TODO: write unit tests
    @domain.setter
    def domain(self, domain: SampleSpace) -> None:
        """Set the domain of the random vector.

        If the random vector is not defined on an empty probability space, the new domain must have the same number of sample points as the existing domain, and then the sample spaces of the sigma-algebra and probability measure are updated to the new sample space. If in addition the random vector is not empty (i.e., if it has outputs), then the outputs of the random vector are remapped to the new domain according to the order of sample points in the new domain. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the domain may be set freely, the sigma-algebra is updated to the power-set sigma-algebra on the new domain, and the probability measure is updated to the uniform measure on the new domain.

        Parameters
        ----------
        domain : SampleSpace
            The new domain for the random vector.

        Raises
        ------
        TypeError
            If `domain` is not an instance of `SampleSpace`.
        """
        from ..base.sample_space import SampleSpace

        if not isinstance(domain, SampleSpace):
            raise TypeError("domain must be an instance of SampleSpace.")

        if self.point_outputs is not None:
            self._point_outputs = dict(zip(domain.data, self.point_outputs.values()))
        self._initialize_property_caches(
            exceptions={
                "_point_outputs",
                "_atom_outputs",
                "_atom_data",
                "_dimension",
                "_range",
            }
        )
        self.prob_space.sample_space = domain

    # TODO: write unit tests
    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra on the underlying probability space.

        The `sig_alg` property is settable. If the random vector is not defined on an empty probability space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the probability measure is updated to be the restriction of the existing probability measure to the new sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the new sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the sample space of the sigma-algebra and the probability measure is the uniform measure on the sample space.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on the domain of the random vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.05,
        ...         1: 0.75,
        ...         2: 0.2,
        ...     }
        ... )
        >>> X = RandomVector(Omega, F, P).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     }
        ... )
        >>> print(X.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                      F
        Omega
        0             0
        1             1
        2             2
        3             2
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     }
        ... )
        >>> X.sig_alg = G
        >>> print(X.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                      G
        Omega
        0             0
        1             0
        2             1
        3             1
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         Omega
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'G':
                      G
        Omega
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        G
        0                0.8
        1                0.2
        >>> empty_RV = RandomVector()
        >>> empty_RV.sig_alg = G
        >>> print(empty_RV.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                      G
        Omega
        0             0
        1             0
        2             1
        3             1
        >>> print(empty_RV.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, U)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         Omega
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'G':
                      G
        Omega
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'U':
                probability
        G
        0                0.5
        1                0.5
        """
        return self.prob_space.sig_alg

    # TODO: write unit tests
    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra on the underlying probability space.

        If the random vector is not defined on an empty probability space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the probability measure is updated to be the restriction of the existing probability measure to the new sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the new sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the sample space of the sigma-algebra and the probability measure is the uniform measure on the sample space.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra for the random vector.

        Raises
        ------
        TypeError
            If `sig_alg` is not an instance of `SigmaAlgebra`.
        ValueError
            If the random vector is not measurable with respect to the new sigma-algebra.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")
        if self.data is not None and not self.is_measurable(sig_alg=sig_alg):
            raise ValueError(
                "The random vector is not measurable with respect to the provided sigma-algebra."
            )

        self._initialize_property_caches(
            exceptions={
                "_point_outputs",
                "_data",
                "_dimension",
                "_range",
            }
        )
        self.prob_space.sig_alg = sig_alg

    # TODO: write unit tests
    @property
    def prob_measure(self) -> ProbabilityMeasure | None:
        """Get the probability measure on the underlying probability space.

        The `prob_measure` property is settable. If the random vector is not defined on an empty probability space, the new probability measure must be a probability measure on a sub-sigma-algebra of the existing sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the sub-sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the probability measure may be set freely and the domain is set to the sample space of the probability measure's sigma-algebra and the sigma-algebra is set to the sigma-algebra of the probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure | None
            The probability measure on the domain of the random vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.05,
        ...         1: 0.75,
        ...         2: 0.2,
        ...     }
        ... )
        >>> X = RandomVector(Omega, F, P).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     }
        ... )
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        F
        0               0.05
        1               0.75
        2               0.20
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     }
        ... )
        >>> Q = ProbabilityMeasure(sig_alg=G, name="Q").from_dict(
        ...     {
        ...         0: 0.1,
        ...         1: 0.9,
        ...     }
        ... )
        >>> X.prob_measure = Q
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        G
        0                0.1
        1                0.9
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, Q)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         Omega
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'G':
                      G
        Omega
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'Q':
                probability
        G
        0                0.1
        1                0.9
        >>> empty_RV = RandomVector()
        >>> empty_RV.prob_measure = Q
        >>> print(empty_RV.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        G
        0                0.1
        1                0.9
        >>> print(empty_RV.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, Q)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         Omega
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'G':
                      G
        Omega
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'Q':
                probability
        G
        0                0.1
        1                0.9
        """
        return self.prob_space.prob_measure

    # TODO: write unit tests
    @prob_measure.setter
    def prob_measure(self, prob_measure: ProbabilityMeasure) -> None:
        """Set the probability measure on the underlying probability space.

        If the random vector is not defined on an empty probability space, the new probability measure must be a probability measure on a sub-sigma-algebra of the existing sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the sub-sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the probability measure may be set freely and the domain is set to the sample space of the probability measure's sigma-algebra and the sigma-algebra is set to the sigma-algebra of the probability measure.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure
            The new probability measure for the random vector.

        Raises
        ------
        TypeError
            If `prob_measure` is not an instance of `ProbabilityMeasure`.
        ValueError
            If the random vector is not measurable with respect to the sigma-algebra of the new probability measure.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure

        if not isinstance(prob_measure, ProbabilityMeasure):
            raise TypeError("prob_measure must be an instance of ProbabilityMeasure.")
        if self.data is not None and not self.is_measurable(
            sig_alg=prob_measure.sig_alg
        ):
            raise ValueError(
                "The random vector is not measurable with respect to the sigma-algebra of the provided probability measure."
            )

        self._initialize_property_caches(
            exceptions={
                "_point_outputs",
                "_data",
                "_dimension",
                "_generated_sig_alg",
            }
        )
        self.prob_space.prob_measure = prob_measure

    def with_probability_measure(
        self,
        prob_measure: ProbabilityMeasure | None = None,
        probabilities: Mapping[Hashable, Real] | None = None,
    ) -> RandomVector:
        """Set the probability measure on the domain of the random vector and return self for chaining.

        This method is equivalent to setting the `prob_measure` attribute with an instance of `ProbabilityMeasure`. The method also accepts a dictionary of probabilities as a parameter, allowing the user to bypass constructing an instance of `ProbabilityMeasure`.

        The method takes either the `probabilities` parameter or the `prob_measure` parameter, but not both. If neither parameter is provided, the method defaults to setting the probability measure to the uniform measure.

        Parameters
        ----------
        probabilities : Mapping[Hashable, Real] | None, default=None
            A mapping from sample points in the domain to their corresponding probabilities.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure to set on the domain of the random vector.

        Raises
        ------
        ValueError
            If both `probabilities` and `prob_measure` are provided.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...     }
        ... )
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.05,
        ...         1: 0.95,
        ...     }
        ... )
        >>> _ = X.with_probability_measure(prob_measure=P)
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        F
        0               0.05
        1               0.95
        >>> _ = X.with_probability_measure(
        ...     probabilities={
        ...         0: 0.1,
        ...         1: 0.9,
        ...         2: 0.0,
        ...     }
        ... )
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        Omega
        0               0.1
        1               0.9
        2               0.0
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if probabilities is not None and prob_measure is not None:
            raise ValueError("Cannot specify both probabilities and prob_measure.")

        if probabilities is None and prob_measure is None:
            prob_measure = ProbabilityMeasure.uniform(
                SigmaAlgebra.power_set(self.domain)
            )

        if probabilities is not None:
            prob_measure = ProbabilityMeasure(
                sig_alg=SigmaAlgebra.power_set(self.domain)
            ).from_dict(probabilities)
        self.prob_measure = prob_measure
        return self

    @property
    def is_identity(self) -> bool:
        """Check if the random vector is the identity mapping on its domain.

        Returns
        -------
        is_identity : bool
            `True` if the random vector is the identity mapping, `False` otherwise.
        """
        return self._is_identity

    @property
    def range(self) -> ProbabilitySpace | None:
        r"""Return the range of a random vector as a probability space with the pushforward measure.

        See the Notes section below for the mathematical details.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.7,
        ...         2: 0.1,
        ...     }
        ... )
        >>> X = RandomVector(Omega, F, P).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     }
        ... )
        >>> print(X.range)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (X_range, power_set, P_X)
        ===========================================
        <BLANKLINE>
        * Sample space 'X_range':
         X_0  X_1
           1    2
           3    4
        <BLANKLINE>
        * Sigma algebra 'power_set':
        power_set  power_set_0  power_set_1
        X_0 X_1
        1   2                1            2
        3   4                3            4
        <BLANKLINE>
        * Probability measure 'P_X':
                probability
        X_0 X_1
        1   2            0.2
        3   4            0.8

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F},P)$. The range

        $$
        X(\Omega) = \{ X(\omega) \in \mathbb{R}^d : \omega \in \Omega \}
        $$

        of the random vector is a probability space when equipped with the *pushforward measure* $P_X$ given by

        $$
        P_X(A) = P \left( \{\omega \in \Omega \mid X(\omega) \in A \} \right),
        $$

        for all events $A \subset X(\Omega)$. In SigAlg, the $\sigma$-algebra on $X(\Omega)$ defaults to the power set.
        """
        from ..base.probability_space import ProbabilitySpace
        from ..base.sample_space import SampleSpace
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self._range is None and self.data is not None:
            level_set_probs = {
                output: self.prob_measure(level_set)
                for output, level_set in self.generated_sig_alg.atom_id_to_sample_ids.items()
            }

            range_sample_space = SampleSpace(name=f"{self.name}_range").from_list(
                indices=list(level_set_probs.keys()),
                variable_names=list(self.index) if self.dimension > 1 else [self.name],
            )
            range_sig_alg = SigmaAlgebra.power_set(range_sample_space)
            range_sig_alg.atom_space.variable_names = (
                list(self.index) if self.dimension > 1 else [self.name]
            )

            pushforward = ProbabilityMeasure(
                sig_alg=range_sig_alg, name=f"{self.prob_measure.name}_{self.name}"
            ).from_dict(level_set_probs)

            self._range = ProbabilitySpace(
                sample_space=range_sample_space,
                sig_alg=range_sig_alg,
                prob_measure=pushforward,
            )

        return self._range

    # --------------------- probability space methods --------------------- #

    # TODO: write unit tests
    def is_measurable(self, sig_alg: SigmaAlgebra | None = None) -> bool:
        r"""Check if the random vector is measurable with respect to a given sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to check measurability against. If `None`, checks measurability with respect to the sigma-algebra on the underlying probability space.

        Returns
        -------
        is_measurable : bool
            `True` if the random vector is measurable with respect to the given sigma-algebra, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> outputs_X = dict(zip(Omega, [(1, 2), (3, 4), (3, 4), (3, 4)]))
        >>> outputs_Y = dict(zip(Omega, [(1, 2), (3, 4), (5, 6), (7, 8)]))
        >>> X = RandomVector(domain=Omega, name="X").from_dict(outputs_X)
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> atom_ids = dict(zip(Omega, [0, 1, 1, 2]))
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)
        >>> # X is constant on the atoms of F, so it is measurable
        >>> print(X.is_measurable(F))
        True
        >>> # Y is not constant on the atoms, so it is not measurable
        >>> print(Y.is_measurable(F))
        False

        Notes
        -----
        Let $(\Omega, \mathcal{F})$ be a measurable space and $X: \Omega \to \mathbb{R}^d$ a function. In the case that $\Omega$ is finite (as in SigAlg), the $\sigma$-algebra is determined by its atoms. In this case, the function $X$ is said to be *$\mathcal{F}$-measurable* if $X$ is constant on the atoms of $\mathcal{F}$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra or None.")
        if sig_alg is not None and sig_alg.sample_space != self.domain:
            raise ValueError(
                "The sample space of sig_alg must match the domain of the random vector."
            )

        if sig_alg is None:
            sig_alg = self.sig_alg
        if sig_alg.is_power_set:
            return True

        df = pd.concat([self.data, sig_alg.data], axis=1).drop_duplicates()
        return len(df) == sig_alg.num_atoms

    def to_random_variable(self) -> RandomVariable:
        """Convert a 1-dimensional random vector to an instance of `RandomVariable`.

        Raises
        ------
        ValueError
            If the random vector has dimension > 1.

        Returns
        -------
        rv : RandomVariable
            The converted `RandomVariable`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [10, 20]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs=outputs)
        >>> X_var = X.to_random_variable()
        >>> X_var # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        Omega
        0      10
        1      20
        """
        from .random_variable import RandomVariable

        if self.dimension != 1:
            raise ValueError(
                "Can only convert a 1-dimensional RandomVector to RandomVariable."
            )

        return RandomVariable(*self.prob_space, name=self.name).from_pandas(self.data)

    # --------------------- data methods --------------------- #

    def __call__(self, key: Hashable | Event) -> Hashable | FeatureVector:
        """Evaluate a random vector on a sample point or an atom in the sigma-algebra.

        Parameters
        ----------
        key : Hashable | Event
            A sample point in the domain or an atom in the sigma-algebra of the random vector.

        Raises
        ------
        ValueError
            If the random vector has no outputs, or if `key` is not in the domain or the sigma-algebra of the random vector, or if `key` is an event that is not an atom in the sigma-algebra.
        TypeError
            If `key` is not a Hashable (i.e., a sample point) or an Event (i.e., an atom in the sigma-algebra).

        Returns
        -------
        output : Hashable | FeatureVector
            If `key` is a sample point, returns the output of the random vector at that sample point. If `key` is an atom in the sigma-algebra, returns the output of the random vector on the atom.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: "a",
        ...         1: "b",
        ...         2: "b",
        ...         3: "c",
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         "a": 0.2,
        ...         "b": 0.5,
        ...         "c": 0.3,
        ...     }
        ... )
        >>> X = RandomVector(Omega, F, P).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     }
        ... )
        >>> # Call on a sample point
        >>> print(X(0))  # doctest: +NORMALIZE_WHITESPACE
        Feature vector 'X(0)':
                 0
        X
        X_0      1
        X_1      2
        >>> # Call on an atom
        >>> atom = F.get_event([1, 2])
        >>> print(X(atom))  # doctest: +NORMALIZE_WHITESPACE
        Feature vector 'X(A)':
                 A
        X
        X_0      3
        X_1      4
        >>> Y = RandomVariable(Omega, F, P, name="Y").from_dict(
        ...     {
        ...         0: 1,
        ...         1: 3,
        ...         2: 3,
        ...         3: 5,
        ...     }
        ... )
        >>> print(Y(0))
        1
        >>> print(Y(atom))
        3
        """
        from ..base.event import Event
        from ..base.feature_vector import FeatureVector

        if self.data is None:
            raise ValueError("Cannot evaluate a random vector without outputs.")

        if not isinstance(key, (Hashable, Event)):
            raise TypeError(
                "key must be a Hashable (i.e., a sample point) or Event (i.e., an atom in the sigma-algebra)."
            )

        if isinstance(key, Event):
            if key not in self.sig_alg:
                raise ValueError(
                    "The provided event is not in the sigma-algebra of the random vector."
                )
            if not key.is_atom:
                raise ValueError(
                    "The provided event is not an atom in the sigma-algebra of the random vector."
                )
            sample_point = key[0]
            output_name = key.name
        else:
            if key not in self.domain:
                raise ValueError(
                    "The provided sample point is not in the domain of the random vector."
                )
            sample_point = key
            output_name = key

        data = self.data.loc[sample_point]

        if not isinstance(data, pd.Series):
            result = data
        else:
            name = f"{self.name}({output_name})" if self.name is not None else None
            result = FeatureVector(random_vector=self, name=name).from_sample_point(
                sample_point,
                overwrite_name=False,
            )
            result._data.name = output_name

        return result

    def __iter__(self) -> Iterator[RandomVariable]:
        """Iterate over the components of the random vector.

        Returns
        -------
        iterator : Iterator[RandomVariable]
            An iterator over the components of the random vector.
        """
        return iter(self.components)

    # TODO: write unit tests
    def restrict_to(
        self, event: Event | list, event_name: Hashable | None = "A"
    ) -> RandomVector:
        r"""Restrict the random vector to an event.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : Event | list
            The event to restrict the random vector to.
        event_name : Hashable | None, default="A"
            The name to use for the event in the name of the resulting restricted random vector. This parameter is only used if `event` is a list of sample points, and is otherwise ignored if `event` is an `Event` instance.

        Raises
        ------
        TypeError
            If `event` is not an `Event` or a list of sample points.
        ValueError
            If `event` is not in the sigma-algebra of the random vector.

        Returns
        -------
        restricted_rv : RandomVector
            A new `RandomVector` representing the restriction of the original random vector to the given event.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     }
        ... )
        >>> X = RandomVector(Omega, F, P).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     }
        ... )
        >>> A = F.get_event([1, 2, 3])
        >>> X_A = X.restrict_to(A)
        >>> print(X_A)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        X        X_0  X_1
        A
        1          3    4
        2          3    4
        3          5    6
        >>> print(X_A.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (A, F_A, P_A)
        ===============================
        <BLANKLINE>
        * Sample space 'A':
         A
         1
         2
         3
        <BLANKLINE>
        * Sigma algebra 'F_A':
                    F_A
        A
        1             1
        2             1
        3             2
        <BLANKLINE>
        * Probability measure 'P_A':
                probability
        F_A
        1              0.625
        2              0.375
        >>> X_B = X.restrict_to([1, 2], event_name="B")
        >>> print(X_B)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|B':
        X        X_0  X_1
        B
        1          3    4
        2          3    4
        >>> print(X_B.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (B, F_B, P_B)
        ===============================
        <BLANKLINE>
        * Sample space 'B':
         B
         1
         2
        <BLANKLINE>
        * Sigma algebra 'F_B':
                    F_B
        B
        1             1
        2             1
        <BLANKLINE>
        * Probability measure 'P_B':
                probability
        F_B
        1                1.0

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F}, P)$. If $A\in \mathcal{F}$ is an event, then we may restrict the random vector to obtain the function $X|_A : A \to \mathbb{R}^d$ on $A$. If $A$ is an event of nonzero probability, then $A$ carries the conditional probability distribution $P_A$, defined so that $P_A(B) = P(B) / P(A)$, for $B\subset A$.
        """
        from ..base.event import Event
        from ..base.probability_space import ProbabilitySpace

        if not isinstance(event, (Event, list)):
            raise TypeError("event must be an Event or a list of sample points.")

        if isinstance(event, list):
            try:
                event = self.sig_alg.get_event(event, name=event_name)
            except ValueError as e:
                raise ValueError(
                    "Event must be in the sigma-algebra of the random vector."
                ) from e
        elif isinstance(event, Event) and event not in self.sig_alg:
            raise ValueError("Event must be in the sigma-algebra of the random vector.")

        event_prob_space = ProbabilitySpace.from_event(
            event=event, prob_measure=self.prob_measure
        )
        event_data = self.data.loc[event.indices]
        event_data.index.name = event.name
        name = (
            f"{self.name}|{event.name}"
            if (self.name is not None and event.name is not None)
            else None
        )
        result = RandomVector(*event_prob_space, name=name).from_pandas(event_data)

        if result.dimension == 1:
            result = result.to_random_variable()

        return result

    # TODO: write unit tests
    def get_component_rv(self, index: Hashable) -> RandomVariable:
        r"""Get a component random variable of the random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        index : Hashable
            The feature index for which to get the component random variable.

        Returns
        -------
        component_rv : RandomVariable
            A new `RandomVariable` representing the component random variable.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          3    4
        >>> X_1 = X.get_component_rv("X_1")
        >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_1':
                X_1
        Omega
        0         2
        1         4

        Notes
        -----
        Given a random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F}, P)$, for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega), X_2(\omega), \ldots, X_d(\omega)),
        $$

        where $X_j: \Omega \to \mathbb{R}$ are the *component random variables* of $X$.
        """
        component_rv = self.get_sub_vector([index]).to_random_variable()
        component_rv.name = index
        return component_rv

    # TODO: write unit tests
    def get_sub_vector(self, feature_indices: list[Hashable]) -> RandomVector:
        r"""Get a sub-vector of the random vector by selecting a collection of component random variables.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        feature_indices : list[Hashable]
            List of feature indices to select for the sub-vector.

        Returns
        -------
        sub_vector : RandomVector
            A new `RandomVector` containing only the specified feature indices.

        Raises
        ------
        ValueError
            If any feature index is not found.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [(1, 2, 3), (4, 5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1  X_2
        Omega
        0          1    2    3
        1          4    5    6
        >>> X_sub = X.get_sub_vector(["X_0", "X_2"])
        >>> print(X_sub) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X_sub':
        X        X_0  X_2
        Omega
        0          1    3
        1          4    6

        Notes
        -----
        Given a random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F}, P)$, for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega), X_2(\omega), \ldots, X_d(\omega)),
        $$

        where $X_j: \Omega \to \mathbb{R}$ are the component random variables of $X$. We may create a *sub-vector* by choosing a collection of the component random variables to get a random vector of smaller dimension. For example, we may select the first and last random variables to create the $2$-dimensional random vector

        $$
        \omega \mapsto (X_1 (\omega), X_d(\omega)).
        $$
        """
        if self.dimension == 1:
            raise ValueError("Cannot get sub-vector of a 1-dimensional RandomVector.")
        invalid_features = [fi for fi in feature_indices if fi not in self.index]
        if invalid_features:
            raise ValueError(f"Feature indices {invalid_features} not found.")

        sub_data = self.data[feature_indices]
        name = f"{self.name}_sub" if self.name is not None else None
        return RandomVector(*self.prob_space, name=name).from_pandas(sub_data)

    def item(self) -> Hashable | FeatureVector:
        """Get the output value of a constant random vector.

        Returns
        -------
        output : Hashable | FeatureVector
            The single output value of the random vector. If the dimension of the random vector is > 1, then the return value is an instance of `FeatureVector`; otherwise, the return value is a `Hashable`.

        Raises
        ------
        ValueError
            If the random vector is not constant.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 2), (1, 2)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          1    2
        >>> print(X.item()) # doctest: +NORMALIZE_WHITESPACE
        Feature vector 'X_item':
              sample_point
        X
        X_0              1
        X_1              2
        >>> outputs_Y = dict(zip(Omega, [1, 1]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        Omega
        0       1
        1       1
        >>> print(Y.item())
        1
        """
        sample_point = self.domain[0]
        item = self(sample_point)

        if self.dimension != 1:
            if self.data.nunique().sum() != self.dimension:
                raise ValueError(
                    "item() can only be called on a constant random vector."
                )
            item.name = f"{self.name}_item" if self.name is not None else None
            item.data.name = "sample_point"
        else:
            if self.data.nunique() != 1:
                raise ValueError(
                    "item() can only be called on a constant random vector."
                )

        return item

    def round(self, decimals: int = 0) -> RandomVector:
        """Round the feature vectors of the random vector to a specified number of decimal places.

        Parameters
        ----------
        decimals : int, default=0
            The number of decimal places to round to. Must be a non-negative integer.

        Raises
        ------
        ValueError
            If `decimals` is not a non-negative integer, or if the random vector's data is not set.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [(0, np.pi), (np.pi / 2, 3 * np.pi / 2)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(np.sin(X).round()) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'sin(X)':
        sin(X)   sin(X_0)  sin(X_1)
        Omega
        0             0.0       0.0
        1             1.0      -1.0
        """
        if not isinstance(decimals, int) or decimals < 0:
            raise ValueError("decimals must be a non-negative integer.")
        if self._data is None:
            raise ValueError("Data must be set to round the random vector.")

        self._data = self.data.round(decimals=decimals)
        return self

    def iter_features(self):
        """Iterate over sample points and their feature vectors.

        Yields tuples of `(sample_index, FeatureVector)` for each sample point in the domain, allowing iteration over the random vector's entire domain.

        Yields
        ------
        sample_index : Hashable
            Index of the sample point.
        features : FeatureVector
            Feature vector of the sample point.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2, prefix="s")
        >>> X = RandomVector(domain=Omega).from_dict(outputs={"s_0": (1, 2), "s_1": (3, 4)})
        >>> for _, features in X.iter_features():
        ...     print(features) # doctest: +NORMALIZE_WHITESPACE
        Feature vector 'X(s_0)':
                 s_0
        X
        X_0        1
        X_1        2
        Feature vector 'X(s_1)':
                 s_1
        X
        X_0        3
        X_1        4
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs={"s_0": 1, "s_1": 2})
        >>> for idx, features in Y.iter_features():
        ...     print(f"Feature of {idx}: ", features)
        Feature of s_0:  1
        Feature of s_1:  2
        """
        for sample_index in self.data.index:
            yield sample_index, self(sample_index)

    # TODO: write unit tests
    def apply_to_features(
        self, function: Callable[[FeatureVector | Hashable], any]
    ) -> RandomVariable:
        """Apply a function to the feature vector of each sample point.

        Applies the given function to each sample point's feature vector,
        returning an instance of `RandomVariable`.

        Parameters
        ----------
        function : Callable[[FeatureVector | Hashable], any]
            Function that takes a `FeatureVector` object (in dimension > 1) or a `Hashable` (in dimension 1) and returns a value.

        Returns
        -------
        results : RandomVariable
            An instance of `RandomVariable` whose values are given by the function applied to each feature vector of the random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 2), (3, 4)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X        X_0  X_1
        Omega
        0          1    2
        1          3    4
        >>> X.apply_to_features(lambda f: f.sum() + 2) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_apply':
                X_apply
        Omega
        0             5
        1             9
        >>> outputs_Y = dict(zip(Omega, [5, 10]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        Omega
        0        5
        1       10
        >>> Y.apply_to_features(lambda x: x * 2) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y_apply':
                Y_apply
        Omega
        0            10
        1            20
        """
        from ..base.feature_vector import FeatureVector
        from .random_variable import RandomVariable

        if self.dimension > 1:

            def wrapper(row):
                sp = FeatureVector.from_pandas(data=row)
                return function(sp)

            data = self.data.apply(wrapper, axis=1)
        else:
            data = self.data.apply(function)

        name = f"{self.name}_apply"
        rv = RandomVariable(domain=self.domain, name=name).from_pandas(data)

        return rv

    # --------------------- equality --------------------- #

    # TODO: write unit tests
    def __eq__(self, other: RandomVector, rtol=1e-5, atol=1e-8) -> bool:
        r"""Check equality with another random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector
            Another random vector to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `RandomVector` with the same domain, feature index, and data.

        Notes
        -----
        Two random vector $X,Y: \Omega \to \mathbb{R}^d$ on the same probability space $(\Omega, \mathcal{F}, P)$ are equal if $X(\omega) = Y(\omega)$ for all $\omega \in \Omega$.
        """
        if not isinstance(other, RandomVector):
            return False
        if not self.domain == other.domain:
            return False
        return np.allclose(
            self.data.to_numpy(), other.data.to_numpy(), rtol=rtol, atol=atol
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the random vector.

        Returns
        -------
        repr_str : str
            The string representation of the random vector.
        """
        if self.data is None:
            return f"Random vector '{self.name}': empty"
        else:
            if self.dimension == 1:
                data = self.data.to_frame()
                data.columns = [self.name]
            else:
                data = self.data

            if self.is_identity:
                return f"Identity random vector '{self.name}':\n{self.data.to_string(index=False)}"
            else:
                return f"Random vector '{self.name}':\n{data}"

    # --------------------- arithmetic operations --------------------- #

    @staticmethod
    def _type(x):
        from ...processes.base.stochastic_process import StochasticProcess

        if isinstance(x, Real):
            return "Number"
        elif isinstance(x, StochasticProcess):
            return "StochasticProcess"
        else:
            return type(x).__name__

    def _apply_operation(
        self,
        other: RandomVector | Real,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
    ) -> RandomVector:
        """Apply a binary operation to this random vector.

        Parameters
        ----------
        self : RandomVector
            The left operand (or right if reverse=True).
        other : RandomVector | Real
            The right operand (or left if reverse=True).
        operation : Callable
            The pandas operation to apply (e.g., `lambda a, b: a + b`).
        op_symbol : str
            Symbol representing the operation (e.g., '+', '-', '*').
        reverse : bool, default=False
            Whether this is a reverse operation (e.g., __radd__ vs __add__).

        Returns
        -------
        result : RandomVector
            A new random vector representing the result of the operation.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If operating on two `RandomVector` instances with different domains or dimensions.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from ..base.index import Index
        from .random_variable import RandomVariable

        types = {self._type(self), self._type(other)}

        if types == {"RandomVariable"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} RandomVariables on incompatible probability spaces."
                )

            if reverse:
                new_name = (
                    f"({other.name}{op_symbol}{self.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(other.data, self.data).rename(new_name)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(self.data, other.data).rename(new_name)

            result = RandomVariable(*super_space, name=new_name).from_pandas(
                data=new_values
            )

            return result

        elif types == {"StochasticProcess"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} StochasticProcesses on incompatible probability spaces."
                )
            if len(self) != len(other):
                raise ValueError(
                    "The length of the StochasticProcesses must be the same."
                )
            if self.time != other.time:
                raise ValueError(
                    "The time indices of the StochasticProcesses must be the same"
                )

            if reverse:
                new_name = (
                    f"({other.name}{op_symbol}{self.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(other.data, self.data).rename(new_name)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(self.data, other.data).rename(new_name)

            result = StochasticProcess(
                *super_space,
                name=new_name,
                time=self.time,
                is_discrete_state=self.is_discrete_state,
            ).from_pandas(data=new_values)

            return result

        elif types == {"RandomVector"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} RandomVectors on incompatible probability spaces."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")

            self_data = self.data.copy()
            other_data = other.data.copy()
            if self.dimension > 1:
                self_data.columns = pd.RangeIndex(self.dimension)
                other_data.columns = pd.RangeIndex(other.dimension)

            if reverse:
                new_name = (
                    f"({other.name}{op_symbol}{self.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(other_data, self_data)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(self_data, other_data)

            result = RandomVector(*super_space, name=new_name).from_pandas(
                data=new_values
            )
            result.index = Index(name=new_name).from_sequence(
                size=self.dimension, prefix=new_name
            )

            return result

        elif types == {"Number", "RandomVariable"}:
            if reverse:
                new_name = (
                    f"({other}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
                new_values = operation(other, self.data).rename(new_name)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other})"
                    if self.name is not None
                    else None
                )
                new_values = operation(self.data, other).rename(new_name)

            result = RandomVariable(*self.prob_space, name=new_name).from_pandas(
                data=new_values
            )

            return result

        elif types == {"Number", "StochasticProcess"}:
            if reverse:
                new_name = (
                    f"({other}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
                new_values = operation(other, self.data).rename(new_name)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other})"
                    if self.name is not None
                    else None
                )
                new_values = operation(self.data, other).rename(new_name)

            result = StochasticProcess(
                *self.prob_space,
                name=new_name,
                time=self.time,
                is_discrete_state=self.is_discrete_state,
            ).from_pandas(data=new_values)

            return result

        elif types == {"Number", "RandomVector"}:
            if reverse:
                new_name = (
                    f"({other}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
                new_values = operation(other, self.data)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other})"
                    if self.name is not None
                    else None
                )
                new_values = operation(self.data, other)

            result = RandomVector(*self.prob_space, name=new_name).from_pandas(
                data=new_values
            )
            result.index = Index(name=new_name).from_sequence(
                size=self.dimension, prefix=new_name
            )

            return result

        elif types == {"RandomVariable", "RandomVector"}:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

        elif types == {"RandomVariable", "StochasticProcess"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} a RandomVariable with a StochasticProcess on different probability spaces."
                )

            if self._type(self) == "RandomVariable":
                if reverse:
                    new_name = (
                        f"({other.name}{op_symbol}{self.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )
                    new_values = operation(
                        other.data, self.data.values.reshape(-1, 1)
                    ).rename(new_name)
                else:
                    new_name = (
                        f"({self.name}{op_symbol}{other.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )
                    new_values = operation(
                        self.data.values.reshape(-1, 1), other.data
                    ).rename(new_name)

                result = StochasticProcess(
                    *self.prob_space, name=new_name, time=other.time
                ).from_pandas(data=new_values)

            else:
                if reverse:
                    new_name = (
                        f"({other.name}{op_symbol}{self.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )
                    new_values = operation(
                        other.data.values.reshape(-1, 1), self.data
                    ).rename(new_name)
                else:
                    new_name = (
                        f"({self.name}{op_symbol}{other.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )
                    new_values = operation(
                        self.data, other.data.values.reshape(-1, 1)
                    ).rename(new_name)

                result = StochasticProcess(
                    *self.prob_space, name=new_name, time=other.time
                ).from_pandas(data=new_values)

            return result

        elif types == {"RandomVector", "StochasticProcess"}:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

        else:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

    def __add__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector."""
        return self._apply_operation(other, lambda a, b: a + b, "+")

    def __radd__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a + b, "+", reverse=True)

    def __sub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract another random vector or a scalar from this random vector."""
        return self._apply_operation(other, lambda a, b: a - b, "-")

    def __rsub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract this random vector from another random vector or a scalar (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a - b, "-", reverse=True)

    def __mul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a * b, "*")

    def __rmul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a * b, "*", reverse=True)

    def __truediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a / b, "/")

    def __rtruediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a / b, "/", reverse=True)

    def __pow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a**b, "**")

    def __rpow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a**b, "**", reverse=True)

    def __array_ufunc__(
        self, ufunc, method, *inputs, **kwargs
    ) -> RandomVector | StochasticProcess | RandomVariable:
        """Override NumPy ufuncs to operate on RandomVector instances.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc object that was called.
        method : str
            A string indicating which ufunc method was called (e.g., '__call__', 'reduce', etc.).
        inputs : tuple
            A tuple of the input arguments to the ufunc.
        kwargs : dict
            A dictionary of keyword arguments passed to the ufunc.

        Returns
        -------
        result : RandomVector | StochasticProcess | RandomVariable
            A new instance of `RandomVector`, `StochasticProcess`, or `RandomVariable`
            containing the result of applying the ufunc to the inputs.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from ..base.index import Index
        from .random_variable import RandomVariable

        if method != "__call__":
            return NotImplemented

        new_inputs = [
            input.data if isinstance(input, RandomVector) else input for input in inputs
        ]
        result_data = getattr(ufunc, method)(*new_inputs, **kwargs)
        new_name = f"{ufunc.__name__}({self.name})" if self.name is not None else None

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                *self.prob_space, name=new_name, time=self.time
            ).from_pandas(data=result_data)
        elif isinstance(self, RandomVariable):
            result = RandomVariable(*self.prob_space, name=new_name).from_pandas(
                data=result_data
            )
            result.data.name = new_name
            return result
        else:
            if self.dimension > 1 and self.name is not None:
                new_index = Index(name=new_name).from_list(
                    [f"{ufunc.__name__}({idx_name})" for idx_name in self.index]
                )
                result_data.columns = new_index.data
            return RandomVector(*self.prob_space, name=new_name).from_pandas(
                data=result_data
            )

    # --------------------- comparison methods --------------------- #

    def __bool__(self) -> bool:
        """Prevent ambiguous boolean conversion of a random vector.

        Raises
        ------
        ValueError
            Always raised to prevent ambiguous boolean evaluation.
            Use explicit methods like .all() or .any() instead.
        """
        raise ValueError(
            "The truth value of a RandomVector is ambiguous. "
            "Use .all() or .any() methods, or check specific conditions explicitly."
        )

    def all(self) -> bool:
        """Check if all values in the random vector are True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for all sample points and all components.

        Returns
        -------
        all_true : bool
            True if all values across all samples and features are True.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 1), (1, 1)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X.all())
        True
        >>> outputs_Y = dict(zip(Omega, [(1, 0), (1, 0)]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y.all())
        False
        """
        return bool(self.data.all().all() if self.dimension > 1 else self.data.all())

    def any(self) -> bool:
        """Check if any value in the random vector is True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for at least one sample point or component.

        Returns
        -------
        any_true : bool
            True if any value across all samples and features is True.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 0), (0, 0)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X.any())
        True
        >>> outputs_Y = dict(zip(Omega, [(0, 0), (0, 0)]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y.any())
        False
        """
        return bool(self.data.any().any() if self.dimension > 1 else self.data.any())

    def _apply_comparison(
        self,
        other: RandomVector | Real,
        op: Callable,
        op_symbol: str,
    ) -> RandomVector:
        """Apply a comparison operation to this random vector.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.
        op : Callable
            The numpy comparison to apply (e.g., ``operator.lt``).
        op_symbol : str
            Symbol representing the comparison (e.g., '<', '<=', '>', '>=').

        Returns
        -------
        result : RandomVector
            A new random vector of booleans representing the comparison result.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or scalar.
        ValueError
            If the random vectors do not have the same domain or dimension.
        """
        from ...core.base.index import Index
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if not isinstance(other, RandomVector) and isinstance(other, Real):
            other = RandomVector(
                *self.prob_space, index=self.index, name=other
            ).from_constant(constant=other)
        elif not isinstance(other, RandomVector):
            raise TypeError("other must be a RandomVector")
        if self.domain != other.domain:
            raise ValueError("Random vectors must have the same domain")
        if self.dimension != other.dimension:
            raise ValueError("Random vectors must have the same dimension")

        comparison_arr = op(self.data.to_numpy(), other.data.to_numpy())
        name = (
            f"({self.name} {op_symbol} {other.name})"
            if self.name and other.name
            else None
        )

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                *self.prob_space, name=name, time=self.time
            ).from_numpy(array=comparison_arr)

        elif isinstance(self, RandomVariable):
            result = RandomVariable(*self.prob_space, name=name).from_numpy(
                array=comparison_arr.flatten()
            )
            result.data.name = name
            return result
        else:
            result = RandomVector(*self.prob_space, name=name).from_numpy(
                array=comparison_arr
            )
            if name is not None:
                index = Index(name=name).from_sequence(size=self.dimension, prefix=name)
                result._index = index
                result.data.columns = index.data
            return result

    def __lt__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is less than another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_lt: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) < Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) < c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.lt, "<")

    def __le__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is less than or equal to another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_le: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than or equal to the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) \leq Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) \leq c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.le, "<=")

    def __gt__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is greater than another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_gt: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is greater than the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) > Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) > c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.gt, ">")

    def __ge__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is greater than or equal another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_ge: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is greater than or equal the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) \geq Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) \geq c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.ge, ">=")
