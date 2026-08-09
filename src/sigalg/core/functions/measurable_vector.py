"""A class representing a measurable vector."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from itertools import combinations
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .operators import OperatorsMethods

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..indices.index import Index
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measurable_set import MeasurableSet
    from ..spaces.measurable_space import MeasurableSpace
    from ..spaces.measure_space import MeasureSpace
    from .measurable_function import MeasurableFunction


class MeasurableVector(OperatorsMethods):
    r"""A class representing a measurable vector.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The domain of the underlying measurable space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra of the underlying measurable space.
    measure : Measure | None, default=None
        An optional measure carried by the measurable vector.
    mapping : MappingLike | None, default=None
        The mapping defining the measureable vector.
    index : IndexLike | None, default=None
        The index of the measurable vector.
    name : Hashable, default="f"
        The name of the measurable vector.

    Examples
    --------
    >>> from sigalg.core import (
    ...     Domain,
    ...     MeasurableSpace,
    ...     MeasurableVector,
    ...     SigmaAlgebra,
    ... )

    Generate a 2-dimensional measurable vector on a pre-existing domain from a dictionary mapping. The power-set sigma-algebra is automatically generated.

    >>> X = Domain.from_sequence(size=3)
    >>> f = MeasurableVector(
    ...     domain=X,
    ...     mapping={
    ...         0: (1, 1),
    ...         1: (1, 1),
    ...         2: (2, 2),
    ...     },
    ...     name="f",
    ... )
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Measurable vector 'f':
    index   0  1
    point
    0       1  1
    1       1  1
    2       2  2
    >>> print(f.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'power_set':
            point
    point
    0           0
    1           1
    2           2

    Generate a measurable vector on a pre-existing measurable space.

    >>> F = SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     },
    ... )
    >>> measurable_space = MeasurableSpace(X, F)
    >>> g = MeasurableVector(
    ...     *measurable_space,
    ...     mapping={
    ...         0: (1, 1),
    ...         1: (1, 1),
    ...         2: (2, 2),
    ...     },
    ...     name="g",
    ... )
    >>> print(g.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
            atom_ID
    point
    0             0
    1             0
    2             1

    Attempt to define a measurable vector that is not measurable.

    >>> h = MeasurableVector(
    ...     *measurable_space,
    ...     mapping={
    ...         0: (1, 2),
    ...         1: (3, 4),
    ...         2: (5, 6),
    ...     },
    ...     name="h",
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: Function h is not measurable.

    Generate a 2-dimensional measurable vector from a function on a domain and a custom index.

    >>> S = Domain([(0, 1), (1, 2)], variable_names=["x", "y"], name="S")
    >>> def mapping(*, x, y):  # noqa: D103
    ...     return (x + y, x)
    >>> v = MeasurableVector(
    ...     domain=S,
    ...     mapping=mapping,
    ...     index=[1, 2],
    ...     name="v",
    ... )
    >>> print(v)  # doctest: +NORMALIZE_WHITESPACE
    Measurable vector 'v':
    index  1  2
    x y
    0 1    1  0
    1 2    3  1

    Notes
    -----
    Given a measurable space $(X,\mathcal{F})$, a *measurable vector* is an $\mathcal{F}$-measurable function $f: X \to \mathbb{R}^d$, where $d$ is the *dimension* of the vector and $\mathbb{R}^d$ is equipped with its Borel $\sigma$-algebra. If $X$ is finite (as it always is, in SigAlg), then $f$ is $\mathcal{F}$-measurable if and only if $f$ is constant on the atoms of $\mathcal{F}$.
    """

    _properties = [
        "_dimension",
        "_components",
        "_atom_data",
        "_component_names",
        "_generated_sig_alg",
        "_range",
        "_is_identity",
    ]
    _repr_name = "MeasurableVector"
    _str_name = "Measurable vector"
    _default_name = "f"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        mapping: MappingLike | None = None,
        index: IndexLike | None = None,
        name: Hashable = "f",
    ) -> None:
        from ...processes.base.stochastic_process import StochasticProcess
        from ...validation.mapping_validator import MappingValidator
        from ..indices.index import Index
        from ..indices.time import Time
        from ..measures.probability_measure import ProbabilityMeasure
        from ..spaces.domain import Domain
        from ..spaces.measurable_space import MeasurableSpace
        from ..spaces.measure_space import MeasureSpace
        from .measurable_function import MeasurableFunction
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

        if domain is not None and not isinstance(domain, Domain):
            domain = Domain(domain)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)

        v = MappingValidator(
            mapping=mapping,
            domain=domain,
            output_name=name,
            index=index,
            index_kind="time" if isinstance(self, StochasticProcess) else "any",
            multi_dim_outputs=True,
            domain_kind="sample_space" if isinstance(self, RandomVector) else "any",
            name=name,
        )
        self._data = v.data
        self._index = v.index
        self._name = v.name
        domain = v.domain

        self._initialize_property_caches()

        if measure is not None:
            self._measure_space = MeasureSpace(
                domain=domain,
                sig_alg=sig_alg,
                measure=measure,
            )
            self._measurable_space = self._measure_space.measurable_space
        else:
            self._measurable_space = MeasurableSpace(
                domain=domain,
                sig_alg=sig_alg,
            )
            self._measure_space = None

        if self.dimension == 1 and not isinstance(self, MeasurableFunction):
            self._data = (
                self._data.squeeze(axis=1)
                if isinstance(self._data, pd.DataFrame)
                else self._data
            )
            self._data.name = self._name
            self._index = None
            self.__class__ = MeasurableFunction

        if measure is not None and isinstance(measure, ProbabilityMeasure):
            if self.dimension > 1:
                self.__class__ = RandomVector
            else:
                self.__class__ = RandomVariable

        if isinstance(self.index, Time) and not isinstance(self, StochasticProcess):
            self.__class__ = StochasticProcess

        if self.sig_alg is not None and not self.sig_alg.is_power_set:
            combined_data = pd.concat(
                [self._data, self.sig_alg.data], axis=1
            ).drop_duplicates()
            if len(combined_data) != self.sig_alg.num_atoms:
                raise ValueError(f"Function {self._name} is not measurable.")

    def _initialize_property_caches(self, exceptions: set | None = None) -> None:
        if exceptions is None:
            exceptions = set()
        for property in set(self._properties) - exceptions:
            setattr(self, property, None)

    @classmethod
    def from_constant(
        cls,
        domain: IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        constant: Hashable | None = None,
        index: IndexLike | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Create a `MeasurableVector` that maps every point in the domain to the same constant output vector.

        Parameters
        ----------
        domain: IndexLike
            The domain of the measurable vector.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying measurable space. If `None`, the power set sigma-algebra is used.
        measure: Measure | None, default=None
            An optional measure carried by the measurable vector.
        constant : Hashable | None, default=None
            The constant output vector that every point in the domain maps to.
        index : IndexLike | None, default=None
            The index of the measurable vector.
        name : Hashable | None, default=None
            The name of the measurable vector. If `None`, a default will be generated.

        Raises
        ------
        TypeError
            If `constant` is not a `Hashable`.
        ValueError
            If `constant` is a tuple and its length does not match the length of `index`.

        Returns
        -------
        vector : MeasurableVector
            A measurable vector mapping every sample point in the domain to the same constant output vector.

        Examples
        --------
        Create a constant 2-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=3)
        >>> f = MeasurableVector.from_constant(domain=X, constant=(1, 2), index=[1, 2])
        >>> print(f) # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index    1  2
        point
        0        1  2
        1        1  2
        2        1  2

        Create a constant 1-dimensional measurable function.

        >>> g = MeasurableVector.from_constant(domain=X, constant=2, name="g")
        >>> print(g) # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        point
        0       2
        1       2
        2       2
        """
        from ..indices.index import Index
        from ..spaces.domain import Domain

        if domain is not None and not isinstance(domain, Domain):
            domain = Domain(domain)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)
        if name is None:
            name = cls._default_name

        if not isinstance(constant, Hashable):
            raise TypeError("constant must be a Hashable.")
        if (
            index is not None
            and isinstance(constant, tuple)
            and len(constant) != len(index)
        ):
            raise ValueError(
                "Length of constant tuple must match the length of the index."
            )

        if constant is not None:
            if isinstance(constant, tuple):
                mapping = dict.fromkeys(domain.data, constant)
            elif index is not None:
                mapping = dict.fromkeys(domain.data, (constant,) * len(index))
            else:
                mapping = dict.fromkeys(domain.data, constant)
        else:
            mapping = None

        return cls(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    # TODO: an identify vector tracks its domain both as the `domain` attribute and as the mapping. very redundant. find ways around this
    @classmethod
    def from_identity(
        cls,
        domain: IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        index: IndexLike | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Create a measurable vector that maps every point in the domain to itself.

        For this construction method, the sigma-algebra must be the power set.

        Parameters
        ----------
        domain: IndexLike
            The domain of the measurable vector.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying measurable space. The sigma-algebra must be the power-set. This parameter is here only for consistency with other constructors.
        measure: Measure | None, default=None
            An optional measure carried by the measurable vector.
        index : IndexLike | None, default=None
            The index of the measurable vector.
        name : Hashable | None, default=None
            The name of the measurable vector. If `None`, a default will be generated.

        Raises
        ------
        ValueError
            If the sigma-algebra is not the power set (if given), or if the length of the index (if given) does not match the dimension of the domain.

        Returns
        -------
        vector : MeasurableVector
            A measurable vector mapping every point in the domain to itself.

        Examples
        --------
        Create an identity vector on a 2-dimensional domain.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.cartesian_power(
        ...     [0, 1], n=2, name="X", variable_names=["x", "y"]
        ... )
        >>> f = MeasurableVector.from_identity(domain=X)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index  0  1
        x y
        0 0    0  0
          1    0  1
        1 0    1  0
          1    1  1

        Print its range.

        >>> print(f.range)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, power_set)
        ===============================
        * Domain 'X':
         x  y
         0  0
         0  1
         1  0
         1  1
        <BLANKLINE>
        * Sigma algebra 'power_set':
             x  y
        x y
        0 0  0  0
          1  0  1
        1 0  1  0
          1  1  1

        Now define an identity vector on a 1-dimensional domain and print its range.

        >>> S = Domain(indices=["a", "b"], name="S")
        >>> g = MeasurableVector.from_identity(domain=S, name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        point
        a       a
        b       b
        >>> print(g.range)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (S, power_set)
        ===============================
        <BLANKLINE>
        * Domain 'S':
         point
             a
             b
        <BLANKLINE>
        * Sigma algebra 'power_set':
               point
        point
        a          a
        b          b
        """
        from ..indices.index import Index
        from ..spaces.domain import Domain

        if sig_alg is not None and not sig_alg.is_power_set:
            raise ValueError(
                "The sigma-algebra must be the power set for an identity measurable vector."
            )
        if index is not None and len(index) != domain.dimension:
            raise ValueError(
                "The length of the index must match the dimension of the domain."
            )
        if name is None:
            name = cls._default_name

        if domain is not None and not isinstance(domain, Domain):
            domain = Domain(domain)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)

        mapping = domain.data.to_frame()
        if mapping.shape[1] == 1:
            mapping = mapping.squeeze(axis=1)
            mapping.name = name
        else:
            mapping = domain.data.to_frame()
            if index is None:
                index = Index.from_sequence(size=mapping.shape[1])
            mapping.columns = index.data

        vector = cls(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            name=name,
        )

        vector._range = vector.measurable_space
        vector._is_identity = True

        return vector

    @classmethod
    def from_randint(
        cls,
        domain: IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        diff_values: int = 0,
        low: int = 0,
        high: int = 2,
        dim: int | None = None,
        index: IndexLike | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Generate a measurable vector with integer outputs uniformly sampled from the range [low, high).

        Parameters
        ----------
        domain: IndexLike
            The domain of the measurable vector.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying measurable space. If `None`, the power set sigma-algebra is used.
        measure: Measure | None, default=None
            An optional measure carried by the measurable vector.
        diff_values : int, default=0
            If nonzero, the vector is randomly generated so that it is measurable with respect to a randomly generated sub-sigma-algebra of `sig_alg`. Then `diff_values = sig_alg.num_atoms - sub_sig_alg.num_atoms`. See the Examples section.
        low : int, default=0
            The lower bound (inclusive) of the random integers.
        high : int, default=2
            The upper bound (exclusive) of the random integers.
        dim : int | None, default=None
            The dimension of the measurable vector. Either `dim` or `index` may be provided to set the dimension of the measurable vector. If neither is provided, `dim` will default to `1`.
        index : IndexLike | None, default=None
            The index of the measurable vector. Either `dim` or `index` may be provided to set the dimension of the measurable vector. If neither is provided, `dim` will default to `1`.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for a random number generator.
        name : Hashable | None, default=None
            The name of the measurable vector. If `None`, a default will be generated.

        Raises
        ------
        TypeError
            If `low` or `high` are not integers, if `sig_alg` is not an instance of `SigmaAlgebra` or `None`, or if `random_state` is not an integer, `np.random.Generator`, or `None`.
        ValueError
            If `dim` is not positive, if both `dim` and `index` are provided, or if `low` is greater than or equal to `high`.

        Returns
        -------
        vector : MeasurableVector
            A measurable vector with integer outputs uniformly sampled from the range [low, high).

        Examples
        --------
        Create a 2-dimensional measurable vector with integer outputs uniformly sampled from the range [0, 5).

        >>> import numpy as np
        >>> from sigalg.core import Domain, MeasurableVector, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...         5: 3,
        ...     },
        ... )
        >>> f = MeasurableVector.from_randint(
        ...     domain=X,
        ...     sig_alg=F,
        ...     low=0,
        ...     high=5,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index  0  1
        point
        0      0  3
        1      0  3
        2      3  2
        3      2  4
        4      2  4
        5      0  3

        The maximum number of unique values of a measurable vector is equal to the number of atoms of the underlying sigma-algebra. Notice that this last vector achieves this upper bound. We can decrease the number of unique values by generating the vector so that it is measurable with respect to a sub-sigma-algebra by specifying a nonzero value for the `diff_values` parameter. This parameter is equal to `diff_values = sig_alg.num_atoms - sub_sig_alg.num_atoms`.

        >>> g = MeasurableVector.from_randint(
        ...     domain=X,
        ...     sig_alg=F,
        ...     diff_values=2,
        ...     low=0,
        ...     high=5,
        ...     dim=2,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        index  0  1
        point
        0      3  3
        1      3  3
        2      2  0
        3      2  0
        4      2  0
        5      2  0
        """
        from ..indices.index import Index
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain

        if not isinstance(domain, Domain):
            domain = Domain(domain)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)
        if name is None:
            name = cls._default_name
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra, if given.")
        if dim is not None and not isinstance(dim, int):
            raise TypeError("dim must be a positive integer, if given.")
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive, if given.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if dim is not None and index is not None:
            raise ValueError("Both dim and index cannot be provided.")

        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError("low and high must be integers.")
        if low >= high:
            raise ValueError("low must be less than high.")

        if dim is None and index is None:
            dim = 1
        if dim is None:
            dim = len(index)
        if index is None:
            index = Index.from_sequence(size=dim)

        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(domain)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        if diff_values > 0:
            sub_sig_alg = SigmaAlgebra.from_rand(
                super=sig_alg,
                num_atoms=sig_alg.num_atoms - diff_values,
                random_state=rng,
            )
        else:
            sub_sig_alg = sig_alg

        mapping = rng.integers(low, high, size=(sub_sig_alg.num_atoms, dim))
        mapping = pd.DataFrame(
            mapping, index=sub_sig_alg.atom_space.data, columns=index.data
        )

        sub_sig_alg_data = cls._to_df(sub_sig_alg.data)

        if sub_sig_alg.is_power_set:
            mapping = pd.merge(
                left=sub_sig_alg_data, right=mapping, left_index=True, right_index=True
            ).drop(columns=list(sub_sig_alg_data.columns))
        else:
            mapping = pd.merge(
                left=sub_sig_alg_data,
                right=mapping,
                left_on=list(sub_sig_alg_data.columns),
                right_index=True,
            ).drop(columns=list(sub_sig_alg_data.columns))

        return cls(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    @classmethod
    def from_randnorm(
        cls,
        domain: IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        diff_values: int = 0,
        loc: float = 0.0,
        scale: float = 1.0,
        dim: int | None = None,
        index: IndexLike | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Generate a measurable vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        Parameters
        ----------
        domain: IndexLike
            The domain of the measurable vector.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying measurable space. If `None`, the power set sigma-algebra is used.
        measure: Measure | None, default=None
            An optional measure carried by the measurable vector.
        diff_values : int, default=0
            If nonzero, the vector is randomly generated so that it is measurable with respect to a randomly generated sub-sigma-algebra of `sig_alg`. Then `diff_values = sig_alg.num_atoms - sub_sig_alg.num_atoms`. See the Examples section.
        loc : float, default=0.0
            The mean of the normal distribution.
        scale : float, default=1.0
            The standard deviation of the normal distribution.
        dim : int | None, default=None
            The dimension of the measurable vector. Either `dim` or `index` may be provided to set the dimension of the measurable vector. If neither is provided, `dim` will default to `1`.
        index : IndexLike | None, default=None
            The index of the measurable vector. Either `dim` or `index` may be provided to set the dimension of the measurable vector. If neither is provided, `dim` will default to `1`.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for a random number generator, or a `np.random.Generator` instance to use directly.
        name : Hashable | None, default=None
            The name of the measurable vector. If `None`, a default will be generated.

        Returns
        -------
        self : MeasurableVector
            A measurable vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        Examples
        --------
        Create a 2-dimensional measurable vector with floating-point outputs sampled from a standard normal distribution.

        >>> import numpy as np
        >>> from sigalg.core import Domain, MeasurableVector, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...         5: 3,
        ...     },
        ... )
        >>> f = MeasurableVector.from_randnorm(
        ...     domain=X,
        ...     sig_alg=F,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(f) # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index          0         1
        point
        0      0.304717 -1.039984
        1      0.304717 -1.039984
        2      0.750451  0.940565
        3     -1.951035 -1.302180
        4     -1.951035 -1.302180
        5      0.127840 -0.316243

        The maximum number of unique values of a measurable vector is equal to the number of atoms of the underlying sigma-algebra. Notice that this last vector achieves this upper bound. We can decrease the number of unique values by generating the vector so that it is measurable with respect to a sub-sigma-algebra by specifying a nonzero value for the `diff_values` parameter. This parameter is equal to `diff_values = sig_alg.num_atoms - sub_sig_alg.num_atoms`.

        >>> g = MeasurableVector.from_randnorm(
        ...     domain=X,
        ...     sig_alg=F,
        ...     diff_values=2,
        ...     dim=2,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g) # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        index         0         1
        point
        0      0.777792  0.066031
        1      0.777792  0.066031
        2      1.127241  0.467509
        3      1.127241  0.467509
        4      1.127241  0.467509
        5      0.777792  0.066031
        """
        from ..indices.index import Index
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain

        if not isinstance(domain, Domain):
            domain = Domain(indices=domain)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)
        if name is None:
            name = cls._default_name
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra, if given.")
        if dim is not None and not isinstance(dim, int):
            raise TypeError("dim must be a positive integer, if given.")
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive, if given.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if dim is not None and index is not None:
            raise ValueError("Both dim and index cannot be provided.")

        if not isinstance(loc, Real) or not isinstance(scale, Real):
            raise TypeError("loc and scale must be real numbers.")
        if scale <= 0:
            raise ValueError("scale must be positive.")

        if dim is None and index is None:
            dim = 1
        if dim is None:
            dim = len(index)
        if index is None:
            index = Index.from_sequence(size=dim)

        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(domain)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        if diff_values > 0:
            sub_sig_alg = SigmaAlgebra.from_rand(
                super=sig_alg,
                num_atoms=sig_alg.num_atoms - diff_values,
                random_state=rng,
            )
        else:
            sub_sig_alg = sig_alg

        mapping = rng.normal(loc, scale, size=(sub_sig_alg.num_atoms, dim))
        mapping = pd.DataFrame(
            mapping, index=sub_sig_alg.atom_space.data, columns=index.data
        )

        sub_sig_alg_data = cls._to_df(sub_sig_alg.data)

        if sub_sig_alg.is_power_set:
            mapping = pd.merge(
                left=sub_sig_alg_data, right=mapping, left_index=True, right_index=True
            ).drop(columns=list(sub_sig_alg_data.columns))
        else:
            mapping = pd.merge(
                left=sub_sig_alg_data,
                right=mapping,
                left_on=list(sub_sig_alg_data.columns),
                right_index=True,
            ).drop(columns=list(sub_sig_alg_data.columns))

        return cls(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    @classmethod
    def concatenate(
        cls,
        factors: list[MeasurableFunction | MeasurableVector | Real],
        index: IndexLike | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Concatenate a list of measurable vectors or scalars into a single measurable vector.

        Parameters
        ----------
        factors : list[MeasurableFunction | MeasurableVector | Real]
            A list of measurable vectors or scalars to combine.
        index : IndexLike | None, default=None
            The index of the resulting measurable vector. If `None`, the index will be generated by concatenating the indices of the input measurable vectors, provided that they are disjoint; otherwise, a new default index will be generated.
        name : Hashable | None, default=None
            The name of the resulting measurable vector. If `None`, the name will be generated by concatenating the names of the input measurable vectors.

        Raises
        ------
        TypeError
            If `factors` is not a list, if any element of `factors` is not a `MeasurableFunction`, `MeasurableVector`, or scalar, or if `name` is not a `Hashable` or `None`.
        ValueError
            If there is not at least one `MeasurableVector` instance in `factors`, or if the measurable vectors in `factors` are not defined on the same measurable space.

        Returns
        -------
        concatenation : MeasurableVector
            A new measurable vector created by combining the input measurable vectors.

        Examples
        --------
        Generate a measure space.

        >>> from sigalg.core import (
        ...     Domain,
        ...     Index,
        ...     Measure,
        ...     MeasurableFunction,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     random_state=42,
        ... )
        >>> mu = Measure(domain=F, mapping={0: 1, 1: 2, 2: 3})

        Generate two measurable vectors with disjoint indices. One has a measure, the other does not.

        >>> I = Index([0, 1, 2])
        >>> f = MeasurableVector.from_randint(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     index=I,
        ...     random_state=42,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1  2
        point
        0       0  1  1
        1       0  0  1
        2       0  1  1
        3       0  1  0
        >>> J = Index([3, 4], name="J")
        >>> g = MeasurableVector.from_randint(
        ...     domain=X,
        ...     sig_alg=F,
        ...     index=J,
        ...     random_state=42,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        index   3  4
        point
        0       0  1
        1       1  0
        2       0  1
        3       0  1

        Concatenate the two vectors. The measure of the one will propagate to the concatenation.

        >>> fg = MeasurableVector.concatenate([f, g])
        >>> print(fg)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'fg':
        index   0  1  2  3  4
        point
        0       0  1  1  0  1
        1       0  0  1  1  0
        2       0  1  1  0  1
        3       0  1  0  0  1
        >>> print(fg.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu':
             measure
        atom_ID
        1          2
        0          1
        2          3

        Generate a measurable function.

        >>> h = MeasurableFunction.from_randint(
        ...     domain=X,
        ...     sig_alg=F,
        ...     random_state=42,
        ...     name="h",
        ... )
        >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'h':
                h
        point
        0       0
        1       1
        2       0
        3       1

        Concatenate measurable functions and vectors, along with scalars using the `|` operator.

        >>> fh2Y = f | h | 2 | g
        >>> print(fh2Y)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'fh2g':
        index   0  1  2  3  4  5  6
        point
        0       0  1  1  0  2  0  1
        1       0  0  1  1  2  1  0
        2       0  1  1  0  2  0  1
        3       0  1  0  1  2  0  1

        From a concatenation with a custom index and name.

        >>> k = MeasurableVector.concatenate([0, h, f], index=[0, 1, 2, 3, 4], name="k")
        >>> print(k)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'k':
        index   0  1  2  3  4
        point
        0       0  0  0  1  1
        1       0  1  0  0  1
        2       0  0  0  1  1
        3       0  1  0  1  0
        """
        from ..indices.index import Index
        from .measurable_function import MeasurableFunction

        if not isinstance(factors, list):
            raise TypeError(
                "factors must be a list of instances of MeasurableVector and scalars."
            )
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)
        actual_rvs = [rv for rv in factors if isinstance(rv, MeasurableVector)]
        if not actual_rvs:
            raise ValueError(
                "There must be at least one measurable vector in `factors`."
            )
        measurable_space = actual_rvs[0].measurable_space
        if any(rv.measurable_space != measurable_space for rv in actual_rvs):
            raise ValueError(
                "All MeasurableVector instances must be defined on the same measurable space."
            )
        measure = cls._check_for_consistent_measures(actual_rvs)
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be a Hashable.")

        try:
            factors = [
                MeasurableFunction.from_constant(
                    *measurable_space, measure=measure, constant=rv, name=rv
                )
                if not isinstance(rv, MeasurableVector)
                else rv
                for rv in factors
            ]
        except TypeError as e:
            raise TypeError(
                "Cannot form constant measurable functions from the factors."
            ) from e

        indices = [
            rv.index if not isinstance(rv, MeasurableFunction) else Index([rv.name])
            for rv in factors
        ]

        ignore_index = any(
            len(idx1 & idx2) >= 1 for idx1, idx2 in combinations(indices, 2)
        )

        if name is None:
            name = "".join(str(rv.name) for rv in factors)

        combined_data = pd.concat(
            [rv.data for rv in factors], axis=1, ignore_index=ignore_index
        )
        if index is not None:
            if not isinstance(index, Index):
                index = Index(indices=index)
            combined_data.columns = index.data

        return cls(
            *factors[0].measurable_space,
            measure=measure,
            mapping=combined_data,
            index=index,
            name=name,
        )

    def __or__(
        self, other: MeasurableVector | Real | MeasurableSet
    ) -> MeasurableVector:
        """Concatenate the current instance with a second measurable vector, a constant measurable function (represented as a `Real`), or restrict the measurable vector to a measurable subset.

        Calls `MeasurableVector.concatenate` if `other` is a `MeasurableVector`, `MeasurableFunction`, or scalar, or calls `MeasurableVector.restrict_to` if `other` is a `MeasurableSet`. See the documentation for those methods for more details.
        """
        from ..spaces.measurable_set import MeasurableSet

        if isinstance(other, MeasurableSet):
            return self.restrict_to(measurable_set=other)
        else:
            return type(self).concatenate([self, other])

    def __ror__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Concatenate the current instance with a second measurable vector or a constant measurable function (represented as a `Real`).

        Calls `MeasurableVector.concatenate`.
        """
        return type(self).concatenate([other, self])

    @classmethod
    def cartesian_product(
        cls,
        factors: list[MeasurableVector],
        index: IndexLike | None = None,
        name: Hashable | None = None,
        domain_name: Hashable | None = None,
        sig_alg_name: Hashable | None = None,
        measure_name: Hashable | None = None,
    ) -> MeasurableVector:
        r"""Form the Cartesian product of a list of measurable vectors.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        factors : list[MeasurableVector]
            The factors of the Cartesian product.
        index : IndexLike | None, default=None
            The index of the Cartesian product. If `None`, a default index will be generated.
        name : Hashable | None, default=None
            The name of the Cartesian product. If `None`, a default will be generated.
        domain_name : Hashable | None, default=None
            The name of the domain of the Cartesian product. If `None`, a default will be generated.
        sig_alg_name : Hashable | None, default=None
            The name of the sigma-algebra of the Cartesian product. If `None`, a default will be generated.
        measure_name : Hashable | None, default=None
            The name of the measure of the Cartesian product. If `None`, a default will be generated.

        Raises
        ------
        TypeError
            If `factors` is not a list of measurable vectors.

        Returns
        -------
        product : MeasurableVector
            The Cartesian product of the measurable vectors.

        Examples
        --------
        Define the first of two random probability spaces.

        >>> import numpy as np
        >>> from sigalg.core import ProbabilitySpace, RandomVector
        >>> rng = np.random.default_rng(42)
        >>> prob_space1 = ProbabilitySpace.from_rand(
        ...     domain_size=3,
        ...     domain_variable_names=["s"],
        ...     domain_name="S",
        ...     num_atoms=2,
        ...     sig_alg_name="F",
        ...     sig_alg_variable_names=["u"],
        ...     random_state=rng,
        ...     measure_name="P",
        ... )
        >>> print(prob_space1)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (S, F, P)
        ===========================
        <BLANKLINE>
        * Sample space 'S':
         s
         2
         0
         1
        <BLANKLINE>
        * Sigma algebra 'F':
                 u
        s
        2        1
        0        1
        1        0
        <BLANKLINE>
        * Probability measure 'P':
           probability
        u
        1     0.507458
        0     0.492542

        Define the second of the two random probability spaces.

        >>> prob_space2 = ProbabilitySpace.from_rand(
        ...     domain_size=3,
        ...     domain_variable_names=["t"],
        ...     domain_name="T",
        ...     num_atoms=2,
        ...     sig_alg_name="G",
        ...     sig_alg_variable_names=["v"],
        ...     random_state=rng,
        ...     measure_name="Q",
        ... )
        >>> print(prob_space2)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (T, G, Q)
        ===========================
        <BLANKLINE>
        * Sample space 'T':
         t
         2
         0
         1
        <BLANKLINE>
        * Sigma algebra 'G':
                 v
        t
        2        1
        0        0
        1        0
        <BLANKLINE>
        * Probability measure 'Q':
           probability
        v
        1     0.182651
        0     0.817349

        Define a 2-dimensional random vector.

        >>> X = RandomVector.from_randint(
        ...    *prob_space1,
        ...    high=10,
        ...    dim=2,
        ...    random_state=rng,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index  0  1
        s
        2      5  4
        0      5  4
        1      4  2

        Define a 3-dimensional random vector.

        >>> Y = RandomVector.from_randint(
        ...     *prob_space2,
        ...     high=10,
        ...     dim=3,
        ...     random_state=rng,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        index  0  1  2
        t
        2      0  5  8
        0      0  8  8
        1      0  8  8

        Form the Cartesian product of the two random vectors using the `@` operator.

        >>> X_times_Y = X @ Y
        >>> print(X_times_Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X x Y':
        index  0  1  2  3  4
        s t
        2 2    5  4  0  5  8
          0    5  4  0  8  8
          1    5  4  0  8  8
        0 2    5  4  0  5  8
          0    5  4  0  8  8
          1    5  4  0  8  8
        1 2    4  2  0  5  8
          0    4  2  0  8  8
          1    4  2  0  8  8

        Print the underlying probability space of the Cartesian product.

        >>> print(X_times_Y.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (S x T, F x G, P x Q)
        =======================================
        <BLANKLINE>
        * Domain 'S x T':
         s  t
         2  2
         2  0
         2  1
         0  2
         0  0
         0  1
         1  2
         1  0
         1  1
        <BLANKLINE>
        * Sigma algebra 'F x G':
             u  v
        s t
        2 2  1  1
          0  1  0
          1  1  0
        0 2  1  1
          0  1  0
          1  1  0
        1 2  0  1
          0  0  0
          1  0  0
        <BLANKLINE>
        * Probability measure 'P x Q':
             probability
        u v
        1 1     0.092688
          0     0.414771
        0 1     0.089963
          0     0.402579

        Notes
        -----
        Given one measurable vector $f: X \to \mathbb{R}^d$ on a measurable space $(X,\mathcal{F})$, and a second measurable vector $g: Y \to \mathbb{R}^e$ on a measurable space $(Y,\mathcal{G})$, their *Cartesian product*, denoted $f \times g$, is the $(\mathcal{F} \times \mathcal{G})$-measurable measurable vector defined

        $$
        (f \times g) : X \times Y \to \mathbb{R}^{d+e}, \quad (f\times g)(x, y) = (f(x),g(y)).
        $$

        Here, $\mathcal{F} \times \mathcal{G}$ is the product $\sigma$-algebra.
        """
        from ..indices.index import Index
        from ..measures.measure import Measure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain

        if index is not None and not isinstance(index, Index):
            index = Index(index)

        if not isinstance(factors, list) or not all(
            isinstance(rv, MeasurableVector) for rv in factors
        ):
            raise TypeError("factors must be a list of MeasurableVectors.")

        mapping = factors[0].data
        domain = Domain.cartesian_product(
            [rv.domain for rv in factors], name=domain_name
        )

        for rv in factors[1:]:
            mapping = pd.merge(
                left=mapping,
                right=rv.data,
                how="cross",
            )
        mapping.index = domain.data

        if index is None:
            index = Index(indices=list(range(mapping.shape[1])))
        mapping.columns = index.data

        if name is None:
            name = " x ".join([rv.name for rv in factors])

        measures = [rv.measure for rv in factors if rv.measure is not None]
        all_measures = len(measures) == len(factors)

        if all_measures:
            measure = Measure.tensor_product(measures, name=measure_name)
            sig_alg = measure.sig_alg
            sig_alg.name = sig_alg_name if sig_alg_name is not None else sig_alg.name
        else:
            measure = None
            sig_alg = SigmaAlgebra.cartesian_product(
                [rv.sig_alg for rv in factors], name=sig_alg_name
            )

        if all(rv.is_identity for rv in factors):
            return MeasurableVector.from_identity(
                domain=domain,
                sig_alg=sig_alg,
                measure=measure,
                name=name,
                index=index,
            )
        else:
            return MeasurableVector(
                domain=domain,
                sig_alg=sig_alg,
                measure=measure,
                mapping=mapping,
                index=index,
                name=name,
            )

    def __matmul__(self, other: MeasurableVector) -> MeasurableVector:
        """Form the Cartesian product of a pair of measurable vectors.

        Calls the `MeasurableVector.cartesian_product` method. See the documentation of that method for details.
        """
        return type(self).cartesian_product([self, other])

    @classmethod
    def cartesian_power(
        cls,
        vector: MeasurableVector,
        n: int,
        index: IndexLike | None = None,
    ) -> MeasurableVector:
        """Form the Cartesian power of a measurable vector.

        Parameters
        ----------
        vector : MeasurableVector
            The base of the Cartesian power.
        n : int
            The power of the Cartesian power.
        index : IndexLike | None, default=None
            The index of the Cartesian power. If `None`, a default index will be generated.

        Raises
        ------
        TypeError
            If `vector` is not a `MeasurableVector` or if `n` is not an integer.
        ValueError
            If `n` is not positive.

        Examples
        --------
        Define a 2-dimensional random vector `X`.

        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.4,
        ...         2: 0.4,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Compute the second Cartesian power of the measurable vector `f` and print its measure space.

        >>> f_2 = MeasurableVector.cartesian_power(f, 2)
        >>> print(f_2)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f ^ 2':
        index    0  1  2  3
        x_0 x_1
        0   0    1  2  1  2
            1    1  2  3  4
            2    1  2  3  4
            3    1  2  5  6
        1   0    3  4  1  2
            1    3  4  3  4
            2    3  4  3  4
            3    3  4  5  6
        2   0    3  4  1  2
            1    3  4  3  4
            2    3  4  3  4
            3    3  4  5  6
        3   0    5  6  1  2
            1    5  6  3  4
            2    5  6  3  4
            3    5  6  5  6
        >>> print(f_2.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X ^ 2, F ^ 2, mu ^ 2)
        ====================================
        <BLANKLINE>
        * Domain 'X ^ 2':
         x_0  x_1
           0    0
           0    1
           0    2
           0    3
           1    0
           1    1
           1    2
           1    3
           2    0
           2    1
           2    2
           2    3
           3    0
           3    1
           3    2
           3    3
        <BLANKLINE>
        * Sigma algebra 'F ^ 2':
                 u_0  u_1
        x_0 x_1
        0   0      0    0
            1      0    1
            2      0    1
            3      0    2
        1   0      1    0
            1      1    1
            2      1    1
            3      1    2
        2   0      1    0
            1      1    1
            2      1    1
            3      1    2
        3   0      2    0
            1      2    1
            2      2    1
            3      2    2
        <BLANKLINE>
        * Measure 'mu ^ 2':
                measure
        u_0 u_1
        0   0      0.04
            1      0.08
            2      0.08
        1   0      0.08
            1      0.16
            2      0.16
        2   0      0.08
            1      0.16
            2      0.16

        Compute the third Cartesian power using the `^` operator.

        >>> f_3 = f ^ 3
        >>> print(f_3)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f ^ 3':
        index        0  1  2  3  4  5
        x_0 x_1 x_2
        0   0   0    1  2  1  2  1  2
                1    1  2  1  2  3  4
                2    1  2  1  2  3  4
                3    1  2  1  2  5  6
            1   0    1  2  3  4  1  2
        ...         .. .. .. .. .. ..
        3   2   3    5  6  3  4  5  6
            3   0    5  6  5  6  1  2
                1    5  6  5  6  3  4
                2    5  6  5  6  3  4
                3    5  6  5  6  5  6
        <BLANKLINE>
        [64 rows x 6 columns]
        """
        name = f"{vector.name} ^ {n}"
        domain_name = f"{vector.domain.name} ^ {n}"
        sig_alg_name = f"{vector.sig_alg.name} ^ {n}"
        measure_name = (
            f"{vector.measure.name} ^ {n}" if vector.measure is not None else None
        )
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)
        return cls.cartesian_product(
            factors=[vector] * n,
            name=name,
            domain_name=domain_name,
            sig_alg_name=sig_alg_name,
            measure_name=measure_name,
            index=index,
        )

    def __xor__(self, power: int) -> MeasurableVector:
        """Form the Cartesian power of this instance of `MeasurableVector`.

        Calls the `MeasurableVector.cartesian_power` method. See the documentation of that method for details.
        """
        return type(self).cartesian_power(vector=self, n=power)

    @classmethod
    def indicator_of(
        cls,
        measurable_set: MeasurableSet,
        measure: Measure | None = None,
        dim: int = 1,
        index: IndexLike | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        r"""Create the indicator measurable vector of a given measurable set of a given dimension.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        measurable_set : MeasurableSet
            The measurable set for which the indicator measurable vector is to be created.
        measure: Measure | None, default=None
            An optional measure carried by the measurable vector.
        dim : int, default=1
            The dimension of the indicator measurable vector.
        index : IndexLike | None, default=None
            The index of the indicator measurable vector. If `None`, a default index will be generated.
        name : Hashable | None, default=None
            The name of the indicator measurable vector. If `None`, a default name will be generated.

        Raises
        ------
        TypeError
            If `measurable_set` is not an instance of `MeasurableSet`, if `dim` is not an integer, or if `name` is not hashable (if given).
        ValueError
            If `dim` is not a positive integer.

        Returns
        -------
        indicator_vector : MeasurableVector
            The indicator measurable vector of the given event.

        Examples
        --------
        Get a measurable set from a sigma-algebra.

        >>> from sigalg.core import Domain, MeasureSpace, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(X)
        >>> measure_space = MeasureSpace(X, F)
        >>> A = measure_space.get_set([0, 1])

        Create an indicator vector with default name.

        >>> I_A = MeasurableVector.indicator_of(A)
        >>> print(I_A)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'I_A':
                I_A
        point
        0         1
        1         1
        2         0

        Create a 2-dimensional indicator vector with custom name and index.

        >>> ind = MeasurableVector.indicator_of(A, dim=2, index=[1, 2], name="ind")
        >>> print(ind)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'ind':
        index   1  2
        point
        0       1  1
        1       1  1
        2       0  0

        Notes
        -----
        Let $(X, \mathcal{F})$ be a measurable space. Given a set $A\in \mathcal{F}$ and a dimension $d$, the *indicator vector* is the vector $I_A: X \to \mathbb{R}^d$ such that

        $$
        I_A(x) = \begin{cases}
        (1, 1, \ldots, 1) & : x \in A,\\
        (0, 0, \ldots, 0) & : x \notin A.
        \end{cases}
        $$
        """
        from ..indices.index import Index
        from ..spaces.measurable_set import MeasurableSet

        if not isinstance(measurable_set, MeasurableSet):
            raise TypeError("measurable_set must be an instance of MeasurableSet.")
        if index is not None and not isinstance(index, Index):
            index = Index(index)
        if not isinstance(dim, int):
            raise TypeError("dim must be an integer.")
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")

        ones = pd.DataFrame(
            np.ones(shape=(len(measurable_set), dim), dtype=int),
            index=measurable_set.data,
        )
        mapping = ones.reindex(measurable_set.domain.data, fill_value=0)

        if name is None:
            name = f"I_{measurable_set.name}"

        return cls(
            domain=measurable_set.domain,
            sig_alg=measurable_set.sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    @staticmethod
    def _check_for_consistent_measures(
        vectors: list[MeasurableVector | Real],
    ) -> Measure | None:
        """Check that all measurable vectors in the list have consistent measures.

        Parameters
        ----------
        vectors : list[MeasurableVector | Real]
            A list of measurable vectors to check for consistent measures.
        """
        measures = [
            v.measure
            for v in vectors
            if hasattr(v, "measure") and v.measure is not None
        ]

        if len(measures) == 0:
            return None
        else:
            max_measure = measures[0]
            for measure in measures[1:]:
                if max_measure <= measure:
                    max_measure = measure
                elif not measure <= max_measure:
                    raise ValueError(
                        "All measurable vectors must have consistent measures."
                    )

            return max_measure

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series | pd.DataFrame | None:
        """Get the underlying data of the measurable vector.

        Returns
        -------
        data : pd.Series | pd.DataFrame | None
            A `pd.Series` (if the measurable vector is 1-dimensional) or `pd.DataFrame` (if the measurable vector is 2-dimensional or higher), or `None`.

        Examples
        --------
        Get the `data` objects of a 2-dimensional measurable vector and a measurable function.

        >>> from sigalg.core import Domain, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(f.data)  # doctest: +NORMALIZE_WHITESPACE
        index   0  1
        point
        0       1  2
        1       3  4
        2       3  4
        >>> g = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ...     name="g",
        ... )
        >>> print(g.data)  # doctest: +NORMALIZE_WHITESPACE
        point
        0    1
        1    2
        2    2
        Name: g, dtype: int64
        """
        return self._data

    @property
    def atom_data(self) -> pd.Series | pd.DataFrame | None:
        """Get the underlying data of the measurable vector, grouped by atom identifiers.

        Returns
        -------
        atom_data : pd.Series | pd.DataFrame | None
            A `pd.Series` (if the measurable vector is 1-dimensional) or `pd.DataFrame` (if the measurable vector is 2-dimensional or higher), or `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(f.atom_data)  # doctest: +NORMALIZE_WHITESPACE
        index    0  1
        atom_ID
        0        1  2
        1        3  4
        >>> g = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ...     name="g",
        ... )
        >>> print(g.atom_data)  # doctest: +NORMALIZE_WHITESPACE
        atom_ID
        0    1
        1    2
        Name: g, dtype: int64
        """
        if self._atom_data is None and self.data is not None:
            sig_alg_data = self._to_df(self.sig_alg.data)

            self._atom_data = (
                pd.concat([self.data, sig_alg_data], axis=1)
                .drop_duplicates()
                .set_index(list(sig_alg_data.columns))
            ).squeeze(axis=1)

            if self.index is not None:
                self._atom_data.columns = self.index.data

        return self._atom_data

    @staticmethod
    def _to_df(
        data: pd.Series | pd.DataFrame, suffix: str | None = None
    ) -> pd.DataFrame:
        if suffix is None:
            suffix = ""
        if isinstance(data, pd.DataFrame):
            return data.add_suffix(suffix)
        else:
            return data.to_frame().add_suffix(suffix)

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the measurable vector.

        Returns
        -------
        dimension : int | None
            The dimension of the measurable vector, or `None`.
        """
        if self._dimension is None and self.data is not None:
            if isinstance(self.data, pd.Series):
                self._dimension = 1
            else:
                self._dimension = self.data.shape[1]

        return self._dimension

    @property
    def components(self) -> list[MeasurableFunction] | None:
        r"""Get the component measurable functions of the measurable vector.

        See the Notes section below for the mathematical details.

        Raises
        ------
        ValueError
            If `self` has an empty `data` attribute.

        Returns
        -------
        components : list[MeasurableFunction] | None
            A list of the component measurable functions of the measurable vector.

        Examples
        --------
        Extract the component functions of a 2-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=3)
        >>> f = MeasurableVector.from_randint(
        ...     domain=X,
        ...     low=0,
        ...     high=3,
        ...     dim=2,
        ...     random_state=42,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1
        point
        0       0  2
        1       1  1
        2       1  2
        >>> for component in f.components:
        ...     print(component)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_0':
                f_0
        point
        0         0
        1         1
        2         1
        Measurable function 'f_1':
                f_1
        point
        0         2
        1         1
        2         2
        >>> g = MeasurableVector.from_randint(
        ...     domain=X,
        ...     low=0,
        ...     high=3,
        ...     dim=1,
        ...     random_state=42,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        point
        0       0
        1       2
        2       1
        >>> for component in g.components:
        ...     print(component)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        point
        0       0
        1       2
        2       1

        Notes
        -----
        If $f: X \to \mathbb{R}^d$ is a measurable vector, then for each $x \in X$ we may write

        $$
        f(x) = (f_1(x),f_2(x),\ldots, f_d(x))
        $$

        where $f_j: X \to \mathbb{R}$ is the *$j$-th component measurable function* of $f$.
        """
        if self._components is None and self.data is not None:
            if self.dimension == 1:
                self._components = [self]
            else:
                self._components = [
                    self.get_component(idx).with_name(name)
                    for idx, name in zip(self.index, self.component_names)
                ]
        return self._components

    @property
    def component_names(self) -> list[Hashable] | None:
        """Get the names of the component functions of the measurable vector.

        Returns
        -------
        component_names : list[Hashable] | None
            A list of the names of the component functions of the measurable vector, or `None`.
        """
        if self._component_names is None and self.data is not None:
            if self.index is not None:
                self._component_names = [
                    f"{self.name}_{idx}".replace(".", "_") for idx in self.index
                ]
            else:
                self._component_names = [self.name]

        return self._component_names

    @property
    def name(self) -> Hashable:
        """Get the name of the measurable vector.

        Returns
        -------
        name : Hashable
            The name of the measurable vector.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name of the measurable vector.

        Parameters
        ----------
        name : Hashable
            The new name for the measurable vector.

        Raises
        ------
        TypeError
            If `name` is not hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")

        self._name = name
        self._components = None
        self._component_names = None
        self._generated_sig_alg = None
        self._range = None
        if isinstance(self._data, pd.Series):
            self._data.name = name

    def with_name(self, name: Hashable) -> MeasurableVector:
        """Set the name of the measurable vector and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the measurable vector.

        Returns
        -------
        self : MeasurableVector
            Returns self to allow method chaining.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=3)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 2),
        ...         1: (1, 1),
        ...         2: (1, 2),
        ...     },
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1
        point
        0       0  2
        1       1  1
        2       1  2
        >>> g = f.with_name("g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        index   0  1
        point
        0       0  2
        1       1  1
        2       1  2
        """
        self.name = name
        return self

    @property
    def index(self) -> Index | None:
        """Get the index of the measurable vector.

        Returns
        -------
        index : Index | None
            The index of the measurable vector, or `None` if the measurable vector is 1-dimensional or has not been set.

        Examples
        --------
        Define a 2-dimensional measurable vector and print its index.

        >>> from sigalg.core import Domain, Index, MeasurableVector
        >>> X = Domain.from_sequence(size=3)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...     },
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1
        point
        0       1  2
        1       3  4
        2       5  6
        >>> print(f.index)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
         index
             0
             1

        Set the index to a new one.

        >>> J = Index(["a", "b"], variable_names=["letter"], name="J")
        >>> f.index = J
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        letter  a  b
        point
        0       1  2
        1       3  4
        2       5  6
        >>> print(f.index)  # doctest: +NORMALIZE_WHITESPACE
        Index 'J':
         letter
              a
              b

        Print the index of a measurable function.

        >>> g = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        point
        0       1
        1       2
        2       3
        >>> print(g.index)
        None
        """
        return self._index

    @index.setter
    def index(self, index: IndexLike) -> None:
        """Set the index of the measurable vector.

        Parameters
        ----------
        index : IndexLike
            The new index for the measurable vector.

        Raises
        ------
        TypeError
            If `index` cannot be converted to an instance of `Index`.
        ValueError
            If the measurable vector has a non-empty `data` attribute and the length of `index` does not match the dimension of the measurable vector.
        """
        from ..indices.index import Index

        if not isinstance(index, Index):
            index = Index(index)

        if self.data is not None:
            if len(index) != self.dimension:
                raise ValueError(
                    "index size must match the dimension of the measurable vector."
                )
            self.data.columns = index.data
            self.atom_data.columns = index.data

        self._components = None
        self._component_names = None
        self._generated_sig_alg = None
        self._range = None
        self._index = index

    @property
    def generated_sig_alg(self) -> SigmaAlgebra | None:
        r"""Get the sigma-algebra generated by a measurable vector.

        See the Notes section below for the mathematical details.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra induced by the measurable vector.

        Examples
        --------
        Extract the generated sigma-algebra from a 2-dimensional measurable vector. Note that the atom identifiers are exactly the values of the vector.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> sig_f = f.generated_sig_alg
        >>> print(sig_f)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(f)':
               f_0  f_1
        point
        0        1    2
        1        3    4
        2        3    4
        3        3    4
        >>> print(sig_f <= F)
        True

        Notes
        -----
        A measurable vector $f: X \to \mathbb{R}^d$ on a measure space $(X, \mathcal{F},\mu)$ generates a $\sigma$-algebra denoted $\sigma(f)$. On a finite domain $X$, this $\sigma$-algebra is determined by its atoms, which are the nonempty preimages

        $$
        \{ x \in X : f(x) = y\},
        $$

        for $y\in \mathbb{R}^d$. The atom identifiers may thus be taken as the vectors $y\in \mathbb{R}^d$ in the range of $f$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self._generated_sig_alg is None and self.data is not None:
            self._generated_sig_alg = SigmaAlgebra.from_measurable_vector(self)
        return self._generated_sig_alg

    @property
    def measurable_space(self) -> MeasurableSpace | None:
        """Get the measurable space on which the measurable vector is defined.

        Returns
        -------
        measurable_space : MeasurableSpace | None
            The measurable space on which the measurable vector is defined.

        Examples
        --------
        Extract the underlying measurable space of a 2-dimensional measurable vector.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableSpace,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> measurable_space = MeasurableSpace(X, F)
        >>> f = MeasurableVector(
        ...     *measurable_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(f.measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, F)
        =======================
        <BLANKLINE>
        * Domain 'X':
            point
                0
                1
                2
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             1
        2             1
        """
        return self._measurable_space

    @property
    def measure_space(self) -> MeasureSpace | None:
        """Get the measure space on which the measurable vector is defined.

        Returns
        -------
        measure_space : MeasureSpace | None
            The measure space on which the measurable vector is defined.

        Examples
        --------
        Extract the underlying measure space of a 2-dimensional measurable vector.

        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasureSpace,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 2,
        ...         1: 8,
        ...     },
        ... )
        >>> measure_space = MeasureSpace(X, F, mu)
        >>> f = MeasurableVector(
        ...     *measure_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(f.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, F, mu)
        ========================
        <BLANKLINE>
        * Domain 'X':
         point
             0
             1
             2
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             1
        2             1
        <BLANKLINE>
        * Measure 'mu':
                    measure
        atom_ID
        0                 2
        1                 8
        """
        return self._measure_space

    @property
    def domain(self) -> Domain | None:
        """Get the domain of the measurable vector.

        The `domain` property is settable. If the measurable vector is not defined on an empty measurable space, the new domain must have the same number of points as the existing domain and the domain of the sigma-algebra is updated to the new domain. If in addition the measurable vector is not empty (i.e., if it has outputs), then the outputs of the measurable vector are remapped to the new domain according to the order of points in the new domain. If the measurable vector is defined on an empty measure space (and therefore also has no outputs), then the domain may be set freely, the sigma-algebra is updated to the power-set sigma-algebra on the new domain, and the measure (if it exists) is updated to the uniform measure on the new domain.

        Returns
        -------
        domain : Domain | None
            The domain of the measurable vector.

        Examples
        --------
        Define a 2-dimensional measurable vector and print its domain.

        >>> from sigalg.core import Domain, Measure, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 25,
        ...         1: 75,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(f.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         point
             0
             1
             2
             3
        >>> print(f.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, F, mu)
        ========================
        <BLANKLINE>
        * Domain 'X':
         point
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Measure 'mu':
                    measure
        atom_ID
        0                25
        1                75

        Set the domain of the vector to a new domain in bijective correspondence with the first.

        >>> Y = Domain(["a", "b", "c", "d"], name="Y")
        >>> f.domain = Y
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1
        point
        a       1  2
        b       1  2
        c       3  4
        d       3  4
        >>> print(f.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Y':
         point
             a
             b
             c
             d
        >>> print(f.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (Y, F, mu)
        ========================
        <BLANKLINE>
        * Domain 'Y':
         point
             a
             b
             c
             d
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        a             0
        b             0
        c             1
        d             1
        <BLANKLINE>
        * Measure 'mu':
                    measure
        atom_ID
        0                25
        1                75

        Define an empty measurable function and set its domain. Notice the default sigma-algebra.

        >>> empty_vec = MeasurableVector(name="empty_vec")
        >>> empty_vec.domain = Y
        >>> print(empty_vec.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Y':
         point
             a
             b
             c
             d
        >>> print(empty_vec.measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (Y, power_set)
        ===============================
        <BLANKLINE>
        * Domain 'Y':
         point
             a
             b
             c
             d
        <BLANKLINE>
        * Sigma algebra 'power_set':
                 point
        point
        a            a
        b            b
        c            c
        d            d
        """
        return self.measurable_space.domain

    @domain.setter
    def domain(self, domain: IndexLike) -> None:
        """Set the domain of the measurable vector.

        If the measurable vector is not defined on an empty measurable space, the new domain must have the same number of points as the existing domain and the domain of the sigma-algebra is updated to the new domain. If in addition the measurable vector is not empty (i.e., if it has outputs), then the outputs of the measurable vector are remapped to the new domain according to the order of points in the new domain. If the measurable vector is defined on an empty measure space (and therefore also has no outputs), then the domain may be set freely, the sigma-algebra is updated to the power-set sigma-algebra on the new domain, and the measure (if it exists) is updated to the uniform measure on the new domain.

        Parameters
        ----------
        domain : IndexLike
            The new domain for the measurable vector.
        """
        from ..spaces.domain import Domain

        if not isinstance(domain, Domain):
            domain = Domain(domain)

        if self.measure_space is not None:
            self.measure_space.domain = domain
        else:
            self.measurable_space.domain = domain

        if self.data is not None:
            self.data.index = self.measurable_space.domain.data

        new = type(self)(
            *self.measurable_space,
            measure=self.measure,
            mapping=self.data if self.data is not None else None,
            index=self.index,
            name=self.name,
        )

        self.__dict__.update(new.__dict__)

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra on the underlying measure space.

        The `sig_alg` property is settable. If the measurable vector is not defined on an empty measurable space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the measure (if it exists) is updated to be the restriction of the existing measure to the new sigma-algebra. If in addition the measurable vector is not empty (i.e., if it has outputs), then the measurable vector must be measurable with respect to the new sigma-algebra. If the measurable vector is defined on an empty measurable space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the domain of the sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on the domain of the measurable vector.

        Examples
        --------
        Define a 2-dimensional measurable vector.

        >>> from sigalg.core import Domain, Measure, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 5,
        ...         1: 75,
        ...         2: 2,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(f.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             1
        2             2
        3             2

        Set the existing sigma-algebra to a new one.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> f.sig_alg = G
        >>> print(f.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             1
        >>> print(f.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, G, mu|G)
        ==========================
        <BLANKLINE>
        * Domain 'X':
         point
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Measure 'mu|G':
                    measure
        atom_ID
        0                80
        1                 2

        Define an empty measurable vector and set the sigma-algebra. Notice the default domain.

        >>> empty_vec = MeasurableVector(name="empty_vec")
        >>> empty_vec.sig_alg = G
        >>> print(empty_vec.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             1
        >>> print(empty_vec.measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, G)
        =======================
        <BLANKLINE>
        * Domain 'X':
         point
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             1
        """
        return self.measurable_space.sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra on the underlying measure space.

        If the measurable vector is not defined on an empty measurable space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the measure (if it exists) is updated to be the restriction of the existing measure to the new sigma-algebra. If in addition the measurable vector is not empty (i.e., if it has outputs), then the measurable vector must be measurable with respect to the new sigma-algebra. If the measurable vector is defined on an empty measurable space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the domain of the sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra for the measurable vector.

        Raises
        ------
        TypeError
            If `sig_alg` is not an instance of `SigmaAlgebra`.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")

        if self.measure_space is not None:
            self.measure_space.sig_alg = sig_alg
        else:
            self.measurable_space.sig_alg = sig_alg

        new = type(self)(
            *self.measurable_space,
            measure=self.measure,
            mapping=self.data if self.data is not None else None,
            index=self.index,
            name=self.name,
        )

        self.__dict__.update(new.__dict__)

    @property
    def measure(self) -> Measure | None:
        """Get the measure on the underlying measure space.

        The `measure` property is settable. If the measurable vector is not defined on an empty measurable space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the measure (if it exists) is updated to be the restriction of the existing measure to the new sigma-algebra. If in addition the measurable vector is not empty (i.e., if it has outputs), then the measurable vector must be measurable with respect to the new sigma-algebra. If the measurable vector is defined on an empty measurable space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the domain of the sigma-algebra.

        Returns
        -------
        measure : Measure | None
            The measure on the domain of the measurable vector.

        Examples
        --------
        Define a 2-dimensional measurable vector.

        >>> from sigalg.core import Domain, Measure, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 5,
        ...         1: 75,
        ...         2: 20,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(f.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu':
                    measure
        atom_ID
        0                 5
        1                75
        2                20

        Set the measure to a new one.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> nu = Measure(
        ...     domain=G,
        ...     mapping={
        ...         0: 1,
        ...         1: 9,
        ...     },
        ...     name="nu",
        ... )
        >>> f.measure = nu
        >>> print(f.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'nu':
                    measure
        atom_ID
        0                 1
        1                 9
        >>> print(f.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, G, nu)
        ========================
        <BLANKLINE>
        * Domain 'X':
         point
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Measure 'nu':
                    measure
        atom_ID
        0                 1
        1                 9

        Define an empty measurable vector and set its measure. Notice the default domain and sigma-algebra.

        >>> empty_vec = MeasurableVector(name="empty_vec")
        >>> empty_vec.measure = nu
        >>> print(empty_vec.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'nu':
                    measure
        atom_ID
        0                 1
        1                 9
        >>> print(empty_vec.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, G, nu)
        ========================
        <BLANKLINE>
        * Domain 'X':
         point
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Measure 'nu':
                    measure
        atom_ID
        0                 1
        1                 9
        """
        return self.measure_space.measure if self.measure_space is not None else None

    @measure.setter
    def measure(self, measure: Measure) -> None:
        """Set the measure on the underlying measure space (if it exists) or add a measure to the underlying measurable space to create one.

        If the measurable vector is not defined on an empty measurable space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the measure (if it exists) is updated to be the restriction of the existing measure to the new sigma-algebra. If in addition the measurable vector is not empty (i.e., if it has outputs), then the measurable vector must be measurable with respect to the new sigma-algebra. If the measurable vector is defined on an empty measurable space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the domain of the sigma-algebra.

        Parameters
        ----------
        measure : Measure
            The new measure for the measurable vector.

        Raises
        ------
        TypeError
            If `measure` is not an instance of `Measure`.
        """
        from ..measures.measure import Measure
        from ..spaces.measure_space import MeasureSpace

        if not isinstance(measure, Measure):
            raise TypeError("measure must be an instance of Measure.")

        if self.measure_space is not None:
            self.measure_space.measure = measure
        else:
            self._measure_space = MeasureSpace(self.domain, self.sig_alg, measure)
            self._measurable_space = self._measure_space.measurable_space

        new = MeasurableVector(
            *self.measure_space,
            mapping=self.data if self.data is not None else None,
            index=self.index,
            name=self.name,
        )

        self.__dict__.update(new.__dict__)

    @property
    def sample_space(self) -> Domain | None:
        """Get the domain of the underlying measurable space.

        This property is an alias for the `domain` property. This property is intended to be called on instances of `RandomVector`, but this is not enforced.

        Returns
        -------
        domain : Domain | None
            The domain of the underlying measurable space.
        """
        return self.domain

    @property
    def prob_space(self) -> MeasureSpace | None:
        """Get the underlying measure space.

        This property is an alias for the `measure_space` property. This property is intended to be called on instances of `RandomVector`, but this is not enforced.

        Returns
        -------
        measure_space : MeasureSpace | None
            The underlying probability space.
        """
        return self.measure_space

    @property
    def prob_measure(self) -> Measure | None:
        """Get the measure of the underlying measure space.

        This property is an alias for the `measure` property.

        Returns
        -------
        measure : Measure | None
            The underlying measure.
        """
        from ..measures.probability_measure import ProbabilityMeasure

        if not isinstance(self.measure, ProbabilityMeasure):
            raise TypeError(
                "The measure of the measurable vector is not a ProbabilityMeasure."
            )

        return self.measure

    @property
    def is_identity(self) -> bool:
        """Check if the measurable vector is the identity mapping on its domain.

        Returns
        -------
        is_identity : bool
            `True` if the measurable vector is the identity mapping, `False` otherwise.
        """
        return self._is_identity

    @property
    def range(self) -> MeasurableSpace | None:
        r"""Return the range of a measurable vector as a measurable space with the power-set sigma-algebra.

        See the Notes section below for the mathematical details.

        Returns
        -------
        range : MeasureSpace | None
            The range of the measurable vector as a measure space with the pushforward measure. If the measurable vector is empty (i.e., if it has no outputs), then `None` is returned.

        Examples
        --------
        Define a 2-dimensional measurable vector.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(f.range)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (f_range, power_set)
        =====================================
        <BLANKLINE>
        * Domain 'f_range':
         f_0  f_1
           1    2
           3    4
        <BLANKLINE>
        * Sigma algebra 'power_set':
                f_0  f_1
        f_0 f_1
        1   2     1    2
        3   4     3    4
        """
        from ..spaces.domain import Domain
        from ..spaces.measurable_space import MeasurableSpace
        from .random_vector import RandomVector

        if self._range is None and self.data is not None:
            range_list = (
                list(self.data.drop_duplicates().apply(tuple, axis=1))
                if self.dimension > 1
                else list(self.data.drop_duplicates())
            )

            domain = Domain(
                range_list,
                variable_names=self.component_names,
                name=f"{self.name}_range",
            )
            domain._data = domain._data.sort_values()

            if isinstance(self, RandomVector):
                domain = domain.to_sample_space()

            self._range = MeasurableSpace(domain)

        return self._range

    # --------------------- methods --------------------- #

    def is_measurable(self, sig_alg: SigmaAlgebra | None = None) -> bool:
        r"""Check if the measurable vector is measurable with respect to a given sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra to check measurability against.

        Returns
        -------
        is_measurable : bool
            `True` if the measurable vector is measurable with respect to the given sigma-algebra, `False` otherwise.

        Examples
        --------
        Define two 2-dimensional vectors and a sigma-algebra. The first is constant on the atoms of the sigma-algebra and hence measurable, while the second is not.

        >>> from sigalg.core import Domain, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> g = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...         3: (7, 8),
        ...     },
        ...     name="g",
        ... )
        >>> print(f.is_measurable(F))
        True
        >>> print(g.is_measurable(F))
        False

        Notes
        -----
        Let $(X, \mathcal{F})$ be a measurable space and $f: X \to \mathbb{R}^d$ a function. In the case that $X$ is finite (as in SigAlg), the $\sigma$-algebra is determined by its atoms, and the function $f$ is said to be *$\mathcal{F}$-measurable* if $f$ is constant on the atoms of $\mathcal{F}$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra.")
        if sig_alg is not None and sig_alg.domain != self.domain:
            raise ValueError(
                "The domain of sig_alg must match the domain of the measurable vector."
            )

        if sig_alg.is_power_set:
            return True

        return self.generated_sig_alg <= sig_alg

    # --------------------- probability methods --------------------- #

    def sample(
        self,
        size: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Generate random samples from the range space of this random vector.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for the random number generator.

        Returns
        -------
        sample : pd.Series | pd.DataFrame
            If the random vector is 1-dimensional, then a `pd.Series` is returned containing the random sample. Otherwise, if the random vector is multi-dimensional, a `pd.DataFrame` is returned whose rows contain the random sample and has columns indexed by the index of the random vector.

        Examples
        --------
        Generate a random probability space and sample from a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=Omega,
        ...     num_atoms=4,
        ...     random_state=rng,
        ... )
        >>> P = ProbabilityMeasure.from_rand(domain=F, random_state=rng)
        >>> X = RandomVector.from_randint(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     high=10,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        0       2  6
        1       2  6
        2       1  7
        3       2  6
        4       7  3
        5       2  6
        6       0  9
        7       2  6
        8       0  9
        9       2  6
        >>> print(X.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
            sample
                0
                1
                2
                3
                4
                5
                6
                7
                8
                9
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             1
        1             1
        2             3
        3             1
        4             2
        5             1
        6             0
        7             1
        8             0
        9             1
        <BLANKLINE>
        * Probability measure 'P':
                    probability
        atom_ID
        1           0.049134
        3           0.207580
        2           0.082504
        0           0.660782
        >>> X_sample = X.sample(size=10, random_state=rng)
        >>> print(X_sample)  # doctest: +NORMALIZE_WHITESPACE
           X_0  X_1
        0    2    6
        1    1    7
        2    0    9
        3    0    9
        4    0    9
        5    0    9
        6    1    7
        7    1    7
        8    7    3
        9    0    9


        Sample from a 1-dimensional random variable.

        >>> Y = RandomVector.from_randint(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     high=10,
        ...     random_state=rng,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       9
        1       9
        2       3
        3       9
        4       0
        5       9
        6       4
        7       9
        8       4
        9       9
        >>> Y_sample = Y.sample(size=10, random_state=rng)
        >>> print(Y_sample)  # doctest: +NORMALIZE_WHITESPACE
        0    3
        1    3
        2    4
        3    3
        4    4
        5    4
        6    4
        7    4
        8    4
        9    4
        Name: Y, dtype: int64
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .operators import Operators

        if not isinstance(self.measure, ProbabilityMeasure):
            raise TypeError("Cannot sample from a non-random-vector.")

        if self.data is not None:
            return Operators.pushforward(vec=self, measure=self.measure).sample(
                size=size, random_state=random_state
            )
        else:
            raise ValueError("Cannot sample from an empty measurable vector instance.")

    # --------------------- data methods --------------------- #

    def get_inverse_image(
        self, value: Hashable | tuple[Hashable] | pd.Series
    ) -> MeasurableSet:
        """Get the inverse image of a value under the measurable vector.

        Parameters
        ----------
        value : Hashable | tuple[Hashable] | pd.Series
            The value to find the inverse image of. If the measurable vector is 1-dimensional, `value` should be a Hashable. If the measurable vector is multi-dimensional, `value` should be a tuple of hashables or a `pd.Series` with an index matching the variable names of the measurable vector.

        Raises
        ------
        ValueError
            If `value` is not in the range of the measurable vector.

        Returns
        -------
        event : MeasurableSet
            The event in the sigma-algebra corresponding to the inverse image of `value` under the measurable vector.

        Examples
        --------
        Generate a 2-dimensional measurable vector.

        >>> import numpy as np
        >>> import pandas as pd
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(101)
        >>> X = Domain.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     random_state=rng,
        ... )
        >>> f = MeasurableVector.from_randint(
        ...     domain=X, sig_alg=F, dim=2, random_state=rng
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1
        point
        0      1  1
        1      0  1
        2      0  1
        3      0  1
        4      0  0
        5      1  1
        6      0  0
        7      0  1
        8      0  1
        9      0  1

        Get an inverse image using the `get_inverse_image` method.

        >>> inv_1 = f.get_inverse_image((1, 1))
        >>> print(inv_1)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set '{f = (1, 1)}':
         point
             0
             5

        Get an inverse image using the overloaded operator `==`.

        >>> inv_2 = f == (0, 1)
        >>> print(inv_2)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set '{f = (0, 1)}':
         point
             1
             2
             3
             7
             8
             9

        Get an inverse image using the overloaded operator `==` and a `pd.Series`.

        >>> s = pd.Series([0, 0], index=f.index)
        >>> inv_3 = f == s
        >>> print(inv_3)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set '{f = (0, 0)}':
         point
             4
             6
        """
        if not isinstance(value, (Hashable, tuple, pd.Series)):
            raise TypeError(
                "value must be a Hashable, tuple, or pd.Series corresponding to the output of the measurable vector."
            )

        if self.data is None:
            raise ValueError(
                "Cannot get inverse image of a measurable vector without outputs."
            )

        if isinstance(value, pd.Series):
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    "The measurable vector is 1-dimensional, but the provided value is a pd.Series."
                )
            if not value.index.equals(self.index.data):
                raise ValueError(
                    "The index of the provided value does not match the index of the measurable vector."
                )
            value = tuple(value)
        if isinstance(value, tuple) and len(value) != self.dimension:
            raise ValueError(
                "The dimension of the provided value does not match the dimension of the measurable vector."
            )

        mask = (
            (self.data == value).all(axis=1)
            if isinstance(value, tuple)
            else self.data == value
        )
        name = f"{{{self.name} = {value}}}"

        return self.sig_alg.get_set(list(self.data.index[mask]), name=name)

    def __call__(self, key: Hashable | MeasurableSet) -> Hashable | pd.Series:
        """Evaluate a measurable vector on a point or an atom in the sigma-algebra.

        Parameters
        ----------
        key : Hashable | MeasurableSet
            A point in the domain or an atom in the sigma-algebra of the measurable vector.

        Raises
        ------
        ValueError
            If the measurable vector has no outputs, or if `key` is not in the domain or the sigma-algebra of the measurable vector, or if `key` is a measurable set that is not an atom in the sigma-algebra.
        TypeError
            If `key` is not a Hashable (i.e., a point) or an MeasurableSet (i.e., an atom in the sigma-algebra).

        Returns
        -------
        output : Hashable | pd.Series
            If `key` is a point, returns the output of the measurable vector at that point. If `key` is an atom in the sigma-algebra, returns the output of the measurable vector on the atom.

        Examples
        --------
        Define a 2-dimensional measurable vector.

        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasurableFunction,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: "a",
        ...         1: "b",
        ...         2: "b",
        ...         3: "c",
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         "a": 2,
        ...         "b": 5,
        ...         "c": 3,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Call the measurable vector on a point.

        >>> print(f(0))  # doctest: +NORMALIZE_WHITESPACE
        index
        0    1
        1    2
        Name: 0, dtype: int64

        Call the measurable vector on an atom of the underlying sigma-algebra.

        >>> A = F.get_set([1, 2])
        >>> print(f(A))  # doctest: +NORMALIZE_WHITESPACE
        index
        0    3
        1    4
        Name: A, dtype: int64

        Define a measurable function.

        >>> g = MeasurableFunction(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping={
        ...         0: 1,
        ...         1: 3,
        ...         2: 3,
        ...         3: 5,
        ...     },
        ...     name="g",
        ... )

        Call the measurable function on a point.

        >>> print(g(0))
        1

        Call the measurable function on the atom.

        >>> print(g(A))
        3
        """
        from ..spaces.measurable_set import MeasurableSet

        if self.data is None:
            raise ValueError("Cannot evaluate a measurable vector without outputs.")

        if not isinstance(key, (Hashable, MeasurableSet)):
            raise TypeError(
                "key must be a Hashable (i.e., a point) or MeasurableSet (i.e., an atom in the sigma-algebra)."
            )

        if isinstance(key, MeasurableSet):
            is_measurable, is_atom, _, _ = MeasurableSet.is_measurable(
                candidate=key,
                sig_alg=self.sig_alg,
                verbose=True,
            )
            if not is_measurable:
                raise ValueError(
                    "The provided set is not in the sigma-algebra of the measurable vector."
                )
            if not is_atom:
                raise ValueError(
                    "The provided set is not an atom in the sigma-algebra of the measurable vector."
                )
            sample_point = key[0]
            output_name = key.name
        else:
            if key not in self.domain:
                raise ValueError(
                    "The provided point is not in the domain of the measurable vector."
                )
            sample_point = key
            output_name = key

        result = self.data.loc[sample_point]

        if isinstance(result, pd.Series):
            return result.rename(output_name)
        else:
            return result.astype(Real)

    def __iter__(self) -> Iterator[MeasurableFunction]:
        """Iterate over the components of the measurable vector.

        Returns
        -------
        iterator : Iterator[MeasurableFunction]
            An iterator over the components of the measurable vector.
        """
        return iter(self.components)

    def restrict_to(
        self,
        measurable_set: MeasurableSet | list,
        set_name: Hashable | None = "A",
    ) -> MeasurableVector:
        r"""Restrict the measurable vector to a measurable set.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        measurable_set : MeasurableSet | list
            The set to restrict the measurable vector to.
        set_name : Hashable | None, default="A"
            The name to use for the measurable set in the name of the resulting restricted measurable vector. This parameter is only used if `measurable_set` is a list of points, and is otherwise ignored if `measurable_set` is a `MeasurableSet` instance.

        Raises
        ------
        TypeError
            If `measurable_set` is not an `MeasurableSet` or a list of points.
        ValueError
            If `measurable_set` is not in the sigma-algebra of the measurable vector.

        Returns
        -------
        restricted_vec : MeasurableVector
            A new `MeasurableVector` representing the restriction of the original measurable vector to the given set.

        Examples
        --------
        Define a 2-dimensional measurable vector.

        >>> from sigalg.core import Domain, Measure, MeasurableVector, SampleSpace, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 2,
        ...         1: 5,
        ...         2: 3,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Restrict the measurable vector to a set using the `restrict_to` method.

        >>> A = F.get_set([1, 2, 3])
        >>> f_A = f.restrict_to(A)
        >>> print(f_A)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f|A':
        index   0  1
        point
        1       3  4
        2       3  4
        3       5  6
        >>> print(f_A.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (A, F_A, mu_A)
        ============================
        <BLANKLINE>
        * Domain 'A':
         point
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'F_A':
                atom_ID
        point
        1             1
        2             1
        3             2
        <BLANKLINE>
        * Measure 'mu_A':
                    measure
        atom_ID
        1                 5
        2                 3

        Compute the same restriction using the overloaded `|` operator.

        >>> f_A = f | A
        >>> print(f_A)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f|A':
        index   0  1
        point
        1       3  4
        2       3  4
        3       5  6

        Restrict the measurable vector using a `list` with a custom name.

        >>> f_B = f.restrict_to([1, 2], set_name="B")
        >>> print(f_B)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f|B':
        index   0  1
        point
        1       3  4
        2       3  4
        >>> print(f_B.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (B, F_B, mu_B)
        ============================
        <BLANKLINE>
        * Domain 'B':
         point
             1
             2
        <BLANKLINE>
        * Sigma algebra 'F_B':
                atom_ID
        point
        1             1
        2             1
        <BLANKLINE>
        * Measure 'mu_B':
                    measure
        atom_ID
        1                 5

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a measurable vector on a measure space $(X, \mathcal{F}, \mu)$. If $A\in \mathcal{F}$ is an measurable set, then we may restrict the measurable vector to obtain the function $f|_A : A \to \mathbb{R}^d$ on $A$.
        """
        from ..spaces.measurable_set import MeasurableSet
        from ..spaces.measure_space import MeasureSpace
        from .random_vector import RandomVector

        if not isinstance(measurable_set, (MeasurableSet, list)):
            raise TypeError(
                "measurable_set must be an MeasurableSet or a list of points."
            )

        if isinstance(measurable_set, list):
            try:
                measurable_set = self.sig_alg.get_set(measurable_set, name=set_name)
            except ValueError as e:
                raise ValueError(
                    "measurable_set must be in the sigma-algebra of the measurable vector."
                ) from e
        elif (
            isinstance(measurable_set, MeasurableSet)
            and measurable_set not in self.sig_alg
        ):
            raise ValueError(
                "measurable_set must be in the sigma-algebra of the measurable vector."
            )

        mapping = self.data.loc[measurable_set.data]
        mapping.index = measurable_set.data
        name = f"{self.name}|{measurable_set.name}"
        mapping.name = name
        set_space = MeasureSpace.from_set(
            measurable_set=measurable_set,
            measure=self.measure,
            normalize=isinstance(self, RandomVector),
        )

        return type(self)(*set_space, mapping=mapping, name=name)

    def __getitem__(self, *args) -> MeasurableFunction:
        """Get a sub-vector of the measurable vector by selecting a collection of component functions, or a single component function if only one index is provided.

        Calls `get_sub_vector` with the provided indices. See the documentation of that method for details.

        Parameters
        ----------
        *args : Hashable | tuple[Hashable]
            The indices of the component functions to select for the sub-vector.

        Returns
        -------
        sub_vector : MeasurableVector
            A new `MeasurableVector` containing only the specified component functions.
        """
        indices = list(*args) if isinstance(args[0], tuple) else list(args)
        return self.get_sub_vector(indices=indices)

    def get_sub_vector(self, indices: list[Hashable]) -> MeasurableVector:
        r"""Get a sub-vector of the measurable vector by selecting a collection of component functions.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        indices : list[Hashable]
            List of indices to select for the sub-vector.

        Returns
        -------
        sub_vector : MeasurableVector
            A new `MeasurableVector` containing only the specified component functions.

        Raises
        ------
        ValueError
            If any index is not found or if the measurable vector is 1-dimensional.

        Examples
        --------
        Define a 3-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=2)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2, 3),
        ...         1: (4, 5, 6),
        ...     },
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1  2
        point
        0       1  2  3
        1       4  5  6

        Get a sub-vector by using the `get_sub_vector` method.

        >>> f_sub = f.get_sub_vector([1, 2])
        >>> print(f_sub)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector '(f_1, f_2)':
        index   1  2
        point
        0       2  3
        1       5  6

        Get a sub-vector by using subscript notation.

        >>> f_sub = f[0, 1]
        >>> print(f_sub)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector '(f_0, f_1)':
        index   0  1
        point
        0       1  2
        1       4  5

        Notes
        -----
        Given a measurable vector $f: X \to \mathbb{R}^d$ on a measure space $(X, \mathcal{F}, \mu)$, for each $x\in X$ we may write

        $$
        f(x) = (f_1(x), f_2(x), \ldots, f_d(x)),
        $$

        where $f_j: X \to \mathbb{R}$ are the component functions of $f$. We may create a *sub-vector* by choosing a collection of the component functions to get a measurable vector of smaller dimension. For example, we may select the first and last components to create the $2$-dimensional measurable vector

        $$
        x \mapsto (f_1 (x), f_d(x)).
        $$
        """
        from .measurable_function import MeasurableFunction
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

        if self.dimension == 1:
            raise ValueError(
                "Cannot get sub-vector of a 1-dimensional MeasurableVector."
            )
        invalid_features = [
            invalid_feature
            for invalid_feature in indices
            if invalid_feature not in self.index
        ]
        if invalid_features:
            raise ValueError(
                f"Indices {invalid_features} not found when forming the sub-vector"
            )

        sub_data = self.data[indices]

        if len(indices) == 1:
            name = f"{self.name}_{indices[0]}".replace(".", "_")
            if isinstance(self, RandomVector):
                sub_vec = RandomVariable(
                    *self.measurable_space,
                    measure=self.measure,
                    mapping=sub_data,
                    name=name,
                )
            else:
                sub_vec = MeasurableFunction(
                    *self.measurable_space,
                    measure=self.measure,
                    mapping=sub_data,
                    name=name,
                )
        else:
            name = (
                "("
                + ", ".join([f"{self.name}_{idx}".replace(".", "_") for idx in indices])
                + ")"
            )
            sub_vec = type(self)(
                *self.measurable_space,
                measure=self.measure,
                mapping=sub_data,
                name=name,
            )
            sub_vec._component_names = [
                f"{self.name}_{idx}".replace(".", "_") for idx in indices
            ]

        return sub_vec

    def get_component(self, index: Hashable) -> MeasurableFunction:
        r"""Get a component function of the measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        index : Hashable
            The index for which to get the component function.

        Returns
        -------
        component : MeasurableFunction
            The desired component function.

        Examples
        --------
        Define a 3-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=2)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2, 3),
        ...         1: (4, 5, 6),
        ...     },
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index   0  1  2
        point
        0       1  2  3
        1       4  5  6

        Get a component function using the `get_component` method.

        >>> f_1 = f.get_component(1)
        >>> print(f_1)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_1':
                f_1
        point
        0         2
        1         5

        Get a component function using subscript notation.

        >>> f_0 = f[0]
        >>> print(f_0)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_0':
                f_0
        point
        0         1
        1         4

        Notes
        -----
        Given a measurable vector $f: X \to \mathbb{R}^d$ on a measurable space $(X, \mathcal{F})$, for each $x \in X$ we may write

        $$
        f(x) = (f_1(x), f_2(x), \ldots, f_d(x)),
        $$

        where $f_j: X \to \mathbb{R}$ are the component functions of $f$.
        """
        return self.get_sub_vector([index])

    def item(self) -> Hashable | pd.Series:
        """Get the output value of a constant measurable vector.

        Returns
        -------
        output : Hashable | pd.Series
            The single output value of the measurable vector.

        Raises
        ------
        ValueError
            If the measurable vector is not constant.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> X = MeasurableVector(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...     },
        ... )
        >>> print(X.item())  # doctest: +NORMALIZE_WHITESPACE
        index
        0    1
        1    2
        dtype: int64
        >>> Y = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...     },
        ...     name="Y",
        ... )
        >>> print(Y.item())
        1
        """
        if self.data is None:
            raise ValueError("Cannot retrieve the item of an empty measurable vector.")

        if len(self.data.drop_duplicates()) != 1:
            raise ValueError(
                "Can only retrieve the item of a constant measurable vector."
            )

        item = self(self.domain[0])

        if isinstance(item, pd.Series):
            item.name = None

        return item

    def round(self, decimals: int = 0) -> MeasurableVector:
        """Round the outputs of the measurable vector to a specified number of decimal places.

        Parameters
        ----------
        decimals : int, default=0
            The number of decimal places to round to. Must be a non-negative integer.

        Raises
        ------
        ValueError
            If `decimals` is not a non-negative integer, or if the measurable vector's data is not set.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> mapping = dict(zip(Omega, [(0, np.pi), (np.pi / 2, 3 * np.pi / 2)]))
        >>> X = RandomVector.with_uniform(domain=Omega, mapping=mapping)
        >>> print(np.sin(X).round())  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'sin(X)':
        index     0    1
        sample
        0       0.0  0.0
        1       1.0 -1.0
        """
        if not isinstance(decimals, int) or decimals < 0:
            raise ValueError("decimals must be a non-negative integer.")
        if self._data is None:
            raise ValueError("Data must be set to round the measurable vector.")

        self._data = self.data.round(decimals=decimals)

        return self

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the measurable vectors's data as a NumPy array.

        Parameters
        ----------
        dtype : data-type | None, default=None
            The desired data-type for the array. If `None`, the data-type of the underlying data is used.
        copy : bool | None, default=None
            Whether to return a copy of the data. If `None`, the default behavior is used.

        Returns
        -------
        np.ndarray
            The measurable vector's data as a NumPy array.
        """
        arr = self.data.values
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        if copy:
            arr = arr.copy()

        return arr

    def to_numpy(self, dtype=None, copy=None) -> np.ndarray:
        """Return the measurable vector's data as a NumPy array.

        Parameters
        ----------
        dtype : data-type | None, default=None
            The desired data-type for the array. If `None`, the data-type of the underlying data is used.
        copy : bool | None, default=None
            Whether to return a copy of the data. If `None`, the default behavior is used.

        Returns
        -------
        np.ndarray
            The measurable vector's data as a NumPy array.
        """
        return self.__array__(dtype=dtype, copy=copy)

    # --------------------- equality --------------------- #

    def __eq__(
        self,
        other: MeasurableVector | Hashable | tuple[Hashable] | pd.Series,
        rtol=1e-5,
        atol=1e-8,
    ) -> bool:
        r"""Check equality with another measurable vector or compute an inverse image of a value under the measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : MeasurableVector | Hashable | tuple[Hashable] | pd.Series
            Another measurable vector to compare with, or a value for which to compute the inverse image.
        rtol : float, default=1e-5
            The relative tolerance for comparing two measurable vectors. This is used only when `other` is a `MeasurableVector`.
        atol : float, default=1e-8
            The absolute tolerance for comparing two measurable vectors. This is used only when `other` is a `MeasurableVector`.

        Returns
        -------
        output : bool | MeasurableSet
            If `other` is a `MeasurableVector`, returns `True` if the two measurable vectors are equal, and `False` otherwise. If `other` is a value, returns the measurable set corresponding to the inverse image of that value under the measurable vector.
        """
        if not isinstance(other, MeasurableVector):
            try:
                return self.get_inverse_image(other)
            except TypeError as e:
                raise TypeError(
                    "If comparing a MeasurableVector to a non-MeasurableVector, the other object must be a Hashable, tuple[Hashable], or pd.Series corresponding to a possible output of the measurable vector."
                ) from e

        if self.domain != other.domain:
            return False
        if self.index != other.index:
            return False

        if isinstance(other.data.index, pd.MultiIndex):
            other_data = other.data.reorder_levels(self.data.index.names)
        else:
            other_data = other.data

        if other.index is not None:
            other_data = other_data.reindex(columns=self.data.columns)
        else:
            other_data = other_data

        self_sorted = self.data.sort_index()
        other_sorted = other_data.sort_index()

        return np.allclose(self_sorted, other_sorted, rtol=rtol, atol=atol)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the measurable vector.

        Returns
        -------
        repr_str : str
            The string representation of the measurable vector.
        """
        if self.data is None:
            return type(self)._repr_name + "(empty)"
        if self.measure is not None:
            return (
                type(self)._repr_name + f"(domain={self.domain.name}, "
                f"sig_alg={self.sig_alg.name}, "
                f"measure={self.measure.name}, "
                f"name={self.name})"
            )
        else:
            return (
                type(self)._repr_name + f"(domain={self.domain.name}, "
                f"sig_alg={self.sig_alg.name}, "
                f"name={self.name})"
            )

    def __str__(self) -> str:
        """Return a detailed string representation of the measurable vector.

        Returns
        -------
        repr_str : str
            The string representation of the measurable vector.
        """
        if self.data is None:
            return f"{type(self)._str_name} '{self.name}': empty"
        else:
            if isinstance(self.data, pd.Series):
                data = self.data.to_frame()
                data.columns = [self.name]
            else:
                data = self.data

            return f"{type(self)._str_name} '{self.name}':\n{data}"

    # --------------------- arithmetic operations --------------------- #

    def _apply_operation(
        self,
        other: MeasurableVector | Real,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
    ) -> MeasurableVector:
        """Apply a binary operation to this measurable vector.

        Parameters
        ----------
        self : MeasurableVector
            The left operand (or right if reverse=True).
        other : MeasurableVector | Real
            The right operand (or left if reverse=True).
        operation : Callable
            The pandas operation to apply (e.g., `lambda a, b: a + b`).
        op_symbol : str
            Symbol representing the operation (e.g., '+', '-', '*').
        reverse : bool, default=False
            Whether this is a reverse operation (e.g., __radd__ vs __add__).

        Returns
        -------
        result : MeasurableVector
            A new measurable vector representing the result of the operation.
        """
        from .measurable_function import MeasurableFunction
        from .parametrized_measurable_function import ParametrizedMeasurableFunction

        if isinstance(self, MeasurableFunction) and isinstance(
            other, MeasurableFunction
        ):
            if self.sig_alg <= other.sig_alg:
                super_sig_alg = other.sig_alg
            elif self.sig_alg > other.sig_alg:
                super_sig_alg = self.sig_alg
            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable functions on incompatible measurable spaces."
                )

            if reverse:
                new_name = f"({other.name}{op_symbol}{self.name})"
                new_values = operation(other.data, self.data).rename(new_name)
            else:
                new_name = f"({self.name}{op_symbol}{other.name})"
                new_values = operation(self.data, other.data).rename(new_name)

            measure = self._check_for_consistent_measures([self, other])

            return MeasurableFunction(
                domain=self.domain,
                sig_alg=super_sig_alg,
                measure=measure,
                mapping=new_values,
                name=new_name,
            )

        # TODO: this needs to be tested!
        if isinstance(self, MeasurableFunction) and isinstance(
            other, ParametrizedMeasurableFunction
        ):
            if self.sig_alg <= other.sig_alg:
                super_sig_alg = other.sig_alg
            elif self.sig_alg > other.sig_alg:
                super_sig_alg = self.sig_alg
            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable functions on incompatible measurable spaces."
                )

            if reverse:
                new_name = f"({other.name}{op_symbol}{self.name})"
                new_values = operation(other.data, self.data).rename(new_name)
            else:
                new_name = f"({self.name}{op_symbol}{other.name})"
                new_values = operation(self.data, other.data).rename(new_name)

            measure = self._check_for_consistent_measures([self, other])

            result = ParametrizedMeasurableFunction(
                domain=other.domain,
                mapping=new_values.rename(new_name),
                output_name=new_name,
                name=new_name,
            )
            result._init_measurable_attrs(
                measurable_domain=self.domain,
                sig_alg=super_sig_alg,
                measure=measure,
            )

            return result

        elif isinstance(self, MeasurableVector) and isinstance(other, MeasurableVector):
            if self.sig_alg <= other.sig_alg:
                super_sig_alg = other.sig_alg
            elif self.sig_alg > other.sig_alg:
                super_sig_alg = self.sig_alg
            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable vectors on incompatible measurable spaces."
                )
            if self.index != other.index:
                raise ValueError(
                    f"Cannot {op_symbol} measurable vectors with different indices."
                )

            if reverse:
                new_name = f"({other.name}{op_symbol}{self.name})"
                new_values = operation(other.data, self.data)
            else:
                new_name = f"({self.name}{op_symbol}{other.name})"
                new_values = operation(self.data, other.data)

            measure = self._check_for_consistent_measures([self, other])

            return MeasurableVector(
                domain=self.domain,
                sig_alg=super_sig_alg,
                measure=measure,
                mapping=new_values,
                name=new_name,
                index=self.index,
            )

        elif isinstance(self, MeasurableFunction) and isinstance(other, Real):
            if reverse:
                new_name = f"({other}{op_symbol}{self.name})"
                new_values = operation(other, self.data).rename(new_name)
            else:
                new_name = f"({self.name}{op_symbol}{other})"
                new_values = operation(self.data, other).rename(new_name)

            return MeasurableFunction(
                *self.measurable_space,
                measure=self.measure,
                mapping=new_values,
                name=new_name,
            )

        elif isinstance(self, MeasurableVector) and isinstance(other, Real):
            if reverse:
                new_name = f"({other}{op_symbol}{self.name})"
                new_values = operation(other, self.data)
            else:
                new_name = f"({self.name}{op_symbol}{other})"
                new_values = operation(self.data, other)

            return MeasurableVector(
                *self.measurable_space,
                measure=self.measure,
                mapping=new_values,
                index=self.index,
                name=new_name,
            )

        else:
            raise TypeError("Unsupported types for arithmetic operations.")

    def __add__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Add another measurable vector or a scalar to this measurable vector."""
        return self._apply_operation(other, lambda a, b: a + b, "+")

    def __radd__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Add another measurable vector or a scalar to this measurable vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a + b, "+", reverse=True)

    def __sub__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Subtract another measurable vector or a scalar from this measurable vector."""
        return self._apply_operation(other, lambda a, b: a - b, "-")

    def __rsub__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Subtract this measurable vector from another measurable vector or a scalar (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a - b, "-", reverse=True)

    def __mul__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Multiply this measurable vector by another measurable vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a * b, "*")

    def __rmul__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Multiply another measurable vector or a scalar by this measurable vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a * b, "*", reverse=True)

    def __truediv__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Divide this measurable vector by another measurable vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a / b, "/")

    def __rtruediv__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Divide another measurable vector or a scalar by this measurable vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a / b, "/", reverse=True)

    def __pow__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Exponentiate this measurable vector by another measurable vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a**b, "**")

    def __rpow__(self, other: MeasurableVector | Real) -> MeasurableVector:
        """Exponentiate another measurable vector or a scalar by this measurable vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a**b, "**", reverse=True)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> MeasurableVector:
        """Override NumPy ufuncs to operate on MeasurableVector instances.

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
        result : MeasurableVector
            A new instance of `MeasurableVector` containing the result of applying the ufunc to the inputs.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if method != "__call__":
            return NotImplemented

        new_inputs = [
            input.data if isinstance(input, MeasurableVector) else input
            for input in inputs
        ]
        result_data = getattr(ufunc, method)(*new_inputs, **kwargs)

        if isinstance(result_data, pd.Series):
            result_data.name = None

        new_name = f"{ufunc.__name__}({self.name})" if self.name is not None else None

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                *self.measure_space, name=new_name, time=self.time
            ).from_pandas(data=result_data)

        elif isinstance(self, RandomVariable):
            result_data.name = None
            return RandomVariable(
                *self.measure_space, mapping=result_data, name=new_name
            )

        else:
            return MeasurableVector(
                *self.measurable_space,
                measure=self.measure,
                mapping=result_data,
                name=new_name,
            )

    def __neg__(self) -> MeasurableVector:
        """Negate this measurable vector."""
        return (-1) * self

    # --------------------- comparison methods --------------------- #

    def __bool__(self) -> bool:
        """Prevent ambiguous boolean conversion of a measurable vector.

        Raises
        ------
        ValueError
            Always raised to prevent ambiguous boolean evaluation.
            Use explicit methods like .all() or .any() instead.
        """
        raise ValueError(
            "The truth value of a MeasurableVector is ambiguous. "
            "Use .all() or .any() methods, or check specific conditions explicitly."
        )

    def all(self) -> bool:
        """Check if all values in the measurable vector are `True`.

        This method is typically used after a comparison operation to verify
        that the comparison holds for all points and all components.

        Returns
        -------
        all_true : bool
            `True` if all values across all outputs are `True`.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=2)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 1),
        ...         1: (1, 1),
        ...     },
        ... )
        >>> print(f.all())
        True
        >>> g = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 0),
        ...         1: (0, 1),
        ...     },
        ...     name="g",
        ... )
        >>> print(g.all())
        False
        """
        return bool(self.data.all().all() if self.dimension > 1 else self.data.all())

    def any(self) -> bool:
        """Check if any value in the measurable vector is `True`.

        This method is typically used after a comparison operation to verify
        that the comparison holds for at least one point or component.

        Returns
        -------
        any_true : bool
            `True` if any value across all outputs is `True`.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=2)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 0),
        ...     },
        ... )
        >>> print(f.any())
        True
        >>> g = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 0),
        ...         1: (0, 0),
        ...     },
        ...     name="g",
        ... )
        >>> print(g.any())
        False
        """
        return bool(self.data.any().any() if self.dimension > 1 else self.data.any())

    def _apply_comparison(
        self,
        other: MeasurableVector | Real,
        op: Callable,
        op_symbol: str,
    ) -> MeasurableVector:
        """Apply a comparison operation to this measurable vector.

        Parameters
        ----------
        other : MeasurableVector | Real
            The measurable vector or scalar to compare with.
        op : Callable
            The numpy comparison to apply (e.g., ``operator.lt``).
        op_symbol : str
            Symbol representing the comparison (e.g., '<', '<=', '>', '>=').

        Returns
        -------
        result : MeasurableVector
            A new measurable vector of booleans representing the comparison result.

        Raises
        ------
        TypeError
            If `other` is not a `MeasurableVector` or scalar.
        ValueError
            If the measurable vectors do not have the same domain or dimension.
        """
        from .measurable_function import MeasurableFunction

        if isinstance(other, Real):
            other = MeasurableVector.from_constant(
                *self.measure_space, index=self.index, name=other, constant=other
            )
        elif not isinstance(other, MeasurableVector):
            raise TypeError("other must be a MeasurableVector or a scalar.")

        if self.measure_space != other.measure_space:
            raise ValueError(
                "The measurable vectors must have the same measure space in order to be compared."
            )
        if self.index != other.index:
            raise ValueError(
                "The measurable vectors must have the same index in order to be compared."
            )

        comparison_arr = op(self.data.to_numpy(), other.data.to_numpy())
        name = (
            f"({self.name} {op_symbol} {other.name})"
            if self.name and other.name
            else None
        )

        if isinstance(self, MeasurableFunction):
            result = MeasurableFunction(
                *self.measure_space, name=name, mapping=comparison_arr.flatten()
            )
            result.data.name = name
            return result

        else:
            return MeasurableVector(
                *self.measure_space, name=name, mapping=comparison_arr
            )

    def __lt__(self, other: MeasurableVector | Real) -> MeasurableVector:
        r"""Check if this measurable vector is less than another measurable vector or scalar.

        Parameters
        ----------
        other : MeasurableVector | Real
            The measurable vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `MeasurableVector`.
        ValueError
            If the measurable vectors do not have the same domain or dimension.

        Returns
        -------
        is_lt: MeasurableVector
            A new `MeasurableVector` of booleans indicating where this measurable vector is less than the other measurable vector or scalar.
        """
        import operator

        return self._apply_comparison(other, operator.lt, "<")

    def __le__(self, other: MeasurableVector | Real) -> MeasurableVector:
        r"""Check if this measurable vector is less than or equal to another measurable vector or scalar.

        Parameters
        ----------
        other : MeasurableVector | Real
            The measurable vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `MeasurableVector`.
        ValueError
            If the measurable vectors do not have the same domain or dimension.

        Returns
        -------
        is_le: MeasurableVector
            A new `MeasurableVector` of booleans indicating where this measurable vector is less than or equal to the other measurable vector or scalar.
        """
        import operator

        return self._apply_comparison(other, operator.le, "<=")

    def __gt__(self, other: MeasurableVector | Real) -> MeasurableVector:
        r"""Check if this measurable vector is greater than another measurable vector or scalar.

        Parameters
        ----------
        other : MeasurableVector | Real
            The measurable vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `MeasurableVector`.
        ValueError
            If the measurable vectors do not have the same domain or dimension.

        Returns
        -------
        is_gt: MeasurableVector
            A new `MeasurableVector` of booleans indicating where this measurable vector is greater than the other measurable vector or scalar.
        """
        import operator

        return self._apply_comparison(other, operator.gt, ">")

    def __ge__(self, other: MeasurableVector | Real) -> MeasurableVector:
        r"""Check if this measurable vector is greater than or equal another measurable vector or scalar.

        Parameters
        ----------
        other : MeasurableVector | Real
            The measurable vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `MeasurableVector`.
        ValueError
            If the measurable vectors do not have the same domain or dimension.

        Returns
        -------
        is_ge: MeasurableVector
            A new `MeasurableVector` of booleans indicating where this measurable vector is greater than or equal the other measurable vector or scalar.
        """
        import operator

        return self._apply_comparison(other, operator.ge, ">=")
