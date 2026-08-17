"""A class representing a measurable vector."""

from __future__ import annotations

from functools import cached_property
from numbers import Real
from typing import TYPE_CHECKING, Literal

from .function import Function
from .operators import OperatorsMethods

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    import numpy as np
    import pandas as pd

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.measurable_space import MeasurableSpace
    from ..spaces.measure_space import MeasureSpace
    from ..spaces.set import Set
    from .measurable_function import MeasurableFunction

    PandasLike = pd.Series | pd.DataFrame


class MeasurableVector(Function, OperatorsMethods):
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
    i  0  1
    x
    0  1  1
    1  1  1
    2  2  2
    >>> print(f.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
       R
    x
    0  0
    1  1
    2  2

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
       F
    x
    0  0
    1  0
    2  1

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
    i    1  2
    x y
    0 1  1  0
    1 2  3  1

    Notes
    -----
    Given a measurable space $(X,\mathcal{F})$, a *measurable vector* is an $\mathcal{F}$-measurable function $f: X \to \mathbb{R}^d$, where $d$ is the *dimension* of the vector and $\mathbb{R}^d$ is equipped with its Borel $\sigma$-algebra. If $X$ is finite (as it always is, in SigAlg), then $f$ is $\mathcal{F}$-measurable if and only if $f$ is constant on the atoms of $\mathcal{F}$.
    """

    _properties = []
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
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> None:
        import pandas as pd

        from ...validation.measurable_func_normalizer import MeasurableFuncNormalizer

        PandasLike = pd.Series | pd.DataFrame

        super().__init__(
            domain=domain,
            mapping=mapping,
            kind="any",
            domain_kind=domain_kind,
            domain_name=domain_name,
            multi_dim_outputs=True,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

        w = MeasurableFuncNormalizer(
            domain=self.domain,
            sig_alg=sig_alg,
            measure=measure,
        )

        self.sig_alg = w.sig_alg
        self.measure = w.measure

        if (
            sig_alg is not None
            and not sig_alg.is_power_set
            and isinstance(self.data, PandasLike)
            and sig_alg not in self.lattice
        ):
            raise ValueError(f"Function {name} is not measurable.")

        self._reset_class()

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Series | pd.DataFrame | Callable,
        sig_alg: SigmaAlgebra,
        measure: Measure | None,
        index_kind: Literal["Index", "Time"],
        index_name: Hashable | None,
        name: Hashable,
        **kwargs,
    ) -> MeasurableVector:

        if measure is not None:
            sig_alg = measure.sig_alg

        vector = super()._from_validated(
            data=data,
            kind="any",
            name=name,
            domain_kind=sig_alg.domain_kind,
            domain_name=sig_alg.domain_name,
            index_kind=index_kind,
            index_name=index_name,
        )

        vector.sig_alg = sig_alg
        vector.measure = measure

        vector._reset_class()

        return vector

    @classmethod
    def from_constant(
        cls,
        domain: IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        *,
        constant: Hashable | None,
    ) -> MeasurableVector:
        """Create a measurable vector that maps every point in the domain to the same constant output vector.

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

        Returns
        -------
        const_vec : MeasurableVector
            A measurable vector mapping every point in the domain to the same constant output vector.

        Examples
        --------
        Create a constant 2-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=3)
        >>> f = MeasurableVector.from_constant(domain=X, constant=(1, 2), index=[1, 2])
        >>> print(f) # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i  1  2
        x
        0  1  2
        1  1  2
        2  1  2

        Create a constant 1-dimensional measurable function.

        >>> g = MeasurableVector.from_constant(domain=X, constant=2, name="g")
        >>> print(g) # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        x
        0       2
        1       2
        2       2
        """
        from ...validation.measurable_func_normalizer import MeasurableFuncNormalizer

        v = MeasurableFuncNormalizer(domain=domain, sig_alg=sig_alg, measure=measure)

        domain = v.domain
        sig_alg = v.sig_alg
        measure = v.measure

        return super().from_constant(
            domain=domain,
            constant=constant,
            domain_kind=domain_kind,
            domain_name=domain_name,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            sig_alg=sig_alg,
            measure=measure,
        )

    @classmethod
    def from_identity(
        cls,
        domain: IndexLike,
        measure: Measure | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Create a measurable vector that maps every point in the domain to itself.

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

        Returns
        -------
        vector : MeasurableVector
            A measurable vector mapping every point in the domain to itself.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, MeasurableVector

        Create an identity vector on a 2-dimensional domain.

        >>> X = Domain.cartesian_power(
        ...     [0, 1], n=2, name="X", variable_names=["x_0", "x_1"]
        ... )
        >>> mu = Measure(domain=X, mapping=dict(zip(X, [1, 2, 3, 0])))
        >>> f = MeasurableVector.from_identity(domain=X, measure=mu)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i        0  1
        x_0 x_1
        0   0    0  0
            1    0  1
        1   0    1  0
            1    1  1

        Print its range.

        >>> print(f.range)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x_0  x_1
           0    0
           0    1
           1    0
           1    1

        Now define an identity vector on a 1-dimensional domain and print its range.

        >>> S = Domain(indices=["a", "b"], name="S")
        >>> g = MeasurableVector.from_identity(domain=S, name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
           g
        x
        a  a
        b  b
        >>> print(g.range)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'S':
         x
         a
         b
        """
        from ...validation.measurable_func_normalizer import MeasurableFuncNormalizer

        v = MeasurableFuncNormalizer(domain=domain, sig_alg=None, measure=measure)

        domain = v.domain
        sig_alg = v.sig_alg
        measure = v.measure

        if measure is not None and not measure.sig_alg.is_power_set:
            raise ValueError(
                "For the from_identity method, the sigma-algebra of the measure must be the power-set sigma-algebra."
            )

        return super().from_identity(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            sig_alg=sig_alg,
            measure=measure,
        )

    @classmethod
    def from_rand(
        cls,
        domain: IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        dim: int | None = None,
        diff_values: int = 0,
        distribution: Literal["uniform", "normal"] = "uniform",
        min_value: int = 0,
        max_value: int = 10,
        loc: float = 0.0,
        scale: float = 1.0,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
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

        Returns
        -------
        vector : MeasurableVector
            A measurable vector with integer outputs uniformly sampled from the range [low, high).

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, MeasurableVector, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Create a 2-dimensional measurable vector with integer outputs uniformly sampled from the range [0, 5).

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
        >>> f = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     min_value=0,
        ...     max_value=5,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i  0  1
        x
        0  0  3
        1  0  3
        2  3  2
        3  2  4
        4  2  4
        5  0  3

        Create a 2-dimensional measurable vector with values drawn from a standard normal distribution.

        >>> g = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     distribution="normal",
        ...     dim=2,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        i         0         1
        x
        0 -1.951035 -1.302180
        1 -1.951035 -1.302180
        2  0.127840 -0.316243
        3 -0.016801 -0.853044
        4 -0.016801 -0.853044
        5  0.879398  0.777792

        The maximum number of unique values of a measurable vector is equal to the number of atoms of the underlying sigma-algebra. Notice that this last vector achieves this upper bound. We can decrease the number of unique values by generating the vector so that it is measurable with respect to a sub-sigma-algebra by specifying a nonzero value for the `diff_values` parameter. This parameter is equal to `diff_values = sig_alg.num_atoms - sub_sig_alg.num_atoms`.

        >>> h = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     diff_values=2,
        ...     distribution="normal",
        ...     dim=2,
        ...     name="h",
        ...     random_state=rng,
        ... )
        >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'h':
        i         0         1
        x
        0 -0.859292  0.368751
        1 -0.859292  0.368751
        2 -0.958883  0.878450
        3 -0.958883  0.878450
        4 -0.958883  0.878450
        5 -0.859292  0.368751
        """
        import numpy as np
        import pandas as pd

        from ...validation.domain_index_validator import DomainIndexValidator
        from ...validation.measurable_func_normalizer import MeasurableFuncNormalizer
        from .._utils.function_helpers import sig_alg_func_to_measurable_func
        from ..indices.index import Index
        from ..indices.time import Time
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        u = MeasurableFuncNormalizer(domain=domain, sig_alg=sig_alg, measure=measure)

        domain = u.domain
        sig_alg = u.sig_alg
        measure = u.measure

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        v = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        domain = v.domain
        domain_kind = v.domain_kind
        domain_name = v.domain_name
        index = v.index
        index_kind = v.index_kind
        index_name = v.index_name

        if name is None:
            name = cls._default_name
        if output_name is None:
            output_name = name

        if diff_values > 0:
            sub_sig_alg = SigmaAlgebra.from_rand(
                super=sig_alg,
                num_atoms=sig_alg.num_atoms - diff_values,
                random_state=rng,
            )

        else:
            sub_sig_alg = sig_alg

        if index is not None:
            if dim is not None and len(index) != dim:
                raise ValueError(
                    "If both index and dim are given, the length of the former must equal the latter."
                )
            dim = len(index)

        else:
            if dim is None:
                raise ValueError("One or the other of dim or index must be given.")
            index_class = Index if index_kind == "Index" else Time
            index = index_class.from_sequence(size=dim, name=index_name)

        if distribution == "normal":
            arr = rng.normal(loc, scale, size=(sub_sig_alg.num_atoms, dim))
        else:
            arr = rng.integers(min_value, max_value, size=(sub_sig_alg.num_atoms, dim))

        if sub_sig_alg.is_canonical_power_set:
            data = pd.DataFrame(arr, index=domain.data, columns=index.data)

        else:
            sig_alg_data = pd.DataFrame(
                arr, index=sub_sig_alg.atom_space.data, columns=index.data
            )
            data = sig_alg_func_to_measurable_func(
                self_data=sig_alg_data,
                sig_alg_data=sub_sig_alg.data,
                parameter_names=[],
            )

        if isinstance(data, pd.Series):
            data.name = output_name

        return cls._from_validated(
            data=data,
            sig_alg=sig_alg,
            measure=measure,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

    @classmethod
    def concatenate(
        cls,
        factors: list[MeasurableFunction | MeasurableVector | Real],
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
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
        >>> f = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     max_value=2,
        ...     index=I,
        ...     random_state=42,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i       0  1  2
        x
        0       0  1  1
        1       0  0  1
        2       0  1  1
        3       0  1  0
        >>> J = Index([3, 4], name="J")
        >>> g = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     index=J,
        ...     max_value=2,
        ...     random_state=42,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        i       3  4
        x
        0       0  1
        1       1  0
        2       0  1
        3       0  1

        Concatenate the two vectors. The measure of the one will propagate to the concatenation.

        >>> fg = MeasurableVector.concatenate([f, g])
        >>> print(fg)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'fg':
        i       0  1  2  3  4
        x
        0       0  1  1  0  1
        1       0  0  1  1  0
        2       0  1  1  0  1
        3       0  1  0  0  1
        >>> print(fg.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu':
             mu
        F
        1     2
        0     1
        2     3

        Generate a measurable function.

        >>> h = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     dim=1,
        ...     max_value=2,
        ...     random_state=42,
        ...     name="h",
        ... )
        >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'h':
                h
        x
        0       0
        1       1
        2       0
        3       1

        Concatenate measurable functions and vectors, along with scalars using the `|` operator.

        >>> fh2Y = f | h | 2 | g
        >>> print(fh2Y)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'fh2g':
        i       0  1  2  3  4  5  6
        x
        0       0  1  1  0  2  0  1
        1       0  0  1  1  2  1  0
        2       0  1  1  0  2  0  1
        3       0  1  0  1  2  0  1

        From a concatenation with a custom index and name.

        >>> k = MeasurableVector.concatenate([0, h, f], index=[0, 1, 2, 3, 4], name="k")
        >>> print(k)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'k':
        i       0  1  2  3  4
        x
        0       0  0  0  1  1
        1       0  1  0  0  1
        2       0  0  0  1  1
        3       0  1  0  1  0
        """
        actual_funcs = [func for func in factors if isinstance(func, Function)]
        measure = cls._check_for_consistent_measures(actual_funcs)
        sig_alg = measure.sig_alg

        return super().concatenate(
            factors=factors,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            sig_alg=sig_alg,
            measure=measure,
        )

    @classmethod
    def cartesian_product(
        cls,
        factors: list[MeasurableVector],
        domain_name: Hashable | None = None,
        sig_alg_name: Hashable | None = None,
        measure_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
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
        >>> from sigalg.core import (
        ...     MeasurableVector,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define two probability measures on two sigma-algebras on the same sample space.

        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(domain=Omega, mapping=dict(zip(Omega, [0, 0, 1])))
        >>> G = SigmaAlgebra(domain=Omega, mapping=dict(zip(Omega, [0, 1, 1])), name="G")
        >>> P = ProbabilityMeasure(domain=F, mapping=dict(zip(F.atom_ids, [0.4, 0.6])))
        >>> Q = ProbabilityMeasure(domain=G, mapping=dict(zip(G.atom_ids, [0.25, 0.75])), name="Q")

        Define two random variables on the two probability spaces.

        >>> X = RandomVariable(Omega, F, P, mapping=dict(zip(Omega, [1, 1, 0])))
        >>> Y = RandomVariable(Omega, G, Q, mapping=dict(zip(Omega, [2, 3, 3])), name="Y")

        Form the Cartesian product of the two random variables.

        >>> product = MeasurableVector.cartesian_product([X, Y])
        >>> print(product)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X x Y':
        i        0  1
        s_0 s_1
        0   0    1  2
            1    1  3
            2    1  3
        1   0    1  2
            1    1  3
            2    1  3
        2   0    0  2
            1    0  3
            2    0  3

        Print the measure space of the Cartesian product.

        >>> print(product.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega x Omega, F x G, P x Q)
        ===============================================
        <BLANKLINE>
        * Domain 'Omega x Omega':
         s_0  s_1
           0    0
           0    1
           0    2
           1    0
           1    1
           1    2
           2    0
           2    1
           2    2
        <BLANKLINE>
        * Sigma algebra 'F x G':
        i        0  1
        s_0 s_1
        0   0    0  0
            1    0  1
            2    0  1
        1   0    0  0
            1    0  1
            2    0  1
        2   0    1  0
            1    1  1
            2    1  1
        <BLANKLINE>
        * Probability measure 'P x Q':
             P x Q
        F G
        0 0   0.10
          1   0.30
        1 0   0.15
          1   0.45

        Form the same Cartesian product using the `@` operator.

        >>> print(X @ Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X x Y':
        i        0  1
        s_0 s_1
        0   0    1  2
            1    1  3
            2    1  3
        1   0    1  2
            1    1  3
            2    1  3
        2   0    0  2
            1    0  3
            2    0  3

        Notes
        -----
        Given one measurable vector $f: X \to \mathbb{R}^d$ on a measurable space $(X,\mathcal{F})$, and a second measurable vector $g: Y \to \mathbb{R}^e$ on a measurable space $(Y,\mathcal{G})$, their *Cartesian product*, denoted $f \times g$, is the $(\mathcal{F} \times \mathcal{G})$-measurable measurable vector defined

        $$
        (f \times g) : X \times Y \to \mathbb{R}^{d+e}, \quad (f\times g)(x, y) = (f(x),g(y)).
        $$

        Here, $\mathcal{F} \times \mathcal{G}$ is the product $\sigma$-algebra.
        """
        from ..measures.measure import Measure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        measures = [func.measure for func in factors if func.measure]
        all_measures = len(measures) == len(factors)

        if all_measures:
            measure = Measure.tensor_product(measures, name=measure_name)
            sig_alg = measure.sig_alg
            sig_alg.name = sig_alg_name if sig_alg_name else sig_alg.name
            sig_alg.domain.name = domain_name if domain_name else sig_alg.domain.name
        else:
            measure = None
            sig_alg = SigmaAlgebra.cartesian_product(
                [factor.sig_alg for factor in factors],
                domain_name=domain_name,
                name=sig_alg_name,
            )

        return super().cartesian_product(
            factors=factors,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            sig_alg=sig_alg,
            measure=measure,
        )

    @classmethod
    def cartesian_power(
        cls,
        vector: MeasurableVector,
        n: int,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
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
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )

        Define a 2-dimensional measurable vector `f`.

        >>> X = Domain.from_sequence(size=4, variable_name="x")
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

        >>> cart_pow = MeasurableVector.cartesian_power(f, 2)
        >>> print(cart_pow)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f ^ 2':
        i        0  1  2  3
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
        >>> print(cart_pow.measure_space)  # doctest: +NORMALIZE_WHITESPACE
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
        i        0  1
        x_0 x_1
        0   0    0  0
            1    0  1
            2    0  1
            3    0  2
        1   0    1  0
            1    1  1
            2    1  1
            3    1  2
        2   0    1  0
            1    1  1
            2    1  1
            3    1  2
        3   0    2  0
            1    2  1
            2    2  1
            3    2  2
        <BLANKLINE>
        * Measure 'mu ^ 2':
                mu ^ 2
        F_0 F_1
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

        >>> print(f ^ 3)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f ^ 3':
        i            0  1  2  3  4  5
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

        return cls.cartesian_product(
            factors=[vector] * n,
            name=name,
            domain_name=domain_name,
            sig_alg_name=sig_alg_name,
            measure_name=measure_name,
            index=index,
            index_kind=index_kind,
        )

    # --------------------- utils --------------------- #

    def _reset_class(self) -> None:
        import pandas as pd

        from ..measures.probability_measure import ProbabilityMeasure
        from .measurable_function import MeasurableFunction
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

        if self.dimension == 1:
            if isinstance(self.measure, ProbabilityMeasure):
                self.__class__ = RandomVariable
            else:
                self.__class__ = MeasurableFunction
            self.data = (
                self.data.squeeze(axis=1)
                if isinstance(self.data, pd.DataFrame)
                else self.data
            )

        elif isinstance(self.measure, ProbabilityMeasure):
            self.__class__ = RandomVector

        else:
            self.__class__ = MeasurableVector

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

    @cached_property
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
         x
         0
         1
         2
        <BLANKLINE>
        * Sigma algebra 'F':
            F
        x
        0   0
        1   1
        2   1
        """
        from ..spaces.measurable_space import MeasurableSpace

        return MeasurableSpace._from_validated(sig_alg=self.sig_alg)

    @cached_property
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
         x
         0
         1
         2
        <BLANKLINE>
        * Sigma algebra 'F':
             F
        x
        0    0
        1    1
        2    1
        <BLANKLINE>
        * Measure 'mu':
              mu
        F
        0      2
        1      8
        """
        from ..spaces.measure_space import MeasureSpace

        return (
            MeasureSpace._from_validated(measure=self.measure) if self.measure else None
        )

    # --------------------- probability methods --------------------- #

    def sample(
        self,
        size: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> PandasLike:
        """Generate random samples from the range space of this random vector.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for the random number generator.

        Returns
        -------
        sample : PandasLike
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

    def restrict_to(
        self,
        measurable_set: Set | list,
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
        from ..spaces.measure_space import MeasureSpace
        from ..spaces.set import Set
        from .random_vector import RandomVector

        if not isinstance(measurable_set, (Set, list)):
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
        elif isinstance(measurable_set, Set) and measurable_set not in self.sig_alg:
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
        import pandas as pd

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
        import numpy as np

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
        self, other: MeasurableVector | Hashable | tuple[Hashable] | pd.Series
    ) -> bool:
        r"""Check equality with another measurable vector or compute an inverse image of a value under the measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : MeasurableVector | Hashable | tuple[Hashable] | pd.Series
            Another measurable vector to compare with, or a value for which to compute the inverse image.

        Returns
        -------
        output : bool | MeasurableSet
            If `other` is a `MeasurableVector`, returns `True` if the two measurable vectors are equal, and `False` otherwise. If `other` is a value, returns the measurable set corresponding to the inverse image of that value under the measurable vector.
        """
        import pandas as pd

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

        return self_sorted.equals(other_sorted)

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
        import pandas as pd

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

            return ParametrizedMeasurableFunction.from_domains(
                complete_domain=other.domain,
                mapping=new_values.rename(new_name),
                name=new_name,
                measurable_domain=self.domain,
                sig_alg=super_sig_alg,
                measure=measure,
                parameter_domain_name=other.parameter_domain_name,
            )

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
        import pandas as pd

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
