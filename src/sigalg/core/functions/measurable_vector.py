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
    from ..indices.index import Index
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
        from .measurable_function import MeasurableFunction

        v = MeasurableFuncNormalizer(domain=domain, sig_alg=sig_alg, measure=measure)

        domain = v.domain
        sig_alg = v.sig_alg
        measure = v.measure

        if cls is MeasurableFunction:
            if (isinstance(constant, tuple) and len(constant) > 1) or (
                index is not None
            ):
                raise ValueError(
                    "Cannot create a constant instance of MeasurableFunction from a constant with more than 1 dimension."
                )
            if isinstance(constant, tuple):
                constant = constant[0]

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
        from .._utils.function_helpers import ascend_from_atom_space
        from ..indices.index import Index
        from ..indices.time import Time
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .measurable_function import MeasurableFunction
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

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
        if cls is MeasurableFunction or cls is RandomVariable:
            dim = 1

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
            data = ascend_from_atom_space(
                self_data=sig_alg_data,
                sig_alg_data=sub_sig_alg.data,
                parameter_names=[],
            )

        if isinstance(data, pd.Series):
            data.name = output_name

        if cls is RandomVariable or cls is RandomVector:
            if measure is not None:
                if not isinstance(measure, ProbabilityMeasure):
                    raise ValueError(
                        "A random variable/vector may only be created with a probability measure."
                    )
            else:
                measure = ProbabilityMeasure.from_rand(domain=sig_alg, random_state=rng)

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
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasurableFunction,
        ...     MeasurableVector,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Generate a measure space.

        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     random_state=42,
        ... )
        >>> mu = Measure(domain=F, mapping={0: 1, 1: 2, 2: 3})

        Generate two measurable vectors. One has a measure, the other does not.

        >>> f = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     index=[0, 1, 2],
        ...     max_value=2,
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
        >>> g = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     index=[3, 4],
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

        Together with the `pushforward` method, the `concatenation` method is SigAlg's mechanism for generating the joint distributions of random variables. We demonstrate.

        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> P = ProbabilityMeasure(domain=Omega, mapping=dict(zip(Omega, [0.1, 0.2, 0.3, 0.4])))
        >>> X = RandomVariable(domain=Omega, measure=P, mapping=dict(zip(Omega, [1, 1, 0, 1])))
        >>> Y = RandomVariable(domain=Omega, measure=P, mapping=dict(zip(Omega, [1, 1, 1, 0])), name="Y")

        The joint distribution of the two random variables comes by pushing the measure `P` forward along the random vector `X | Y`.

        >>> joint = P >> (X | Y)
        >>> joint
        ProbabilityMeasure(domain=XY_range, sig_alg=R, name=P_XY)
        >>> print(joint)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_XY':
             P_XY
        X Y
        0 1   0.3
        1 0   0.4
          1   0.3
        """
        actual_funcs = [func for func in factors if isinstance(func, Function)]
        measure = cls._get_max_measure(actual_funcs)
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
        i                0  1
        omega_0 omega_1
        0       0        1  2
                1        1  3
                2        1  3
        1       0        1  2
                1        1  3
                2        1  3
        2       0        0  2
                1        0  3
                2        0  3

        Print the measure space of the Cartesian product.

        >>> print(product.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega x Omega, F x G, P x Q)
        ===============================================
        <BLANKLINE>
        * Domain 'Omega x Omega':
         omega_0  omega_1
               0        0
               0        1
               0        2
               1        0
               1        1
               1        2
               2        0
               2        1
               2        2
        <BLANKLINE>
        * Sigma algebra 'F x G':
        i                0  1
        omega_0 omega_1
        0       0        0  0
                1        0  1
                2        0  1
        1       0        0  0
                1        0  1
                2        0  1
        2       0        1  0
                1        1  1
                2        1  1
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
        i                0  1
        omega_0 omega_1
        0       0        1  2
                1        1  3
                2        1  3
        1       0        1  2
                1        1  3
                2        1  3
        2       0        0  2
                1        0  3
                2        0  3

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

        measures = [func.measure for func in factors if func.measure is not None]
        all_measures = len(measures) == len(factors)

        if all_measures:
            measure = Measure.tensor_product(measures, name=measure_name)
            sig_alg = measure.sig_alg
            sig_alg.name = sig_alg_name if sig_alg_name else sig_alg.name
            sig_alg.domain.name = domain_name if domain_name else sig_alg.domain.name
        else:
            measure = None
            sig_alg = SigmaAlgebra.tensor_product(
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

            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.squeeze(axis=1)

        elif isinstance(self.measure, ProbabilityMeasure):
            self.__class__ = RandomVector

        else:
            self.__class__ = MeasurableVector

    @staticmethod
    def _get_max_measure(
        vectors: list[MeasurableVector | Real],
    ) -> Measure | None:
        """Check that all measurable vectors in the list have linearly-ordered measures, and return the maximum.

        The set of all sigma-algebras on a fixed domain is a lattice, and one may define a measure on any one of the sigma-algebras in this lattice. Given two such measure/sigma-algebra pairs `(mu, F)` and `(nu, G)`, let us say that `(mu, F) <= (nu, G)` provided that `F` is a subset of `G` and `mu` is the restriction of `nu` to `F`. This is a partial order on the set of all measure/sigma-algebra pairs.

        This method checks that the collection of measure/sigma-algebra pairs of the measurable vectors forms a chain with respect to this partial order, i.e., a linearly-ordered subset. If so, the method returns the maximum measure in the chain; otherwise, it returns `None`.

        Parameters
        ----------
        vectors : list[MeasurableVector | Real]
            A list of measurable vectors to check for consistent measures.

        Returns
        -------
        max_measure : Measure | None
            The maximum element in the chain of measures, or `None` if it does not exist.
        """
        measures = [
            vec.measure
            for vec in vectors
            if hasattr(vec, "measure") and vec.measure is not None
        ]

        if len(measures) == 0:
            return None
        else:
            max_measure = measures[0]
            for measure in measures[1:]:
                if max_measure.is_restriction_of(measure):
                    max_measure = measure
                elif not measure.is_restriction_of(max_measure):
                    return None

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
            MeasureSpace._from_validated(measure=self.measure)
            if self.measure is not None
            else None
        )

    # --------------------- probability methods --------------------- #

    def sample(
        self,
        size: int,
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
        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, RandomVector, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Generate a random probability space and sample from a 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=Omega,
        ...     num_atoms=4,
        ...     random_state=rng,
        ... )
        >>> P = ProbabilityMeasure.from_rand(domain=F, random_state=rng)
        >>> X = RandomVector.from_rand(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     max_value=10,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        i      0  1
        omega
        0      9  4
        1      9  4
        2      8  6
        3      9  4
        4      7  7
        5      9  4
        6      1  3
        7      9  4
        8      1  3
        9      9  4
        >>> print(X.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         omega
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
              F
        omega
        0     1
        1     1
        2     3
        3     1
        4     2
        5     1
        6     0
        7     1
        8     0
        9     1
        <BLANKLINE>
        * Probability measure 'P':
                  P
        F
        1  0.751070
        3  0.244804
        2  0.000026
        0  0.004100
        >>> X_sample = X.sample(size=1_000, random_state=rng)
        >>> print(X_sample.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'C':
                   C
        X_0 X_1
        9   4    738
        8   6    258
        1   3      4

        Sample from a 1-dimensional random variable.

        >>> Y = RandomVariable.from_rand(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     max_value=10,
        ...     random_state=rng,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
               Y
        omega
        0      4
        1      4
        2      7
        3      4
        4      3
        5      4
        6      7
        7      4
        8      7
        9      4
        >>> Y_sample = Y.sample(size=1_000, random_state=rng)
        >>> print(Y_sample.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'C':
             C
        Y
        4  736
        7  264
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

    # --------------------- function methods --------------------- #

    def restrict_to(
        self,
        subset: Set | list,
        normalize: bool = False,
        subset_name: Hashable | None = "A",
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
        >>> from sigalg.core import Domain, Measure, MeasurableVector, Set, SigmaAlgebra

        Define a 2-dimensional measurable vector with a measure.

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

        >>> A = Set([1, 2, 3], domain=X)
        >>> f_A = f.restrict_to(A)
        >>> print(f_A)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f|A':
        i  0  1
        x
        1  3  4
        2  3  4
        3  5  6

        Print its measure space consisting of the restricted sigma-algebra and measure.

        >>> print(f_A.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (A, F|A, mu|A)
        ============================
        <BLANKLINE>
        * Domain 'A':
         x
         1
         2
         3
        <BLANKLINE>
        * Sigma algebra 'F|A':
                F|A
        x
        1         1
        2         1
        3         2
        <BLANKLINE>
        * Measure 'mu|A':
             mu|A
        F
        1       5
        2       3

        Compute the same restriction using the overloaded `|` operator.

        >>> print(f | A)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f|A':
        i   0  1
        x
        1   3  4
        2   3  4
        3   5  6

        Restrict the measurable vector using a `list` with a custom name. Pass `normalize=True` to create a probability measure.

        >>> f_B = f.restrict_to([1, 2], normalize=True, subset_name="B")
        >>> print(f_B)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'f|B':
        i   0  1
        x
        1   3  4
        2   3  4
        >>> print(f_B.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (B, F|B, mu|B)
        ================================
        <BLANKLINE>
        * Domain 'B':
         x
         1
         2
        <BLANKLINE>
        * Sigma algebra 'F|B':
                F|B
        x
        1         1
        2         1
        <BLANKLINE>
        * Probability measure 'mu|B':
                    mu|B
        F
        1            1.0

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a measurable vector on a measure space $(X, \mathcal{F}, \mu)$. If $A\in \mathcal{F}$ is an measurable set, then we may restrict the measurable vector to obtain the function $f|_A : A \to \mathbb{R}^d$ on $A$.
        """
        restricted_sig_alg = self.sig_alg.restrict_to(
            subset=subset, subset_name=subset_name
        )
        if self.measure is not None:
            restricted_measure = self.measure.restrict_to(
                subset, normalize=normalize, subset_name=subset_name
            )
        else:
            restricted_measure = None

        return super().restrict_to(
            subset=subset,
            subset_name=subset_name,
            sig_alg=restricted_sig_alg,
            measure=restricted_measure,
        )

    def __round__(self, ndigits: int = None) -> MeasurableVector:
        """Round the outputs of the measurable vector to a specified number of decimal places.

        Parameters
        ----------
        decimals : int, default=0
            The number of decimal places to round to. Must be a non-negative integer.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableFunction, Measure, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1])))
        >>> mu = Measure(domain=F, mapping=dict(zip(F.atom_space, [1, 2])))
        >>> f = MeasurableFunction(
        ...     domain=X, sig_alg=F, measure=mu, mapping=dict(zip(X, [0.1, 0.45, 0.45]))
        ... )
        >>> rounded_f = round(f, 2)
        >>> print(rounded_f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'round(f)':
           round(f)
        x
        0      0.10
        1      0.45
        2      0.45
        >>> print(rounded_f.measure_space)  # doctest: +NORMALIZE_WHITESPACE
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
        0  0
        1  1
        2  1
        <BLANKLINE>
        * Measure 'mu':
           mu
        F
        0   1
        1   2
        """
        return super().__round__(
            ndigits=ndigits,
            sig_alg=self.sig_alg,
            measure=self.measure,
        )

    # --------------------- conversion methods --------------------- #

    def to_function(self) -> Function:
        """Promote to a `Function` instance."""
        return Function._from_validated(
            data=self.data,
            kind="any",
            domain_kind=type(self.domain).__name__,
            domain_name=self.domain.name,
            index_kind=type(self.index).__name__ if self.index is not None else "Index",
            index_name=self.index.name if self.index is not None else None,
            name=self.name,
        )

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
            parameter_list = ", ".join(self.variable_names)
            return (
                type(self)._repr_name + f"(parameters=({parameter_list}), "
                f"domain={self.domain.name}, "
                f"sig_alg={self.sig_alg.name}, "
                f"measure={self.measure.name}, "
                f"name={self.name})"
            )
        else:
            parameter_list = ", ".join(self.variable_names)
            return (
                type(self)._repr_name + f"(parameters=({parameter_list}), "
                f"domain={self.domain.name}, "
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

    def _apply_binary_operation(
        self,
        other: Function | Real,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> Function:
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

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableFunction,
        ...     MeasurableVector,
        ...     Measure,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)

        Define two functions on a measurable space with 2-dimensional outputs and print their sum.

        >>> X = Domain([(1, 2), (3, 4), (5, 6), (7, 8)], variable_names=["u", "v"])
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2])))
        >>> f = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i    0  1
        u v
        1 2  0  7
        3 4  6  4
        5 6  6  4
        7 8  4  8
        >>> g = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     dim=2,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        i    0  1
        u v
        1 2  0  6
        3 4  2  0
        5 6  2  0
        7 8  5  9
        >>> print(f + g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector '(f + g)':
        i    0   1
        u v
        1 2  0  13
        3 4  8   4
        5 6  8   4
        7 8  9  17

        Since both functions have the same sigma-algebra, the sigma-algebra passes through to the sum.

        >>> (f + g).measurable_space
        MeasurableSpace(domain=X, sig_alg=F)

        The same is true for differences of functions with 1-dimensional outputs, for example.

        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
             f
        u v
        1 2  7
        3 4  7
        5 6  7
        7 8  7
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
             g
        u v
        1 2  7
        3 4  5
        5 6  5
        7 8  1
        >>> print(f - g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f - g)':
             (f - g)
        u v
        1 2        0
        3 4        2
        5 6        2
        7 8        6
        >>> (f - g).measurable_space
        MeasurableSpace(domain=X, sig_alg=F)

        Arithmetic operations between two measurable functions does not strictly require that they are both defined on the same sigma-algebra. If one sigma-algebra is a sub-sigma-algebra of another, then the result of an arithmetic operation will be defined on the larger sigma-algebra.

        >>> G = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 1])), name="G")
        >>> G <= F
        True
        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
             f
        u v
        1 2  8
        3 4  4
        5 6  4
        7 8  5
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=G,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
             g
        u v
        1 2  3
        3 4  1
        5 6  1
        7 8  1
        >>> print(f * g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f * g)':
             (f * g)
        u v
        1 2       24
        3 4        4
        5 6        4
        7 8        5
        >>> (f * g).measurable_space
        MeasurableSpace(domain=X, sig_alg=F)

        If two measurable functions carry the same measure, this measure will pass through to the result of an arithmetic operation between them.

        >>> mu = Measure(domain=F, mapping=dict(zip(F.atom_space, [4, 2, 7])))
        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
             f
        u v
        1 2  9
        3 4  7
        5 6  7
        7 8  6
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
             g
        u v
        1 2  4
        3 4  8
        5 6  8
        7 8  5
        >>> print(f / g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f / g)':
             (f / g)
        u v
        1 2    2.250
        3 4    0.875
        5 6    0.875
        7 8    1.200
        >>> (f / g).measure_space
        MeasureSpace(domain=X, sig_alg=F, measure=mu)

        Again, the arithmetic operations do not strictly require that measurable functions carry the same measure, as long as one is defined on a sub-sigma-algebra of another and is the restriction of the measure on the larger sigma-algebra. Then the result of an arithmetic operation will carry the larger sigma-algebra and its measure.

        >>> nu = Measure(domain=G, mapping=dict(zip(G.atom_space, [4, 9])), name="nu")
        >>> nu.is_restriction_of(mu)
        True

        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
             f
        u v
        1 2  4
        3 4  4
        5 6  4
        7 8  2
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=G,
        ...     measure=nu,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
             g
        u v
        1 2  0
        3 4  5
        5 6  5
        7 8  5
        >>> print(f**g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f ** g)':
             (f ** g)
        u v
        1 2       1.0
        3 4    1024.0
        5 6    1024.0
        7 8      32.0
        >>> (f**g).measure_space
        MeasureSpace(domain=X, sig_alg=F, measure=mu)
        """
        from .parametrized_measurable_function import ParametrizedMeasurableFunction

        if isinstance(other, Real):
            sig_alg = self.sig_alg
            measure = self.measure
            complete_domain_name = None
            parameter_domain_name = None
            parameter_names = None
            self_promoted = self

        elif isinstance(other, MeasurableVector):
            if self.sig_alg <= other.sig_alg:
                sig_alg = other.sig_alg

            elif self.sig_alg > other.sig_alg:
                sig_alg = self.sig_alg

            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable functions on incompatible measurable spaces."
                )

            measure = self._get_max_measure([self, other])
            complete_domain_name = None
            parameter_domain_name = None
            parameter_names = None
            self_promoted = self

        elif isinstance(other, ParametrizedMeasurableFunction):
            if self.sig_alg <= other.sig_alg:
                sig_alg = other.sig_alg

            elif self.sig_alg > other.sig_alg:
                sig_alg = self.sig_alg

            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable functions on incompatible measurable spaces."
                )

            measure = self._get_max_measure([self, other])
            complete_domain_name = other.domain.name
            parameter_domain_name = other.parameter_domain_name
            parameter_names = other.parameter_names
            self_promoted = ParametrizedMeasurableFunction._from_validated(
                data=self.data,
                sig_alg=sig_alg,
                measure=measure,
                complete_domain_name=complete_domain_name,
                parameter_domain_name=parameter_domain_name,
                parameter_names=parameter_names,
                name=self.name,
            )

        elif isinstance(other, Function):
            if self.sig_alg in other.lattice:
                sig_alg = self.sig_alg
                measure = self.measure
                complete_domain_name = None
                parameter_domain_name = None
                parameter_names = None
                self_promoted = self

            else:
                sig_alg = None
                measure = None
                complete_domain_name = None
                parameter_domain_name = None
                parameter_names = None
                self_promoted = self.to_function()

        else:
            raise NotImplementedError(
                f"Arithmetic not implemented between MeasurableVector and {type(other).__name__}."
            )

        return Function._apply_binary_operation(
            self=self_promoted,
            other=other,
            operation=operation,
            op_symbol=op_symbol,
            reverse=reverse,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            sig_alg=sig_alg,
            measure=measure,
            complete_domain_name=complete_domain_name,
            parameter_domain_name=parameter_domain_name,
            parameter_names=parameter_names,
        )
