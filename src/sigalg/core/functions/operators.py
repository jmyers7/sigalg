"""Class for operators on random vectors, such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of measures."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ..measures.measure import Measure
    from ..measures.parametrized_measure import (
        ParametrizedMeasure,
    )
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.set import Set
    from .function import Function
    from .measurable_function import MeasurableFunction
    from .measurable_vector import MeasurableVector
    from .parametrized_measurable_function import ParametrizedMeasurableFunction
    from .random_variable import RandomVariable
    from .random_vector import RandomVector


class Operators:
    """Class containing methods such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of measures."""

    # --------------------- general methods --------------------- #

    @classmethod
    def sum(
        cls,
        vec: MeasurableVector,
        name: Hashable | None = None,
    ) -> MeasurableFunction:
        """Compute the sum of the components of a measurable vector.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector whose components are to be summed.
        name : Hashable | None, default=None
            The name of the resulting measurable function. If `None`, a default name will be generated.

        Returns
        -------
        summed_vec : MeasurableFunction
            A measurable function representing the sum of the components of the input measurable vector.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector
        >>> D = Domain.from_sequence(size=2, variable_name="flip", name="D")
        >>> X = (D ^ 3).with_name("X")
        >>> f = MeasurableVector.from_identity(domain=X)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i                     0  1  2
        flip_0 flip_1 flip_2
        0      0      0       0  0  0
                      1       0  0  1
               1      0       0  1  0
                      1       0  1  1
        1      0      0       1  0  0
                      1       1  0  1
               1      0       1  1  0
                      1       1  1  1
        >>> g = f.sum(name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                              g
        flip_0 flip_1 flip_2
        0      0      0       0
                      1       1
               1      0       1
                      1       2
        1      0      0       1
                      1       2
               1      0       2
                      1       3
        """
        from ..functions.measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")

        data_trans = vec.data.copy()
        data_trans = data_trans.sum(axis=1)

        if name is None:
            name = f"{vec.name}_sum"

        return MeasurableVector._from_validated(
            data=data_trans,
            sig_alg=vec.sig_alg,
            measure=vec.measure,
            index_kind="Index",
            index_name=None,
            name=name,
        )

    # TODO: add Notes section
    @classmethod
    def transform(
        cls,
        vec: MeasurableVector,
        functions: list[Callable[[MeasurableVector], MeasurableFunction]],
        index: IndexLike | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Apply a transformation to a measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector to transform.
        functions : list[Callable[[MeasurableVector], MeasurableFunction]]
            A list of functions to apply to the measurable vector.
        index : IndexLike | None, default=None
            The new index for the transformed vector. If `None`, the original index of `vec` will be used.
        name : Hashable | None, default=None
            The name of the transformed vector. If `None`, a default name will be generated.

        Returns
        -------
        transformed_vector : MeasurableVector
            The transformed measurable vector.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Operators, RandomVariable, Time
        >>> from sigalg.processes import IIDProcess, StochasticProcess
        >>> T = Time.discrete(start=0, length=2)
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.5),
        ...     support=[0, 1],
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        t       0  1  2
        omega
        0       0  0  0
        1       0  0  1
        2       0  1  0
        3       0  1  1
        4       1  0  0
        5       1  0  1
        6       1  1  0
        7       1  1  1
        >>> S = Time.discrete(start=4, stop=5)
        >>> def f4(process: StochasticProcess) -> RandomVariable:
        ...     X0, X1, _ = X
        ...     return X0 + X1
        >>> def f5(process: StochasticProcess) -> RandomVariable:
        ...     _, X1, X2 = X
        ...     return X1 + X2
        >>> X_transform = Operators.transform(vec=X, functions=[f4, f5], index=S)
        >>> print(X_transform)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_transform':
        t       4  5
        omega
        0       0  0
        1       0  1
        2       1  1
        3       1  2
        4       1  0
        5       1  1
        6       2  1
        7       2  2
        """
        from ..indices.index import Index
        from .measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")
        if not isinstance(functions, list):
            raise TypeError("functions must be a list.")
        if not all(isinstance(f, Callable) for f in functions):
            raise TypeError("Each element in functions must be callable.")
        if not isinstance(index, Index):
            index = Index(index) if index is not None else None
        if index is not None and len(functions) != len(index):
            raise ValueError("The number of functions must match the length of index.")

        if index is not None and not isinstance(index, Index):
            index = Index(index)

        if index is None:
            index = vec.index

        transformed_vecs = {}

        for f, i in zip(functions, index):
            transformed_vecs[i] = f(vec).data

        data = pd.DataFrame(transformed_vecs, index=vec.domain.data, columns=index.data)

        if name is None:
            name = f"{vec.name}_transform"

        return MeasurableVector._from_validated(
            data=data,
            sig_alg=vec.sig_alg,
            measure=vec.measure,
            index_kind=type(index).__name__,
            index_name=index.name,
            name=name,
        )

    # TODO: add Notes section
    @classmethod
    def pointwise_map(
        cls,
        vec: MeasurableVector,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Apply a function pointwise to the values of a measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector to which the function will be applied.
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to each value in the measurable vector.
        name : Hashable | None, default=None
            The name of the transformed measurable vector. If `None`, a default name will be generated.

        Returns
        -------
        mapped_vector : MeasurableVector
            A new measurable vector with the function applied pointwise to its values.

        Examples
        --------
        >>> from sigalg.core import Operators, Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t       0  1  2
        omega
        0       3  2  1
        1       3  2  3
        2       3  4  3
        3       3  4  5
        >>> def f(x):
        ...     return x + 1
        >>> X_mapped = Operators.pointwise_map(vec=X, function=f)
        >>> print(X_mapped)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_mapped':
        t       0  1  2
        omega
        0       4  3  2
        1       4  3  4
        2       4  5  4
        3       4  5  6
        """
        from .measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")
        if not isinstance(function, Callable):
            raise TypeError("function must be a callable object.")

        data_trans = vec.data.copy()
        data_trans = data_trans.map(function)
        if name is None:
            name = f"{vec.name}_mapped"

        return MeasurableVector._from_validated(
            data=data_trans,
            sig_alg=vec.sig_alg,
            measure=vec.measure,
            index_kind=vec.index_kind,
            index_name=vec.index_name,
            name=name,
        )

    # TODO: add Notes section
    @classmethod
    def cumsum(
        cls,
        vec: MeasurableVector,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Compute the cumulative sum of a measurable vector along its index.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector for which to compute the cumulative sum.
        name : Hashable | None, default=None
            The name of the transformed measurable vector. If `None`, a default name will be generated.

        Returns
        -------
        cumsum_vector : MeasurableVector
            A new measurable vector representing the cumulative sum of the input vector.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Operators, Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        t           1  2  3
        omega
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X_cumsum = Operators.cumsum(X)
        >>> print(X_cumsum) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        t           1  2  3
        omega
        0           0  0  0
        1           0  0  1
        2           0  1  1
        3           0  1  2
        4           1  1  1
        5           1  1  2
        6           1  2  2
        7           1  2  3
        """
        from .measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")

        data_trans = vec.data.copy()
        data_trans = data_trans.cumsum(axis=1)
        if name is None:
            name = f"{vec.name}_cumsum"

        return MeasurableVector._from_validated(
            data=data_trans,
            sig_alg=vec.sig_alg,
            measure=vec.measure,
            index_kind=vec.index_kind,
            index_name=vec.index_name,
            name=name,
        )

    # TODO: add Notes section
    @classmethod
    def cumprod(
        cls,
        vec: MeasurableVector,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Compute the cumulative product of a measurable vector along its index.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector for which to compute the cumulative product.
        name : Hashable | None, default=None
            The name of the transformed vector. If `None`, a default name will be generated.

        Returns
        -------
        cumprod_vector : MeasurableVector
            A new measurable vector representing the cumulative product of the input vector.

        Examples
        --------
        >>> from sigalg.core import Operators, Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=3)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=3, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t           0  1  2  3
        omega
        0           3  2  1  0
        1           3  2  1  2
        2           3  2  3  2
        3           3  2  3  4
        4           3  4  3  2
        5           3  4  3  4
        6           3  4  5  4
        7           3  4  5  6
        >>> X_cumprod = Operators.cumprod(X)
        >>> print(X_cumprod) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumprod':
        t           0   1   2    3
        omega
        0           3   6   6    0
        1           3   6   6   12
        2           3   6  18   36
        3           3   6  18   72
        4           3  12  36   72
        5           3  12  36  144
        6           3  12  60  240
        7           3  12  60  360
        """
        from .measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")

        data_trans = vec.data.copy()
        data_trans = data_trans.cumprod(axis=1)
        if name is None:
            name = f"{vec.name}_cumprod"

        return MeasurableVector._from_validated(
            data=data_trans,
            sig_alg=vec.sig_alg,
            measure=vec.measure,
            index_kind=vec.index_kind,
            index_name=vec.index_name,
            name=name,
            mapping=data_trans,
            index=vec.index,
        )

    @classmethod
    def max_value(cls, vec: MeasurableVector) -> Real:
        """Get the maximum value across all outputs and indices of a measurable vector.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector for which to find the maximum value.

        Returns
        -------
        max_value : Real
            The maximum value found in the measurable vector.

        Examples
        --------
        >>> from sigalg.core import Operators, Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t           0  1  2
        omega
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> max_value = Operators.max_value(X)
        >>> print(max_value)
        5
        """
        from .measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")

        return vec.data.values.max()

    @classmethod
    def min_value(cls, vec: MeasurableVector) -> Real:
        """Get the minimum value across all outputs and indices of a measurable vector.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector for which to find the minimum value.

        Returns
        -------
        min_value : Real
            The minimum value found in the measurable vector.

        Examples
        --------
        >>> from sigalg.core import Operators, Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t           0  1  2
        omega
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> min_value = Operators.min_value(X)
        >>> print(min_value)
        1
        """
        from .measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")
        return vec.data.values.min()

    # --------------------- measure-related methods --------------------- #

    @classmethod
    def integrate(
        cls,
        function: Function | ParametrizedMeasurableFunction,
        subset: Set | list[Hashable] | None = None,
        measure: Measure | ParametrizedMeasure | None = None,
        variables: tuple[Hashable, Hashable] | None = None,
        subset_name: Hashable | None = None,
    ) -> Real | pd.Series | Function:
        r"""Compute the Lebesgue integral of a measurable vector with respect to a measure over an (optional) set.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        function : MeasurableVector | ParametrizedMeasurableFunction
            The measurable vector or parametrized measurable function to integrate.
        subset: Set | list[Hashable] | None, default=None
            The optional set over which to integrate. If `None`, the integral will be taken over the entire domain of the measurable vector.
        measure : Measure | ParametrizedMeasure | None, default=None
            The measure or parametrized measure with respect to which to integrate. If `None`, the measure of the underlying measure space is used (if it exists) carried by the measurable vector or parametrized measurable function.
        variables : tuple[Hashable, Hashable] | None, default=None
            A pair of explicit domain variables over which the integral should be taken. The first item of the tuple should correspond to the variable name of the function's domain, while the second item should correspond to the variable name of the measure's domain.
        subset_name : Hashable | None, default=None
            If the `subset` is passed as a list, the name that will be assigned to the subset. Ignored otherwise.

        Returns
        -------
        integral : Real | pd.Series | Function
            Returns the following:

            * If `function` is a `MeasurableFunction` and `measure` is a `Measure`, returns a `Real` representing the integral of the function with respect to the measure over the specified set.

            * If `function` is a `MeasurableVector` of dimension > 1 and `measure` is a `Measure`, returns a `pd.Series` representing the integral of each component of the vector with respect to the measure over the specified set.

            * If `function` is a `ParametrizedMeasurableFunction` and `measure` is a `Measure`, returns a `Function` representing the integral of the function with respect to the measure over the specified set for each parameter value.

            * If `function` is a `MeasurableFunction` and `measure` is a `ParametrizedMeasure`, returns a `Function` representing the integral of the function with respect to the measure over the specified set for each parameter value.

            * If `function` is a `ParametrizedMeasurableFunction` and `measure` is a `ParametrizedMeasure`, returns a `Function` representing the integral of the function with respect to the measure over the specified set for each parameter value.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableFunction,
        ...     Measure,
        ...     MeasureSpace,
        ...     Operators,
        ...     ParametrizedMeasurableFunction,
        ...     ParametrizedMeasure,
        ...     Set,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)

        Define a measure space and a measurable function.

        >>> measure_space = MeasureSpace.from_rand(
        ...     domain_size=100,
        ...     num_atoms=27,
        ...     num_null_atoms=12,
        ...     random_state=rng,
        ... )
        >>> X, F, mu = measure_space
        >>> f = MeasurableFunction.from_rand(
        ...     *measure_space,
        ...     distribution="normal",
        ...     diff_values=24,
        ...     random_state=rng,
        ... )

        Get a measurable set from the sigma-algebra, compute the integral over this set, and check that it agrees with the defining formula for the Lebesgue integral.

        >>> U = F.get_random_set(num_atoms=4, name="U", random_state=rng)
        >>> I_U = U.indicator
        >>> integrate = Operators.integrate
        >>> np.allclose(integrate(f, U), sum(I_U(A) * f(A) * mu(A) for A in F))
        True

        Check that the integral over a null set is 0.

        >>> N = measure_space.get_random_set(
        ...     num_atoms=3,
        ...     is_null=True,
        ...     name="N",
        ...     random_state=rng,
        ... )
        >>> I_N = N.indicator
        >>> integrate(f, N)
        0.0

        Define a new measure space and measurable function to demonstrate integration against parametrized objects.

        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 1),
        ...         2: (1, 1),
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         (0, 1): 2,
        ...         (1, 1): 3,
        ...     },
        ... )
        >>> f = MeasurableFunction(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ... )

        Define a parametrized measure and parametrized measurable function over the same parameter domain.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> nu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping={
        ...         (0, 0, 1): 3,  # (theta, F_0, F_1) = (0, 0, 0), etc ...
        ...         (0, 1, 1): 4,
        ...         (1, 0, 1): 1,
        ...         (1, 1, 1): 2,
        ...     },
        ...     name="nu",
        ... )
        >>> g = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping={
        ...         (0, 0): 2,  # (theta, X) = (0, 0)
        ...         (0, 1): 4,
        ...         (0, 2): 4,
        ...         (1, 0): 1,
        ...         (1, 1): -1,
        ...         (1, 2): -1,
        ...     },
        ...     name="g",
        ... )

        Extract a measurable set from the sigma-algebra.

        >>> U = Set([1, 2], domain=X, name="U")

        It is convenient to conceptualize a parametrized measure as a family of measures. Then integration of a measurable function against a parametrized measure returns a function of the parameters whose values are the integrals of the function against the measures. Iteration over the parametrized measure yields the measures, allowing us to check that these integrals all match.

        >>> all(integrate(f, U, nu)(**param) == integrate(f, U, measure) for param, measure in nu)
        True

        Likewise, it is convenient to conceptualize a parametrized measurable function as a family of measurable functions. Then integration of a parametrized measurable function against a measure returns a function of the parameters whose values are the integrals of the functions against the measure. Iteration over the parametrized measurable function yields the functions, allowing us to check that these integrals match.

        >>> all(integrate(g, U, mu)(**param) == integrate(function, U, mu) for param, function in g)
        True

        Finally, it is possible to integrate a parametrized measurable function against a parametried measure as long as their parameter domains agree. We leave the reader to guess the meaning of the following verification.

        >>> all(
        ...     integrate(g, U, nu)(**param) == integrate(function, U, measure)
        ...     for (param, function), (_, measure) in zip(g, nu)
        ... )
        True

        Notes
        -----
        Let $f: X \to \mathbb{R}$ be a measurable function on a measure space $(X, \mathcal{F}, \mu)$. Assuming $X$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{F}$ is determined by its set $\alpha(\mathcal{F})$ of atoms. Let $U$ be a measurable set in $\mathcal{F}$, and write $I_U$ for its indicator function. Since both $f$ and $I_U$ are $\mathcal{F}$-measurable, they take constant values on each atom $A\in \alpha(\mathcal{F})$ that we write as $f(A)$ and $I_U(A)$, respectively. Then the *Lebesgue integral* of $f$ over $U$ is the number

        $$
        \int_U f \, d\mu = \sum_{A\in \alpha(\mathcal{F})} I_U(A)f(A) \mu(A).
        $$

        If $f:X \to \mathbb{R}^d$ is instead a measurable vector of dimension $d>1$, with components

        $$
        f = (f_1, f_2, \ldots, f_d),
        $$

        then we define the *Lebesgue integral* of $f$ over $U$ to be the $d$-dimensional vector whose entries are the separate Lebesgue integrals $\int_U f_j \, d\mu$, for $j=1,2,\ldots,d$.
        """
        from .._utils.function_helpers import compute_integral
        from ..measures.measure import Measure
        from ..measures.parametrized_measure import ParametrizedMeasure
        from ..sigma_algebras.lattice import NonMeasurableError
        from ..spaces.set import Set
        from .function import Function
        from .parametrized_measurable_function import ParametrizedMeasurableFunction

        if not isinstance(function, Function):
            raise TypeError(
                "function must be a Function or ParametrizedMeasurableFunction instance."
            )

        if measure is not None and not isinstance(
            measure, Measure | ParametrizedMeasure
        ):
            raise TypeError(
                "If given, measure must be a Measure or ParametrizedMeasure instance."
            )

        if function.dimension > 1 and isinstance(measure, ParametrizedMeasure):
            raise TypeError(
                "Cannot integrate a function with outputs of dimension > 1 against a parametrized measure."
            )

        indicator_data = None

        if variables is None:
            if measure is None:
                if hasattr(function, "measure"):
                    measure = function.measure
                else:
                    raise ValueError(
                        "The function does not carry a measure and the measure parameter of the integrate method is None."
                    )

            elif hasattr(function, "sig_alg"):
                if function.sig_alg <= measure.sig_alg:
                    measure = measure | function.sig_alg
                else:
                    raise ValueError(
                        "If given, measure must be defined on the sigma-algebra of the measurable vector."
                    )

            try:
                function_atom_data = function.atom_data(measure.sig_alg)
            except NonMeasurableError as e:
                raise NonMeasurableError(
                    "The function is not measurable with respect to the sigma-algebra carried by the measure."
                ) from e

            measure_data = measure.data

            if subset is not None:
                if not isinstance(subset, Set):
                    subset = Set(
                        indices=subset, domain=measure.sig_alg.domain, name=subset_name
                    )
                if subset not in measure.sig_alg:
                    raise ValueError(
                        "If given, the subset must be in the sigma-algebra of the measure."
                    )

                indicator_data = subset.lattice.get_atom_data(measure.sig_alg)

        else:
            if function.dimension > 1:
                raise ValueError(
                    "Integration over explicit variables is not implemented for functions with multi-dimensional outputs."
                )

            if subset is not None:
                raise TypeError(
                    "Integration over explict variables is not implmented over subsets."
                )

            function_atom_data = function.data.rename_axis(index={variables[0]: "var"})
            measure_data = measure.data.rename_axis(index={variables[1]: "var"})
            function_atom_data = function_atom_data.reindex(
                measure_data.index, fill_value=0.0
            )
            measure_data = measure_data.reindex(
                function_atom_data.index, fill_value=0.0
            )

            if isinstance(function, ParametrizedMeasurableFunction):
                function_atom_data = function_atom_data.unstack(
                    level=function.parameter_names
                )

        if subset is None:
            name = f"int {function.name} d{measure.name}"
        else:
            name = f"int_{subset.name} {function.name} d{measure.name}"

        data = compute_integral(
            function_atom_data=function_atom_data,
            measure_data=measure_data,
            indicator_data=indicator_data,
            function_parameter_names=getattr(function, "parameter_names", None),
            measure_parameter_names=getattr(measure, "parameter_names", None),
        )

        if isinstance(function, ParametrizedMeasurableFunction):
            return Function._from_validated(
                data=data.rename(name),
                kind="any",
                domain_kind="Domain",
                domain_name=function.parameter_domain_name,
                index_kind="Index",
                index_name=None,
                name=name,
            )

        elif isinstance(function, Function) and isinstance(
            measure, ParametrizedMeasure
        ):
            return Function._from_validated(
                data=data.rename(name),
                kind="any",
                domain_kind="Domain",
                domain_name=measure.parameter_domain_name,
                index_kind="Index",
                index_name=None,
                name=name,
            )

        else:
            if isinstance(data, pd.Series):
                return data.rename(name)
            else:
                return data.astype(Real)

    @classmethod
    def pushforward(
        cls,
        vec: MeasurableVector,
        measure: Measure | ParametrizedMeasure | None = None,
        name: Hashable | None = None,
    ) -> Measure | ParametrizedMeasure:
        r"""Push forward a (parametrized) measure on the domain of a measurable vector to a measure on its range.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector along which to push forward the measure.
        measure : Measure | ParametrizedMeasure | None, default=None
            Measure to push forward. If `None`, the measure carried by the measurable vector is used.
        name : Hashable | None, default=None
            The name of the pushforward measure. If `None`, a default name is generated.

        Returns
        -------
        pushforward : Measure | ParametrizedMeasure
            The measure pushed forward along the measurable vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableVector,
        ...     Measure,
        ...     Operators,
        ...     ParametrizedProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a measure space.

        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ... )

        Define a 2-dimensional measurable vector and pushforward the measure `mu`.

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
        >>> mu_f = Operators.pushforward(f, mu)
        >>> print(mu_f)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu_f':
                    mu_f
        f_0 f_1
        1   2          1
        3   4          5

        Now define a measurable space with a sample space.

        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )

        Define a parametrized probability measure on the sigma-algebra.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> mapping = {
        ...     (0, 0): 0.1,  # (theta, F) = (0, 0), etc ...
        ...     (0, 1): 0.2,
        ...     (0, 2): 0.7,
        ...     (1, 0): 0.4,
        ...     (1, 1): 0.5,
        ...     (1, 2): 0.1,
        ... }
        >>> P = ParametrizedProbabilityMeasure.from_domains(
        ...     measure_domain=F, parameter_domain=Theta, mapping=mapping
        ... )

        Define a 2-dimensional random vector and pushforward the parametrized probability measure `P`.

        >>> X = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 1),
        ...         1: (1, 1),
        ...         2: (3, 1),
        ...         3: (3, 1),
        ...     },
        ... )
        >>> P_X = Operators.pushforward(X, P)
        >>> print(P_X)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P_X':
        theta      0    1
        X_0 X_1
        1   1    0.1  0.4
        3   1    0.9  0.6

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a measurable vector on a measure space $(X, \mathcal{F}, \mu)$. Then we define a measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(A) = \mu\left( \{x \in X : f(x) \in A\}\right),
        $$

        for all Borel subsets $A\subset \mathbb{R}^d$.

        If $\mu$ is a parametrized measure on $X$ with parameter domain $\Theta$, then we define a parametrized measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(\theta, A) = \mu\left(\theta, \{x \in X : f(x) \in A\}\right),
        $$

        for all $\theta \in \Theta$ and all Borel subsets $A\subset \mathbb{R}^d$.
        """
        from .._utils.utils import to_df
        from ..functions.measurable_vector import MeasurableVector
        from ..measures.measure import Measure
        from ..measures.parametrized_measure import (
            ParametrizedMeasure,
        )
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be a MeasurableVector.")
        if measure is not None and not isinstance(
            measure, Measure | ParametrizedMeasure
        ):
            raise TypeError("measure must be a Measure or ParametrizedMeasure.")
        if measure is not None and vec.sig_alg != measure.sig_alg:
            raise ValueError("vec must have the same sigma-algebra as that of measure.")

        if measure is None:
            if vec.measure is None:
                raise ValueError(
                    "If measure is not given, then the measurable vector must carry a measure."
                )
            measure = vec.measure

        if name is None:
            name = f"{measure.name}_{vec.name}"

        vec_data = to_df(vec.atom_data(), suffix="_vec")

        if isinstance(measure, ParametrizedMeasure):
            measure_data_unstacked = measure.data.unstack(level=measure.parameter_names)

            data = (
                pd.concat(
                    [vec_data, measure_data_unstacked],
                    axis=1,
                )
                .groupby(list(vec_data.columns))
                .sum()
            )
            data.index.names = vec.generated_sig_alg.variable_names
            data.columns = measure_data_unstacked.columns

            data = (
                data.stack(level=measure.parameter_names)
                .reorder_levels(
                    measure.parameter_names + vec.generated_sig_alg.variable_names
                )
                .sort_index()
                .rename(name)
            )

            sig_alg = SigmaAlgebra.power_set(vec.range)

            return ParametrizedMeasure._from_validated(
                data=data,
                sig_alg=sig_alg,
                kind="param_probability"
                if isinstance(measure, ParametrizedProbabilityMeasure)
                else "param_measure",
                complete_domain_name=f"{measure.parameter_domain.name} x {vec.name}_range",
                parameter_domain_name=measure.parameter_domain.name,
                parameter_names=measure.parameter_names,
                name=name,
            )

        else:
            data = (
                pd.concat([vec_data, measure.data], axis=1)
                .groupby(list(vec_data.columns))[measure.name]
                .sum()
                .rename(name)
            )

            data.index.names = vec.generated_sig_alg.variable_names

            sig_alg = SigmaAlgebra.power_set(vec.range)

            return Measure._from_validated(
                data=data,
                kind=measure.kind,
                sig_alg=sig_alg,
                name=name,
            )

    # --------------------- probability-related methods --------------------- #

    # TODO: add to mixin
    # TODO: missing docstring
    @classmethod
    def density(
        cls,
        rv: RandomVariable,
        given: SigmaAlgebra | RandomVector | None = None,
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> MeasurableFunction | ParametrizedMeasurableFunction:
        """Pass."""
        from .random_vector import RandomVector

        if given is None:
            pushforward = cls.pushforward(vec=rv)

        else:
            given_name = given.name
            if isinstance(given, RandomVector):
                given = given.generated_sig_alg
            pushforward = cls.pushforward(vec=rv, measure=rv.measure.conditional(given))

        if name is None:
            if given is None:
                name = f"f_{rv.name}"
            else:
                name = f"f_({rv.name}|{given_name})"

        return pushforward.derivative(name=name, tol=tol)

    @classmethod
    def expectation(
        cls,
        rv: RandomVector,
        given: SigmaAlgebra | RandomVector | None = None,
        measure: ProbabilityMeasure | None = None,
        name: Hashable | None = None,
    ) -> RandomVector:
        r"""Compute the expectation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the expectation.
        given : SigmaAlgebra | RandomVector | None, default=None
            The sigma-algebra or random vector to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.
        name : Hashable | None, default=None
            The name to assign to the resulting expected value random vector. If `None`, a default name is generated.

        Returns
        -------
        exp : RandomVector
            The expected value of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Function,
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: (1, 3),
        ...         1: (2, 4),
        ...         2: (3, 1),
        ...         3: (1, 0),
        ...         4: (5, 0),
        ...         5: (0, 8),
        ...     },
        ...     name="Y",
        ... )

        Give aliases to the `integrate` and `expectation` methods, and get the constant random variable whose unique value is `1`.

        >>> E = Operators.expectation
        >>> integrate = Operators.integrate
        >>> one = Function.from_constant(domain=Omega, constant=1)

        Check that the unconditional expectation of the random variable `X` is equal to the constant random variable whose unique value is the Lebesgue integral of the random variable.

        >>> E(X) == integrate(X) * one
        True

        Compute the unconditional expectation of the random vector `Y`, and check that its components are the unconditional expectations of the components of `Y`.

        >>> all(E_Y_i == integrate(Y_i) * one for E_Y_i, Y_i in zip(E(Y), Y))
        True

        Define a sigma-algebra for conditional expectations.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional expectation of the random variable `X` is equal to its Fourier expansion.

        >>> E(X, G) == sum(integrate(X, B) / P(B) * B.indicator for B in G if P(B) != 0)
        True

        Check the same for the components of the conditional expectation of the random vector `Y`.

        >>> all(
        ...     E_Y_i_G == sum(integrate(Y_i, B) / P(B) * B.indicator for B in G if P(B) != 0)
        ...     for E_Y_i_G, Y_i in zip(E(Y, G), Y)
        ... )
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional expectation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $E(X\mid \mathcal{G})$ for which

        $$
        \int_V E(X\mid \mathcal{G}) \, dP = \int_V X \, dP,
        $$

        for all $V\in \mathcal{G}$. All such random variables are equal almost surely.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional expectation called a *Fourier expansion*:

        $$
        E(X\mid \mathcal{G}) = \sum_B \frac{\int_B X \, dP}{P(B)} I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability and $I_B$ is the indicator function of $B$.

        The *unconditional expectation* of $X$, denoted $E(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. In this case $E(X)$ is the constant random variable with

        $$
        E(X)(\omega) = \int_\Omega X \, dP,
        $$

        for all $\omega\in \Omega$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional expectation* to be the $d$-dimensional vector whose entries are the separate conditional expectations $E(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        # from .._utils.function_helpers import sig_alg_func_to_measurable_func
        # from .._utils.utils import add_subscript, to_df
        from .._utils.function_helpers import compute_expectation
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_vector import RandomVector

        if isinstance(given, RandomVector):
            given = given.generated_sig_alg

        cls._validate_univariate_parameters(rv=rv, sig_alg=given, measure=measure)

        if measure is None:
            measure = rv.prob_measure

        if given is None:
            name = f"E({rv.name})"
            exp = rv.atom_data().multiply(measure.data, axis=0).sum()

            if isinstance(exp, pd.Series):
                data = pd.DataFrame(
                    [exp] * len(rv.domain), index=rv.domain.data, columns=exp.index
                )
            else:
                data = pd.Series(exp, index=rv.domain.data, name=name)

            sig_alg = SigmaAlgebra._from_validated(
                data=pd.Series(0, name=name, index=rv.domain.data),
                variable_names=["T"],
                domain_kind=type(rv.domain).__name__,
                domain_name=rv.domain.name,
                index_kind="Index",
                index_name=None,
                name="T",
            )

            measure = ProbabilityMeasure._from_validated(
                data=pd.Series(
                    1.0,
                    index=pd.Index([0], name="T"),
                    name=f"{measure.name}|T",
                ),
                kind="probability",
                sig_alg=sig_alg,
                name=f"{measure.name}|T",
            )

            return RandomVector._from_validated(
                data=data,
                sig_alg=sig_alg,
                measure=measure,
                index_kind=type(rv.index).__name__ if rv.index is not None else "Index",
                index_name=type(rv.index) if rv.index is not None else None,
                name=name,
            )

        data = compute_expectation(
            rv_atom_data=rv.atom_data(),
            given_data=given.data,
            given_variable_names=given.variable_names,
            atom_data=given.up_lattice.get_atom_data(rv.sig_alg),
            measure_data=measure.data,
            measure_data_on_given=(measure | given).data,
        )

        if name is None:
            if given.name.startswith("sigma(") and given.name.endswith(")"):
                name = f"E({rv.name}|{given.name[6:-1]})"
            else:
                name = f"E({rv.name}|{given.name})"

        if isinstance(data, pd.Series):
            data.name = name
        else:
            data.columns = rv.index.data

        return RandomVector._from_validated(
            data=data,
            sig_alg=given,
            measure=measure | given,
            index_kind=type(rv.index) if rv.index is not None else "Index",
            index_name=rv.index if rv.index is not None else None,
            name=name,
        )

    # TODO: slow reference implementation
    @classmethod
    def variance(
        cls,
        rv: RandomVector,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the variance of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        **User note**: This is a naive reference implementation. It is slow.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the variance.
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        var : RandomVector
            The variance of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: (1, 3),
        ...         1: (2, 4),
        ...         2: (3, 1),
        ...         3: (1, 0),
        ...         4: (5, 0),
        ...         5: (0, 8),
        ...     },
        ...     name="Y",
        ... )

        Give aliases to the `variance` and `expectation` methods.

        >>> V = Operators.variance
        >>> E = Operators.expectation

        Check that the variance may be computed via the "short-cut" formula.

        >>> V(X) == E(X**2) - E(X) ** 2
        True

        Check the same for `Y`.

        >>> V(Y) == E(Y**2) - E(Y) ** 2
        True

        Define a sigma-algebra for conditional variances.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional variance of the random variable `X` is equal to a linear combination of indicators weighted by unconditional variances.

        >>> V(X, G) == sum(V(X | B).item() * B.indicator for B in G if P(B) > 0)
        True

        Check the same for the random vector `Y`.

        >>> all(
        ...     V_Y_i_G == sum(V(Y_i | B).item() * B.indicator for B in G if P(B) > 0)
        ...     for V_Y_i_G, Y_i in zip(V(Y, G), Y)
        ... )
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F}, P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional variance* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        V(X\mid \mathcal{G}) = E\left[ (X-E(X\mid \mathcal{G}))^2 \mid \mathcal{G}\right].
        $$

        The *unconditional variance* of $X$, denoted $V(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional variance is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional variance:

        $$
        V(X\mid \mathcal{G}) = \sum_B V(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $V(X|_B)$ is the unconditional variance of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional variance* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional variances $V(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        cls._validate_univariate_parameters(rv=rv, sig_alg=given, measure=measure)

        result = (
            cls.expectation(
                rv=rv**2,
                given=given,
                measure=measure,
            )
            - cls.expectation(rv=rv, given=given, measure=measure) ** 2
        )

        name = f"V({rv.name}|{given.name})" if given is not None else f"V({rv.name})"

        if isinstance(result.data, pd.Series):
            result.data.name = name

        result.name = name

        return result

    # TODO: slow reference implementation
    @classmethod
    def std(
        cls,
        rv: RandomVector,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the standard deviation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        **User note**: This is a naive reference implementation. It is slow.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the standard deviation.
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        std : RandomVector
            The standard deviation of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: (1, 3),
        ...         1: (2, 4),
        ...         2: (3, 1),
        ...         3: (1, 0),
        ...         4: (5, 0),
        ...         5: (0, 8),
        ...     },
        ...     name="Y",
        ... )

        Give aliases to the `variance` and `std` methods.

        >>> V = Operators.variance
        >>> std = Operators.std

        Check that the unconditional standard deviation is equal to its definition.

        >>> std(X) == V(X) ** 0.5
        True

        Check the same for `Y`.

        >>> std(Y) == V(Y) ** 0.5
        True

        Define a sigma-algebra for conditional standard deviations.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional standard deviation of the random variable `X` is equal to a linear combination of indicators weighted by unconditional standard deviations.

        >>> std(X, G) == sum(std(X | B).item() * B.indicator for B in G if P(B) > 0)
        True

        Check the same for the random vector `Y`.

        >>> all(
        ...     std_Y_i_G == sum(std(Y_i | B).item() * B.indicator for B in G if P(B) != 0)
        ...     for std_Y_i_G, Y_i in zip(std(Y, G), Y)
        ... )
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional standard deviation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $\sigma(X \mid \mathcal{G})$ that is equal almost surely to the random variable

        $$
        \sigma(X\mid \mathcal{G}) = \sqrt{V(X\mid \mathcal{G})}.
        $$

        The *unconditional standard deviation* of $X$, denoted $\sigma(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional standard deviation is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional standard deviation:

        $$
        \sigma(X\mid \mathcal{G}) = \sum_B \sigma(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B)$ is the unconditional standard deviation of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional standard deviation* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional standard deviations $\sigma(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        cls._validate_univariate_parameters(rv=rv, sig_alg=given, measure=measure)

        result = cls.variance(rv, given, measure) ** 0.5
        result.data = result.data.fillna(0.0)

        name = (
            (f"std({rv.name}|{given.name})") if given is not None else f"std({rv.name})"
        )

        if isinstance(result.data, pd.Series):
            result.data.name = name

        result.name = name

        return result

    # TODO: slow reference implementation
    @classmethod
    def cov(
        cls,
        rv1: RandomVariable,
        rv2: RandomVariable,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> RandomVariable:
        r"""Compute the covariance of two random variables, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        **User note**: This is a naive reference implementation. It is slow.

        Parameters
        ----------
        rv1 : RandomVariable
            The first random variable for which to compute the covariance.
        rv2 : RandomVariable
            The second random variable for which to compute the covariance
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability used to compute the covariance. If `None`, the common probability measure carried by the random variables is used (accessed through their `prob_measure` attribute).

        Returns
        -------
        cov : RandomVariable
            The covariance of the random variables.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with two random variables.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 3,
        ...         1: 4,
        ...         2: 1,
        ...         3: 0,
        ...         4: 0,
        ...         5: 8,
        ...     },
        ...     name="Y",
        ... )

        Give aliases to the `expectation` and `cov` methods.

        >>> E = Operators.expectation
        >>> cov = Operators.cov

        Check that the unconditional covariance is equal to its definition.

        >>> cov(X, Y) == E(X * Y) - E(X) * E(Y)
        True

        Define a sigma-algebra for conditional covariance.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional covariance of the random variables is equal to a linear combination of indicators weighted by unconditional covariances.

        >>> np.allclose(cov(X, Y, G), sum(cov(X | B, Y | B).item() * B.indicator for B in G if P(B) > 0))
        True

        Notes
        -----
        Let $X,Y:\Omega \to \mathbb{R}$ be two random variables on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional covariance* of $X$ and $Y$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        \sigma(X,Y\mid \mathcal{G}) = E(XY \mid \mathcal{G}) - E(X\mid \mathcal{G})E(Y\mid \mathcal{G}).
        $$

        The *unconditional covariance* of $X$ and $Y$, denoted $\sigma(X, Y)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional covariance is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional covariance:

        $$
        \sigma(X,Y\mid \mathcal{G}) = \sum_B \sigma(X|_B, Y|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B, Y|_B)$ is the unconditional covariance of the restricted random variables $X|_B, Y|_B:B\to \mathbb{R}$.
        """
        cls._validate_bivariate_parameters(
            rv1=rv1, rv2=rv2, sig_alg=given, measure=measure
        )

        result = cls.expectation(rv1 * rv2, given, measure) - cls.expectation(
            rv1, given, measure
        ) * cls.expectation(rv2, given, measure)

        name = (
            f"cov({rv1.name}, {rv2.name}|{given.name})"
            if given is not None
            else f"cov({rv1.name}, {rv2.name})"
        )

        if isinstance(result.data, pd.Series):
            result.data.name = name

        result.name = name

        return result

    # TODO: slow reference implementation
    @classmethod
    def corr(
        cls,
        rv1: RandomVariable,
        rv2: RandomVariable,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> RandomVariable:
        r"""Compute the correlation of two random variables, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        **User note**: This is a naive reference implementation. It is slow.

        Parameters
        ----------
        rv1 : RandomVariable
            The first random variable for which to compute the correlation.
        rv2 : RandomVariable
            The second random variable for which to compute the correlation
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability used to compute the correlation. If `None`, the common probability measure carried by the random variables is used (accessed through their `prob_measure` attribute).

        Returns
        -------
        corr : RandomVariable
            The correlation of the two random variables.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with two random variables.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 3,
        ...         1: 4,
        ...         2: 1,
        ...         3: 0,
        ...         4: 0,
        ...         5: 8,
        ...     },
        ...     name="Y",
        ... )

        Give aliases to the `std`, `cov`, and `corr` methods.

        >>> std = Operators.std
        >>> cov = Operators.cov
        >>> corr = Operators.corr

        Check that the unconditional correlation is equal to its definition.

        >>> corr(X, Y) == cov(X, Y) / (std(X) * std(Y))
        True

        Define a sigma-algebra for conditional correlation.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional correlation of the random variables is equal to a linear combination of indicators weighted by unconditional correlations.

        >>> np.allclose(corr(X, Y, G), sum(corr(X | B, Y | B).item() * B.indicator for B in G if P(B) > 0))
        True

        Notes
        -----
        Let $X,Y:\Omega \to \mathbb{R}$ be two random variables on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional correlation* of $X$ and $Y$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        \rho(X,Y\mid \mathcal{G}) = \frac{\sigma(X, Y \mid \mathcal{G})}{\sigma(X \mid \mathcal{G}) \sigma(Y \mid \mathcal{G})}.
        $$

        The *unconditional correlation* of $X$ and $Y$, denoted $\rho(X, Y)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional correlation is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional correlation:

        $$
        \rho(X,Y\mid \mathcal{G}) = \sum_B \rho(X|_B, Y|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\rho(X|_B, Y|_B)$ is the unconditional correlation of the restricted random variables $X|_B, Y|_B:B\to \mathbb{R}$.
        """
        cls._validate_bivariate_parameters(
            rv1=rv1, rv2=rv2, sig_alg=given, measure=measure
        )

        result = cls.cov(rv1, rv2, given, measure) / (
            cls.std(rv1, given, measure) * cls.std(rv2, given, measure)
        )
        result.data = result.data.fillna(0.0)

        name = (
            f"corr({rv1.name}, {rv2.name}|{given.name})"
            if given is not None
            else f"corr({rv1.name}, {rv2.name})"
        )

        if isinstance(result.data, pd.Series):
            result.data.name = name

        result.name = name

        return result

    @staticmethod
    def _validate_univariate_parameters(
        rv: RandomVector,
        sig_alg: SigmaAlgebra | None,
        measure: ProbabilityMeasure | None,
    ):
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")

        if measure is None and (
            rv.measure is None or not isinstance(rv.measure, ProbabilityMeasure)
        ):
            raise ValueError(
                "If measure is not given, then the random vector must carry a probability measure."
            )

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("If given, sig_alg must be a SigmaAlgebra instance.")

        if measure is not None and not isinstance(measure, ProbabilityMeasure):
            raise TypeError("If given, measure must be a ProbabilityMeasure instance.")

        if sig_alg is not None and not sig_alg <= rv.sig_alg:
            raise ValueError(
                "If given, sig_alg must be a sub-sigma-algebra of the random vector's sigma-algebra."
            )

        if measure is not None and measure.sig_alg != rv.sig_alg:
            raise ValueError(
                "If given, measure must be defined on the sigma-algebra of the random vector."
            )

    @staticmethod
    def _validate_bivariate_parameters(
        rv1: RandomVariable,
        rv2: RandomVariable,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> None:
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable

        if not isinstance(rv1, RandomVariable) or not isinstance(rv2, RandomVariable):
            raise TypeError("rv1 and rv2 must be RandomVariable instances.")
        if rv1.measurable_space != rv2.measurable_space:
            raise ValueError(
                "rv1 and rv2 must be defined on the same measurable space."
            )
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra) or not sig_alg <= rv1.sig_alg
        ):
            raise TypeError(
                "sig_alg must be a SigmaAlgebra or None, and it must be a sub-sigma-algebra of the sigma-algebra of the random variables."
            )
        if measure is None:
            if (
                rv1.measure != rv2.measure
                or rv1.measure is None
                or not isinstance(rv1.measure, ProbabilityMeasure)
            ):
                raise ValueError(
                    "If measure is not passed, the random variables must have the same probability measures."
                )
            else:
                measure = rv1.measure
        else:
            if not isinstance(measure, ProbabilityMeasure):
                raise TypeError("measure must be a ProbabilityMeasure or None.")
            if measure.sig_alg != rv1.sig_alg:
                raise ValueError(
                    "If measure is passed, it must be defined on the sigma-algebra of the random variables."
                )

    # --------------------- information-theoretic methods --------------------- #

    # TODO: missing docstring
    # TODO: add to mixin
    @classmethod
    def surprisal(
        cls,
        rv: RandomVariable,
        given: SigmaAlgebra | RandomVector | None = None,
        base: Literal["e", "2", "10"] = "e",
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> MeasurableFunction | ParametrizedMeasurableFunction:
        """Pass."""
        from .random_vector import RandomVector

        if given is None:
            pushforward = cls.pushforward(vec=rv)

        else:
            given_name = given.name
            if isinstance(given, RandomVector):
                given = given.generated_sig_alg
            pushforward = cls.pushforward(vec=rv, measure=rv.measure.conditional(given))

        if name is None:
            if given is None:
                name = f"s({rv.name})"
            else:
                name = f"s({rv.name}|{given_name})"

        return pushforward.surprisal(
            base=base,
            name=name,
            tol=tol,
        )

    # TODO: Notes section missing
    # TODO: tol unused parameter
    # TODO: add to mixin
    @classmethod
    def entropy(
        cls,
        rv: RandomVariable,
        given: SigmaAlgebra | RandomVector | None = None,
        base: Literal["e", "2", "10"] = "e",
        tol: float = 1e-8,
    ) -> Real:
        """Compute the entropy of a random variable with respect to a base measure, optionally conditioned on a sigma-algebra or random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVariable
            The random variable whose entropy is to be computed.
        given : SigmaAlgebra | RandomVector | None, default=None
            The optional sigma-algebra or random vector on which to condition the entropy.
        base : Literal["e", "2", "10"], default="e"
            The base of the logarithm used to compute the entropy.
        tol : float, default=1e-8
            Tolerance for testing for absolute continuity.

        Returns
        -------
        entropy : Real
            The entropy of the random variable.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a random variable on a probability space.

        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 3,
        ...         4: 3,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.5,
        ...         1: 0.4,
        ...         2: 0.1,
        ...         3: 0.0,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 5,
        ...         2: 4,
        ...         3: -1,
        ...         4: -1,
        ...     },
        ... )

        Compute the entropy of the random variable.

        >>> H = Operators.entropy
        >>> H(X)
        0.9433483923290392

        We may check that the entropy is the integral of the surprisal of the pushforward measure.

        >>> P_X = P >> X
        >>> H(X) == (P_X).surprisal().integrate(measure=P_X)
        True

        Define a sub-sigma-algebra for conditional entropy.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )

        Compute the conditional entropy.

        >>> H(X, G)
        0.6182654189375909

        Check that the conditional entropy agrees with its mathematical definition as a double integral.

        >>> P_X_G = P.conditional(G) >> X
        >>> H(X, G) == P_X_G.surprisal().integrate(measure=P_X_G).ascend(G).integrate(measure=P)
        True
        """
        from .._utils.function_helpers import compute_integral
        from .._utils.measure_helpers import compute_entropy
        from ..measures.measure import Measure
        from .random_vector import RandomVector

        if given is None:
            pushforward = cls.pushforward(vec=rv)

        else:
            if isinstance(given, RandomVector):
                given = given.generated_sig_alg
            pushforward = cls.pushforward(vec=rv, measure=rv.measure.conditional(given))

        base_measure = Measure.counting(pushforward.domain)

        data = compute_entropy(
            self_data=pushforward.data,
            base_measure_data=base_measure.data,
            sig_alg_data=pushforward.sig_alg.data,
            parameter_names=getattr(given, "variable_names", None),
            base=base,
        )

        if given is None:
            return data.astype(Real)

        else:
            return compute_integral(
                function_atom_data=data,
                measure_data=(rv.measure | given).data,
            ).astype(Real)

    # TODO: Notes section missing
    # TODO: tol unused parameter
    # TODO: add to mixin
    @classmethod
    def cross_entropy(
        cls,
        rv1: RandomVariable,
        rv2: RandomVariable,
        given: SigmaAlgebra | RandomVector | None = None,
        base: Literal["e", "2", "10"] = "e",
        tol: float = 1e-8,
    ) -> Real:
        """Compute the cross entropy from an initial random variable to a second one, optionally conditioned on a sigma-algebra or random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv1 : RandomVariable
            The intial random variable of the cross entropy. See the `Notes` section for an explanation of the "initial" terminology.
        rv2 : RandomVariable
            The terminal random variable of the cross entropy. See the `Notes` section for an explanation of the "terminal" terminology.
        given : SigmaAlgebra | RandomVector | None, default=None
            The optional sigma-algebra or random vector on which to condition the cross entropy.
        base : Literal["e", "2", "10"], default="e"
            The base of the logarithm used to compute the cross entropy.
        tol : float, default=1e-8
            Tolerance for testing for absolute continuity.

        Returns
        -------
        cross_entropy : Real
            The cross entropy from the intitial random variable to the terminal one.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a pair of random variables on a probability space.

        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 3,
        ...         4: 3,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.5,
        ...         1: 0.4,
        ...         2: 0.1,
        ...         3: 0.0,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 5,
        ...         2: 4,
        ...         3: -1,
        ...         4: -1,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping={
        ...         0: 2,
        ...         1: 1,
        ...         2: 5,
        ...         3: 5,
        ...         4: 5,
        ...     },
        ...     name="Y",
        ... )

        Compute the cross entropy from `X` to `Y`.

        >>> H = Operators.cross_entropy
        >>> H(X, Y)
        1.3791794031346958
        """
        from .._utils.function_helpers import compute_integral
        from .._utils.measure_helpers import compute_surprisal
        from ..measures.measure import Measure
        from .random_vector import RandomVector

        if given is None:
            pushforward1 = cls.pushforward(vec=rv1)
            pushforward2 = cls.pushforward(vec=rv2)

        else:
            if isinstance(given, RandomVector):
                given = given.generated_sig_alg
            pushforward1 = cls.pushforward(
                vec=rv1, measure=rv1.measure.conditional(given)
            )
            pushforward2 = cls.pushforward(
                vec=rv2, measure=rv2.measure.conditional(given)
            )

        base_measure = Measure.counting(pushforward2.domain)

        surprisal_data = compute_surprisal(
            self_data=pushforward2.data,
            base_measure_data=base_measure.data,
            sig_alg_data=pushforward2.sig_alg.data,
            parameter_names=getattr(given, "variable_names", None),
            base=base,
        ).rename("surprisal")

        surprisal_pushforward_merged = pd.merge(
            left=surprisal_data.rename_axis(index={rv2.name: "S"}).reset_index(),
            right=pushforward1.data.rename("pushforward")
            .rename_axis(index={rv1.name: "S"})
            .reset_index(),
            how="outer",
        ).fillna(0.0)

        surprisal_pushforward_merged["product"] = (
            surprisal_pushforward_merged["surprisal"]
            * surprisal_pushforward_merged["pushforward"]
        )

        if given is None:
            return surprisal_pushforward_merged["product"].sum().astype(Real)

        else:
            inner_integral_data = surprisal_pushforward_merged.groupby(
                given.variable_names
            )["product"].sum()

            return compute_integral(
                function_atom_data=inner_integral_data,
                measure_data=(rv1.measure | given).data,
            ).astype(Real)


class OperatorsMethods:
    """Mixin class to add operators to `MeasurableVector`."""

    # --------------------- general methods --------------------- #

    def sum(self, name: Hashable | None = None) -> MeasurableFunction:
        """Compute the sum of the components of the measurable vector.

        Internally calls `Operators.sum`.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the resulting measurable function. If `None`, a default name will be generated.

        Returns
        -------
        summed_vec : MeasurableFunction
            The measurable function representing the sum of the components of the measurable vector.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector
        >>> D = Domain.from_sequence(size=2, variable_name="flip", name="D")
        >>> X = (D ^ 3).with_name("X")
        >>> f = MeasurableVector.from_identity(domain=X)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i                     0  1  2
        flip_0 flip_1 flip_2
        0      0      0       0  0  0
                      1       0  0  1
               1      0       0  1  0
                      1       0  1  1
        1      0      0       1  0  0
                      1       1  0  1
               1      0       1  1  0
                      1       1  1  1
        >>> g = f.sum(name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                              g
        flip_0 flip_1 flip_2
        0      0      0       0
                      1       1
               1      0       1
                      1       2
        1      0      0       1
                      1       2
               1      0       2
                      1       3
        """
        return Operators.sum(vec=self, name=name)

    def transform(
        self,
        functions: list[Callable[[MeasurableVector], MeasurableFunction]],
        index: IndexLike | None = None,
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Apply a transformation to the measurable vector.

        Internally calls `Operators.transform`.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        functions : list[Callable[[MeasurableVector], MeasurableFunction]]
            A list of functions to apply to the measurable vector.
        index : IndexLike | None, default=None
            The new index for the transformed vector. If `None`, the original index of the measurable vector is used.
        name : Hashable | None, default=None
            The name of the transformed vector. If `None`, a default name will be generated.

        Returns
        -------
        transformed_vector : MeasurableVector
            The transformed measurable vector.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import IIDProcess, StochasticProcess
        >>> T = Time.discrete(start=0, length=2)
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.5),
        ...     support=[0, 1],
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        t       0  1  2
        omega
        0       0  0  0
        1       0  0  1
        2       0  1  0
        3       0  1  1
        4       1  0  0
        5       1  0  1
        6       1  1  0
        7       1  1  1
        >>> S = Time.discrete(start=4, stop=5)
        >>> def f4(process: StochasticProcess) -> RandomVariable:
        ...     X0, X1, _ = X
        ...     return X0 + X1
        >>> def f5(process: StochasticProcess) -> RandomVariable:
        ...     _, X1, X2 = X
        ...     return X1 + X2
        >>> X_transform = X.transform(functions=[f4, f5], index=S)
        >>> print(X_transform)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_transform':
        t       4  5
        omega
        0       0  0
        1       0  1
        2       1  1
        3       1  2
        4       1  0
        5       1  1
        6       2  1
        7       2  2
        """
        return Operators.transform(
            vec=self, functions=functions, index=index, name=name
        )

    def pointwise_map(
        self,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> MeasurableVector:
        """Apply a function pointwise to the values of the measurable vector.

        Internally calls `Operators.pointwise_map`.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to each value in the measurable vector.
        name : Hashable | None, default=None
            The name of the transformed measurable vector. If `None`, a default name will be generated.

        Returns
        -------
        mapped_vector : MeasurableVector
            A new measurable vector with the function applied pointwise to its values.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t       0  1  2
        omega
        0       3  2  1
        1       3  2  3
        2       3  4  3
        3       3  4  5
        >>> def f(x):
        ...     return x + 1
        >>> X_mapped = X.pointwise_map(function=f)
        >>> print(X_mapped)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_mapped':
        t       0  1  2
        omega
        0       4  3  2
        1       4  3  4
        2       4  5  4
        3       4  5  6
        """
        return Operators.pointwise_map(vec=self, function=function, name=name)

    def cumsum(self, name: Hashable | None = None) -> MeasurableVector:
        """Compute the cumulative sum of the measurable vector along its index.

        Internally calls `Operators.cumsum`.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed measurable vector. If `None`, a default name will be generated.

        Returns
        -------
        cumsum_vector : MeasurableVector
            A new measurable vector representing the cumulative sum of the input vector.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        t           1  2  3
        omega
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X_cumsum = X.cumsum()
        >>> print(X_cumsum) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        t           1  2  3
        omega
        0           0  0  0
        1           0  0  1
        2           0  1  1
        3           0  1  2
        4           1  1  1
        5           1  1  2
        6           1  2  2
        7           1  2  3
        """
        return Operators.cumsum(vec=self, name=name)

    def cumprod(self, name: Hashable | None = None) -> MeasurableVector:
        """Compute the cumulative product of the measurable vector along its index.

        Internally calls `Operators.cumprod`.

        See the Notes section below for the mathematical details.


        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed vector. If `None`, a default name will be generated.

        Returns
        -------
        cumprod_vector : MeasurableVector
            A new measurable vector representing the cumulative product of the input vector.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=3)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=3, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t           0  1  2  3
        omega
        0           3  2  1  0
        1           3  2  1  2
        2           3  2  3  2
        3           3  2  3  4
        4           3  4  3  2
        5           3  4  3  4
        6           3  4  5  4
        7           3  4  5  6
        >>> X_cumprod = X.cumprod()
        >>> print(X_cumprod) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumprod':
        t           0   1   2    3
        omega
        0           3   6   6    0
        1           3   6   6   12
        2           3   6  18   36
        3           3   6  18   72
        4           3  12  36   72
        5           3  12  36  144
        6           3  12  60  240
        7           3  12  60  360
        """
        return Operators.cumprod(vec=self, name=name)

    def max_value(self) -> Real:
        """Get the maximum value across all outputs and indices of the measurable vector.

        Internally calls `Operators.max_value`.

        Returns
        -------
        max_value : Real
            The maximum value found in the measurable vector.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t           0  1  2
        omega
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> max_value = X.max_value()
        >>> print(max_value)
        5
        """
        return Operators.max_value(vec=self)

    def min_value(self) -> Real:
        """Get the minimum value across all outputs and indices of the measurable vector.

        Internally calls `Operators.min_value`.

        Returns
        -------
        min_value : Real
            The minimum value found in the measurable vector.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        t           0  1  2
        omega
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> min_value = X.min_value()
        >>> print(min_value)
        1
        """
        return Operators.min_value(vec=self)

    # --------------------- measure-related methods --------------------- #

    def integrate(
        self,
        subset: Set | list[Hashable] | None = None,
        measure: Measure | ParametrizedMeasure | None = None,
        variables: tuple[Hashable, Hashable] | None = None,
        subset_name: Hashable | None = None,
    ) -> Real | pd.Series | Function:
        r"""Compute the Lebesgue integral of a measurable vector with respect to a measure over an (optional) set.

        See the Notes section below for the mathematical details.

        Calls `Operators.integrate` with appropriate arguments.

        Parameters
        ----------
        subset: MeasurableSet | None, default=None
            The optional set over which to integrate. If `None`, the integral will be taken over the entire domain of the measurable vector.
        measure : Measure | ParametrizedMeasure | None, default=None
            The measure or parametrized measure with respect to which to integrate. If `None`, the measure of the underlying measure space is used (if it exists) carried by the measurable vector.
        variables : tuple[Hashable, Hashable] | None, default=None
            A pair of explicit domain variables over which the integral should be taken. The first item of the tuple should correspond to the variable name of the function's domain, while the second item should correspond to the variable name of the measure's domain.
        subset_name : Hashable | None, default=None
            If the `subset` is passed as a list, the name that will be

        Returns
        -------
        integral : Real | pd.Series | Function
            Returns the following:

            * If `function` is a `MeasurableFunction` and `measure` is a `Measure`, returns a `Real` representing the integral of the function with respect to the measure over the specified set.

            * If `function` is a `MeasurableVector` of dimension > 1 and `measure` is a `Measure`, returns a `pd.Series` representing the integral of each component of the vector with respect to the measure over the specified set.

            * If `function` is a `MeasurableFunction` and `measure` is a `ParametrizedMeasure`, returns a `Function` representing the integral of the function with respect to the measure over the specified set for each parameter value.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableFunction,
        ...     Measure,
        ...     MeasureSpace,
        ...     ParametrizedMeasurableFunction,
        ...     ParametrizedMeasure,
        ...     Set,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)

        Define a measure space and a measurable function.

        >>> measure_space = MeasureSpace.from_rand(
        ...     domain_size=100,
        ...     num_atoms=27,
        ...     num_null_atoms=12,
        ...     random_state=rng,
        ... )
        >>> X, F, mu = measure_space
        >>> f = MeasurableFunction.from_rand(
        ...     *measure_space,
        ...     distribution="normal",
        ...     diff_values=24,
        ...     random_state=rng,
        ... )

        Get a measurable set from the sigma-algebra, compute the integral over this set, and check that it agrees with the defining formula for the Lebesgue integral.

        >>> U = F.get_random_set(num_atoms=4, name="U", random_state=rng)
        >>> I_U = U.indicator
        >>> np.allclose(f.integrate(U), sum(I_U(A) * f(A) * mu(A) for A in F))
        True

        Check that the integral over a null set is 0.

        >>> N = measure_space.get_random_set(
        ...     num_atoms=3,
        ...     is_null=True,
        ...     name="N",
        ...     random_state=rng,
        ... )
        >>> I_N = N.indicator
        >>> f.integrate(N)
        0.0

        Define a new measure space and measurable function to demonstrate integration against parametrized objects.

        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 1),
        ...         2: (1, 1),
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         (0, 1): 2,
        ...         (1, 1): 3,
        ...     },
        ... )
        >>> f = MeasurableFunction(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ... )

        Define a parametrized measure and parametrized measurable function over the same parameter domain.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> nu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping={
        ...         (0, 0, 1): 3,  # (theta, F_0, F_1) = (0, 0, 0), etc ...
        ...         (0, 1, 1): 4,
        ...         (1, 0, 1): 1,
        ...         (1, 1, 1): 2,
        ...     },
        ...     name="nu",
        ... )
        >>> g = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping={
        ...         (0, 0): 2,  # (theta, X) = (0, 0)
        ...         (0, 1): 4,
        ...         (0, 2): 4,
        ...         (1, 0): 1,
        ...         (1, 1): -1,
        ...         (1, 2): -1,
        ...     },
        ...     name="g",
        ... )

        Extract a measurable set from the sigma-algebra.

        >>> U = Set([1, 2], domain=X, name="U")

        It is convenient to conceptualize a parametrized measure as a family of measures. Then integration of a measurable function against a parametrized measure returns a function of the parameters whose values are the integrals of the function against the measures. Iteration over the parametrized measure yields the measures, allowing us to check that these integrals all match.

        >>> all(f.integrate(U, nu)(**param) == f.integrate(U, measure) for param, measure in nu)
        True

        Likewise, it is convenient to conceptualize a parametrized measurable function as a family of measurable functions. Then integration of a parametrized measurable function against a measure returns a function of the parameters whose values are the integrals of the functions against the measure. Iteration over the parametrized measurable function yields the functions, allowing us to check that these integrals match.

        >>> all(g.integrate(U, mu)(**param) == function.integrate(U, mu) for param, function in g)
        True

        Finally, it is possible to integrate a parametrized measurable function against a parametried measure as long as their parameter domains agree. We leave the reader to guess the meaning of the following verification.

        >>> all(
        ...     g.integrate(U, nu)(**param) == function.integrate(U, measure)
        ...     for (param, function), (_, measure) in zip(g, nu)
        ... )
        True

        Notes
        -----
        Let $f: X \to \mathbb{R}$ be a measurable function on a measure space $(X, \mathcal{F}, \mu)$. Assuming $X$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{F}$ is determined by its set $\alpha(\mathcal{F})$ of atoms. Let $U$ be a measurable set in $\mathcal{F}$, and write $I_U$ for its indicator function. Since both $f$ and $I_U$ are $\mathcal{F}$-measurable, they take constant values on each atom $A\in \alpha(\mathcal{F})$ that we write as $f(A)$ and $I_U(A)$, respectively. Then the *Lebesgue integral* of $f$ over $U$ is the number

        $$
        \int_U f \, d\mu = \sum_{A\in \alpha(\mathcal{F})} I_U(A)f(A) \mu(A).
        $$

        If $f:X \to \mathbb{R}^d$ is instead a measurable vector of dimension $d>1$, with components

        $$
        f = (f_1, f_2, \ldots, f_d),
        $$

        then we define the *Lebesgue integral* of $f$ over $U$ to be the $d$-dimensional vector whose entries are the separate Lebesgue integrals $\int_U f_j \, d\mu$, for $j=1,2,\ldots,d$.
        """
        return Operators.integrate(
            function=self,
            subset=subset,
            measure=measure,
            variables=variables,
            subset_name=subset_name,
        )

    def pushforward(
        self,
        measure: Measure | ParametrizedMeasure | None = None,
        name: Hashable | None = None,
    ) -> Measure | ParametrizedMeasure:
        r"""Push forward a (parametrized) measure on the domain of a measurable vector to a measure on its range.

        See the Notes section below for the mathematical details.

        Calls `Operators.pushforward` with appropriate arguments.

        Parameters
        ----------
        measure : Measure | ParametrizedMeasure | None, default=None
            Measure to push forward. If `None`, the measure carried by the measurable vector is used.
        name : Hashable | None, default=None
            The name of the pushforward measure. If `None`, a default name is generated.

        Returns
        -------
        pushforward : Measure | ParametrizedMeasure
            The measure pushed forward along the measurable vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableVector,
        ...     Measure,
        ...     ParametrizedProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a measure space.

        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ... )

        Define a 2-dimensional measurable vector and pushforward the measure `mu`.

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
        >>> mu_f = f.pushforward(mu)
        >>> print(mu_f)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu_f':
                    mu_f
        f_0 f_1
        1   2          1
        3   4          5

        Now define a measurable space with a sample space.

        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )

        Define a parametrized probability measure on the sigma-algebra.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> mapping = {
        ...     (0, 0): 0.1,  # (theta, F) = (0, 0), etc ...
        ...     (0, 1): 0.2,
        ...     (0, 2): 0.7,
        ...     (1, 0): 0.4,
        ...     (1, 1): 0.5,
        ...     (1, 2): 0.1,
        ... }
        >>> P = ParametrizedProbabilityMeasure.from_domains(
        ...     measure_domain=F, parameter_domain=Theta, mapping=mapping
        ... )

        Define a 2-dimensional random vector and pushforward the parametrized probability measure `P`.

        >>> X = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 1),
        ...         1: (1, 1),
        ...         2: (3, 1),
        ...         3: (3, 1),
        ...     },
        ... )
        >>> P_X = X.pushforward(P)
        >>> print(P_X)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P_X':
        theta      0    1
        X_0 X_1
        1   1    0.1  0.4
        3   1    0.9  0.6

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a measurable vector on a measure space $(X, \mathcal{F}, \mu)$. Then we define a measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(A) = \mu\left( \{x \in X : f(x) \in A\}\right),
        $$

        for all Borel subsets $A\subset \mathbb{R}^d$.

        If $\mu$ is a parametrized measure on $X$ with parameter domain $\Theta$, then we define a parametrized measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(\theta, A) = \mu\left(\theta, \{x \in X : f(x) \in A\}\right),
        $$

        for all $\theta \in \Theta$ and all Borel subsets $A\subset \mathbb{R}^d$.
        """
        return Operators.pushforward(
            vec=self,
            measure=measure,
            name=name,
        )

    # --------------------- probability-related methods --------------------- #

    def expectation(
        self,
        given: SigmaAlgebra | RandomVector | None = None,
        measure: ProbabilityMeasure | None = None,
        name: Hashable | None = None,
    ) -> RandomVector:
        r"""Compute the expectation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Calls `Operators.expectation` with appropriate arguments.

        Parameters
        ----------
        given : SigmaAlgebra | RandomVector | None, default=None
            The sigma-algebra or random vector to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.
        name : Hashable | None, default=None
            The name to assign to the resulting expected value random vector. If `None`, a default name is generated.

        Returns
        -------
        exp : RandomVector
            The expectation of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Function,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: (1, 3),
        ...         1: (2, 4),
        ...         2: (3, 1),
        ...         3: (1, 0),
        ...         4: (5, 0),
        ...         5: (0, 8),
        ...     },
        ...     name="Y",
        ... )

        Get the constant random variable whose unique value is `1`.

        >>> one = Function.from_constant(domain=Omega, constant=1)

        Check that the unconditional expectation of the random variable `X` is equal to the constant random variable whose unique value is the Lebesgue integral of the random variable.

        >>> X.expectation() == X.integrate() * one
        True

        Compute the unconditional expectation of the random vector `Y`, and check that its components are the unconditional expectations of the components of `Y`.

        >>> all(E_Y_i == Y_i.integrate() * one for E_Y_i, Y_i in zip(Y.expectation(), Y))
        True

        Define a sigma-algebra for conditional expectations.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional expectation of the random variable `X` is equal to its Fourier expansion.

        >>> X.expectation(given=G) == sum(X.integrate(B) / P(B) * B.indicator for B in G if P(B) != 0)
        True

        Check the same for the components of the conditional expectation of the random vector `Y`.

        >>> all(
        ...     E_Y_i_G == sum(Y_i.integrate(B) / P(B) * B.indicator for B in G if P(B) != 0)
        ...     for E_Y_i_G, Y_i in zip(Y.expectation(G), Y)
        ... )
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional expectation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $E(X\mid \mathcal{G})$ for which

        $$
        \int_V E(X\mid \mathcal{G}) \, dP = \int_V X \, dP,
        $$

        for all $V\in \mathcal{G}$. All such random variables are equal almost surely.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional expectation called a *Fourier expansion*:

        $$
        E(X\mid \mathcal{G}) = \sum_B \frac{\int_B X \, dP}{P(B)} I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability and $I_B$ is the indicator function of $B$.

        The *unconditional expectation* of $X$, denoted $E(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. In this case $E(X)$ is the constant random variable with

        $$
        E(X)(\omega) = \int_\Omega X \, dP,
        $$

        for all $\omega\in \Omega$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional expectation* to be the $d$-dimensional vector whose entries are the separate conditional expectations $E(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        return Operators.expectation(
            rv=self,
            given=given,
            measure=measure,
            name=name,
        )

    def variance(
        self,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the variance of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Calls `Operators.variance` with appropriate arguments.

        **User note**: This is a naive reference implementation. It is slow.

        Parameters
        ----------
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        var : RandomVector
            The variance of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: (1, 3),
        ...         1: (2, 4),
        ...         2: (3, 1),
        ...         3: (1, 0),
        ...         4: (5, 0),
        ...         5: (0, 8),
        ...     },
        ...     name="Y",
        ... )

        Check that the variance may be computed via the "short-cut" formula.

        >>> X.variance() == (X**2).expectation() - X.expectation() ** 2
        True

        Check the same for `Y`.

        >>> Y.variance() == (Y**2).expectation() - Y.expectation() ** 2
        True

        Define a sigma-algebra for conditional variances.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional variance of the random variable `X` is equal to a linear combination of indicators weighted by unconditional variances.

        >>> X.variance(G) == sum((X | B).variance().item() * B.indicator for B in G if P(B) > 0)
        True

        Check the same for the random vector `Y`.

        >>> all(
        ...     V_Y_i_G == sum((Y_i | B).variance().item() * B.indicator for B in G if P(B) > 0)
        ...     for V_Y_i_G, Y_i in zip(Y.variance(G), Y)
        ... )
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F}, P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional variance* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        V(X\mid \mathcal{G}) = E\left[ (X-E(X\mid \mathcal{G}))^2 \mid \mathcal{G}\right].
        $$

        The *unconditional variance* of $X$, denoted $V(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional variance is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional variance:

        $$
        V(X\mid \mathcal{G}) = \sum_B V(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $V(X|_B)$ is the unconditional variance of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional variance* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional variances $V(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        return Operators.variance(
            rv=self,
            given=given,
            measure=measure,
        )

    def std(
        self,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the standard deviation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Calls `Operators.std` with appropriate arguments.

        **User note**: This is a naive reference implementation. It is slow.

        Parameters
        ----------
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        std : RandomVector
            The standard deviation of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.0,
        ...         2: 0.2,
        ...         3: 0.2,
        ...         4: 0.15,
        ...         5: 0.45,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 1,
        ...         4: 5,
        ...         5: 0,
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     domain=Omega,
        ...     measure=P,
        ...     mapping={
        ...         0: (1, 3),
        ...         1: (2, 4),
        ...         2: (3, 1),
        ...         3: (1, 0),
        ...         4: (5, 0),
        ...         5: (0, 8),
        ...     },
        ...     name="Y",
        ... )

        Check that the unconditional standard deviation is equal to its definition.

        >>> X.std() == X.variance() ** 0.5
        True

        Check the same for `Y`

        >>> Y.std() == Y.variance() ** 0.5
        True

        Define a sub-sigma-algebra of `F` for conditional standard deviations.

        Define a sigma-algebra for conditional standard deviations.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ...     name="G",
        ... )

        Check that the conditional standard deviation of the random variable `X` is equal to a linear combination of indicators weighted by unconditional standard deviations.

        >>> X.std(G) == sum((X | B).std().item() * B.indicator for B in G if P(B) > 0)
        True

        Check the same for the random vector `Y`.

        >>> all(
        ...     std_Y_i_G == sum((Y_i | B).std().item() * B.indicator for B in G if P(B) != 0)
        ...     for std_Y_i_G, Y_i in zip(Y.std(G), Y)
        ... )
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional standard deviation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $\sigma(X \mid \mathcal{G})$ that is equal almost surely to the random variable

        $$
        \sigma(X\mid \mathcal{G}) = \sqrt{V(X\mid \mathcal{G})}.
        $$

        The *unconditional standard deviation* of $X$, denoted $\sigma(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional standard deviation is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional standard deviation:

        $$
        \sigma(X\mid \mathcal{G}) = \sum_B \sigma(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B)$ is the unconditional standard deviation of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional standard deviation* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional standard deviations $\sigma(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        return Operators.std(
            rv=self,
            given=given,
            measure=measure,
        )
