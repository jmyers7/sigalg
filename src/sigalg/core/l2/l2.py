"""A class representing an L2-space of random variables defined on a given measure space."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable
    from numbers import Real

    from ...typing.index_like import IndexLike
    from ..functions.function import Function
    from ..functions.measurable_function import MeasurableFunction
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain


class L2:
    r"""A class representing an L2-space of measurable functions defined on a given measure space.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The domain on which the L2-space is defined.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra over which the L2-space is defined. If `None`, the power-set sigma-algebra on the domain is used.
    measure : Measure | None, default=None
        The measure over which the L2-space is defined. If `None`, the counting measure on the domain is used.
    name : Hashable, default="H"
        The name of the L2-space.

    Examples
    --------
    >>> from sigalg.core import Domain, L2, Measure, SigmaAlgebra

    Define a measure space and the L2-space over it.

    >>> X = Domain.from_sequence(size=4)
    >>> F = SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 0,
    ...         3: 1,
    ...     },
    ... )
    >>> mu = Measure(
    ...     domain=F,
    ...     mapping={
    ...         0: 6,
    ...         1: 3,
    ...     },
    ... )
    >>> H = L2(X, F, mu)

    Print the L2-space.

    >>> print(H)  # doctest: +NORMALIZE_WHITESPACE
    H = L2(X, F, mu)
    ================
    <BLANKLINE>
    * Domain 'X':
     x
     0
     1
     2
     3
    <BLANKLINE>
    * Sigma algebra 'F':
         F
    x
    0    0
    1    1
    2    0
    3    1
    <BLANKLINE>
    * Measure 'mu':
            mu
    F
    0        6
    1        3

    Notes
    -----
    Let $(X,\mathcal{F},\mu)$ be a measure space. We define $L^2(X,\mathcal{F},\mu)$ to be the set of all $\mathcal{F}$-measurable functions $f: X \to \mathbb{R}$ such that

    $$
    \int_X f^2 \, d\mu < \infty.
    $$

    When the domain $X$ and measure $\mu$ are fixed and understood, we will write $L^2(\mathcal{F})$ in place of $L^2(X,\mathcal{F},\mu)$. We agree to identify two functions $f$ and $g$ in $L^2(\mathcal{F})$ provided that they are equal almost everywhere, i.e., the set of points on which they are not equal has measure $0$.

    The set $L^2(\mathcal{F})$ is a (real) vector space under the standard point-wise operations. Even more, it is also a Hilbert space when equipped with the inner product

    $$
    \langle f, g \rangle \stackrel{\text{def}}{=} \int_X fg \, d\mu,
    $$

    for $f,g\in L^2(\mathcal{F})$.

    In the case that $X$ is finite (as it always is, in SigAlg), the square-integrable condition is automatically satisfied, so $L^2(\mathcal{F})$ is simply the vector space of all $\mathcal{F}$-measurable functions.
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: Domain | IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        name: Hashable = "H",
    ) -> None:
        from ..spaces.measure_space import MeasureSpace

        self.measure_space = MeasureSpace(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
        )

        self.name = name

    # --------------------- properties --------------------- #

    @cached_property
    def basis_df(self) -> pd.DataFrame | None:
        """Return a `pd.DataFrame` whose columns are the orthonormal basis vectors of the L2-space.

        Returns
        -------
        basis_df : pd.DataFrame | None
            A `pd.DataFrame` whose columns are the orthonormal basis vectors of the L2-space, or `None` if the underlying measure space is empty.

        Examples
        --------
        >>> from sigalg.core import Domain, L2, Measure, SigmaAlgebra

        Define an L2-space.

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
        ...         0: 0.0,
        ...         1: 5,
        ...         2: 4,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Print the basis dataframe.

        >>> print(H.basis_df)  # doctest: +NORMALIZE_WHITESPACE
                  1    2
        x
        0  0.000000  0.0
        1  0.447214  0.0
        2  0.000000  0.5
        3  0.000000  0.5
        """
        if self.sig_alg is not None and self.measure is not None:
            if isinstance(self.sig_alg.data, pd.DataFrame):
                sig_alg_data = self.sig_alg.data.apply(tuple, axis=1)
            else:
                sig_alg_data = self.sig_alg.data

            basis_df = pd.get_dummies(sig_alg_data).astype(int)
            measure_data = self.measure.data.reindex(basis_df.columns)

            return (basis_df.mul(1 / measure_data**0.5, axis=1)).dropna(axis=1)

        else:
            return None

    @cached_property
    def basis(self) -> dict[Hashable, MeasurableFunction] | None:
        r"""Return an orthonormal basis of the L2-space.

        See the Notes section below for the mathematical details.

        Returns
        -------
        basis : dict[Hashable, MeasurableFunction] | None
            A dictionary mapping the atom ID to the corresponding basis vector of the L2-space.

        Examples
        --------
        >>> from sigalg.core import Domain, L2, Measure, SigmaAlgebra

        Define an L2-space.

        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 7,
        ...         2: 2,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Print the basis vectors.

        >>> for atom_id, phi in H.basis.items():
        ...     print(f"Atom identifier: {atom_id}")
        ...     print(f"Basis function:\n{phi}\n")  # doctest: +NORMALIZE_WHITESPACE
        Atom identifier: 1
        Basis function:
        Measurable function 'phi_1':
                  phi_1
        x
        0      0.000000
        1      0.000000
        2      0.377964
        3      0.377964
        4      0.000000
        5      0.000000
        <BLANKLINE>
        Atom identifier: 2
        Basis function:
        Measurable function 'phi_2':
                  phi_2
        x
        0      0.000000
        1      0.000000
        2      0.000000
        3      0.000000
        4      0.707107
        5      0.707107

        Notes
        -----
        Let $(X,\mathcal{F},\mu)$ be a measure space and set $H = L^2(X, \mathcal{F}, \mu)$. In the case that $X$ is finite, so that the $\sigma$-algebra $\mathcal{F}$ is determined by its set $\{A_i\}_{i\in I}$ of (finitely many) atoms, the vector space $H$ has an orthonormal basis given by the normalized indicator functions of the atoms of nonzero measure.

        Indeed, if we suppose $i\neq j$, then the product $I_{A_i}I_{A_j}$ of indicator functions is $0$ since $A_i$ and $A_j$ are disjoint. Thus, we have

        $$
        \langle I_{A_i}, I_{A_j} \rangle = \int_X I_{A_i} I_{A_j} \, d\mu = 0,
        $$

        which proves that $\{I_{A_i}\}_{i\in I}$ is an orthogonal set.

        If $f$ is an $\mathcal{F}$-measurable function, then it must be constant on each atom $A_i$. Thus, we have

        $$
        f = \sum_{i\in I} c_i I_{A_i}
        $$

        for some real numbers $c_i$. If an atom $A_i$ has measure $0$, then the corresponding summand $c_i I_{A_i}$ may be dropped, which still yields an equality in the $L^2$-space $H$ since $f$ and the linear combination on the right-hand side are still equal almost everywhere.

        Thus, the set of indicator functions of atoms with nonzero measure form an orthogonal basis $H$. To obtain an orthonormal basis, we first compute the norms of the indicator functions:

        $$
        \|I_{A_i}\|^2 = \langle I_{A_i}, I_{A_i} \rangle = \int_X I_{A_i}^2 \, d\mu = \mu(A_i).
        $$

        Thus, provided that $\mu(A_i)\neq 0$, we obtain a normalized basis function:

        $$
        \phi_i \stackrel{\text{def}}{=} \frac{I_{A_i}}{\sqrt{\mu(A_i)}}.
        $$
        """
        from ..functions.measurable_function import MeasurableFunction

        if self.basis_df is not None:
            basis = {}

            for atom_id, data in self.basis_df.items():
                name = f"phi_{atom_id}"
                basis[atom_id] = MeasurableFunction._from_validated(
                    data=data.rename(name),
                    sig_alg=self.sig_alg,
                    measure=self.measure,
                    index_kind="Index",
                    index_name=None,
                    name=name,
                )

            return basis

        else:
            return None

    @property
    def dim(self) -> int | None:
        """The dimension of the L2-space.

        Returns
        -------
        dim : int | None
            The dimension of the L2-space, or `None` if the basis is not defined.

        Examples
        --------
        >>> from sigalg.core import Domain, L2, Measure, SigmaAlgebra

        Define an L2-space.

        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 7,
        ...         2: 2,
        ...     },
        ... )
        >>> H = L2(X, F, mu)
        >>> print(H.dim)
        2
        """
        return len(self.basis) if self.basis is not None else None

    @property
    def domain(self) -> Domain | None:
        """The underlying domain on which the L2-space is defined.

        Returns
        -------
        domain : Domain | None
            The domain on which the L2-space is defined, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, L2, Measure, SigmaAlgebra

        Define an L2-space.

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
        ...         0: 0,
        ...         1: 5,
        ...         2: 4,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Print its domain.

        >>> print(H.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2
         3
        """
        return self.measure_space.domain

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """The underlying sigma-algebra on which the L2-space is defined.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on which the L2-space is defined, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, L2, Measure, SigmaAlgebra

        Define an L2-space.

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
        ...         0: 0.1,
        ...         1: 4,
        ...         2: 5,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Print its sigma-algebra.

        >>> print(H.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
               F
        x
        0      0
        1      1
        2      2
        3      2
        """
        return self.measure_space.sig_alg

    @property
    def measure(self) -> Measure | None:
        """The underlying measure on which the L2-space is defined.

        Returns
        -------
        measure : Measure | None
            The measure on which the L2-space is defined, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, L2, Measure, SigmaAlgebra

        Define an L2-space.

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
        ...         0: 1,
        ...         1: 4,
        ...         2: 5,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Print its measure.

        >>> print(H.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu':
                 mu
        F
        0         1
        1         4
        2         5
        """
        return self.measure_space.measure

    # --------------------- methods --------------------- #

    def __contains__(self, function: Function) -> bool:
        """Determine whether a function is in the L2-space.

        Parameters
        ----------
        function : Function
            The measurable function.

        Returns
        -------
        is_in : bool
            `True` if the function is in the L2-space; `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     Function,
        ...     L2,
        ...     Measure,
        ...     SigmaAlgebra,
        ... )

        Define an L2-space.

        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.7,
        ...         1: 0.3,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Define a function and test whether it is in the L2-space.

        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...     },
        ... )
        >>> f in H
        False

        Define a second function and test whether it is in the L2-space.

        >>> g = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 2,
        ...     },
        ...     name="g",
        ... )
        >>> g in H
        True
        """
        from ..functions.function import Function

        if not isinstance(function, Function):
            raise TypeError("function must be an instance of Function.")
        if function.domain != self.domain:
            raise ValueError(
                "The domain of function must match the domain of the L2-space."
            )
        return function.is_measurable(self.sig_alg)

    # --------------------- Hilbert space methods --------------------- #

    def fourier_coefficients(
        self, function: MeasurableFunction
    ) -> dict[Hashable, Real]:
        r"""Compute the Fourier coefficients of a measurable function with respect to the orthonormal basis of the L2-space contained in the `basis` attribute.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        function : MeasurableFunction
            The measurable function whose Fourier coefficients are to be computed.

        Returns
        -------
        coefficients : dict[Hashable, Real]
            A dictionary mapping the name of each basis vector to the corresponding Fourier coefficient of `function` with respect to that basis vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     L2,
        ...     Measure,
        ...     MeasurableFunction,
        ...     SigmaAlgebra,
        ... )

        Define an L2-spaces.

        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: "b",
        ...         1: "b",
        ...         2: "a",
        ...         3: "a",
        ...         4: "c",
        ...         5: "c",
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         "b": 7,
        ...         "a": 0,
        ...         "c": 2,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Define a function and compute its Fourier coefficients.

        >>> f = MeasurableFunction(
        ...     domain=X,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 3,
        ...         3: 3,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ... )
        >>> c = H.fourier_coefficients(function=f)
        >>> I = c.keys()

        Get the basis vectors of the L2-space and compute the Fourier expansion of the function.

        >>> phi = H.basis
        >>> f_fourier = sum(c[i] * phi[i] for i in I).with_name("f_fourier")

        Print the Fourier expansion and check that it is equal to the original function almost everywhere.

        >>> print(f_fourier)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_fourier':
               f_fourier
        x
        0           -1.0
        1           -1.0
        2            0.0
        3            0.0
        4            1.0
        5            1.0
        >>> print(mu.equal_almost_everywhere(f, f_fourier))
        True

        Notes
        -----
        Let $(X,\mathcal{F},\mu)$ be a measure space and set $H = L^2(X, \mathcal{F},\mu)$. Provided that $X$ is finite (as it always is, in SigAlg), the vector space $H$ has an orthonormal basis $\{\phi_i\}_{i\in I}$ consisting of the normalized indicator functions of the atoms of $\mathcal{F}$ of nonzero measure. Thus, given a function $f\in H$, we have its *generalized Fourier expansion*:

        $$
        f = \sum_{i\in I} \langle f, \phi_i \rangle \phi_i.
        $$

        The coefficients $c_i = \langle f,\phi_i \rangle$ are called the *Fourier coefficients* of $f$.
        """
        if function not in self:
            raise ValueError("The function must be in the L2-space.")

        function_times_indicators = self.sig_alg.atom_indicator_df.mul(
            function.data, axis=0
        ).drop_duplicates()
        measure_data = self.measure.data.reindex(function_times_indicators.columns)
        coefficients_series = function_times_indicators.mul(
            measure_data**0.5, axis=1
        ).sum()
        coefficients_series = coefficients_series[coefficients_series.abs() > 1e-10]

        return coefficients_series.to_dict()

    def inner(self, first: MeasurableFunction, second: MeasurableFunction) -> Real:
        """Compute the inner product of two measurable functions.

        Parameters
        ----------
        first : MeasurableFunction
            The first measurable function.
        second : MeasurableFunction
            The second measurable function.

        Returns
        -------
        inner_product : Real
            The inner product of the two measurable functions.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     L2,
        ...     Measure,
        ...     MeasurableFunction,
        ...     SigmaAlgebra,
        ... )

        Define an L2-space.

        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: "b",
        ...         1: "b",
        ...         2: "a",
        ...         3: "a",
        ...         4: "c",
        ...         5: "c",
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         "b": 7,
        ...         "a": 0,
        ...         "c": 2,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Define a function.

        >>> f = MeasurableFunction(
        ...     *H.measure_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 3,
        ...         3: 3,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ... )

        Get the basis vectors of the L2-space and compute the Fourier expansion of the function using the inner products of the Hilbert space structure.

        >>> phi = H.basis
        >>> I = phi.keys()
        >>> f_fourier = sum(H.inner(f, phi[i]) * phi[i] for i in I).with_name("f_fourier")

        Print the Fourier expansion and check that it is equal to the original function almost everywhere.

        >>> print(f_fourier)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_fourier':
               f_fourier
        x
        0           -1.0
        1           -1.0
        2            0.0
        3            0.0
        4            1.0
        5            1.0
        >>> print(mu.equal_almost_everywhere(f, f_fourier))
        True
        """
        if first not in self or second not in self:
            raise ValueError("Both measurable functions must be in the L2-space.")
        return (first * second).integrate()

    def norm(self, function: MeasurableFunction) -> Real:
        """Compute the norm of a measurable function.

        Parameters
        ----------
        function : MeasurableFunction
            The measurable function whose norm is to be computed.

        Returns
        -------
        norm : Real
            The norm of the measurable function in the L2-space.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Domain,
        ...     L2,
        ...     Measure,
        ...     MeasurableFunction,
        ...     SigmaAlgebra,
        ... )

        Define an L2-space.

        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 7,
        ...         2: 2,
        ...     },
        ... )
        >>> H = L2(X, F, mu)

        Get the indicator functions of the atoms of the sigma-algebra and check that their squared norms are equal to the measures of the atoms.

        >>> all(np.allclose(H.norm(A.indicator) ** 2, mu(A)) for A in F)
        True
        """
        from ..functions.operators import Operators

        if function not in self:
            raise ValueError("The function must be in the L2-space.")
        return Operators.integrate(function**2, measure=self.measure) ** 0.5

    def metric(self, first: MeasurableFunction, second: MeasurableFunction) -> Real:
        r"""Compute the distance between two measurable functions.

        See the Notes for the mathematical details.

        Parameters
        ----------
        first : MeasurableFunction
            The first measurable function.
        second : MeasurableFunction
            The second measurable function.

        Returns
        -------
        distance : Real
            The distance between the two measurable functions in the L2-space.

        Examples
        --------
        >>> from sigalg.core import (
        ...     L2,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define an L2-space over a probability space, a random variable, and a sub-sigma-algebra.

        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.6,
        ...         2: 0.15,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 2,
        ...         3: 2,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 0,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ...     name="G",
        ... )

        Compute the conditional expectation of the random variable given the sub-sigma-algebra and compute the L2-distance between the random variable and its conditional expectation.

        >>> E = X.expectation(given=G)
        >>> squared_distance = H.metric(X, E)
        >>> print(round(squared_distance, 2))
        1.26

        The distance between the random variable and its conditional expectation is less than (or equal to) the distance between it and any other random variable that is measurable with respect to the sub-sigma-algebra. For example, consider the random variable `Y` defined below, which is measurable with respect to the sub-sigma-algebra.

        >>> Y = RandomVariable(
        ...     domain=Omega,
        ...     sig_alg=G,
        ...     measure=P | G,
        ...     mapping={
        ...         0: 2,
        ...         1: 2,
        ...         2: 2,
        ...         3: 2,
        ...         4: -4,
        ...         5: -4,
        ...     },
        ...     name="Y",
        ... )
        >>> squared_distance = H.metric(X, Y)
        >>> print(round(squared_distance, 2))
        2.45

        Notes
        -----
        Any normed vector space $H$ has a norm-induced metric $d$ given by

        $$
        d(f,g) = \|f - g \|,
        $$

        for $f,g\in H$. In particular, if $H$ is a vector space of the form $L^2(X,\mathcal{F},\mu)$, then the induced metric is given by

        $$
        d(f,g) = \|f-g\| = \sqrt{\int_X (f-g)^2 \, d\mu},
        $$

        for $f,g\in L^2(X, \mathcal{F},\mu)$.
        """
        if first not in self or second not in self:
            raise ValueError("The functions must be in the L2-space.")

        return self.inner((first - second), (first - second)) ** 0.5

    def proj(
        self,
        function: MeasurableFunction,
        subspace: list[MeasurableFunction],
        name: Hashable | None = None,
    ) -> tuple[MeasurableFunction, np.ndarray, int]:
        r"""Compute the orthogonal projection of a measurable function onto the subspace spanned by a set of measurable functions.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        function : MeasurableFunction
            The measurable function to be projected.
        subspace : list[MeasurableFunction]
            A list of measurable functions spanning the subspace onto which `function` is to be projected.
        name : Hashable | None, default=None
            The name of the projection. If `None`, a default will be generated.

        Returns
        -------
        proj : MeasurableFunction
            The orthogonal projection of `function` onto the subspace spanned by `subspace`.
        coefficients : np.ndarray
            The coefficients of the projection of `function` onto the subspace spanned by the measurable functions in `subspace`. See the Notes section below for the mathematical details.
        dim : int
            The dimension of the subspace spanned by `subspace`.

        Examples
        --------
        >>> from sigalg.core import (
        ...     L2,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define an L2-space over a probability space and a random variable in it.

        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra.power_set(Omega, name="F")
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.4,
        ...         2: 0.2,
        ...         3: 0.2,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> X = RandomVariable(
        ...     *H.measure_space,
        ...     mapping={
        ...         0: 2.0,
        ...         1: 3.0,
        ...         2: 5.0,
        ...         3: 7.0,
        ...     },
        ... )

        For a quadratic regression example, we will project a random variable `Y` onto the subspace spanned by `1`, `X`, and `X**2`.

        >>> one = RandomVariable.from_constant(*H.measure_space, constant=1, name="one")
        >>> Y = RandomVariable(
        ...     *H.measure_space,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 3.0,
        ...         2: 2.0,
        ...         3: 4.0,
        ...     },
        ...     name="Y",
        ... )
        >>> proj, c, dim = H.proj(function=Y, subspace=[one, X, X**2])

        The `c` coefficients returned by the `proj` method are the coefficients of the orthogonal projection of `Y` onto the subspace spanned by `1`, `X`, and `X**2`. We can verify that the projection is correct by computing the expected projection using the coefficients and comparing it to the projection returned by the `proj` method.

        >>> expected_proj = sum([c[k] * X**k for k in range(dim)])
        >>> print(proj == expected_proj)
        True

        Notes
        -----
        Let $(X, \mathcal{F}, \mu)$ be a measure space, set $H = L^2(X, \mathcal{F},\mu)$, and suppose for simplicitly that $X$ is finite, so that $H$ is finite-dimensional. Suppose $g$ is a function in $H$ and that $\{f_1,f_2,\ldots,f_n\} \subset H$ spans a subspace $V$ of $H$. The *orthogonal projection* of $g$ onto $V$ is the unique function $\widehat{g}\in V$ such that

        $$
        \|g - \widehat{g}\| \leq \|g - h\|,
        $$

        for all $h\in V$. The existence and uniqueness of $\widehat{g}$ is a consequence of the Projection Theorem for Hilbert spaces.

        In applications, one computes $\widehat{g}$ by identifying $\widehat{g}$ as the global minimizer of the objective function

        $$
        \psi: \mathbb{R}^n \to \mathbb{R}, \quad \psi(c) = \frac{1}{2} \left \|\sum_{j=1}^n c_j f_j - g \right\|^2.
        $$

        The first (Fréchet) derivative $D\psi(c)$ of $\psi$ at $c$ is given by

        $$
        D\psi(c)h = \sum_{j=1}^n \left[ \sum_{k=1}^n c_k \langle f_j,f_j \rangle - \langle f_j, g \rangle \right]h_j,
        $$

        for $h\in \mathbb{R}^n$. At the minimizer $\widehat{g} = \sum_{k=1}^n c_k f_k$, we must have $D\psi(c)=0$, which yields the linear system of equations

        $$
        \begin{bmatrix}
        \langle f_1, f_1 \rangle & \cdots & \langle f_1, f_n \rangle \\
        \vdots & \ddots & \vdots \\
        \langle f_n, f_1 \rangle & \cdots & \langle f_n, f_n \rangle \\
        \end{bmatrix} \begin{bmatrix} c_1 \\ \vdots \\ c_n \end{bmatrix} =
        \begin{bmatrix} \langle f_1, g \rangle \\ \vdots \\ \langle f_n,g \rangle \end{bmatrix},
        $$

        for the unknown $c$. The coefficient matrix on the left is the *Gram matrix* $G$ of the functions $f_1,\ldots,f_n$ as vectors in the Hilbert space $H$. Provided that $G$ is invertible (which is equivalent to linear independence of the $f_j$'s in $H$), there is a unique solution $c$. Otherwise, if $G$ is not invertible, then there are infinitely many choices for $c$. In this case, the method `proj` returns the $c$ for which $\sum_{k=1}^n c_k^2$ is minimum.

        Finally, for a solution $c$ for which $\widehat{g} = \sum_{k=1}^n c_k f_k$, note that the linear system above can equivalently be expressed as the system

        $$
        \langle f_1, \widehat{g} - g \rangle = \cdots = \langle f_n, \widehat{g}-g \rangle = 0.
        $$

        These are called the *normal equations*, which confirm that $\widehat{g}$ really is the orthogonal projection of $g$ onto the subspace spanned by the $f_j$'s.
        """
        from ..functions.random_variable import RandomVariable

        if function not in self:
            raise ValueError("The function to be projected must be in the L2-space.")
        if subspace is None or len(subspace) == 0:
            raise ValueError("The subspace must be nonempty.")
        for subspace_function in subspace:
            if subspace_function not in self:
                raise ValueError(
                    "All functions in the subspace must be in the L2-space."
                )

        A = np.zeros((self.dim, len(subspace)))
        for j, subspace_function in enumerate(subspace):
            coefficients = np.fromiter(
                self.fourier_coefficients(function=subspace_function).values(),
                dtype=float,
            )
            A[:, j] = coefficients

        rv_vec = np.fromiter(
            self.fourier_coefficients(function=function).values(), dtype=float
        )
        c, _, dim, _ = np.linalg.lstsq(A, rv_vec, rcond=None)

        rv_data = pd.DataFrame({rv.name: rv.data for rv in subspace})
        data = rv_data.multiply(c, axis=1).sum(axis=1)

        if name is None:
            name = f"{function.name}_proj"

        proj = RandomVariable._from_validated(
            data=data.rename(name),
            sig_alg=self.sig_alg,
            measure=self.measure,
            index_kind="Index",
            index_name=None,
            name=name,
        )

        return proj, c, dim

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the L2-space.

        Returns
        -------
        repr_str : str
            A string representation showing the L2-space's component names.
        """
        return (
            f"{self.name} = L2("
            f"{self.domain.name}, "
            f"{self.sig_alg.name}, "
            f"{self.measure.name}, "
            f"{self.name})"
        )

    def __str__(self) -> str:
        """Return a detailed string representation of the L2-space.

        Returns
        -------
        repr_str : str
            A formatted string showing the L2-space header and detailed
            representations of its components.
        """
        header = (
            f"{self.name} = L2("
            f"{self.domain.name}, "
            f"{self.sig_alg.name}, "
            f"{self.measure.name})"
        )
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + str(self.domain)
            + "\n\n* "
            + str(self.sig_alg)
            + "\n\n* "
            + str(self.measure)
        )
