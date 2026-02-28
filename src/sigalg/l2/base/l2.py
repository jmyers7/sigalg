"""Classes for modeling L2-spaces of random variables.

Classes
-------
L2
    A class representing the L2-space of random variables defined on a given probability space.
"""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...core.base.probability_space import ProbabilitySpace
    from ...core.base.sample_space import SampleSpace
    from ...core.probability_measures.probability_measure import ProbabilityMeasure
    from ...core.random_objects.random_variable import RandomVariable
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra


class L2:
    r"""A class representing the L2-space of random variables defined on a given probability space.

    Suppose given a probability space $(\Omega, \mathcal{F}, P)$, where $\Omega$ is a sample space, $\mathcal{F}$ is a $\sigma$-algebra on $\Omega$, and $P$ is a probability measure on $\mathcal{F}$. In the mathematical literature, the space $L^2(\Omega, \mathcal{F}, P)$ is defined as the set of all (equivalence classes of) random variables $X$ defined on $\Omega$ that are $\mathcal{F}$-measurable and satisfy $E(X^2) < \infty$.

    The equivalence relation is defined as follows: two random variables $X$ and $Y$ are considered equivalent if they are *equal almost surely*, i.e., if $P(X \neq Y) = 0$. In other words, in the mathematical definition of an $L^2$-space, two random variables that differ only on a set of probability zero are identified as the same element of $L^2$.

    The class `L2` is SigAlg's model of these $L^2$-spaces of random variables.

    Mimicking the mathematical definition of an $L^2$-space described above, an instance `H` of `L2` is initialized with a `SampleSpace`, a `SigmaAlgebra` on that sample space, and a `ProbabilityMeasure`.

    Since all sample spaces in SigAlg are finite, the condition $E(X^2) < \infty$ is automatically satisfied for all random variables defined on the sample space. Therefore, the only condition for an instance `X` of `RandomVariable` to be in an instance `H` of `L2` is that `X` is measurable with respect to the `SigmaAlgebra` attribute of `H`. This can be checked via the `in` operator by writing `X in H`.

    Note that the `in` operator acts on a random variable itself, not on the equivalence class of the random variable described above. Indeed, these equivalence classes are not explicitly modeled in SigAlg.

    Besides providing the `in` operator, an instance of `L2` also provides several Hilbert space methods, including `inner` for computing the inner product of two random variables, `norm` for computing the norm of a random variable, and `metric` for computing the (norm induced) distance between two random variables.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space on which the L2-space is defined.
    sigma_algebra : SigmaAlgebra | None, default=None
        The sigma algebra on which the L2-space is defined. If `None`, the power set sigma-algebra on the sample space is used.
    probability_measure : ProbabilityMeasure | None, default=None
        The probability measure on which the L2-space is defined. If `None`, the uniform probability measure on the sample space is used.
    name : Hashable | None, default="H"
        The name of the L2-space.

    Raises
    ------
    TypeError
        If `sample_space` is not an instance of `SampleSpace`, or if `sigma_algebra` is not an instance of `SigmaAlgebra` or `None`, or if `probability_measure` is not an instance of `ProbabilityMeasure` or `None`. If `sigma_algebra` is not `None`, it must be defined on the same sample space as the L2-space. If `probability_measure` is not `None`, it must be defined on the same sample space as the L2-space.

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
    >>> from sigalg.l2 import L2
    >>> Omega = SampleSpace().from_sequence(size=4)
    >>> atom_ids = {
    ...     0: 0,
    ...     1: 0,
    ...     2: 1,
    ...     3: 1,
    ... }
    >>> F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)
    >>> probabilities = {
    ...     0: 0.2,
    ...     1: 0.1,
    ...     2: 0.4,
    ...     3: 0.3,
    ... }
    >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(probabilities)
    >>> H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)
    >>> outputs_X = {
    ...     0: 3,
    ...     1: 3,
    ...     2: 5,
    ...     3: 5,
    ... }
    >>> X = RandomVariable(domain=Omega).from_dict(outputs_X)
    >>> outputs_Y = {
    ...     0: 1,
    ...     1: 3,
    ...     2: 4,
    ...     3: 2,
    ... }
    >>> Y = RandomVariable(domain=Omega, name="Y").from_dict(outputs_Y)
    >>> # X is measurable with respect to F, so it is in H
    >>> print(X in H)
    True
    >>> # Y is not measurable with respect to F, so it is not in H
    >>> print(Y in H)
    False
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
        name: Hashable | None = "H",
    ) -> None:
        from ...core.base.sample_space import SampleSpace
        from ...core.probability_measures.probability_measure import ProbabilityMeasure
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be an instance of SampleSpace.")
        if sigma_algebra is not None and (
            not isinstance(sigma_algebra, SigmaAlgebra)
            or sigma_algebra.sample_space != sample_space
        ):
            raise TypeError(
                "sigma_algebra must be an instance of SigmaAlgebra or None. If not None, it must be defined on the same sample space as the L2-space."
            )
        if probability_measure is not None and (
            not isinstance(probability_measure, ProbabilityMeasure)
            or probability_measure.sample_space != sample_space
        ):
            raise TypeError(
                "probability_measure must be an instance of ProbabilityMeasure or None. If not None, it must be defined on the same sample space as the L2-space."
            )

        self._sample_space = sample_space
        if sigma_algebra is None:
            sigma_algebra = SigmaAlgebra.power_set(sample_space)
        if probability_measure is None:
            probability_measure = ProbabilityMeasure.uniform(sample_space)
        self._sigma_algebra = sigma_algebra
        self._probability_measure = probability_measure
        self._name = name

        # caches
        self._probability_space: ProbabilitySpace | None = None
        self._basis: list[RandomVariable] | None = None
        self._base_df: pd.DataFrame | None = None

    # --------------------- properties --------------------- #

    @property
    def _cached_base_df(self) -> pd.DataFrame:
        if self._base_df is None:
            self._base_df = pd.concat(
                [self.sigma_algebra.data, self.probability_measure.data], axis=1
            )
            self._base_df["prob_by_atom"] = self._base_df.groupby("atom ID")[
                "probability"
            ].transform("sum")
        return self._base_df

    @property
    def basis(self) -> dict[str, RandomVariable]:
        r"""Return a vector space basis of the L2-space.

        Consider the space $H = L^2(\Omega, \mathcal{F}, P)$. In SigAlg, we restrict ourselves to finite sample spaces $\Omega$, so the $\sigma$-algebra $\mathcal{F}$ is completely determined by its *atoms*, i.e., the minimal nonempty events in $\mathcal{F}$. The atoms form a partition of $\Omega$.

        The indicator functions $I_A$, as $A$ ranges over the atoms of $\mathcal{F}$, form an orthogonal basis of the $L^2$-space. Their squared norms are computed as

        $$
        \|I_A\|^2 = E(I_A^2) = \int_{\Omega} I_A^2 \, dP = P(A),
        $$

        since $I_A^2 = I_A$. Therefore, a convenient choice of orthonormal basis of $H$ is given by all normalized indicator functions $I_A / \sqrt{P(A)}$, for which $P(A) > 0$. Note that if $P(A) = 0$, then the indicator function $I_A$ is the zero vector in $H$ (since it is equal to the zero vector almost surely), so it does not contribute to the basis.

        The `basis` attribute of an instance `H` of `L2` returns a dictionary mapping the atom ID to the corresponding normalized indicator function of that atom, for all atoms that have positive probability.

        Returns
        -------
        basis : dict[str, RandomVariable]
            A dictionary mapping the atom ID to the corresponding basis vector of the L2-space.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> from sigalg.l2 import L2
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        >>> # A probability measure assigning nonzero probability to all atoms
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        >>> H = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=P,
        ... )
        >>> e_0, e_1 = H.basis.values()
        >>> e_0 # doctest: +NORMALIZE_WHITESPACE
        Random variable '0':
                       0
        sample
        0       1.195229
        1       1.195229
        2       0.000000
        >>> # A probability measure assigning zero probability to one atom
        >>> Q = ProbabilityMeasure(sample_space=Omega).from_dict({0:0.2, 1:0.8, 2:0})
        >>> G = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=Q,
        ...     name="G"
        ... )
        >>> G.basis # doctest: +NORMALIZE_WHITESPACE
        {np.int64(0): Random variable '0':
                  0
        sample
        0       1.0
        1       1.0
        2       0.0}
        """
        from ...core.random_objects.random_variable import RandomVariable

        if self._basis is None:
            self._basis = {}
            df = self._cached_base_df

            for atom_id in df[df["prob_by_atom"] > 1e-8]["atom ID"].unique():
                mask = df["atom ID"] == atom_id
                atom_prob = df.loc[mask, "prob_by_atom"].iloc[0]

                indicator_data = pd.Series(0.0, index=df.index)
                indicator_data[mask] = 1 / np.sqrt(atom_prob)

                self._basis[atom_id] = RandomVariable(
                    domain=self.sample_space, name=atom_id
                ).from_pandas(indicator_data)

        return self._basis

    @property
    def dim(self) -> int:
        """The dimension of the L2-space, i.e., the number of basis vectors.

        Returns
        -------
        dim : int
            The dimension of the L2-space, i.e., the number of basis vectors.
        """
        return len(self.basis)

    @property
    def sample_space(self) -> SampleSpace:
        """The sample space on which the L2-space is defined.

        Returns
        -------
        sample_space : SampleSpace
            The sample space on which the L2-space is defined.
        """
        return self._sample_space

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        """The sigma-algebra on which the L2-space is defined.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra on which the L2-space is defined.
        """
        return self._sigma_algebra

    # TODO: write unit tests for sigma_algebra setter
    @sigma_algebra.setter
    def sigma_algebra(self, sigma_algebra: SigmaAlgebra) -> None:
        """Set the sigma-algebra on which the L2-space is defined.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra to set for the L2-space. Must be defined on the same sample space as the L2-space.

        Raises
        ------
        TypeError
            If `sigma_algebra` is not an instance of `SigmaAlgebra`.
        ValueError
            If `sigma_algebra` is not defined on the same sample space as the L2-space.
        """
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be an instance of SigmaAlgebra.")
        if sigma_algebra.sample_space != self.sample_space:
            raise ValueError(
                "The sample space of the sigma algebra must match the sample space of the L2-space."
            )
        self._sigma_algebra = sigma_algebra
        self._basis = None
        self._base_df = None

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        """The probability measure on which the L2-space is defined.

        Returns
        -------
        probability_measure : ProbabilityMeasure
            The probability measure on which the L2-space is defined.
        """
        return self._probability_measure

    # TODO: write unit tests for probability_measure setter
    @probability_measure.setter
    def probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        """Set the probability measure on which the L2-space is defined.

        Parameters
        ----------
        probability_measure : ProbabilityMeasure
            The probability measure to set for the L2-space. Must be defined on the same sample space as the L2-space.

        Raises
        ------
        TypeError
            If `probability_measure` is not an instance of `ProbabilityMeasure`.
        ValueError
            If `probability_measure` is not defined on the same sample space as the L2-space.
        """
        from ...core.probability_measures.probability_measure import ProbabilityMeasure

        if not isinstance(probability_measure, ProbabilityMeasure):
            raise TypeError(
                "probability_measure must be an instance of ProbabilityMeasure."
            )
        if probability_measure.sample_space != self.sample_space:
            raise ValueError(
                "The sample space of the probability measure must match the sample space of the L2-space."
            )
        self._probability_measure = probability_measure
        self._basis = None
        self._base_df = None

    @property
    def name(self) -> Hashable:
        """The name of the L2-space.

        Returns
        -------
        name : Hashable
            The name of the L2-space.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        self._name = name

    # --------------------- methods --------------------- #

    def integrate(self, rv: RandomVariable) -> Real:
        """Integrate a random variable with respect to the probability measure of the L2-space.

        Parameters
        ----------
        rv : RandomVariable
            The random variable to be integrated.

        Raises
        ------
        ValueError
            If `rv` is not in the L2-space.

        Returns
        -------
        integral : Real
            The integral of the random variable with respect to the probability measure of the L2-space.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
        >>> from sigalg.l2 import L2
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        >>> H = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=P,
        ... )
        >>> X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        >>> float(round(H.integrate(X), 2))
        1.6
        """
        if rv not in self:
            raise ValueError("The random variable must be in the L2-space.")
        return self.probability_measure.integrate(rv=rv)

    def fourier_coefficients(self, rv: RandomVariable) -> dict[Hashable, Real]:
        """Compute the Fourier coefficients of a random variable with respect to the basis of the L2-space.

        Parameters
        ----------
        rv : RandomVariable
            The random variable whose Fourier coefficients are to be computed.

        Raises
        ------
        ValueError
            If `rv` is not in the L2-space.

        Returns
        -------
        coefficients : dict[Hashable, Real]
            A dictionary mapping the name of each basis vector to the corresponding Fourier coefficient of `rv` with respect to that basis vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
        >>> from sigalg.l2 import L2
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        >>> H = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=P,
        ... )
        >>> # Get the Fourier coefficients of X with respect to the basis of H
        >>> X = RandomVariable(domain=Omega, name="X").from_dict({0: 2, 1: 2, 2: 3})
        >>> coeffs = H.fourier_coefficients(rv=X)
        >>> coeffs
        {0: 1.6733200530681511, 1: 1.6431676725154982}
        >>> # Reconstruct X from its Fourier coefficients and the basis of H
        >>> sum(coeffs[basis_name] * basis_vec for basis_name, basis_vec in H.basis.items()).with_name("X_reconstructed") # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_reconstructed':
        X_reconstructed
        sample
        0                   2.0
        1                   2.0
        2                   3.0
        >>> # Define a new probability measure Q that assigns zero probability to an atom in the sigma algebra, and define a new L2-space
        >>> Q = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.8, 2: 0.0})
        >>> K = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=Q,
        ... )
        >>> # Compute the Fourier coefficients of X with respect to the basis of K, and note that there is only one coefficient
        >>> K.fourier_coefficients(rv=X)
        {0: 2.0}
        >>> # Reconstruct X from its Fourier coefficients and the basis of K, and note that the reconstruction differs from the original X on a set of probability zero
        >>> (2.0 * K.basis[0]).with_name("X_reconstructed") # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_reconstructed':
        X_reconstructed
        sample
        0                   2.0
        1                   2.0
        2                   0.0
        """
        if rv not in self:
            raise ValueError("The random variable must be in the L2-space.")

        df = self._cached_base_df.copy()
        df = df[df["prob_by_atom"] > 1e-8]

        df["contribution"] = (
            rv.data.loc[df.index] * df["probability"] / np.sqrt(df["prob_by_atom"])
        )

        return df.groupby("atom ID")["contribution"].sum().to_dict()

    def __contains__(self, rv: RandomVariable) -> bool:
        """Determine whether a random variable is in the L2-space.

        A random variable is in the L2-space if it is measurable with respect to the sigma algebra.

        Parameters
        ----------
        rv : RandomVariable
            The random variable.

        Raises
        ------
        TypeError
            If `rv` is not an instance of `RandomVariable`.
        ValueError
            If the domain of `rv` does not match the sample space of the L2-space.

        Returns
        -------
        is_in : bool
            `True` if the random variable is in the L2-space; `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
        >>> from sigalg.l2 import L2
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        >>> H = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=P,
        ... )
        >>> V_0, _ = H.basis.values()
        >>> # An indicator of an atom is always in the L2-space
        >>> V_0 in H
        True
        >>> # A random variable which is not in the L2-space.
        >>> X = RandomVariable(domain=Omega).from_dict({0: 0, 1: 1, 2: 2})
        >>> X in H
        False
        """
        from ...core.random_objects.random_variable import RandomVariable

        if not isinstance(rv, RandomVariable):
            raise TypeError("rv must be an instance of RandomVariable.")
        if rv.domain != self.sample_space:
            raise ValueError("The domain of rv must match the sample space.")
        return rv.is_measurable(self.sigma_algebra)

    # --------------------- Hilbert space methods --------------------- #

    def inner(self, first: RandomVariable, second: RandomVariable) -> Real:
        """Compute the inner product of two random variables.

        Both random variables must be in the L2-space.

        Parameters
        ----------
        first : RandomVariable
            The first random variable.
        second : RandomVariable
            The second random variable.

        Raises
        ------
        ValueError
            If one of the random variables is not in the L2-space.

        Returns
        -------
        inner_product : Real
            The inner product of the two random variables.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
        >>> from sigalg.l2 import L2
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        >>> H = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=P,
        ... )
        >>> X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        >>> Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})
        >>> float(H.inner(X, Y))
        8.2
        >>> # Example of orthogonal RVs: two indicator functions of disjoint events
        >>> A, B = F.to_atoms()
        >>> I_A = RandomVariable.indicator_of(A)
        >>> I_B = RandomVariable.indicator_of(B)
        >>> float(H.inner(I_A, I_B))
        0.0
        """
        if first not in self or second not in self:
            raise ValueError("Both random variables must be in the L2-space.")
        return self.probability_measure.integrate(rv=first * second)

    def norm(self, X: RandomVariable) -> Real:
        """Compute the norm of a random variable in the L2-space.

        Parameters
        ----------
        X : RandomVariable
            The random variable whose norm is to be computed.

        Raises
        ------
        ValueError
            The random variable must be in the L2-space.

        Returns
        -------
        norm : Real
            The norm of the random variable in the L2-space.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
        >>> from sigalg.l2 import L2
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        >>> H = L2(
        ...     sample_space=Omega,
        ...     sigma_algebra=F,
        ...     probability_measure=P,
        ... )
        >>> A, _ = F.to_atoms()
        >>> I_A = RandomVariable.indicator_of(A)
        >>> # The squared norm of an indicator function is the probability of the corresponding event
        >>> float(round(H.norm(I_A) ** 2, 1))
        0.7
        """
        if X not in self:
            raise ValueError("The random variable must be in the L2-space.")
        return self.probability_measure.integrate(rv=X**2) ** 0.5

    def metric(self, first: RandomVariable, second: RandomVariable) -> Real:
        """Compute the distance between two random variables in the L2-space.

        Parameters
        ----------
        first : RandomVariable
            The first random variable.
        second : RandomVariable
            The second random variable.

        Raises
        ------
        ValueError
            If one of the two random variables is not in the L2-space.

        Returns
        -------
        distance : Real
            The distance between the two random variables in the L2-space.
        """
        if first not in self or second not in self:
            raise ValueError("The random variables must be in the L2-space.")
        return self.inner((first - second), (first - second)) ** 0.5

    def proj(
        self,
        rv: RandomVariable,
        subspace: list[RandomVariable],
    ) -> tuple[RandomVariable, np.ndarray, int]:
        r"""Compute the orthogonal projection of a random variable onto the subspace spanned by a set of random variables.

        Let $H$ be the given $L^2$-space, suppose that $Y$ is a random variable in $H$, and suppose $\{X_1,X_2,\ldots,X_n\} \subset H$ spans a subspace $V$. The random vector $Y$ is stored in the parameter `rv`, while the $X_k$'s are stored in the parameter `subspace`. The goal is to compute the orthogonal projection $\hat{Y}$ of $Y$ onto $V$. By definition, this is a minimizer of the squared norm $\|Y - \hat{Y}\|^2$, over all $\hat{Y} \in V$. The random variable $\hat{Y}$ is returned as `proj`. The method also computes coefficients $u_1,u_2,\ldots,u_n$ such that

        $$
        \hat{Y} = \sum_{k=1}^n u_k X_k.
        $$

        If $V$ has dimension $d<n$, then there are infinitely many choices of coefficients $u_1,u_2,\ldots,u_n$; in this case, the method returns the particular choice of coefficients for which $\sum_{k=1}^n u_k^2$ is minimized. The coefficients are returned as an `np.ndarray` in the variable `coefficients`, in the same order as the random variables in the input list `subspace`. The method also returns the dimension $d$ of the subspace $V$ as the variable `dim`.

        Parameters
        ----------
        rv : RandomVariable
            The random variable to be projected.
        subspace : list[RandomVariable]
            A list of random variables spanning the subspace onto which `rv` is to be projected.

        Raises
        ------
        ValueError
             If `rv` is not in the L2-space, or if any of the random variables in `subspace` is not in the L2-space, or if `subspace` is empty.

        Returns
        -------
        proj : RandomVariable
            The orthogonal projection of `rv` onto the subspace spanned by `subspace`.
        coefficients : np.ndarray
            The coefficients of the projection of `rv` onto the subspace spanned by the random variables in `subspace`. See the description above for details.
        dim : int
            The dimension of the subspace spanned by `subspace`.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace
        >>> from sigalg.l2 import L2
        >>> # Define a sample space and probability measure
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.4,
        ...         2: 0.2,
        ...         3: 0.2,
        ...     }
        ... )
        >>> # Define an L2 space with default power set sigma-algebra
        >>> H = L2(sample_space=Omega, probability_measure=P)
        >>> # For a quadratic regression example, we will project a random variable Y onto the subspace spanned by 1, X, and X^2
        >>> one = RandomVariable(domain=Omega, name="one").from_constant(1)
        >>> X = RandomVariable(domain=Omega, name="X").from_dict(
        ...     {
        ...         0: 2.0,
        ...         1: 3.0,
        ...         2: 5.0,
        ...         3: 7.0,
        ...     }
        ... )
        >>> Y = RandomVariable(domain=Omega, name="Y").from_dict(
        ...     {
        ...         0: 1.0,
        ...         1: 3.0,
        ...         2: 2.0,
        ...         3: 4.0,
        ...     }
        ... )
        >>> # Project Y onto the subspace spanned by 1, X, and X^2
        >>> proj, u, dim = H.proj(rv=Y, subspace=[one, X, X**2])
        >>> expected_proj = sum([u[k] * X**k for k in range(dim)])
        >>> # Check that the projection is correct
        >>> print(proj == expected_proj)
        True
        """
        if rv not in self:
            raise ValueError(
                "The random variable to be projected must be in the L2-space."
            )
        if subspace is None or len(subspace) == 0:
            raise ValueError("The subspace must be nonempty.")
        for subspace_rv in subspace:
            if subspace_rv not in self:
                raise ValueError(
                    "All random variables in the subspace must be in the L2-space."
                )

        A = np.zeros((self.dim, len(subspace)))
        for j, subspace_rv in enumerate(subspace):
            coefficients = np.fromiter(
                self.fourier_coefficients(rv=subspace_rv).values(), dtype=float
            )
            A[:, j] = coefficients

        rv_vec = np.fromiter(self.fourier_coefficients(rv=rv).values(), dtype=float)
        u, _, dim, _ = np.linalg.lstsq(A, rv_vec, rcond=None)

        proj = sum([u[k] * subspace_rv for k, subspace_rv in enumerate(subspace)])
        name = f"{rv.name}_proj" if rv.name is not None else "proj"

        return proj.with_name(name), u, dim
