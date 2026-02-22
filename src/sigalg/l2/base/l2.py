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

    Note that the `in` operator acts on a random variable itself, not on the equivalence class of the random variable described above. Indeed, these equivalence classes are not explicitly modeled in SigAlg. However, an instance of `L2` does provide the `almost_surely_equal` method for determining whether two random variables are equal almost surely.

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

    # --------------------- properties --------------------- #

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
            tol = 1e-8

            df = pd.concat(
                [self.sigma_algebra.data, self.probability_measure.data], axis=1
            )
            atom_probs = df.groupby("atom ID")["probability"].transform("sum")

            for atom_id in df["atom ID"].unique():
                mask = df["atom ID"] == atom_id
                atom_prob = atom_probs[mask].iloc[0]

                if atom_prob < tol:
                    continue

                indicator_data = pd.Series(0.0, index=df.index)
                indicator_data[mask] = 1 / (atom_prob**0.5)

                basis_vec = RandomVariable(
                    domain=self.sample_space, name=atom_id
                ).from_pandas(indicator_data)

                self._basis[atom_id] = basis_vec

        return self._basis

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

    def fourier_coefficients(self, rv: RandomVariable) -> dict[str, Real]:
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
        coefficients : dict[str, Real]
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
        {'V_0': np.float64(1.6733200530681511), 'V_1': np.float64(1.6431676725154982)}
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
        {'V_0': np.float64(2.0)}
        >>> # Reconstruct X from its Fourier coefficients and the basis of K, and note that the reconstruction differs from the original X on a set of probability zero
        >>> (2.0 * K.basis["V_0"]).with_name("X_reconstructed") # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_reconstructed':
        X_reconstructed
        sample
        0                   2.0
        1                   2.0
        2                   0.0
        """
        if rv not in self:
            raise ValueError("The random variable must be in the L2-space.")
        return {
            basis_vec.name: self.inner(rv, basis_vec)
            for basis_vec in self.basis.values()
        }

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

    def almost_surely_equal(
        self, first: RandomVariable, second: RandomVariable, tol: float = 1e-8
    ) -> bool:
        r"""Determine whether two random variables are equal almost surely.

        Two random variables $X,Y:\Omega \to \mathbb{R}$ defined on a probability space $(\Omega, \mathcal{F}, P)$ are equal almost surely if $P(X \neq Y) = 0$. This is equivalent to the condition that the $L^2$ distance between $X$ and $Y$ is zero, i.e., $E((X-Y)^2) = 0$.

        Parameters
        ----------
        first : RandomVariable
            The first random variable.
        second : RandomVariable
            The second random variable.
        tol : float, default=1e-8
            The tolerance below which the L2 distance is deemed to be zero.

        Returns
        -------
        equal_as : bool
            True if the random variables are equal almost surely; False otherwise.
        """
        if first not in self or second not in self:
            raise ValueError("Both random variables must be in the L2-space.")

        return self.metric(first, second) < tol

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
