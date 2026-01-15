"""L2-space module."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.base.probability_space import ProbabilitySpace
    from ...core.base.sample_space import SampleSpace
    from ...core.probability_measures.probability_measure import ProbabilityMeasure
    from ...core.random_objects.random_variable import RandomVariable
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra


class L2:
    """A class representing the L2-space of random variables defined on a given probability space."""

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
        name: Hashable | None = "H",
    ) -> None:
        from ...core.probability_measures.probability_measure import ProbabilityMeasure
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

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
        """Get the vector space basis of the L2-space.

        The basis vectors of the L2-space are the indicator functions of the atoms in the sigma algebra with nonzero probability, each scaled by the reciprocal of the square root of the atom's probability.

        Parameters
        ----------
        tol : float, default=1e-8
            The tolerance below which an atom is deemed to have probability 0.

        Returns
        -------
        basis : dict[str, RandomVariable]
            A dictionary mapping the name of each basis vector to the corresponding basis vector of the L2-space.

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
        >>> V_0, V_1 = H.basis.values()
        >>> V_0 # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V_0':
                    V_0
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
        {'V_0': Random variable 'V_0':
                V_0
        sample
        0       1.0
        1       1.0
        2       0.0}
        """
        from ...core.random_objects.random_variable import RandomVariable

        tol = 1e-8

        if self._basis is None:
            self._basis = {}
            for idx, event in enumerate(self.sigma_algebra.to_atoms()):
                event_prob = self.probability_measure(event)
                if event_prob > tol:
                    name = f"V_{event.name}" if event.name is not None else f"V_{idx}"
                    basis_vec = (
                        RandomVariable.indicator_of(event) / (event_prob**0.5)
                    ).with_name(name)
                    self._basis[name] = basis_vec
        return self._basis

    @property
    def probability_space(self) -> ProbabilitySpace:
        """The probability space on which the L2-space is defined.

        Returns
        -------
        prob_space : ProbabilitySpace
             The probability space on which the L2-space is defined.
        """
        from ...core.base.probability_space import ProbabilitySpace

        if self._probability_space is None:
            self._probability_space = ProbabilitySpace(
                sample_space=self._sample_space,
                sigma_algebra=self._sigma_algebra,
                probability_measure=self._probability_measure,
            )
        return self._probability_space

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

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        """The probability measure on which the L2-space is defined.

        Returns
        -------
        probability_measure : ProbabilityMeasure
            The probability measure on which the L2-space is defined.
        """
        return self._probability_measure

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
    def name(self, name: str) -> None:
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

    def distance(self, first: RandomVariable, second: RandomVariable) -> Real:
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
