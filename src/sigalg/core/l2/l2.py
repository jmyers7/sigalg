"""A class representing an L2-space of random variables defined on a given probability space."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...core.measures.probability_measure import ProbabilityMeasureMethods

if TYPE_CHECKING:
    from ..spaces.probability_space import ProbabilitySpace
    from ..spaces.sample_space import SampleSpace
    from ...core.measures.probability_measure import ProbabilityMeasure
    from ...core.functions.random_variable import RandomVariable
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra


class L2(ProbabilityMeasureMethods):
    r"""A class representing an L2-space of random variables defined on a given probability space.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space on which the L2-space is defined.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra on which the L2-space is defined. If `None`, the power-set sigma-algebra on the sample space is used.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure on which the L2-space is defined. If `None`, the uniform probability measure on the sample space is used.
    name : Hashable | None, default="H"
        The name of the L2-space.

    Raises
    ------
    TypeError
        If `sample_space` is not an instance of `SampleSpace`, or if `sig_alg` is not an instance of `SigmaAlgebra` or `None`, or if `prob_measure` is not an instance of `ProbabilityMeasure` or `None`. If `sig_alg` is not `None`, it must be defined on the same sample space as the L2-space. If `prob_measure` is not `None`, it must be defined on the same sample space as the L2-space.

    Examples
    --------
    >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=4)
    >>> F = SigmaAlgebra(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 0,
    ...         3: 1,
    ...     },
    ... )
    >>> P = ProbabilityMeasure(
    ...     sig_alg=F,
    ...     mapping={
    ...         0: 0.65,
    ...         1: 0.35,
    ...     },
    ... )
    >>> H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
    >>> print(H)  # doctest: +NORMALIZE_WHITESPACE
    H = L2(Omega, F, P)
    ===================
    <BLANKLINE>
    * Sample space 'Omega':
     sample
          0
          1
          2
          3
    <BLANKLINE>
    * Sigma algebra 'F':
            atom_ID
    sample
    0             0
    1             1
    2             0
    3             1
    <BLANKLINE>
    * Probability measure 'P':
            probability
    atom_ID
    0               0.65
    1               0.35

    Notes
    -----
    Let $(\Omega,\mathcal{F},P)$ be a probability space. We define $L^2(\Omega,\mathcal{F},P)$ to be the set of all $\mathcal{F}$-measurable random variables $X: \Omega \to \mathbb{R}$ such that

    $$
    \int_\Omega X^2 \, dP < \infty. \tag{$\ast$}
    $$

    When the sample space $\Omega$ and probability measure $P$ are fixed and understood, we will write $L^2(\mathcal{F})$ in place of $L^2(\Omega,\mathcal{F},P)$. We agree to identify two random variables $X$ and $Y$ in $L^2(\mathcal{F})$ provided that they are equal almost surely, i.e., the set of sample points on which they are not equal has probability $0$.

    The set $L^2(\mathcal{F})$ is a (real) vector space under the standard point-wise operators. Even more, it is also a Hilbert space when equipped with the inner product

    $$
    \langle X, Y \rangle \stackrel{\text{def}}{=} \int_\Omega XY \, dP,
    $$

    for $X,Y\in L^2(\mathcal{F})$.

    In the case that $\Omega$ is finite (as it always is, in SigAlg), the condition $(\ast)$ is automatically satisfied, so $L^2(\mathcal{F})$ is simply the vector space of all $\mathcal{F}$-measurable random variables.
    """

    _properties = [
        "_basis",
        "_basis_df",
    ]

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        name: Hashable = "H",
    ) -> None:
        from ..spaces.probability_space import ProbabilitySpace

        if not isinstance(name, Hashable):
            raise TypeError("Name must be a Hashable.")

        self._prob_space = ProbabilitySpace(
            sample_space=sample_space,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
        )
        self._name = name
        self._initialize_property_caches()

    def _initialize_property_caches(self, exceptions: set | None = None) -> None:
        if exceptions is None:
            exceptions = set()
        for property in set(self._properties) - exceptions:
            setattr(self, property, None)

    # --------------------- properties --------------------- #

    @property
    def basis_df(self) -> pd.DataFrame | None:
        """Return a `pd.DataFrame` whose columns are the orthonormal basis vectors of the L2-space.

        Returns
        -------
        basis_df : pd.DataFrame | None
            A `pd.DataFrame` whose columns are the orthonormal basis vectors of the L2-space, or `None` if the underlying probability space is empty.

        Examples
        --------
        >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.55,
        ...         2: 0.45,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> print(H.basis_df)  # doctest: +NORMALIZE_WHITESPACE
                    1         2
        sample
        0      0.0000  0.000000
        1      1.3484  0.000000
        2      0.0000  1.490712
        3      0.0000  1.490712
        """
        if (
            self._basis_df is None
            and self.sig_alg is not None
            and self.prob_measure is not None
        ):
            if isinstance(self.sig_alg.data, pd.DataFrame):
                sig_alg_data = self.sig_alg.data.apply(tuple, axis=1)
            else:
                sig_alg_data = self.sig_alg.data

            self._basis_df = pd.get_dummies(sig_alg_data).astype(int)

            prob_measure_data = self.prob_measure.data.reindex(self._basis_df.columns)

            self._basis_df = (
                self._basis_df.mul(1 / prob_measure_data**0.5, axis=1)
            ).dropna(axis=1)
        return self._basis_df

    @property
    def basis(self) -> dict[Hashable, RandomVariable] | None:
        r"""Return an orthonormal basis of the L2-space.

        See the Notes section below for the mathematical details.

        Returns
        -------
        basis : dict[Hashable, RandomVariable] | None
            A dictionary mapping the atom ID to the corresponding basis vector of the L2-space.

        Examples
        --------
        >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
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
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.0,  # atom with probability 0
        ...         1: 0.75,
        ...         2: 0.25,
        ...     },
        ... )
        >>> H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> for atom_id, phi in H.basis.items():
        ...     print(f"Atom identifier: {atom_id}")
        ...     print(f"Basis function:\n{phi}\n")  # doctest: +NORMALIZE_WHITESPACE
        Atom identifier: 1
        Basis function:
        Random variable 'phi_1':
                phi_1
        sample
        0       0.000000
        1       0.000000
        2       1.154701
        3       1.154701
        4       0.000000
        5       0.000000
        <BLANKLINE>
        Atom identifier: 2
        Basis function:
        Random variable 'phi_2':
                phi_2
        sample
        0         0.0
        1         0.0
        2         0.0
        3         0.0
        4         2.0
        5         2.0

        Notes
        -----
        Let $(\Omega,\mathcal{F},P)$ be a probability space and set $H = L^2(\Omega, \mathcal{F}, P)$. In the case that $\Omega$ is finite, so that the $\sigma$-algebra $\mathcal{F}$ is determined by its set $\{A_i\}_{i\in I}$ of (finitely many) atoms, the vector space $H$ as an orthonormal basis given by the normalized indicator functions of the atoms of nonzero probability.

        Indeed, if we suppose $i\neq j$, then the product $I_{A_i}I_{A_j}$ of indicator functions is $0$ since $A_i$ and $A_j$ are disjoint. Thus, we have

        $$
        \langle I_{A_i}, I_{A_j} \rangle = \int_\Omega I_{A_i} I_{A_j} \, dP = 0,
        $$

        which proves that $\{I_{A_i}\}_{i\in I}$ is an orthogonal set.

        If $X$ is an $\mathcal{F}$-measurable random variable, then it must be constant on each atom $A_i$. Thus, we have

        $$
        X = \sum_{i\in I} x_i I_{A_i}
        $$

        for some real numbers $x_i$. If an atom $A_i$ has probability $0$, then the corresponding summand $x_i I_{A_i}$ may be dropped, which still yields an equality in the $L^2$-space $H$ since $X$ and the linear combination on the right-hand side are still equal almost surely.

        Thus, the set of indicator functions of atoms with nonzero probability form an orthogonal basis $H$. To obtain an orthonormal basis, we first compute the norms of the indicator functions:

        $$
        \|I_{A_i}\|^2 = \langle I_{A_i}, I_{A_i} \rangle = \int_\Omega I_{A_i}^2 \, dP = P(A_i).
        $$

        Thus, provided that $P(A_i)\neq 0$, we obtain a normalized basis function:

        $$
        \phi_i \stackrel{\text{def}}{=} \frac{I_{A_i}}{\sqrt{P(A_i)}}.
        $$

        The `basis` attribute contains the orthonormal basis $\{\phi_i\}$ indexed by the atoms with nonzero probability.
        """
        from ..functions.random_variable import RandomVariable

        if self._basis is None and self.basis_df is not None:
            self._basis = {}
            for atom_id, data in self.basis_df.items():
                name = f"phi_{atom_id}"
                self._basis[atom_id] = RandomVariable(
                    sample_space=self.sample_space,
                    sig_alg=self.sig_alg,
                    prob_measure=self.prob_measure,
                    mapping=data.rename(name),
                    name=name,
                )

        return self._basis

    @property
    def dim(self) -> int | None:
        """The dimension of the L2-space.

        Returns
        -------
        dim : int | None
            The dimension of the L2-space, or `None` if the basis is not defined.

        Examples
        --------
        >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
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
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.0,  # atom with probability 0
        ...         1: 0.75,
        ...         2: 0.25,
        ...     },
        ... )
        >>> H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> print(H.dim)
        2
        """
        return len(self.basis) if self.basis is not None else None

    @property
    def prob_space(self) -> ProbabilitySpace:
        """The underlying probability space on which the L2-space is defined.

        Returns
        -------
        prob_space : ProbabilitySpace
            The underlying probability space on which the L2-space is defined.

        Examples
        --------
        >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
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
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.0,  # atom with probability 0
        ...         1: 0.75,
        ...         2: 0.25,
        ...     },
        ... )
        >>> H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> print(H.prob_space)  # doctest: +NORMALIZE_WHITESPACE
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
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        4             2
        5             2
        <BLANKLINE>
        * Probability measure 'P':
                probability
        atom_ID
        0               0.00
        1               0.75
        2               0.25
        """
        return self._prob_space

    @property
    def sample_space(self) -> SampleSpace | None:
        """The underlying sample space on which the L2-space is defined.

        The `sample_space` parameter is settable. If the underlying probability space is not empty, the new sample space must contain the same number of sample points as the current sample space, and the sigma-algebra and probability measure will be updated to be defined on the new sample space with the same atom structure and probabilities as before. If the underlying probability space is empty, then setting the sample space will set the sigma-algebra to be the power set sigma-algebra on the new sample space, and the probability measure to be the uniform probability measure on that sigma-algebra.

        Returns
        -------
        sample_space : SampleSpace | None
            The sample space on which the L2-space is defined, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.55,
        ...         2: 0.45,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> print(H.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         sample
              0
              1
              2
              3
        >>> S = SampleSpace(["a", "b", "c", "d"], name="S")
        >>> H.sample_space = S
        >>> print(H.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (S, F, P)
        ===========================
        <BLANKLINE>
        * Sample space 'S':
        sample
            a
            b
            c
            d
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        a             0
        b             1
        c             2
        d             2
        <BLANKLINE>
        * Probability measure 'P':
                probability
        atom_ID
        0               0.00
        1               0.55
        2               0.45
        >>> K = L2(name="K")
        >>> K.sample_space = S
        >>> print(K.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (S, power_set, U)
        ===================================
        <BLANKLINE>
        * Sample space 'S':
        sample
             a
             b
             c
             d
        <BLANKLINE>
        * Sigma algebra 'power_set':
                atom_ID
        sample
        a             a
        b             b
        c             c
        d             d
        <BLANKLINE>
        * Probability measure 'U':
                probability
        sample
        a              0.25
        b              0.25
        c              0.25
        d              0.25
        """
        return self.prob_space.sample_space

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the underlying sample space on which the L2-space is defined.

        If the underlying probability space is not empty, the new sample space must contain the same number of sample points as the current sample space, and the sigma-algebra and probability measure will be updated to be defined on the new sample space with the same atom structure and probabilities as before. If the underlying probability space is empty, then setting the sample space will set the sigma-algebra to be the power set sigma-algebra on the new sample space, and the probability measure to be the uniform probability measure on that sigma-algebra.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space to set for the L2-space.

        Raises
        ------
        TypeError
            If `sample_space` is not an instance of `SampleSpace`.
        """
        from ..spaces.sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be an instance of SampleSpace.")

        self.prob_space.sample_space = sample_space
        self._initialize_property_caches()

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """The underlying sigma-algebra on which the L2-space is defined.

        The `sig_alg` parameter is settable. If the underlying probability space is not empty, the new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra, and the probability measure will be updated to be the restriction of the current probability measure to the new sigma-algebra. If the underlying probability space is empty, then setting the sigma-algebra will set the sample space to be the sample space of the new sigma-algebra, and the probability measure to be the uniform probability measure on the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on which the L2-space is defined, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.1,
        ...         1: 0.45,
        ...         2: 0.45,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> print(H.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
               atom_ID
        sample
        0            0
        1            1
        2            2
        3            2
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> H.sig_alg = G
        >>> print(H.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, P|G)
        =================================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'P|G':
                probability
        atom_ID
        0               0.55
        1               0.45
        >>> K = L2(name="K")
        >>> K.sig_alg = F
        >>> print(K.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, U)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'F':
               atom_ID
        sample
        0            0
        1            1
        2            2
        3            2
        <BLANKLINE>
        * Probability measure 'U':
                 probability
        atom_ID
        0           0.333333
        1           0.333333
        2           0.333333
        """
        return self.prob_space.sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra on which the L2-space is defined.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra to set for the L2-space.

        Raises
        ------
        TypeError
            If `sig_alg` is not an instance of `SigmaAlgebra`.
        """
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")

        self.prob_space.sig_alg = sig_alg
        self._initialize_property_caches()

    @property
    def prob_measure(self) -> ProbabilityMeasure | None:
        """The underlying probability measure on which the L2-space is defined.

        The `prob_measure` parameter is settable. If the underlying probability space is not empty, the new probability measure must be defined on a sub-sigma-algebra of the current sigma-algebra. The sigma-algebra will be updated to be the sigma-algebra of the new probability measure. If the underlying probability space is empty, setting the probability measure will set the sample space to be the sample space of the new probability measure, and the sigma-algebra to be the sigma-algebra of the new probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure | None
            The probability measure on which the L2-space is defined, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import L2, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.1,
        ...         1: 0.45,
        ...         2: 0.45,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> print(H.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                 probability
        atom_ID
        0               0.10
        1               0.45
        2               0.45
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> Q = ProbabilityMeasure(
        ...     sig_alg=G,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.75,
        ...     },
        ...     name="Q",
        ... )
        >>> H.prob_measure = Q
        >>> print(H.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, Q)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
               atom_ID
        sample
        0            0
        1            0
        2            1
        3            1
        <BLANKLINE>
        * Probability measure 'Q':
                 probability
        atom_ID
        0               0.25
        1               0.75
        >>> K = L2(name="K")
        >>> K.prob_measure = P
        >>> print(K.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
             0
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'F':
               atom_ID
        sample
        0            0
        1            1
        2            2
        3            2
        <BLANKLINE>
        * Probability measure 'P':
                 probability
        atom_ID
        0               0.10
        1               0.45
        2               0.45
        """
        return self.prob_space.prob_measure

    @prob_measure.setter
    def prob_measure(self, prob_measure: ProbabilityMeasure) -> None:
        """Set the probability measure on which the L2-space is defined.

        If the underlying probability space is not empty, the new probability measure must be defined on a sub-sigma-algebra of the current sigma-algebra. The sigma-algebra will be updated to be the sigma-algebra of the new probability measure. If the underlying probability space is empty, setting the probability measure will set the sample space to be the sample space of the new probability measure, and the sigma-algebra to be the sigma-algebra of the new probability measure.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure
            The probability measure to set for the L2-space.

        Raises
        ------
        TypeError
            If `prob_measure` is not an instance of `ProbabilityMeasure`.

        """
        from ...core.measures.probability_measure import ProbabilityMeasure

        if not isinstance(prob_measure, ProbabilityMeasure):
            raise TypeError("prob_measure must be an instance of ProbabilityMeasure.")

        self.prob_space.prob_measure = prob_measure
        self._initialize_property_caches()

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
        """Set the name of the L2-space.

        Parameters
        ----------
        name : Hashable
            The name to set for the L2-space.

        Raises
        ------
        TypeError
            If `name` is not a Hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("Name must be a Hashable.")
        self._name = name

    # --------------------- methods --------------------- #

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
        >>> from sigalg.core import (
        ...     L2,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.7,
        ...         1: 0.3,
        ...     },
        ... )
        >>> H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> phi_0, phi_1 = H.basis.values()
        >>> print(phi_0 in H)
        True
        >>> X = RandomVariable(
        ...     Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...     },
        ... )
        >>> print(X in H)
        False
        """
        from ...core.functions.random_variable import RandomVariable

        if not isinstance(rv, RandomVariable):
            raise TypeError("rv must be an instance of RandomVariable.")
        if rv.sample_space != self.sample_space:
            raise ValueError("The domain of rv must match the sample space.")
        return rv.is_measurable(self.sig_alg)

    # --------------------- Hilbert space methods --------------------- #

    def fourier_coefficients(self, rv: RandomVariable) -> dict[Hashable, Real]:
        r"""Compute the Fourier coefficients of a random variable with respect to the orthonormal basis of the L2-space contained in the `basis` attribute.

        See the Notes section below for the mathematical details.

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
        >>> from sigalg.core import (
        ...     L2,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: "b",
        ...         1: "b",
        ...         2: "a",
        ...         3: "a",
        ...         4: "c",
        ...         5: "c",
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         "b": 0.75,
        ...         "a": 0.0,
        ...         "c": 0.25,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> X = RandomVariable(
        ...     *H.prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 3,
        ...         3: 3,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ... )
        >>> c = H.fourier_coefficients(rv=X)
        >>> phi = H.basis
        >>> I = c.keys()
        >>> X_fourier = sum(c[i] * phi[i] for i in I).with_name("X_fourier")
        >>> print(X_fourier)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_fourier':
                X_fourier
        sample
        0            -1.0
        1            -1.0
        2             0.0
        3             0.0
        4             1.0
        5             1.0
        >>> print(P.equal_almost_surely(X, X_fourier))
        True

        Notes
        -----
        Let $(\Omega,\mathcal{F},P)$ be a probability space and set $H = L^2(\Omega, \mathcal{F},P)$. Provided that $\Omega$ is finite (as it always is, in SigAlg), the vector space $H$ has an orthonormal basis $\{\phi_i\}_{i\in I}$ consisting of the normalized indicator functions of the atoms of $\mathcal{F}$ of nonzero probability. Thus, given a random variable $X\in H$, we have its *generalized Fourier expansion*:

        $$
        X = \sum_{i\in I} \langle X, \phi_i \rangle \phi_i.
        $$

        The coefficients $c_i = \langle X,\phi_i \rangle$ are called the *Fourier coefficients* of $X$.
        """
        if rv not in self:
            raise ValueError("The random variable must be in the L2-space.")

        rv_times_indicators = self.sig_alg.atom_indicator_df.mul(
            rv.data, axis=0
        ).drop_duplicates()
        prob_measure_data = self.prob_measure.data.reindex(rv_times_indicators.columns)
        coefficients_series = rv_times_indicators.mul(
            prob_measure_data**0.5, axis=1
        ).sum()
        coefficients_series = coefficients_series[coefficients_series.abs() > 1e-10]

        return coefficients_series.to_dict()

    def inner(self, first: RandomVariable, second: RandomVariable) -> Real:
        """Compute the inner product of two random variables.

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
        >>> from sigalg.core import (
        ...     L2,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: "b",
        ...         1: "b",
        ...         2: "a",
        ...         3: "a",
        ...         4: "c",
        ...         5: "c",
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         "b": 0.75,
        ...         "a": 0.0,
        ...         "c": 0.25,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> X = RandomVariable(
        ...     *H.prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 3,
        ...         3: 3,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ... )
        >>> phi = H.basis
        >>> I = phi.keys()
        >>> X_fourier = sum(H.inner(X, phi[i]) * phi[i] for i in I).with_name("X_fourier")
        >>> print(X_fourier)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_fourier':
               X_fourier
        sample
        0           -1.0
        1           -1.0
        2            0.0
        3            0.0
        4            1.0
        5            1.0
        >>> print(P.equal_almost_surely(X, X_fourier))
        True
        """
        if first not in self or second not in self:
            raise ValueError("Both random variables must be in the L2-space.")
        return self.prob_measure.integrate(rv=first * second)

    def norm(self, X: RandomVariable) -> Real:
        """Compute the norm of a random variable.

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
        >>> from sigalg.core import (
        ...     L2,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
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
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.75,
        ...         2: 0.25,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> indicators = [RandomVariable.indicator_of(A) for A in F.to_atoms]
        >>> for i, I in enumerate(indicators):
        ...     norm = H.norm(I)
        ...     print(f"P(A_{i}) = {round(norm**2, 2)}")
        P(A_0) = 0.0
        P(A_1) = 0.75
        P(A_2) = 0.25
        """
        if X not in self:
            raise ValueError("The random variable must be in the L2-space.")
        return self.prob_measure.integrate(rv=X**2) ** 0.5

    def metric(self, first: RandomVariable, second: RandomVariable) -> Real:
        r"""Compute the distance between two random variables.

        See the Notes for the mathematical details.

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

        Examples
        --------
        >>> from sigalg.core import (
        ...     L2,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
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
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.6,
        ...         2: 0.15,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> # Given a sub-sigma-algebra G of F and a random variable X in L2(Omega, F, P), the conditional expectation E(X|G) minimizes the squared distance from X to the subspace of G-measurable random variables
        >>> X = RandomVariable(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
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
        ...     sample_space=Omega,
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
        >>> E = X.expectation(sig_alg=G)
        >>> print(E)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(X|G)':
                  E(X|G)
        sample
        0       1.117647
        1       1.117647
        2       1.117647
        3       1.117647
        4       1.000000
        5       1.000000
        >>> squared_distance = H.metric(X, E)
        >>> print(round(squared_distance, 2))
        1.26
        >>> # Check that the squared distance between X and the expectation is less than the squared distance from X to another G-measurable random variable Y
        >>> Y = RandomVariable(
        ...     sample_space=Omega,
        ...     sig_alg=G,
        ...     prob_measure=P.restrict_to(G),
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
        d(x,y) = \|x - y \|,
        $$

        for $x,y\in H$. In particular, if $H$ is a vector space of the form $L^2(\Omega,\mathcal{F},P)$, then the induced metric is given by

        $$
        d(X,Y) = \|X-Y\| = \sqrt{\int_\Omega (X-Y)^2 \, dP},
        $$

        for $X,Y\in L^2(\Omega, \mathcal{F},P)$.
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

        See the Notes section below for the mathematical details.

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
            The coefficients of the projection of `rv` onto the subspace spanned by the random variables in `subspace`. See the Notes section below for the mathematical details.
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
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra.power_set(Omega, name="F")
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.4,
        ...         2: 0.2,
        ...         3: 0.2,
        ...     },
        ... )
        >>> H = L2(Omega, F, P)
        >>> # For a quadratic regression example, we will project a random variable Y onto the subspace spanned by 1, X, and X^2
        >>> one = RandomVariable.from_constant(sample_space=Omega, constant=1, name="one")
        >>> X = RandomVariable(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 2.0,
        ...         1: 3.0,
        ...         2: 5.0,
        ...         3: 7.0,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 3.0,
        ...         2: 2.0,
        ...         3: 4.0,
        ...     },
        ...     name="Y",
        ... )
        >>> # Project Y onto the subspace spanned by 1, X, and X^2
        >>> proj, c, dim = H.proj(rv=Y, subspace=[one, X, X**2])
        >>> print(proj)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y_proj':
                 Y_proj
        sample
        0      1.812609
        1      2.238179
        2      3.015762
        3      3.695271
        >>> expected_proj = sum([c[k] * X**k for k in range(dim)]).with_name("expected_proj")
        >>> print(expected_proj)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'expected_proj':
               expected_proj
        sample
        0           1.812609
        1           2.238179
        2           3.015762
        3           3.695271

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space, set $H = L^2(\Omega, \mathcal{F},P)$, and suppose for simplicitly that $\Omega$ is finite, so that $H$ is finite-dimensional. Suppose $Y$ is a random variable in $H$ and that $\{X_1,X_2,\ldots,X_n\} \subset H$ spans a subspace $V$ of $H$. The *orthogonal projection* of $Y$ onto $V$ is the unique random variable $\widehat{Y}\in V$ such that

        $$
        \|Y - \widehat{Y}\| \leq \|Y - Z\|,
        $$

        for all $Z\in V$. The existence and uniqueness of $\widehat{Y}$ is a consequence of the Projection Theorem for Hilbert spaces.

        In applications, one computes $\widehat{Y}$ by identifying $\widehat{Y}$ as the global minimizer of the objective function

        $$
        f: \mathbb{R}^n \to \mathbb{R}, \quad f(c) = \frac{1}{2} \left \|\sum_{j=1}^n c_j X_j - Y \right\|^2.
        $$

        The first (Fréchet) derivative $Df(c)$ of $f$ at $c$ is given by

        $$
        Df(c)h = \sum_{j=1}^n \left[ \sum_{k=1}^n c_k \langle X_j,X_j \rangle - \langle X_j, Y \rangle \right]h_j,
        $$

        for $h\in \mathbb{R}^n$. At the minimizer $\widehat{Y} = \sum_{k=1}^n c_k X_k$, we must have $Df(c)=0$, which yields the linear system of equations

        $$
        \begin{bmatrix}
        \langle X_1, X_1 \rangle & \cdots & \langle X_1, X_n \rangle \\
        \vdots & \ddots & \vdots \\
        \langle X_n, X_1 \rangle & \cdots & \langle X_n, X_n \rangle \\
        \end{bmatrix} \begin{bmatrix} c_1 \\ \vdots \\ c_n \end{bmatrix} =
        \begin{bmatrix} \langle X_1, Y \rangle \\ \vdots \\ \langle X_n,Y \rangle \end{bmatrix},
        $$

        for the unknown $c$. The coefficient matrix on the left is the *Gram matrix* $G$ of the random variables $X_1,\ldots,X_n$ as vectors in the Hilbert space $H$. Provided that $G$ is invertible (which is equivalent to linear independence of the $X_j$'s in $H$), there is a unique solution $c$. Otherwise, if $G$ is not invertible, then there are infinitely many choices for $c$. In this case, the method `proj` returns the $c$ for which $\sum_{k=1}^n c_k^2$ is minimum.

        Finally, for a solution $c$ for which $\widehat{Y} = \sum_{k=1}^n c_k X_k$, note that the linear system above can equivalently be expressed as the system

        $$
        \langle X_1, \widehat{Y} - Y \rangle = \cdots = \langle X_n, \widehat{Y}-Y \rangle = 0.
        $$

        These are called the *normal equations*, which confirm that $\widehat{Y}$ really is the orthogonal projection of $Y$ onto the subspace spanned by the $X_j$'s.
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
        c, _, dim, _ = np.linalg.lstsq(A, rv_vec, rcond=None)

        proj = sum([c[k] * subspace_rv for k, subspace_rv in enumerate(subspace)])
        name = f"{rv.name}_proj" if rv.name is not None else "proj"

        return proj.with_name(name), c, dim

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
            f"{self.sample_space.name}, "
            f"{self.sig_alg.name}, "
            f"{self.prob_measure.name})"
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
            f"{self.sample_space.name}, "
            f"{self.sig_alg.name}, "
            f"{self.prob_measure.name})"
        )
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.sample_space)
            + "\n\n* "
            + repr(self.sig_alg)
            + "\n\n* "
            + repr(self.prob_measure)
        )
