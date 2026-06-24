"""A class representing a probability measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import dirichlet

from ..base.multivariate_function import MultivariateFunction
from ..random_objects.operators import OperatorsMethods

if TYPE_CHECKING:
    from ...validation.mapping_validator import MappingLike
    from ..base.event import Event
    from ..base.sample_space import SampleSpace
    from ..random_objects.random_vector import RandomVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class ProbabilityMeasure(MultivariateFunction, OperatorsMethods):
    r"""A class representing a probability measure on a sigma-algebra.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra on which the probability measure is defined.
    name : Hashable, default="P"
        A name for the probability measure.

    Raises
    ------
    TypeError
        If `sig_alg` is not a `SigmaAlgebra` instance.

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> F = SigmaAlgebra(
    ...    sample_space=Omega,
    ...    mapping={
    ...        0: 0,
    ...        1: 0,
    ...        2: 1,
    ...    },
    ... )
    >>> P = ProbabilityMeasure.on(sig_alg=F, mapping={0: 0.2, 1: 0.8})
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
          probability
    atom
    0             0.2
    1             0.8
    >>> Q = ProbabilityMeasure.on(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0.1,
    ...         1: 0.3,
    ...         2: 0.6,
    ...     },
    ...     name="Q",
    ... )
    >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'Q':
            probability
    sample
    0               0.1
    1               0.3
    2               0.6

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be a measurable space consisting of a $\sigma$-algebra $\mathcal{F}$ on a set $\Omega$. A *probability measure* $P$ is a countably additive function $P: \mathcal{F} \to [0,1]$ such that $P(\Omega) = 1$. Here, *countable additivity* means that

    $$
    P \left( \bigcup_{k=1}^\infty A_k \right) = \sum_{k=1}^\infty P(A_k)
    $$

    for all collections $\{A_k\}_{k=1}^\infty$ of pairwise disjoint measurable sets. If $\Omega$ is finite (as it always is, in SigAlg), then $P$ needs only to be finitely additive in order to be countably additive.

    If $\mathcal{F}$ is the power set of a finite set $\Omega$, then $P$ is completely determined by its values on the finitely many singleton sets $\{\omega\}$ for $\omega \in \Omega$. In this case, we define

    $$
    P(\omega) \stackrel{\text{def}}{=} P(\{\omega\})
    $$

    for each $\omega\in \Omega$. From this viewpoint, $P:\Omega \to [0,1]$ functions as a *probability mass function* on $\Omega$.
    """

    # --------------------- constructors --------------------- #

    @classmethod
    def on(
        cls,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        mapping: MappingLike | Callable | None = None,
        name: Hashable = "P",
    ) -> SigmaAlgebra:
        """Pass."""
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        cls._validate_sig_alg_and_sample_space(
            sig_alg=sig_alg, sample_space=sample_space
        )

        space = sig_alg.atom_space if sig_alg is not None else sample_space
        prob_measure = cls(
            domain=space,
            mapping=mapping,
            output_name="probability",
            name=name,
            kind="probabilities",
        )

        if sig_alg is not None:
            prob_measure._sig_alg = sig_alg
        else:
            prob_measure._sig_alg = SigmaAlgebra.power_set(sample_space)

        return prob_measure

    @staticmethod
    def _validate_sig_alg_and_sample_space(
        sig_alg: SigmaAlgebra | None, sample_space: SampleSpace | None
    ) -> None:
        from ..base.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance, if given.")
        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance, if given.")
        if (sig_alg is not None and sample_space is not None) or (
            sig_alg is None and sample_space is None
        ):
            raise ValueError(
                "One of sig_alg or sample_space must be given, but not both."
            )

    @classmethod
    def uniform(
        cls,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        name: Hashable = "U",
    ) -> ProbabilityMeasure:
        r"""Create a uniform probability measure on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra on which to define the uniform probability measure.
        name : Hashable, default="U"
            A name for the probability measure.

        Raises
        ------
        ValueError
            If the sigma-algebra has no atoms.
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance, or if `name` is not hashable.

        Returns
        -------
        prob_measure: ProbabilityMeasure
            A uniform ProbabilityMeasure instance on the provided sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> U = ProbabilityMeasure.uniform(sig_alg=F)
        >>> print(U)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'U':
              probability
        atom
        0        0.333333
        1        0.333333
        2        0.333333
        >>> V = ProbabilityMeasure.uniform(sample_space=Omega, name="V")
        >>> print(V)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'V':
              probability
        sample
        0            0.25
        1            0.25
        2            0.25
        3            0.25

        Notes
        -----
        Let $(\Omega,\mathcal{F})$ be an event space where $\Omega$ is finite, and suppose that $\mathcal{F}$ has $n$ atoms. The *uniform probability measure* on $\mathcal{F}$ is the probability measure $P$ defined by

        $$
        P(A) = \frac{1}{n},
        $$

        for all atoms $A\in \mathcal{F}$.
        """
        cls._validate_sig_alg_and_sample_space(
            sig_alg=sig_alg, sample_space=sample_space
        )

        space = sig_alg.atom_ids if sig_alg is not None else sample_space

        n = len(space)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on sigma-algebra with no atoms."
            )
        probs = dict.fromkeys(space, 1.0 / n)

        return cls.on(
            sig_alg=sig_alg, sample_space=sample_space, mapping=probs, name=name
        )

    @classmethod
    def from_rand(
        cls,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "P",
    ) -> ProbabilityMeasure:
        """Generate a random probability measure.

        This method generates a random probability measure on the sample space by sampling from a Dirichlet distribution with all concentration parameters equal to 1.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra on which this probability measure is defined.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.
        name : Hashable, default="P"
            The name of this probability measure.

        Raises
        ------
        TypeError
            If `random_state` is not an integer, Generator, or `None`.

        Returns
        -------
        self : ProbabilityMeasure
            A probability measure with randomly generated probabilities.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(seed=42)
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure.from_rand(sig_alg=F, random_state=rng)
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
              probability
        atom
        0        0.507174
        1        0.492826
        >>> Q = ProbabilityMeasure.from_rand(sample_space=Omega, random_state=rng, name="Q")
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        sample
        0          0.866873
        1          0.101707
        2          0.031420
        """
        cls._validate_sig_alg_and_sample_space(
            sig_alg=sig_alg, sample_space=sample_space
        )
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")

        space = sig_alg.atom_ids if sig_alg is not None else sample_space

        probs_arr = dirichlet.rvs(
            alpha=[
                1,
            ]
            * len(space),
            random_state=random_state,
        )
        mapping = dict(zip(space, probs_arr[0]))

        return cls.on(
            sig_alg=sig_alg, sample_space=sample_space, mapping=mapping, name=name
        )

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra:
        """Get the sigma-algebra on which the probability measure is defined.

        The `sig_alg` property is settable. The new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra. The probability measure will be restricted to the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on which the probability measure is defined.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure.on(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.3,
        ...         2: 0.5,
        ...     },
        ... )
        >>> print(P.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
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
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> P.sig_alg = G
        >>> print(P.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
               atom_ID
        sample
        0            0
        1            1
        2            1
        3            1
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
              probability
        atom
        0             0.2
        1             0.8
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra on which the probability measure is defined.

        The new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra. The probability measure will be restricted to the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra on which the probability measure is defined.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance.
        ValueError
            If `sig_alg` is not a sub-sigma-algebra of the current sigma-algebra, or if the current sigma-algebra has no data.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")

        if not sig_alg <= self._sig_alg:
            raise ValueError(
                "sig_alg must be a sub-sigma-algebra of the current sigma-algebra."
            )

        if sig_alg.atom_id_to_sample_ids is not None:
            mapping = {
                atom_id: self(atom)
                for atom_id, atom in sig_alg.atom_id_to_sample_ids.items()
            }
        else:
            raise ValueError("Cannot set sig_alg for a sigma-algebra with no data.")

        new = ProbabilityMeasure.on(sig_alg=sig_alg, mapping=mapping, name=self.name)

        self.__dict__.update(new.__dict__)

    @property
    def sample_space(self) -> SampleSpace:
        """Get the sample space of the probability measure.

        The `sample_space` property is settable. The new sample space must contain the same number of sample points. If the probability measure does not have a sigma-algebra, the sample space cannot be set.

        Returns
        -------
        sample_space : SampleSpace
            The sample space on which the probability measure is defined.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
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
        >>> P = ProbabilityMeasure.on(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.3,
        ...         2: 0.5,
        ...     },
        ... )
        >>> print(P.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         sample
              0
              1
              2
              3
        >>> S = SampleSpace(["a", "b", "c", "d"], name="S")
        >>> P.sample_space = S
        >>> print(P.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'S':
         sample
              a
              b
              c
              d
        """
        return self._sig_alg._sample_space if self.sig_alg is not None else None

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the sample space of the probability measure.

        The new sample space must contain the same number of sample points.

        Parameters
        ----------
        sample_space : SampleSpace
            The new sample space on which the probability measure is defined.

        Raises
        ------
        ValueError
            If the probability measure does not have a sigma-algebra.
        """
        self.sig_alg.sample_space = sample_space

    # TODO: write unit tests
    @property
    def probs(self) -> dict[Hashable, Real] | None:
        """Later."""
        return self.dict

    # --------------------- probability methods --------------------- #

    def conditional_probability(self, event: Event, given: Event) -> Real:
        r"""Compute the conditional probability P(A|B).

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : Event
            The event A.
        given : Event
            The event B.

        Raises
        ------
        ValueError
            If `event` or `given` are from a different sample space than this probability measure's sample space, or if P(B) = 0.

        Returns
        -------
        conditional_prob : Real
            The conditional probability P(A|B).

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=7)
        >>> F = SigmaAlgebra(
        ...    sample_space=Omega,
        ...    mapping={
        ...        0: 0,
        ...        1: 1,
        ...        2: 1,
        ...        3: 2,
        ...        4: 2,
        ...        5: 3,
        ...        6: 3,
        ...    },
        ... )
        >>> P = ProbabilityMeasure.on(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.1,
        ...         1: 0.25,
        ...         2: 0.25,
        ...         3: 0.4,
        ...     },
        ... )
        >>> A = F.get_event([1, 2, 3, 4])
        >>> B = F.get_event([3, 4, 5, 6])
        >>> print(P.conditional_probability(event=A, given=B))
        0.3846153846153846
        >>> # Check
        >>> print(P(A & B) / P(B))
        0.3846153846153846

        Notes
        -----
        Let $A$ and $B$ be two events in a probability space $(\Omega, \mathcal{F}, P)$ with $P(B) > 0$. The *conditional probability* of $A$ given $B$, denoted P(A\mid B)$, is defined as

        $$
        P(A\mid B) = \frac{P(A \cap B)}{P(B)}.
        $$
        """
        if not self.sig_alg.is_power_set and event.sig_alg != self.sig_alg:
            raise ValueError(
                "Event is not measurable with respect to this probability measure's sigma-algebra"
            )
        if not self.sig_alg.is_power_set and given.sig_alg != self.sig_alg:
            raise ValueError(
                "Event is not measurable with respect to this probability measure's sigma-algebra"
            )
        prob_given = self(given)
        if prob_given < 1e-10:
            raise ValueError(
                "Cannot compute conditional probability given event with probability 0."
            )
        return self(event & given) / prob_given

    def are_independent(
        self,
        event1: Event | None = None,
        event2: Event | None = None,
        rv1: RandomVector | None = None,
        rv2: RandomVector | None = None,
        algebra1: SigmaAlgebra | None = None,
        algebra2: SigmaAlgebra | None = None,
        tol: Real = 1e-8,
    ) -> bool:
        r"""Check if two events, two random vectors, or two sigma-algebras are independent.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event1 : Event | None, default=None
            The first event.
        event2 : Event | None, default=None
            The second event.
        rv1: RandomVector | None, default=None
            The first random vector.
        rv2: RandomVector | None, default=None
            The second random vector.
        algebra1 : SigmaAlgebra | None, default=None
            The first sigma-algebra.
        algebra2 : SigmaAlgebra | None, default=None
            The second sigma-algebra.
        tol : Real, default=1e-10
            The numerical tolerance for checking independence.

        Raises
        ------
        ValueError
            If neither events, random vectors, nor sigma-algebras are provided, or if two of these types are provided, or if the provided objects are from a different sample space.
        TypeError
            If the provided objects are not of the correct type.

        Returns
        -------
        is_independent : bool
            `True` if the events, random vectors, or sigma-algebras are independent, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> Omega = SampleSpace.from_product(
        ...     indices1=[0, 1], indices2=[0, 1], variable_names=["flip_1", "flip_2"]
        ... )
        >>> P = ProbabilityMeasure.on(
        ...     sample_space=Omega,
        ...     mapping=lambda *, flip_1, flip_2: (
        ...         0.75 ** (flip_1 + flip_2) * 0.25 ** (2 - flip_1 - flip_2)
        ...     ),
        ... )
        >>> prob_space = ProbabilitySpace(sample_space=Omega, prob_measure=P)
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, power_set, P)
        =======================================
        <BLANKLINE>
        * Sample space 'Omega':
         flip_1  flip_2
             0       0
             0       1
             1       0
             1       1
        <BLANKLINE>
        * Sigma algebra 'power_set':
                      atom_ID
        flip_1 flip_2
        0      0       (0, 0)
               1       (0, 1)
        1      0       (1, 0)
               1       (1, 1)
        <BLANKLINE>
        * Probability measure 'P':
                       probability
        flip_1 flip_2
        0      0            0.0625
               1            0.1875
        1      0            0.1875
               1            0.5625
        >>> A = prob_space.get_event(
        ...     [(0, 0), (0, 1)],
        ...     name="A",
        ... )
        >>> B = prob_space.get_event(
        ...     [(0, 1), (1, 1)],
        ...     name="B",
        ... )
        >>> print(P.are_independent(A, B))
        True
        >>> X = RandomVector.from_identity(*prob_space, index=[1, 2])
        >>> X_1, X_2 = X
        >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_1':
                       X_1
        flip_1 flip_2
        0      0         0
               1         0
        1      0         1
               1         1
        >>> print(X_2)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_2':
                       X_2
        flip_1 flip_2
        0      0         0
               1         1
        1      0         0
               1         1
        >>> print(P.are_independent(rv1=X_1, rv2=X_2))
        True
        >>> Y = (X_1 + X_2).with_name("Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                       Y
        flip_1 flip_2
        0      0       0
               1       1
        1      0       1
               1       2
        >>> print(P.are_independent(rv1=X_1, rv2=Y))
        False

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space, and let $\mathcal{G}$ and $\mathcal{H}$ be two sub-$\sigma$-algebras of $\mathcal{F}$. We say that $\mathcal{G}$ and $\mathcal{H}$ are *independent* if for every $G \in \mathcal{G}$ and $H \in \mathcal{H}$, we have

        $$
        P(G \cap H) = P(G) P(H).
        $$

        In the special case where $\mathcal{G} = \sigma(A)$ and $\mathcal{H} = \sigma(B)$ are the $\sigma$-algebras generated by two events $A$ and $B$ in $\mathcal{F}$, this reduces to the condition that

        $$
        P(A \cap B) = P(A) P(B),
        $$

        and we say that the events $A$ and $B$ are *independent*.
        """
        from ..base.event import Event
        from ..random_objects.random_vector import RandomVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        events_provided = event1 is not None and event2 is not None
        rvs_provided = rv1 is not None and rv2 is not None
        algebras_provided = algebra1 is not None and algebra2 is not None

        if sum((events_provided, rvs_provided, algebras_provided)) != 1:
            raise ValueError(
                "Must provide exactly one of the following pairs of arguments: (event1, event2), (rv1, rv2), or (algebra1, algebra2)."
            )

        if events_provided:
            if not isinstance(event1, Event) or not isinstance(event2, Event):
                raise TypeError("event1 and event2 must be Event instances.")

            if not self.sig_alg.is_power_set:
                for event in (event1, event2):
                    if event.sig_alg != self.sig_alg:
                        raise ValueError(
                            "Event is not measurable with respect to this probability measure's sigma-algebra"
                        )

            if abs(self(event1 & event2) - self(event1) * self(event2)) < tol:
                return True
            else:
                return False

        if rvs_provided or algebras_provided:
            if rvs_provided:
                if not isinstance(rv1, RandomVector) or not isinstance(
                    rv2, RandomVector
                ):
                    raise TypeError("rv1 and rv2 must be RandomVector instances.")
                if (
                    rv1.domain != self.sig_alg.sample_space
                    or rv2.domain != self.sig_alg.sample_space
                ):
                    raise ValueError(
                        "Random vectors must be from this probability measure's sample space."
                    )

                algebra1 = SigmaAlgebra.from_random_vector(rv1)
                algebra2 = SigmaAlgebra.from_random_vector(rv2)

            if not isinstance(algebra1, SigmaAlgebra) or not isinstance(
                algebra2, SigmaAlgebra
            ):
                raise TypeError("algebra1 and algebra2 must be SigmaAlgebra instances.")
            if not (algebra1 <= self.sig_alg and algebra2 <= self.sig_alg):
                raise ValueError(
                    "Both sigma-algebras must be sub-algebras of the probability measure's sigma-algebra"
                )

            for atom1 in algebra1.to_atoms:
                for atom2 in algebra2.to_atoms:
                    event1 = self.sig_alg.get_event(list(atom1), name=atom1.name)
                    event2 = self.sig_alg.get_event(list(atom2), name=atom2.name)
                    if not self.are_independent(event1=event1, event2=event2, tol=tol):
                        return False
            return True

    def almost_surely_equal(
        self,
        first: RandomVector,
        second: RandomVector,
        tol: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two random vectors are equal almost surely.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        first : RandomVector
            The first random vector.
        second : RandomVector
            The second random vector.
        tol : float, default=1e-8
            The tolerance below which a probability is considered to be zero for the purposes of this comparison.
        rtol : float, default=1e-5
            The relative tolerance for `np.isclose` when comparing the random vectors.
        atol : float, default=1e-8
            The absolute tolerance for `np.isclose` when comparing the random vectors.

        Raises
        ------
        TypeError
            If `first` or `second` are not `RandomVector` instances.
        ValueError
            If `first` or `second` are from a different sample space than this probability measure's sample space, or if they have different dimensions.

        Returns
        -------
        equal_as : bool
            True if the random vectors are equal almost surely; False otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> P = ProbabilityMeasure.on(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.4,
        ...         1: 0.6,
        ...         2: 0.0,
        ...     },
        ... )
        >>> # Test on random variables
        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 3.0,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 4.0,
        ...     },
        ...     name="Y",
        ... )
        >>> Z = RandomVariable(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 3.0,
        ...         2: 3.0,
        ...     },
        ...     name="Z",
        ... )
        >>> print(P.almost_surely_equal(X, Y))
        True
        >>> print(P.almost_surely_equal(X, Z))
        False
        >>> # Test on random vectors of dimension > 1
        >>> U = RandomVector(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 2),
        ...     },
        ...     name="U",
        ... )
        >>> V = RandomVector(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (-1, 4),
        ...     },
        ...     name="V",
        ... )
        >>> W = RandomVector(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (-1, 1),
        ...         2: (3, 2),
        ...     },
        ...     name="W",
        ... )
        >>> print(P.almost_surely_equal(U, V))
        True
        >>> print(P.almost_surely_equal(U, W))
        False

        Notes
        -----
        Two random vectors $X,Y:\Omega \to \mathbb{R}^d$ defined on a probability space $(\Omega, \mathcal{F}, P)$ are *equal almost surely* if

        $$
        P \left( \{\omega \in \Omega : X(\omega) \neq Y(\omega)\} \right) = 0.
        $$
        """
        from ..random_objects.random_variable import RandomVector

        if not isinstance(first, RandomVector) or not isinstance(second, RandomVector):
            raise TypeError("first and second must be RandomVector instances.")
        if first.dimension != second.dimension:
            raise ValueError("The random vectors must have the same dimension.")
        if (
            first.domain != self.sig_alg.sample_space
            or second.domain != self.sig_alg.sample_space
        ):
            raise ValueError(
                "Random vectors must be from this probability measure's sample space."
            )

        first_df = (
            pd.concat([self.sig_alg.data, first.data], axis=1)
            .drop_duplicates()
            .set_index("atom_ID")
        )
        second_df = (
            pd.concat([self.sig_alg.data, second.data], axis=1)
            .drop_duplicates()
            .set_index("atom_ID")
        )
        first_arr = first_df.to_numpy()
        second_arr = second_df.to_numpy()
        prob_arr = self.data.to_numpy()

        if first.dimension == 1:
            are_different = (
                ~np.isclose(first_arr, second_arr, rtol=rtol, atol=atol)
            ).squeeze()
        else:
            are_different = ~np.all(
                np.isclose(first_arr, second_arr, rtol=rtol, atol=atol), axis=1
            )

        prob_different = np.sum(are_different.astype(float) * prob_arr)

        return prob_different < tol

    def restrict_to(self, sig_alg: SigmaAlgebra) -> ProbabilityMeasure:
        """Restrict the probability measure to a sub-sigma-algebra and return the restricted measure as a new `ProbabilityMeasure` instance.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sub-sigma-algebra to which to restrict the probability measure.

        Returns
        -------
        restricted_measure : ProbabilityMeasure
            A new `ProbabilityMeasure` instance representing the restriction of this probability measure to `sig_alg`.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )
        >>> P = ProbabilityMeasure.on(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.5,
        ...         1: 0.3,
        ...         2: 0.2,
        ...     },
        ... )
        >>> P_G = P.restrict_to(sig_alg=G)
        >>> print(P_G)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_G':
              probability
        atom
        0             0.8
        1             0.2
        """
        mapping = self.data.copy()
        name = (
            f"{self.name}_{sig_alg.name}"
            if sig_alg.name is not None and self.name is not None
            else "restriction"
        )
        restriction = ProbabilityMeasure.on(
            sig_alg=self.sig_alg, mapping=mapping, name=name
        )
        restriction.sig_alg = sig_alg
        return restriction

    # --------------------- data access methods --------------------- #

    def __call__(self, *args, **kwargs):
        """Get the probability of an event.

        One may pass arguments in one of the following ways:

        * An `Event` instance as a (single) positional argument or a keyword argument named `event`.
        * A list of sample points as a (single) positional argument or a keyword argument named `event`. The list of sample points must correspond to a measurable event in the sigma-algebra of the probability measure.
        * A single sample point as a keyword argument named `sample_point`. The sample point must correspond to a measurable (singleton) event in the sigma-algebra of the probability measure.
        * An atom ID of the sigma-algebra as a keyword argument.

        This method calls the parent `__call__` method of the parent class `MultivariateFunction` and hence allows curried calls. See the docstring of the parent class for details.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Raises
        ------
        ValueError
            If the event is not measurable with respect to the sigma-algebra of the probability measure.

        Returns
        -------
        probability : Real
            The probability of the event.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (0, 2),
        ...         3: (2, 4),
        ...         4: (2, 4),
        ...         5: (2, 4),
        ...     },
        ...     variable_names=["F_0", "F_1"],
        ... )
        >>> P = ProbabilityMeasure.on(
        ...     sig_alg=F,
        ...     mapping={
        ...         (1, 2): 0.2,
        ...         (0, 2): 0.2,
        ...         (2, 4): 0.6,
        ...     },
        ... )
        >>> # Call on `Event` instances as positional or keyword arguments
        >>> A = F.get_event([0, 1, 2])
        >>> print(P(A))
        0.4
        >>> print(P(event=A))
        0.4
        >>> # Call on a list as a positional or keyword argument
        >>> print(P([0, 1, 2]))
        0.4
        >>> print(P(event=[0, 1, 2]))
        0.4
        >>> # Call on a sample point as a keyword argument
        >>> print(P(sample_point=2))
        0.2
        >>> print(P(F_0=0, F_1=2))
        0.2
        >>> # Evaluate the probability of an event using curried calls
        >>> print(P(F_0=0)(F_1=2))
        0.2
        >>> print(P(F_1=2)(F_0=0))
        0.2
        """
        from ..base.event import Event

        event = None
        if len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], Event):
                event = args[0]
            if isinstance(args[0], list):
                event = self.sig_alg.get_event(args[0])
            if isinstance(args[0], Hashable):
                event = self.sig_alg.get_event([args[0]])
        elif "event" in kwargs and len(kwargs) == 1 and len(args) == 0:
            if isinstance(kwargs["event"], Event):
                event = kwargs["event"]
            if isinstance(kwargs["event"], list):
                event = self.sig_alg.get_event(kwargs["event"])
        elif "sample_point" in kwargs and len(kwargs) == 1 and len(args) == 0:
            event = self.sig_alg.get_event([kwargs["sample_point"]])

        if event is not None and isinstance(event, Event):
            if not event.sig_alg <= self.sig_alg:
                raise ValueError(
                    "Event is not in the domain of the probability measure."
                )
            df = pd.concat([event.indicator.data, self.sig_alg.data], axis=1)
            if isinstance(self.sig_alg.data, pd.Series):
                index_name = self.sig_alg.data.name
            else:
                index_name = self.sig_alg.data.columns.to_list()
            atom_indicator = df.drop_duplicates().set_index(index_name).squeeze()
            return self.data[atom_indicator.astype(bool)].sum()

        try:
            return super().__call__(*args, **kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Error while evaluating a probability measure. Perhaps the callable function was not constructed properly due to an invalid parameter name."
            ) from e

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the probability measure.

        Returns
        -------
        repr_str : str
            A string representation of the probability measure.
        """
        if self.data is not None:
            return f"Probability measure '{self.name}':\n{self.data.to_frame()}"
        else:
            return f"Probability measure '{self.name}': empty"

    # --------------------- equality --------------------- #

    def __eq__(self, other: ProbabilityMeasure) -> bool:
        """Check equality with another probability measure.

        Two probability measures are considered equal if they have the same sigma-algebras and identical probability values for each atom. They may have different names and still be considered equal.

        Parameters
        ----------
        other : ProbabilityMeasure
            The other probability measure to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the two probability measures are equal, `False` otherwise.
        """
        if not isinstance(other, ProbabilityMeasure):
            if isinstance(other, MultivariateFunction):
                return super().__eq__(other)
            raise TypeError(
                "Can only compare with another ProbabilityMeasure instance."
            )
        if self.sig_alg != other.sig_alg:
            return False

        self_atom_mapping = {
            atom_id: frozenset(sample_ids)
            for atom_id, sample_ids in self.sig_alg.atom_id_to_sample_ids.items()
        }
        other_atom_mapping = {
            atom_id: frozenset(sample_ids)
            for atom_id, sample_ids in other.sig_alg.atom_id_to_sample_ids.items()
        }

        s1 = self.data.rename(index=self_atom_mapping).sort_index()
        s2 = other.data.rename(index=other_atom_mapping).sort_index()
        return s1.index.equals(s2.index) and (s1 - s2).abs().lt(1e-8).all()


class ProbabilityMeasureMethods:
    """Mixin class providing probability measure methods to other classes."""

    def conditional_probability(self, event: Event, given: Event) -> Real:
        """Compute the conditional probability P(A|B).

        Calls `ProbabilityMeasure.conditional_probability`. See the docstring of `ProbabilityMeasure.conditional_probability` for details.

        Parameters
        ----------
        event : Event
            The event A.
        given : Event
            The event B.

        Returns
        -------
        conditional_prob : Real
            The conditional probability P(A|B).
        """
        return self.prob_measure.conditional_probability(event, given)

    def are_independent(
        self,
        event1: Event | None = None,
        event2: Event | None = None,
        rv1: RandomVector | None = None,
        rv2: RandomVector | None = None,
        algebra1: SigmaAlgebra | None = None,
        algebra2: SigmaAlgebra | None = None,
        tol: Real = 1e-8,
    ) -> bool:
        """Check if two events, two random vectors, or two sigma-algebras are independent.

        Calls `ProbabilityMeasure.are_independent`. See the docstring of `ProbabilityMeasure.are_independent` for details.

        Parameters
        ----------
        event1 : Event | None, default=None
            The first event.
        event2 : Event | None, default=None
            The second event.
        rv1: RandomVector | None, default=None
            The first random vector.
        rv2: RandomVector | None, default=None
            The second random vector.
        algebra1 : SigmaAlgebra | None, default=None
            The first sigma-algebra.
        algebra2 : SigmaAlgebra | None, default=None
            The second sigma-algebra.
        tol : Real, default=1e-10
            The numerical tolerance for checking independence.

        Returns
        -------
        is_independent : bool
            `True` if the events, random vectors, or sigma-algebras are independent, `False` otherwise.
        """
        return self.prob_measure.are_independent(
            event1=event1,
            event2=event2,
            rv1=rv1,
            rv2=rv2,
            algebra1=algebra1,
            algebra2=algebra2,
            tol=tol,
        )

    def almost_surely_equal(
        self,
        first: RandomVector,
        second: RandomVector,
        tol: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two random vectors are equal almost surely.

        Calls `ProbabilityMeasure.almost_surely_equal`. See the docstring of `ProbabilityMeasure.almost_surely_equal` for details.

        Parameters
        ----------
        first : RandomVector
            The first random vector.
        second : RandomVector
            The second random vector.
        tol : float, default=1e-8
            The tolerance below which a probability is considered to be zero for the purposes of this comparison.
        rtol : float, default=1e-5
            The relative tolerance for `np.isclose` when comparing the random vectors.
        atol : float, default=1e-8
            The absolute tolerance for `np.isclose` when comparing the random vectors.

        Returns
        -------
        equal_as : bool
            True if the random vectors are equal almost surely; False otherwise.
        """
        return self.prob_measure.almost_surely_equal(
            first=first, second=second, tol=tol, rtol=rtol, atol=atol
        )
