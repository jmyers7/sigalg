"""A class representing a probability measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import dirichlet

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn
from ..random_objects.operators import OperatorsMethods

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.feature_vector import FeatureVector
    from ..base.sample_space import SampleSpace
    from ..random_objects.random_vector import RandomVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class ProbabilityMeasure(OperatorsMethods):
    r"""A class representing a probability measure on a sigma-algebra.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra on which the probability measure is defined.
    name : Hashable | None, default="P"
        A name for the probability measure.

    Raises
    ------
    TypeError
        If `sig_alg` is not a `SigmaAlgebra` instance (if given), or if `name` is not hashable (if given).

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
    ...     {
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     }
    ... )
    >>> P = ProbabilityMeasure(sig_alg=F).from_atoms(
    ...     {
    ...         0: 0.2,
    ...         1: 0.8,
    ...     }
    ... )
    >>> print(P) # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
            probability
    atom ID
    0               0.2
    1               0.8
    >>> A = F.get_event([0, 1])
    >>> print(P(A))
    0.2

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be an event space consisting of a $\sigma$-algebra $\mathcal{F}$ on a set $\Omega$. A *probability measure* $P$ is a countably additive function $P: \mathcal{F} \to [0,1]$ such that $P(\Omega) = 1$. Here, *countable additivity* means that

    $$
    P \left( \bigcup_{k=1}^\infty A_k \right) = \sum_{k=1}^\infty P(A_k)
    $$

    for all collections $\{A_k\}_{k=1}^\infty$ of pairwise disjoint measurable sets. If $\Omega$ is finite (as it always is, in SigAlg), then $P$ needs only to be finitely additive in order to be countably additive.

    See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sig_alg: SigmaAlgebra | None = None,
        name: Hashable | None = "P",
    ) -> None:
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("If given, sig_alg must be a SigmaAlgebra instance.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")
        self.sig_alg = sig_alg
        self._name = name

        # caches for properties
        self._data: pd.Series | None = None
        self._point_probs: Mapping[Hashable, Real] | None = None
        self._atom_probs: Mapping[Hashable, Real] | None = None

    def from_dict(self, point_probs: Mapping[Hashable, Real]) -> ProbabilityMeasure:
        """Create a probability measure from a dictionary of probabilities of sample points.

        If a `sig_alg` was not provided during initialization, a power-set sigma-algebra will be created from the keys of the provided dictionary. If it was provided, the keys of the dictionary must match the sample space of the sigma-algebra.

        Parameters
        ----------
        point_probs : Mapping[Hashable, Real]
            A mapping from sample points to their probabilities.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     }
        ... )
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.7
        1               0.3
        """
        from ..base.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        v = SampleSpaceMappingIn(
            mapping=point_probs, sample_space=self.sample_space, kind="probabilities"
        )

        if self.sig_alg is None:
            self.sig_alg = SigmaAlgebra.power_set(
                SampleSpace().from_list(list(v.mapping.keys()))
            )
        self._point_probs = v.mapping
        return self

    def from_atoms(self, atom_probs: Mapping[Hashable, Real]) -> ProbabilityMeasure:
        """Create a probability measure from a dictionary of probabilities of atoms.

        The `sig_alg` parameter must be set during construction for this method to be used.

        Parameters
        ----------
        atom_probs : Mapping[Hashable, Real]
            A mapping from atom identifiers to their probabilities.

        """
        if self.sig_alg is None:
            raise ValueError(
                "The sig_alg parameter must be set during construction for the from_atoms method."
            )

        v = SampleSpaceMappingIn(
            mapping=atom_probs,
            sample_space=self.sig_alg.atom_space,
            kind="probabilities",
        )
        self._atom_probs = v.mapping
        return self

    def from_pandas(self, data: pd.Series) -> ProbabilityMeasure:
        """Create a `ProbabilityMeasure` from a `pd.Series`.

        If a `sig_alg` was not provided during initialization, it will be generated as the power-set on the indices of the provided `pd.Series`. If it was provided, its sample space must match the index of the `pd.Series`.

        Parameters
        ----------
        data: pd.Series
            A `pd.Series` with sample space indices as the index and their associated probabilities as values

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> s = pd.Series([0.2, 0.5, 0.3])
        >>> P = ProbabilityMeasure(sig_alg=F).from_pandas(s)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0               0.2
        1               0.5
        2               0.3
        >>> # Create a probability measure without initializing with a sample space
        >>> s = pd.Series([0.2, 0.5, 0.3], index=["a", "b", "c"])
        >>> Q = ProbabilityMeasure(name="Q").from_pandas(s)
        >>> print(Q) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        sample
        a               0.2
        b               0.5
        c               0.3
        >>> print(Q.sig_alg) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'power_set':
                atom ID
        sample
        a             0
        b             1
        c             2
        """
        from ..base.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        v = SampleSpaceMappingIn(
            mapping=data.to_dict(),
            sample_space=self.sample_space,
            kind="probabilities",
        )

        if self.sig_alg is None:
            sample_space = SampleSpace().from_pandas(data.index)
            self.sig_alg = SigmaAlgebra.power_set(sample_space)

        self._data = pd.Series(v.mapping, name="probability")
        self._data.index.name = self.sig_alg.sample_space.data.name
        return self

    @classmethod
    def uniform(cls, sig_alg: SigmaAlgebra, name: Hashable = "P") -> ProbabilityMeasure:
        r"""Create a uniform probability measure on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra on which to define the uniform probability measure.
        name : Hashable, default="P"
            A name for the probability measure.

        Raises
        ------
        ValueError
            If the sample space is empty.
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
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure.uniform(sig_alg=F)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.50
        1               0.25
        2               0.25
        >>> A = F.get_event([0, 1])
        >>> print(P(A))
        0.5

        Notes
        -----
        Let $(\Omega,\mathcal{F})$ be an event space where $\Omega$ is finite of cardinality $n$. The *uniform probability measure* on $\mathcal{F}$ is the probability measure $P$ defined by

        $$
        P(A) = \frac{|A|}{n},
        $$

        for all events $A\in \mathcal{F}$, where $|A|$ denotes the cardinality of $A$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")

        n = len(sig_alg.sample_space)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on empty sample space."
            )
        probabilities = dict.fromkeys(sig_alg.sample_space.data, 1.0 / n)
        return cls(sig_alg=sig_alg, name=name).from_dict(probabilities)

    def from_rand(
        self, random_state: int | np.random.Generator | None = None
    ) -> ProbabilityMeasure:
        """Generate a random probability measure.

        This method generates a random probability measure on the sample space by sampling from a Dirichlet distribution with all concentration parameters equal to 1. For this construction method, the `sig_alg` must be provided at construction.

        Parameters
        ----------
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.

        Raises
        ------
        ValueError
            If the sample space is not provided at construction.
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
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_rand(random_state=rng)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0           0.665304
        1           0.334696
        """
        if self.sig_alg is None:
            raise ValueError("Sigma-algebra must be provided at construction.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        probs_arr = dirichlet.rvs(
            alpha=[
                1,
            ]
            * len(self.sig_alg.sample_space),
            random_state=random_state,
        )
        probs = dict(zip(self.sig_alg.sample_space, probs_arr[0]))
        self.from_dict(probs)
        return self

    @classmethod
    def from_features(
        cls,
        rv: RandomVector,
        pmf: Callable[[FeatureVector | Hashable], Real],
        name: Hashable | None = "P",
    ) -> ProbabilityMeasure:
        """Add a probability measure on the domain of a random vector using a function of the features.

        Parameters
        ----------
        rv : RandomVector
            The random vector whose domain will receive the probability measure.
        pmf : Callable[[FeatureVector | Hashable], Real]
            Function mapping feature vectors (in dimension > 1) or hashable values (in dimension 1) to probability values. Must return non-negative values that sum to 1.
        name: Hashable | None, default="P",
            The name of the probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The resulting probability measure.

        Examples
        --------
        >>> from sigalg.core import FeatureVector, ProbabilityMeasure, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (0, 0),
        ...         1: (0, 1),
        ...         2: (1, 0),
        ...         3: (1, 1),
        ...     }
        ... )
        >>> def pmf(v: FeatureVector) -> Real:
        ...     v0, v1 = v
        ...     return 0.75**v0 * 0.25 ** (1 - v0) * 0.6**v1 * 0.4 ** (1 - v1)
        >>> P = ProbabilityMeasure.from_features(rv=X, pmf=pmf)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0            0.10
        1            0.15
        2            0.30
        3            0.45
        """
        from ..random_objects.random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")
        if not callable(pmf):
            raise TypeError("pmf must be a callable function.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")
        if rv.sig_alg is None:
            raise ValueError("Random vector must have a sigma-algebra.")

        probabilities = {
            sample_index: pmf(sample_features)
            for sample_index, sample_features in rv.iter_features()
        }
        return cls(sig_alg=rv.sig_alg, name=name).from_dict(probabilities)

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        """Get the sample space of the probability measure.

        Returns
        -------
        sample_space : SampleSpace
            The sample space on which the probability measure is defined.
        """
        return self.sig_alg.sample_space if self.sig_alg else None

    @property
    def point_probs(self) -> dict[Hashable, Real]:
        """Get the mapping from sample points to their probabilities.

        Returns
        -------
        probabilities : dict[Hashable, Real]
            A mapping from sample IDs to their probabilities.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     }
        ... )
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.7
        1               0.3
        >>> print(P.point_probs)
        {0: 0.2, 1: 0.5, 2: 0.3}
        """
        return self._point_probs

    @property
    def atom_probs(self) -> dict[Hashable, Real]:
        """Get the mapping from atom identifiers to their probabilities.

        Returns
        -------
        probabilities : dict[Hashable, Real]
            A mapping from atom identifiers to their probabilities.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     }
        ... )
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.7
        1               0.3
        >>> print(P.atom_probs)
        {0: 0.7, 1: 0.3}
        """
        if self._atom_probs is None and self.data is not None:
            self._atom_probs = self.data.to_dict()
        return self._atom_probs

    @property
    def data(self) -> pd.Series:
        """Get the probability values as a `pd.Series`.

        Returns
        -------
        data: pd.Series
            A `pd.Series` with sample space indices as the index and their associated probabilities as values.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     }
        ... )
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.7
        1               0.3
        >>> print(P.data) # doctest: +NORMALIZE_WHITESPACE
        atom ID
        0    0.7
        1    0.3
        Name: probability, dtype: float64
        """
        if self._data is None:
            if self._atom_probs is not None:
                self._data = pd.Series(data=self._atom_probs, name="probability")
                self._data.index.name = self.sig_alg.data.name
            elif self._point_probs is not None:
                point_data = pd.Series(data=self._point_probs, name="probability")
                df = pd.concat([point_data, self.sig_alg.data], axis=1)
                self._data = df.groupby(["atom ID"], sort=False)["probability"].sum()
        return self._data

    @property
    def name(self) -> Hashable:
        """Get the name of the probability measure.

        Returns
        -------
        name: Hashable
            The name of the probability measure.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name of the probability measure.

        Parameters
        ----------
        name: Hashable
            The new name of the probability measure.

        Raises
        ------
        TypeError
            If `name` is not Hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        self._name = name

    def with_name(self, name: Hashable) -> ProbabilityMeasure:
        """Set the name of the probability measure and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the probability measure.

        Returns
        -------
        self : ProbabilityMeasure
            The current instance with the updated name.
        """
        self.name = name
        return self

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
        >>> Omega = SampleSpace().from_sequence(size=7)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...         5: 3,
        ...         6: 3,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     point_probs={
        ...         0: 0.1,
        ...         1: 0.2,
        ...         2: 0.05,
        ...         3: 0.1,
        ...         4: 0.15,
        ...         5: 0.25,
        ...         6: 0.15,
        ...     }
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
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time, SigmaAlgebra
        >>> from sigalg.processes import IIDProcess
        >>> # Flip a biased coin twice, with 0 = tail is shown, 1 = head is shown
        >>> time = Time.discrete(start=1, stop=2)
        >>> coin_flips = IIDProcess(
        ...     distribution=bernoulli(p=0.7),
        ...     support=[0, 1],
        ...     name="coin_flips",
        ...     time=time,
        ... ).from_enumeration()
        >>> print(coin_flips) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'coin_flips':
        time        1  2
        trajectory
        0           0  0
        1           0  1
        2           1  0
        3           1  1
        >>> # Get the underlying sample space and probability measure
        >>> Omega = coin_flips.domain
        >>> P = coin_flips.prob_measure
        >>> # Check independence of the events "first flip is tails" and "second flip is heads"
        >>> first_flip_tails = coin_flips.sig_alg.get_event([0, 1])
        >>> second_flip_heads = coin_flips.sig_alg.get_event([1, 3])
        >>> print(P.are_independent(event1=first_flip_tails, event2=second_flip_heads))
        True
        >>> # Check independence of the random variables representing the first and second flips
        >>> flip1, flip2 = coin_flips
        >>> P.are_independent(rv1=flip1, rv2=flip2)
        True
        >>> # Check independence of the random variable representing the first flip and the random variable representing the total number of heads
        >>> sum_of_heads= flip1 + flip2
        >>> P.are_independent(rv1=flip1, rv2=sum_of_heads)
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

            atoms1 = algebra1.to_atoms()
            atoms2 = algebra2.to_atoms()
            for atom1 in atoms1:
                for atom2 in atoms2:
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
        >>> from sigalg.core import SampleSpace, ProbabilityMeasure, RandomVariable, RandomVector, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.4,
        ...         1: 0.6,
        ...         2: 0.0,
        ...     }
        ... )
        >>> # Test on random variables
        >>> X = RandomVariable(domain=Omega, name="X").from_dict(
        ...     {
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 3.0,
        ...     }
        ... )
        >>> Y = RandomVariable(domain=Omega, name="Y").from_dict(
        ...     {
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 4.0,
        ...     }
        ... )
        >>> Z = RandomVariable(domain=Omega, name="Z").from_dict(
        ...     {
        ...         0: 1.0,
        ...         1: 3.0,
        ...         2: 3.0,
        ...     }
        ... )
        >>> print(P.almost_surely_equal(X, Y))
        True
        >>> print(P.almost_surely_equal(X, Z))
        False
        >>> # Test on random vectors of dimension > 1
        >>> U = RandomVector(domain=Omega, name="U").from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 2),
        ...     }
        ... )
        >>> V = RandomVector(domain=Omega, name="V").from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (-1, 4),
        ...     }
        ... )
        >>> W = RandomVector(domain=Omega, name="W").from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (-1, 1),
        ...         2: (3, 2),
        ...     }
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
            .set_index("atom ID")
        )
        second_df = (
            pd.concat([self.sig_alg.data, second.data], axis=1)
            .drop_duplicates()
            .set_index("atom ID")
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

    # --------------------- data access methods --------------------- #

    def __call__(self, key: Hashable | list[Hashable] | Event) -> Real:
        """Get the probability of an event.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Event
            A sample point, a list of sample points, or an `Event` instance.

        Raises
        ------
        TypeError
            If `key` is not a Hashable, list of Hashables, or Event.
        ValueError
            If `key` is an Event from a different sample space.
        KeyError
            If any sample point in `key` is not found in the sample space.

        Returns
        -------
        probability : Real
            The probability associated with the given sample point(s) or event.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=5)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.1,
        ...         3: 0.1,
        ...         4: 0.1,
        ...     }
        ... )
        >>> # Probability of a single sample point
        >>> print(P(4))
        0.1
        >>> # Probability of multiple sample points
        >>> print(P([0, 1]))
        0.7
        >>> # Probability of an event
        >>> A = F.get_event([2, 3])
        >>> print(P(A))
        0.2
        """
        from ..base import Event

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError(
                "Key must be a sample point, a list of sample points, or an instance of Event."
            )

        if isinstance(key, Event) and key.sig_alg != self.sig_alg:
            raise ValueError("Event is not in the domain of the probability measure.")
        elif isinstance(key, Hashable):
            key = self.sig_alg.get_event([key])
        elif isinstance(key, list):
            key = self.sig_alg.get_event(key)

        df = pd.concat([key.indicator.data, self.sig_alg.data], axis=1)
        atom_indicator = df.drop_duplicates().set_index("atom ID").squeeze()
        return self.data[atom_indicator.astype(bool)].sum()

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the probability measure.

        If the sigma-algebra is the power set, the sample points themselves will be displayed, rather than the atom identifiers of the singleton atoms.

        Returns
        -------
        repr_str : str
            A string representation of the probability measure.
        """
        if self.sig_alg.is_power_set:
            atom_id_mapping = {
                atom_id: sample_point
                for atom_id, (
                    sample_point,
                ) in self.sig_alg.atom_id_to_sample_ids.items()
            }
            print_data = self.data.rename(index=atom_id_mapping)
            print_data.index.name = "sample"
        else:
            print_data = self.data

        return f"Probability measure '{self.name}':\n{print_data.to_frame()}"

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
            return False
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

        return self.data.rename(index=self_atom_mapping).equals(
            other.data.rename(index=other_atom_mapping)
        )


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
