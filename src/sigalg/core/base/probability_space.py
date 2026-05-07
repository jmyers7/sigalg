"""A class representing a probability space."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..probability_measures.probability_measure import ProbabilityMeasureMethods
from ..sigma_algebras.sigma_algebra import SigmaAlgebraMethods

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .event import Event
    from .sample_space import SampleSpace


class ProbabilitySpace(SigmaAlgebraMethods, ProbabilityMeasureMethods):
    r"""A class representing a probability space.

    See the Notes section below for the mathematical details.

    If all three of `sample_space`, `sig_alg`, and `prob_measure` are provided, the `sample_space` of the provided `sig_alg` must match the provided `sample_space`, and the `sig_alg` of the provided `prob_measure` must match the provided `sig_alg`. Otherwise, if only some of the parameters are provided, the missing components will be automatically created to be compatible with the provided ones as follows:

    * If `sample_space` is not provided but `sig_alg` or `prob_measure` is given, then `sample_space` will be taken from the provided `sig_alg` or `prob_measure`.
    * If `sig_alg` is not provided but `sample_space` or `prob_measure` is given, then `sig_alg` will be set to the power set sigma-algebra on the provided `sample_space` (if given) or the sample space of the provided `prob_measure` (if given).
    * If `prob_measure` is not provided but `sample_space` or `sig_alg` is given, then `prob_measure` will be set to the uniform probability measure on the provided `sig_alg` (if given) or the power set sigma-algebra on the provided `sample_space` (if given).

    If none of the parameters are provided, all components will be initialized to `None` and can be set later.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
        The sample space of the probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra of the probability space.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure of the probability space.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace` instance (when provided), `sig_alg` is not a `SigmaAlgebra` instance (when provided), or `prob_measure` is not a `ProbabilityMeasure` instance (when provided).
    ValueError
        If `sample_space` is `None` but `sig_alg` or `prob_measure` is given.

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> # Create with default uniform probability measure and power-set sigma-algebra
    >>> prob_space = ProbabilitySpace(sample_space=Omega)
    >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
    Probability space (Omega, power_set, P)
    =======================================
    <BLANKLINE>
    * Sample space 'Omega':
    [0, 1, 2]
    <BLANKLINE>
    * Sigma algebra 'power_set':
            atom ID
    sample
    0             0
    1             1
    2             2
    <BLANKLINE>
    * Probability measure 'P':
            probability
    sample
    0          0.333333
    1          0.333333
    2          0.333333
    >>> # Create with a custom sigma-algebra and probability measure
    >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
    ...     {
    ...             0: 0,
    ...             1: 1,
    ...             2: 1,
    ...     }
    ... )
    >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
    ...     {
    ...             0: 0.5,
    ...             1: 0.3,
    ...             2: 0.2,
    ...     }
    ... )
    >>> prob_space = ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)
    >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
    Probability space (Omega, F, P)
    ===============================
    <BLANKLINE>
    * Sample space 'Omega':
    [0, 1, 2]
    <BLANKLINE>
    * Sigma algebra 'F':
            atom ID
    sample
    0             0
    1             1
    2             1
    <BLANKLINE>
    * Probability measure 'P':
            probability
    atom ID
    0               0.5
    1               0.5

    Notes
    -----
    A *probability space* is a triple $(\Omega, \mathcal{F}, P)$ constiting of a sample space $\Omega$, a $\sigma$-algebra $\mathcal{F}$ on $\Omega$, and a probability measure $P$ defined on $\mathcal{F}$.

    See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> None:

        self._validate_parameters(sample_space, sig_alg, prob_measure)
        self._sample_space, self._sig_alg, self._prob_measure = (
            self._generate_components(sample_space, sig_alg, prob_measure)
        )

    def _generate_components(self, sample_space, sig_alg, prob_measure):
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        parameter_cases = (
            sample_space is not None,
            sig_alg is not None,
            prob_measure is not None,
        )

        # The only parameter cases that are not handled are (0, 0, 0) and (1, 1, 1)
        if parameter_cases in [(0, 0, 1), (0, 1, 0), (0, 1, 1)]:
            sample_space = (
                sig_alg.sample_space
                if sig_alg is not None
                else prob_measure.sample_space
            )
            if parameter_cases in [(0, 0, 1)]:
                sig_alg = SigmaAlgebra.power_set(sample_space)
            if parameter_cases in [(0, 1, 0)]:
                prob_measure = ProbabilityMeasure.uniform(sig_alg)
        if parameter_cases == (1, 0, 0):
            sig_alg = SigmaAlgebra.power_set(sample_space)
            prob_measure = ProbabilityMeasure.uniform(sig_alg)
        if parameter_cases == (1, 0, 1):
            sig_alg = prob_measure.sig_alg
        if parameter_cases == (1, 1, 0):
            prob_measure = ProbabilityMeasure.uniform(sig_alg)

        return sample_space, sig_alg, prob_measure

    def from_dict(
        self,
        probabilities: dict[Hashable, Real],
    ) -> ProbabilitySpace:
        """Create a probability space from a dictionary of probabilities.

        If a `sample_space` was not provided during initialization, it will be created from the keys of the provided dictionary. If it was provided, it must match the keys of the provided dictionary.

        If a `sig_alg` was not provided during initialization, a power set sigma-algebra will be created on the sample space. If it was provided, its sample space must match the keys of the provided dictionary.

        Parameters
        ----------
        probabilities : dict[Hashable, Real]
            Dictionary mapping sample point indices to their probabilities.
            Probabilities must be non-negative and sum to 1.

        Raises
        ------
        TypeError
            If `probabilities` is not a dictionary.
        ValueError
            If the keys of `probabilities` do not match the sample space (if given).

        Returns
        -------
        probability_space : ProbabilitySpace
            A new probability space with the specified probabilities.

        Examples
        --------
        >>> from sigalg.core import ProbabilitySpace
        >>> prob_space = ProbabilitySpace().from_dict(
        ...     {
        ...             "a": 0.5,
        ...             "b": 0.3,
        ...             "c": 0.2,
        ...     }
        ... )
        >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, power_set, P)
        =======================================
        <BLANKLINE>
        * Sample space 'Omega':
        ['a', 'b', 'c']
        <BLANKLINE>
        * Sigma algebra 'power_set':
                atom ID
        sample
        a             0
        b             1
        c             2
        <BLANKLINE>
        * Probability measure 'P':
                probability
        sample
        a               0.5
        b               0.3
        c               0.2
        """
        from ..probability_measures import ProbabilityMeasure
        from .sample_space import SampleSpace

        if not isinstance(probabilities, dict):
            raise TypeError("probabilities must be a dictionary.")

        if self.sample_space is not None:
            if set(probabilities.keys()) != set(self.sample_space):
                raise ValueError(
                    "If sample_space was given during initialization, its elements must match the keys of the probabilities dictionary."
                )
        else:
            self.sample_space = SampleSpace().from_list(list(probabilities.keys()))

        self.prob_measure = ProbabilityMeasure(sig_alg=self.sig_alg).from_dict(
            point_probs=probabilities
        )

        return self

    @classmethod
    def from_event(
        cls, event: Event, prob_measure: ProbabilityMeasure
    ) -> ProbabilitySpace:
        r"""Create a conditional probability space from an event.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : Event
            The event on which to create the conditional probability space.
        prob_measure : ProbabilityMeasure
            The probability measure to use for the conditional probability space.

        Raises
        ------
        TypeError
            If `event` is not an `Event` instance or `prob_measure` is not a `ProbabilityMeasure` instance.
        ValueError
            If the event is not in the domain (sigma-algebra) of the probability measure, or if the probability of the event under the given probability measure is zero.

        Returns
        -------
        prob_space : ProbabilitySpace
            A new probability space representing the conditional probability space on the event.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=6)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...         5: 0,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.1,
        ...         1: 0.2,
        ...         2: 0.35,
        ...         3: 0.25,
        ...         4: 0.07,
        ...         5: 0.03,
        ...     }
        ... )
        >>> A = F.get_event([1, 2, 3, 4])
        >>> conditional_space = ProbabilitySpace.from_event(event=A, prob_measure=P)
        >>> print(conditional_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (A, F_A, P_A)
        ===============================
        <BLANKLINE>
        * Sample space 'A':
        [1, 2, 3, 4]
        <BLANKLINE>
        * Sigma algebra 'F_A':
                atom ID
        sample
        1             1
        2             1
        3             2
        4             2
        <BLANKLINE>
        * Probability measure 'P_A':
                probability
        atom ID
        1           0.632184
        2           0.367816

        Notes
        -----
        Let $A$ be an event in a probability space $(\Omega,\mathcal{F},P)$. Provided that $A$ has positive probability, we may construct a *conditional probability space* $(A, \mathcal{F}_A, P_A)$ as follows. The $\sigma$-algebra $\mathcal{F}_A$ contains those sets of the form $A\cap B$, for $B\in \mathcal{F}$. The probability measure $P_A$ is defined on events $E\in \mathcal{F}_A$ by setting $P_A(E) = P(E)/P(A)$.

        Provided that $\Omega$ is finite (as it always is, in SigAlg), then the $\sigma$-algebra $\mathcal{F}_A$ is determined uniquely by its atoms, which are just the nonempty intersections of the atoms of $\mathcal{F}$ with $A$. Thus, the same atom identifiers from $\mathcal{F}$ can be used to define $\mathcal{F}_A$.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .event import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")
        if not isinstance(prob_measure, ProbabilityMeasure):
            raise TypeError("prob_measure must be a ProbabilityMeasure instance.")
        if event not in prob_measure.sig_alg:
            raise ValueError(
                "The event must be in the domain (sigma-algebra) of the given probability measure."
            )

        prob_event = prob_measure(event)
        sig_alg = prob_measure.sig_alg
        event_sample_space = event.to_sample_space()

        if prob_event < 1e-8:
            raise ValueError(
                "Cannot create a probability space from an event with 0 probability."
            )

        event_atom_ids = {omega: sig_alg.sample_id_to_atom_id[omega] for omega in event}
        if event.name is not None and sig_alg.name is not None:
            event_sig_alg_name = f"{sig_alg.name}_{event.name}"
        else:
            event_sig_alg_name = "sigma-algebra_event"
        event_sigma_algebra = SigmaAlgebra(
            sample_space=event_sample_space, name=event_sig_alg_name
        ).from_dict(event_atom_ids)

        atom_event_indicator = (
            pd.concat([event.indicator.data, sig_alg.data], axis=1)
            .drop_duplicates()
            .set_index("atom ID")
            .squeeze()
            .astype(bool)
        )
        atom_probs = (prob_measure.data[atom_event_indicator] / prob_event).to_dict()

        if event.name is not None and prob_measure.name is not None:
            event_prob_measure_name = f"{prob_measure.name}_{event.name}"
        else:
            event_prob_measure_name = "prob_event"
        event_probability_measure = ProbabilityMeasure(
            sig_alg=event_sigma_algebra, name=event_prob_measure_name
        ).from_dict(probs=atom_probs)

        return cls(
            sample_space=event_sample_space,
            sig_alg=event_sigma_algebra,
            prob_measure=event_probability_measure,
        )

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        """Get the sample space of the probability space.

        Returns
        -------
        sample_space : SampleSpace
            The sample space of the probability space.
        """
        return self._sample_space

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the sample space of the probability space.

        Setting a new sample space will set the sigma-algebra to the power-set sigma-algebra and the probability measure to the uniform probability measure on the power-set sigma-algebra.

        Parameters
        ----------
        sample_space : SampleSpace
            The new sample space to set.
        """
        self._validate_parameters(sample_space, sig_alg=None, prob_measure=None)

        self._sample_space, self._sig_alg, self._prob_measure = (
            self._generate_components(
                sample_space=sample_space, sig_alg=None, prob_measure=None
            )
        )

    @property
    def sig_alg(self) -> SigmaAlgebra:
        """Get the sigma-algebra of the probability space.

        Returns
        -------
        sig_alge : SigmaAlgebra
            The sigma-algebra of the probability space.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...             0: 0,
        ...             1: 1,
        ...             2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...             0: 0.5,
        ...             1: 0.3,
        ...             2: 0.2,
        ...     }
        ... )
        >>> prob_space = ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> print(prob_space.sig_alg) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        0             0
        1             1
        2             1
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra of the probability space.

        Setting a new sigma-algebra will set the probability measure to the uniform probability measure on the new sigma-algebra. If the sample space was not set during initialization, it will be set to the sample space of the new sigma-algebra. If the sample space was set during initialization, it must match the sample space of the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra to set.
        """
        self._validate_parameters(self.sample_space, sig_alg, prob_measure=None)
        self._sample_space, self._sig_alg, self._prob_measure = (
            self._generate_components(
                sample_space=self.sample_space, sig_alg=sig_alg, prob_measure=None
            )
        )

    @property
    def prob_measure(self) -> ProbabilityMeasure:
        """Get the probability measure of the probability space.

        Returns
        -------
        prob : ProbabilityMeasure
            The probability measure of this probability space.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...             0: 0,
        ...             1: 1,
        ...             2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...             0: 0.5,
        ...             1: 0.3,
        ...             2: 0.2,
        ...     }
        ... )
        >>> prob_space = ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> print(prob_space.prob_measure) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.5
        1               0.5
        """
        return self._prob_measure

    @prob_measure.setter
    def prob_measure(self, prob_measure: ProbabilityMeasure) -> None:
        """Set the probability measure of the probability space.

        The sigma-algebra of the new probability measure must match the sigma-algebra of the probability space (if set during initialization). If the sigma-algebra was not set during initialization, it will be set to the sigma-algebra of the new probability measure. If the sample space was set during initialization, it must match the sample space of the new probability measure.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure
            New probability measure.

        Raises
        ------
        ValueError
            If `sample_space` is `None`.
        """
        self._validate_parameters(
            self.sample_space, sig_alg=self.sig_alg, prob_measure=prob_measure
        )
        self._sample_space, self._sig_alg, self._prob_measure = (
            self._generate_components(
                sample_space=self.sample_space,
                sig_alg=self.sig_alg,
                prob_measure=prob_measure,
            )
        )

    # --------------------- methods --------------------- #

    def sample(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> list[Hashable]:
        """Generate random samples from this probability space.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate. Must be positive.
        random_state : int | np.random.Generator | None, default=None
            Random seed or generator for reproducibility. If `None`, the random state is not set.

        Returns
        -------
        sample : list[Hashable]
            A list of sampled outcomes from the sample space.

        Raises
        ------
        ValueError
            If `size` is not a positive integer, or if the probability space does not have a power set sigma-algebra (sampling is only supported for probability spaces with power set sigma-algebras).
        TypeError
            If `random_state` is not an integer, `np.random.Generator`, or `None`.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.15,
        ...         1: 0.25,
        ...         2: 0.35,
        ...         3: 0.25,
        ...     }
        ... )
        >>> prob_space = ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> rng = np.random.default_rng(seed=42)
        >>> sample = prob_space.sample(size=5, random_state=rng)
        >>> print(sample)
        [3, 2, 3, 2, 0]
        """
        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if not self.sig_alg.is_power_set:
            raise ValueError(
                "Sampling is only supported for probability spaces with power set sigma-algebras."
            )

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        outcomes = list(self.sample_space)
        probabilities = [self.prob_measure(outcome) for outcome in outcomes]

        samples = rng.choice(outcomes, size=size, p=probabilities)

        return samples.tolist()

    # --------------------- data access methods --------------------- #

    def __iter__(self):
        """Allow unpacking of probability space components.

        Enables syntax like: `Omega, F, P = prob_space`, where `Omega` is the sample space of the probability space, and `F` and `P` are its sigma-algebra and probability measure, respectively.

        Yields
        ------
        sample_space : SampleSpace
            The sample space.
        sig_alg : SigmaAlgebra
            The sigma-algebra.
        prob_measure : ProbabilityMeasure
            The probability measure.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...             0: 0,
        ...             1: 1,
        ...             2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...             0: 0.5,
        ...             1: 0.3,
        ...             2: 0.2,
        ...     }
        ... )
        >>> prob_space = ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)
        >>> Omega1, F1, P1 = prob_space
        >>> print(Omega1) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        [0, 1, 2]
        >>> print(F1) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        0             0
        1             1
        2             1
        >>> print(P1) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.5
        1               0.5
        """
        yield self.sample_space
        yield self.sig_alg
        yield self.prob_measure

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        """Check equality with another probability space.

        Two probability spaces are equal if they have the same sample space,
        sigma-algebra, and probability measure.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `ProbabilitySpace` with identical
            components, `False` otherwise.
        """
        if not isinstance(other, ProbabilitySpace):
            return False
        return (
            self.sample_space == other.sample_space
            and self.sig_alg == other.sig_alg
            and self.prob_measure == other.prob_measure
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the probability space.

        Returns
        -------
        repr_str : str
            A string representation showing the probability space's component names.
        """
        if (
            self.sample_space is None
            or self.sig_alg is None
            or self.prob_measure is None
        ):
            return "ProbabilitySpace(empty)"
        return (
            f"ProbabilitySpace("
            f"sample_space={self.sample_space.name}, "
            f"sig_alg={self.sig_alg.name}, "
            f"prob_measure={self.prob_measure.name})"
        )

    def __str__(self) -> str:
        """Return a detailed string representation of the probability space.

        Returns
        -------
        repr_str : str
            A formatted string showing the probability space header and detailed
            representations of its components.
        """
        if (
            self.sample_space is None
            or self.sig_alg is None
            or self.prob_measure is None
        ):
            return "ProbabilitySpace(empty)"
        header = (
            f"Probability space ("
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

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace | None,
        sig_alg: SigmaAlgebra | None,
        prob_measure: ProbabilityMeasure | None,
    ) -> None:
        """Validate probability space construction parameters.

        Parameters
        ----------
        sample_space : SampleSpace | None
            The sample space to validate.
        sig_alg : SigmaAlgebra | None
            The sigma-algebra to validate.
        prob_measure : ProbabilityMeasure | None
            The probability measure to validate.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance, `sig_algebra` is not
            a `SigmaAlgebra` instance (when provided), or `prob_measure` is
            not a `ProbabilityMeasure` instance (when provided).
        ValueError
            If `sig_alg` or `prob_measure` have sample spaces that do
            not match the provided `sample_space`.
        """
        from ..probability_measures import ProbabilityMeasure
        from ..sigma_algebras import SigmaAlgebra
        from .sample_space import SampleSpace

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError("prob_measure must be a ProbabilityMeasure instance.")

        if sig_alg is not None and prob_measure is not None:
            if prob_measure.sig_alg != sig_alg:
                raise ValueError(
                    "If both sig_alg and prob_measure are given, the probability measure must be defined on the given sigma-algebra."
                )

        if sample_space is not None:
            if sig_alg is not None and sig_alg.sample_space != sample_space:
                raise ValueError(
                    "If sig_alg is given, its sample space must match the given sample_space."
                )
            if prob_measure is not None and prob_measure.sample_space != sample_space:
                raise ValueError(
                    "If prob_measure is given, its sample space must match the given sample_space."
                )
