"""A class representing a probability measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import dirichlet

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn
from ..random_objects.operators import OperatorsMethods

if TYPE_CHECKING:
    from ..base.event import Event
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
    >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
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

    In SigAlg, an instance `P` of `ProbabilityMeasure` represents such a probability measure.
    """

    # --------------------- constructors --------------------- #

    _properties = [
        "_atom_probs",
        "_data",
        "_point_probs",
        "_point_data",
    ]

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
        self._sig_alg = sig_alg
        self._name = name
        self._initialize_property_caches()

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    def from_dict(
        self,
        probs: Mapping[Hashable, Real],
        type: str = "atom",
        overwrite_sig_alg: bool = False,
    ) -> ProbabilityMeasure:
        """Create a probability measure from a dictionary of probabilities.

        If the `type` parameter is set to `'point'`, the dictionary is interpreted as mapping sample points to their probabilities. If a `sig_alg` was not provided during initialization, or if it was provided and `overwrite_sig_alg` is True, a power-set sigma-algebra will be created from the keys of the provided dictionary. If a `sig_alg` was provided and `overwrite_sig_alg` is False, the keys of the provided dictionary must match the sample IDs of the sigma-algebra.

        If the `type` parameter is set to `'atom'`, the dictionary is interpreted as mapping atom identifiers to the probabilities of the atoms. In this case, the `sig_alg` parameter must be provided at construction, and the keys of the provided dictionary must match the atom IDs of the sigma-algebra. The `overwrite_sig_alg` parameter is ignored in this case.

        Parameters
        ----------
        probs : Mapping[Hashable, Real]
            A mapping from sample points or atom identifiers to their probabilities.
        type : str, default="atom"
            A string indicating whether the provided dictionary maps sample points (`'point'`) or atom identifiers (`'atom'`) to probabilities.
        overwrite_sig_alg : bool, default=False
            If `type` is `'point'` and a `sig_alg` was provided at construction, whether to overwrite the existing `sig_alg` with a new power-set sigma-algebra generated from the keys of the provided dictionary. Ignored if `type` is `'atom'`.

        Raises
        ------
        ValueError
            If `type` is not `'point'` or `'atom'`, or if `type` is `'atom'` and `sig_alg` is not provided at construction.

        Returns
        -------
        self : ProbabilityMeasure
            The constructed `ProbabilityMeasure` instance.

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
        ...     },
        ...     type="point",
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

        if not isinstance(probs, Mapping):
            raise TypeError(
                "probs must be a mapping from sample points or atom IDs to probabilities."
            )
        if type not in ["point", "atom"]:
            raise ValueError("type must be either 'point' or 'atom'.")
        if not isinstance(overwrite_sig_alg, bool):
            raise TypeError("overwrite_sig_alg must be a boolean.")

        if type == "atom" and self._sig_alg is None:
            raise ValueError(
                "The sig_alg parameter must be set during construction for the from_dict method with type='atom'."
            )

        if type == "point" and overwrite_sig_alg:
            self._sig_alg = None

        reference_space = (
            self.sample_space if type == "point" else self._sig_alg.atom_space
        )
        v = SampleSpaceMappingIn(
            mapping=probs,
            sample_space=reference_space,
            kind="probabilities",
        )

        self._initialize_property_caches()

        if type == "point":
            if self._sig_alg is None:
                self._sig_alg = SigmaAlgebra.power_set(
                    SampleSpace().from_list(list(v.mapping.keys()))
                )
            self._point_probs = v.mapping
        else:
            self._atom_probs = v.mapping

        return self

    def from_pandas(
        self,
        data: pd.Series,
        type: str = "atom",
        overwrite_sig_alg: bool = False,
    ) -> ProbabilityMeasure:
        """Create a `ProbabilityMeasure` from a `pd.Series` of probabilities.

        If the `type` parameter is set to `'point'`, the `pd.Series` is interpreted as mapping sample points to their probabilities. If a `sig_alg` was not provided during initialization, or if it was provided and `overwrite_sig_alg` is True, a power-set sigma-algebra will be created from the index of the provided `pd.Series`. If a `sig_alg` was provided and `overwrite_sig_alg` is False, the index of the provided `pd.Series` must match the sample IDs of the sigma-algebra.

        If the `type` parameter is set to `'atom'`, the `pd.Series` is interpreted as mapping atom identifiers to the probabilities of the atoms. In this case, the `sig_alg` parameter must be provided at construction, and the index of the provided `pd.Series` must match the atom IDs of the sigma-algebra. The `overwrite_sig_alg` parameter is ignored in this case.

        Parameters
        ----------
        data: pd.Series
            A `pd.Series` with sample points or atom identifiers as the index and the probabilities as its values.
        type : str, default="atom"
            A string indicating whether the provided `pd.Series` maps sample points (`'point'`) or atom identifiers (`'atom'`) to probabilities.
        overwrite_sig_alg : bool, default=False
            If `type` is `'point'` and a `sig_alg` was provided at construction, whether to overwrite the existing `sig_alg` with a new power-set sigma-algebra generated from the index of the provided `pd.Series`. Ignored if `type` is `'atom'`.

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
        >>> s = pd.Series([0.8, 0.2])
        >>> P = ProbabilityMeasure(sig_alg=F).from_pandas(s)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0               0.8
        1               0.2
        """
        from ..base.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        if type not in ["point", "atom"]:
            raise ValueError("type must be either 'point' or 'atom'.")
        if not isinstance(overwrite_sig_alg, bool):
            raise TypeError("overwrite_sig_alg must be a boolean.")

        if type == "atom" and self._sig_alg is None:
            raise ValueError(
                "The sig_alg parameter must be set during construction for the from_pandas method with type='atom'."
            )

        if type == "point" and overwrite_sig_alg:
            self._sig_alg = None

        reference_space = (
            self.sample_space if type == "point" else self._sig_alg.atom_space
        )
        _ = SampleSpaceMappingIn(
            mapping=data.to_dict(),
            sample_space=reference_space,
            kind="probabilities",
        )

        self._initialize_property_caches()

        if type == "point":
            if self._sig_alg is None:
                self._sig_alg = SigmaAlgebra.power_set(
                    SampleSpace().from_list(list(data.index))
                )
            self._point_data = data
            self._point_data.index.name = "sample"
            self._point_data.name = "probability"
        else:
            self._data = data
            self._data.index.name = self.sig_alg.atom_space.data.name
            self._data.name = "probability"

        return self

    @classmethod
    def uniform(
        cls, sig_alg: SigmaAlgebra, name: Hashable = "uniform"
    ) -> ProbabilityMeasure:
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
        >>> uniform = ProbabilityMeasure.uniform(sig_alg=F)
        >>> print(uniform) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'uniform':
                probability
        atom ID
        0               0.50
        1               0.25
        2               0.25
        >>> A = F.get_event([0, 1])
        >>> print(uniform(A))
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
        return cls(sig_alg=sig_alg, name=name).from_dict(probabilities, type="point")

    # TODO: make a class method?
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
        self.from_dict(probs, type="point")
        return self

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra on which the probability measure is defined.

        The `sig_alg` property is settable. If a sigma-algebra is already set, the new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra. The probability measure will be restricted to the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on which the probability measure is defined.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.3,
        ...         2: 0.5,
        ...     }
        ... )
        >>> print(P.sig_alg) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        0             0
        1             1
        2             2
        3             2
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
        ...     }
        ... )
        >>> P.sig_alg = G
        >>> print(P.sig_alg) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom ID
        sample
        0             0
        1             1
        2             1
        3             1
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom ID
        0                0.2
        1                0.8
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra on which the probability measure is defined.

        If a sigma-algebra is already set, the new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra. The probability measure will be restricted to the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra on which the probability measure is defined.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance.
        ValueError
            If a sigma-algebra is already set and `sig_alg` is not a sub-sigma-algebra of the current sigma-algebra.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")

        if self.sig_alg is not None:
            if not sig_alg <= self._sig_alg:
                raise ValueError(
                    "sig_alg must be a sub-sigma-algebra of the current sigma-algebra."
                )

            if sig_alg.atom_id_to_sample_ids is not None:
                self._atom_probs = {
                    atom_id: self(event)
                    for atom_id, event in sig_alg.atom_id_to_sample_ids.items()
                }

            self._data = None

        self._sig_alg = sig_alg

    @property
    def sample_space(self) -> SampleSpace | None:
        """Get the sample space of the probability measure.

        The `sample_space` property is settable if the probability measure has a sigma-algebra. In this case, the new sample space must contain the same number of sample points. If the probability measure does not have a sigma-algebra, the sample space cannot be set.

        Returns
        -------
        sample_space : SampleSpace | None
            The sample space on which the probability measure is defined.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.3,
        ...         2: 0.2,
        ...         3: 0.3,
        ...     },
        ...     type="point",
        ... )
        >>> print(P.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        [0, 1, 2, 3]
        >>> Omega_new = SampleSpace(name="Omega_new").from_list(["a", "b", "c", "d"])
        >>> P.sample_space = Omega_new
        >>> print(P.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega_new':
        ['a', 'b', 'c', 'd']
        >>> print(P.point_probs)
        {'a': 0.2, 'b': 0.3, 'c': 0.2, 'd': 0.3}
        """
        return self._sig_alg._sample_space if self._sig_alg else None

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the sample space of the probability measure.

        The new sample space must contain the same number of sample points. If the probability measure does not have a sigma-algebra, the sample space cannot be set.

        Parameters
        ----------
        sample_space : SampleSpace
            The new sample space on which the probability measure is defined.

        Raises
        ------
        ValueError
            If the probability measure does not have a sigma-algebra.
        """
        if self.sig_alg is not None:
            self.sig_alg.sample_space = sample_space
            if self.point_probs is not None:
                self._point_probs = dict(
                    zip(sample_space.data, self.point_probs.values())
                )
        else:
            raise ValueError("Cannot set sample space when sig_alg is not set.")

    # TODO: write unit tests
    @property
    def atom_probs(self) -> dict[Hashable, Real] | None:
        """Get the mapping from atom identifiers to their probabilities.

        Returns
        -------
        probabilities : dict[Hashable, Real] | None
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
        ...     },
        ...     type="point",
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
        if self._atom_probs is None:
            if self._data is not None:
                self._atom_probs = self._data.to_dict()
            elif self._point_probs:
                self._atom_probs = self._point_to_atom_probs(self._point_probs)
            elif self._point_data is not None:
                self._point_probs = self._point_data.to_dict()
                self._atom_probs = self._point_to_atom_probs(self._point_probs)
        return self._atom_probs

    # TODO: write unit tests
    @property
    def data(self) -> pd.Series | None:
        """Get the probability values as a `pd.Series`.

        Returns
        -------
        data: pd.Series | None
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
        ...     },
        ...     type="point",
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
        if self._data is None and self.atom_probs is not None:
            self._data = self._dict_to_pandas(
                self.atom_probs, data_index_name=self.sig_alg.data.name
            )
        return self._data

    # TODO: write unit tests
    @property
    def point_probs(self) -> dict[Hashable, Real] | None:
        """Get the mapping from sample points to their probabilities.

        Returns
        -------
        probabilities : dict[Hashable, Real] | None
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
        ...     },
        ...     type="point",
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
        if self._point_probs is None and self._point_data is not None:
            self._point_probs = self._point_data.to_dict()
        return self._point_probs

    # TODO: write unit tests and docstring
    @property
    def point_data(self) -> pd.Series | None:
        """Later."""
        if self._point_data is None and self.point_probs is not None:
            self._point_data = self._dict_to_pandas(
                self.point_probs, data_index_name=self.sig_alg.data.index.name
            )
        return self._point_data

    def _dict_to_pandas(
        self,
        dict_param: dict,
        data_index_name: str | None = None,
    ) -> pd.Series:
        """Convert dictionary to `pd.Series`."""
        data = pd.Series(dict_param, name="probability")
        data.index = data.index.to_flat_index()
        data.index.name = data_index_name

        return data

    def _point_to_atom_probs(self, point_probs: dict) -> dict:
        """Convert point outputs to atom outputs."""
        return {
            atom_id: sum(
                [
                    point_probs[sample_point]
                    for sample_point in self.sig_alg.atom_id_to_sample_ids[atom_id]
                ]
            )
            for atom_id in self.sig_alg.atom_id_to_sample_ids
        }

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
        ...     {
        ...         0: 0.1,
        ...         1: 0.2,
        ...         2: 0.05,
        ...         3: 0.1,
        ...         4: 0.15,
        ...         5: 0.25,
        ...         6: 0.15,
        ...     },
        ...     type="point",
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
        time  1  2
        0     0  0
        1     0  1
        2     1  0
        3     1  1
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
        >>> Omega = SampleSpace().from_sequence(size=5)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     }
        ... )
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.5,
        ...         1: 0.3,
        ...         2: 0.2,
        ...     }
        ... )
        >>> P_G = P.restrict_to(sig_alg=G)
        >>> print(P_G)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_G':
                probability
        atom ID
        0                0.8
        1                0.2
        """
        data = self.data.copy()
        name = (
            f"{self.name}_{sig_alg.name}"
            if sig_alg.name is not None and self.name is not None
            else "restriction"
        )
        restriction = type(self)(sig_alg=self.sig_alg, name=name).from_pandas(data)
        restriction.sig_alg = sig_alg
        return restriction

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
        ...     },
        ...     type="point",
        ... )
        >>> # Probability of a singleton event
        >>> print(P(4))
        0.1
        >>> # Probability of an event consting of a list of sample points
        >>> print(P([0, 1]))
        0.7
        >>> # Probability of an event as an `Event` instance
        >>> A = F.get_event([2, 3])
        >>> print(P(A))
        0.2
        """
        from ..base import Event

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError(
                "Key must be a sample point, a list of sample points, or an instance of Event."
            )

        if isinstance(key, Event) and not key.sig_alg <= self.sig_alg:
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
