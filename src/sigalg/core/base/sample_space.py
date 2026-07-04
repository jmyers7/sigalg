"""A class representing a sample space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .domain import Domain

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .event_space import EventSpace
    from .probability_space import ProbabilitySpace


class SampleSpace(Domain):
    r"""A class representing a sample space.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        An `IndexLike` object containing the points in the sample space.
    name : Hashable | None, default=None
        Name identifier for the sample space. If `None`, will use the default name `Omega`.
    variable_names : list[Hashable] | None, default=None
        A list of names of the variables for the index. If `None`, a default variable name `sample` will be used.

    Examples
    --------
    Construct a 1-dimensional `SampleSpace` from a list of sample points.

    >>> from sigalg.core import SampleSpace
    >>> import pandas as pd
    >>> indices = ["red", "green", "blue"]
    >>> Omega1 = SampleSpace(indices=indices, name="Omega1")
    >>> print(Omega1)  # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega1':
    sample
       red
     green
      blue

    Construct a 1-dimensional `SampleSpace` from a `pd.Index` object.

    >>> indices = pd.Index(["a", "b", "c"], name="letter")
    >>> Omega2 = SampleSpace(indices=indices, name="Omega2")
    >>> print(Omega2) # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega2':
    letter
         a
         b
         c

    Construct a 2-dimensional `SampleSpace` from a list of ordered pairs.

    >>> indices = [("red", 1), ("green", 2), ("blue", 3)]
    >>> Omega3 = SampleSpace(indices=indices, name="Omega3", variable_names=["color", "number"])
    >>> print(Omega3) # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega3':
     color  number
       red       1
     green       2
      blue       3

    Construct a 2-dimensional `SampleSpace` from a `pd.MultiIndex` object.

    >>> indices = pd.MultiIndex.from_tuples(
    ...     [("a", 1), ("b", 2), ("c", 3)], names=["letter", "number"]
    ... )
    >>> Omega4 = SampleSpace(indices=indices, name="Omega4")
    >>> print(Omega4) # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega4':
     letter  number
          a       1
          b       2
          c       3

    Notes
    -----
    In the abstract, a *sample space* is just a set $\Omega$. However, in the context of probability theory, sample spaces are often conceptualized as the set of all possible outcomes of a random experiment. Each element $\omega \in \Omega$ is called a *sample point* or *outcome*. The sample space serves as the foundational building block for defining events (subsets of sample spaces contained in $\sigma$-algebras) and probability measures (functions that assign probabilities to events).
    """

    _default_name = "Omega"
    _repr_name = "Sample space"
    _variable_names_prefix = "sample"

    # --------------------- conversion methods --------------------- #

    @classmethod
    def from_domain(cls, domain: Domain) -> SampleSpace:
        """Promote an instance of `Domain` to an instance of `SampleSpace`.

        Parameters
        ----------
        domain : Domain
            The instance of `Domain` to promote.

        Raises
        ------
        TypeError
            If `domain` is not an instance of `Domain`.

        Returns
        -------
        sample_space : SampleSpace
            The new instance of `SampleSpace` constructed from `domain`.

        Examples
        --------
        >>> from sigalg.core import Domain, SampleSpace
        >>> D = Domain([1, 2])
        >>> D_sample_space = SampleSpace.from_domain(D)
        >>> isinstance(D_sample_space, SampleSpace)
        True
        """
        from .domain import Domain

        if not isinstance(domain, Domain):
            raise TypeError("domain must be an instance of `Domain`.")

        return cls._promote(domain)

    def make_probability_space(
        self,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilitySpace:
        """Convert this sample space to a probability space by adding a sigma-algebra and probability measure.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.
        prob_measure : ProbabilityMeasure | None, default=None
            Probability measure to use. If `None`, a uniform probability measure will be created.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.

        Returns
        -------
        probability_space : ProbabilitySpace
            A `ProbabilitySpace` object with this sample space.

        Examples
        --------
        Define a sample space.

        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace(indices=["a", "b", "c"])

        Promote to a `ProbabilitySpace` with default power set sigma-algebra and uniform probability measure.

        >>> prob_space = Omega.make_probability_space()
        >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, power_set, U)
        =======================================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              a
              b
              c
        <BLANKLINE>
        * Sigma algebra 'power_set':
               atom_ID
        sample
        a            a
        b            b
        c            c
        <BLANKLINE>
        * Probability measure 'U':
                probability
        sample
        a          0.333333
        b          0.333333
        c          0.333333

        Create a custom sigma-algebra and probability measure, and promote to a `ProbabilitySpace` with these custom objects.

        >>> F = SigmaAlgebra(sample_space=Omega, mapping=
        ...     {
        ...         "a": 0,
        ...         "b": 1,
        ...         "c": 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F, mapping=
        ...     {
        ...         0: 0.2,
        ...         1: 0.8,
        ...     },
        ... )
        >>> prob_space = Omega.make_probability_space(sig_alg=F, prob_measure=P)
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              a
              b
              c
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        a             0
        b             1
        c             1
        <BLANKLINE>
        * Probability measure 'P':
                 probability
        atom_ID
        0                0.2
        1                0.8
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .probability_space import ProbabilitySpace

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("`sig_alg` must be a `SigmaAlgebra` or `None`.")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError("`prob_measure` must be a `ProbabilityMeasure` or `None`.")

        return ProbabilitySpace(
            sample_space=self,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
        )

    def make_event_space(self, sig_alg: SigmaAlgebra | None = None) -> EventSpace:
        """Convert this sample space to an event space by adding a sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` or `None`.

        Returns
        -------
        event_space : EventSpace
            An `EventSpace` object with this sample space.

        Examples
        --------
        Define a sample space.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> S = SampleSpace(indices=["s0", "s1", "s2", "s3"], name="S")

        Promote to an `EventSpace` with default power set sigma-algebra.

        >>> event_space = S.make_event_space()
        >>> print(event_space) # doctest: +NORMALIZE_WHITESPACE
        Event space (S, power_set)
        ==========================
        <BLANKLINE>
        * Sample space 'S':
         sample
             s0
             s1
             s2
             s3
        <BLANKLINE>
        * Sigma algebra 'power_set':
                atom_ID
         sample
             s0      s0
             s1      s1
             s2      s2
             s3      s3

        Create a custom sigma-algebra, and promote to an `EventSpace` with this custom object.

        >>> F = SigmaAlgebra(sample_space=S, mapping={"s0": 0, "s1": 0, "s2": 1, "s3": 1})
        >>> event_space = S.make_event_space(sig_alg=F)
        >>> print(event_space) # doctest: +NORMALIZE_WHITESPACE
        Event space (S, F)
        ==================
        <BLANKLINE>
        * Sample space 'S':
         sample
             s0
             s1
             s2
             s3
        <BLANKLINE>
        * Sigma algebra 'F':
                 atom_ID
         sample
         s0            0
         s1            0
         s2            1
         s3            1
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .event_space import EventSpace

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("`sig_alg` must be a `SigmaAlgebra` or `None`.")
        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(sample_space=self)

        return EventSpace(sample_space=self, sig_alg=sig_alg)
