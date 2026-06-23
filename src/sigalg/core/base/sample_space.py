"""A class representing a sample space."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from ...validation.index_validator import IndexLike
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
    indices : IndexLike
        An `IndexLike` object containing the points in the index.
    name : Hashable, default="Omega"
        Name identifier for the sample space.
    variable_names : list[Hashable] | None, default=None
        A list of names of the variables for the index. See the Examples section below for usage.

    Examples
    --------
    Construct a 1-dimensional `SampleSpace` from a list of sample points.

    >>> from sigalg.core import SampleSpace
    >>> import pandas as pd
    >>> indices = ["red", "green", "blue"]
    >>> Omega1 = SampleSpace(indices=indices, name="Omega1")
    >>> print(Omega1)  # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega1':
     Omega1
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

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        indices: IndexLike | None = None,
        name: Hashable = "Omega",
        variable_names: list[Hashable] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            indices=indices, name=name, variable_names=variable_names, **kwargs
        )

    @classmethod
    def from_sequence(
        cls,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        name: Hashable = "Omega",
        variable_name: Hashable | None = None,
    ) -> SampleSpace:
        """Create a sample space with sequentially numbered items.

        Parameters
        ----------
        size : int
            Number of sample points to generate. Must be positive.
        initial_index : int, default=0
            Starting index for sequential numbering.
        prefix : Hashable | None, default=None
            Prefix for index names. If `None`, then numerical indices are used.
        name : Hashable, default="Omega"
            Name identifier for the sample space.
        variable_name : Hashable | None, default=None
            An optional single element for the variable name. If `None`, the default will be set to the name of the sample space.

        Returns
        -------
        sample_space : SampleSpace
            A new `SampleSpace` with automatically generated indices.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, `prefix` is not hashable, or `variable_name` is not a hashable (if given).

        Examples
        --------
        Build a `SampleSpace` consisting of the numbers 0, 1, 2, with default name and variable name.

        >>> from sigalg.core import SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> print(Omega)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         Omega
             0
             1
             2

        Build a `SampleSpace` consisting of the strings F_0, F_1, F_2.

        >>> Omega2 = SampleSpace.from_sequence(size=3, name="Omega2", prefix="F")
        >>> print(Omega2)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega2':
         Omega2
            F_0
            F_1
            F_2

        Build a `SampleSpace` consisting of the numbers 5 and 6, with a custom variable name.

        >>> Omega3 = SampleSpace.from_sequence(size=2, name="Omega3", initial_index=5, variable_name="x")
        >>> print(Omega3)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega3':
         x
         5
         6
        """
        return super().from_sequence(
            size=size,
            initial_index=initial_index,
            prefix=prefix,
            name=name,
            variable_name=variable_name,
        )

    # --------------------- conversion methods --------------------- #

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
        Omega
            a
            b
            c
        <BLANKLINE>
        * Sigma algebra 'power_set':
             power_set
        Omega
        a            a
        b            b
        c            c
        <BLANKLINE>
        * Probability measure 'U':
                probability
        Omega
        a          0.333333
        b          0.333333
        c          0.333333

        Create a custom sigma-algebra and probability measure, and promote to a `ProbabilitySpace` with these custom objects.

        >>> # Create with custom probability measure and sigma-algebra
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         "a": 0,
        ...         "b": 1,
        ...         "c": 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.8,
        ...     }
        ... )
        >>> prob_space = Omega.make_probability_space(sig_alg=F, prob_measure=P)
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
        Omega
            a
            b
            c
        <BLANKLINE>
        * Sigma algebra 'F':
                      F
        Omega
        a             0
        b             1
        c             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        F
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
         S
         s0
         s1
         s2
         s3
        <BLANKLINE>
        * Sigma algebra 'power_set':
             power_set
         S
        s0          s0
        s1          s1
        s2          s2
        s3          s3

        Create a custom sigma-algebra, and promote to an `EventSpace` with this custom object.

        >>> F = SigmaAlgebra(sample_space=S).from_dict(
        ...     sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1},
        ... )
        >>> event_space = S.make_event_space(sig_alg=F)
        >>> print(event_space) # doctest: +NORMALIZE_WHITESPACE
        Event space (S, F)
        ==================
        <BLANKLINE>
        * Sample space 'S':
         S
         s0
         s1
         s2
         s3
        <BLANKLINE>
        * Sigma algebra 'F':
                      F
         S
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

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        r"""Return a string representation of the sample space.

        Returns
        -------
        repr_str : str
            A formatted string showing the sample space name and its sample points.
        """
        if self.data is None:
            return f"Sample space '{self.name}': empty"
        else:
            return f"Sample space '{self.name}':\n{self.data.to_frame().to_string(index=False)}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        """Check equality with another sample space.

        Two sample spaces are equal if they have the same sample points in the
        same order. They can have different names and different data names and still
        be considered equal.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        equal : bool
            `True` if the other object is a `SampleSpace` with identical values,
            `False` otherwise.
        """
        return isinstance(other, SampleSpace) and super().__eq__(other)
