"""A class representing a sample space."""

from .domain import Domain


class SampleSpace(Domain):
    r"""A class representing a sample space.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        An `IndexLike` object containing the points in the sample space.
    name : Hashable | None, default=None
        Name identifier for the sample space. If `None`, a default will be generated.
    variable_names : list[Hashable] | None, default=None
        A list of names of the variables for the sample space. If `None`, default variables names will be generated.

    Examples
    --------
    Construct a 1-dimensional `SampleSpace` from a list of sample points.

    >>> from sigalg.core import SampleSpace
    >>> import pandas as pd
    >>> indices = ["red", "green", "blue"]
    >>> Omega1 = SampleSpace(indices=indices, name="Omega1")
    >>> print(Omega1)  # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega1':
     omega
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
    """

    _default_name = "Omega"
    _repr_name = "SampleSpace"
    _str_name = "Sample space"
    _variable_names_prefix = "omega"

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the sample space.

        Returns
        -------
        repr_str : str
            String representation of the sample space.
        """
        if self.data is None:
            return f"{type(self)._repr_name}(empty)"
        else:
            return f"{type(self)._repr_name}(size={len(self.data)}, variable_names={self.variable_names}, name={self.name})"
