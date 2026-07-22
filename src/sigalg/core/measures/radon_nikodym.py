"""A class representing a Radon-Nikodym derivative."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..functions.random_variable import RandomVariable

if TYPE_CHECKING:
    from .probability_measure import ProbabilityMeasure


class RadonNikodym(RandomVariable):
    """A class representing a Radon-Nikodym derivative of one probability measure with respect to another.

    The `__init__` method is not meant to be used directly. Instead, use the `from_measures` class method to create an instance of this class.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from sigalg.core import (
    ...     ProbabilityMeasure,
    ...     RadonNikodym,
    ...     SampleSpace,
    ...     SigmaAlgebra,
    ... )
    >>> Omega = SampleSpace.from_sequence(size=10)
    >>> F = SigmaAlgebra.from_rand(
    ...     num_atoms=3,
    ...     sample_space=Omega,
    ...     random_state=42,
    ... )
    >>> P = ProbabilityMeasure(
    ...     sig_alg=F,
    ...     mapping={
    ...         0: 0.2,
    ...         1: 0.8,
    ...         2: 0.0,
    ...     },
    ... )
    >>> Q = ProbabilityMeasure(
    ...     sig_alg=F,
    ...     mapping={
    ...         0: 0.9,
    ...         1: 0.1,
    ...         2: 0.0,
    ...     },
    ...     name="Q",
    ... )
    >>> dQ_dP = RadonNikodym.from_measures(prob_measure=Q, wrt=P)
    >>> print(dQ_dP)  # doctest: +NORMALIZE_WHITESPACE
    Radon-Nikodym derivative 'dQ_dP':
            dQ_dP
    sample
    0       0.000
    1       0.000
    2       0.125
    3       0.000
    4       0.125
    5       4.500
    6       4.500
    7       0.000
    8       0.000
    9       0.000
    >>> for U in F.to_atoms:
    ...     print(dQ_dP.integrate(event=U) == Q(U))
    True
    True
    True
    """

    _repr_name = "Radon-Nikodym derivative"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_measures(
        cls,
        prob_measure: ProbabilityMeasure,
        wrt: ProbabilityMeasure,
        tol: float = 1e-8,
    ) -> RadonNikodym:
        """Compute the Radon-Nikodym derivative of one probability measure with respect to another.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure
            The probability measure for which the Radon-Nikodym derivative is to be computed.
        wrt : ProbabilityMeasure
            The probability measure with respect to which the Radon-Nikodym derivative is to be computed.
        tol : float, default=1e-8
            A tolerance level for checking absolute continuity.

        Raises
        ------
        TypeError
            If `prob_measure` or `wrt` is not an instance of `ProbabilityMeasure`, or if `tol` is not a float.
        ValueError
            If `prob_measure` or `wrt` does not have its `data` attribute set, or if they are not defined on the same sigma-algebra, or if `prob_measure` is not absolutely continuous with respect to `wrt`, or if `tol` is not positive.

        Returns
        -------
        derivative : RadonNikodym
            An instance of the `RadonNikodym` class representing the Radon-Nikodym derivative of `prob_measure` with respect to `wrt`.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RadonNikodym,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     num_atoms=3,
        ...     sample_space=Omega,
        ...     random_state=42,
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.8,
        ...         2: 0.0,
        ...     },
        ... )
        >>> Q = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.9,
        ...         1: 0.1,
        ...         2: 0.0,
        ...     },
        ...     name="Q",
        ... )
        >>> dQ_dP = RadonNikodym.from_measures(prob_measure=Q, wrt=P)
        >>> print(dQ_dP)  # doctest: +NORMALIZE_WHITESPACE
        Radon-Nikodym derivative 'dQ_dP':
                dQ_dP
        sample
        0       0.000
        1       0.000
        2       0.125
        3       0.000
        4       0.125
        5       4.500
        6       4.500
        7       0.000
        8       0.000
        9       0.000
        >>> for U in F.to_atoms:
        ...     print(dQ_dP.integrate(event=U) == Q(U))
        True
        True
        True
        """
        from .probability_measure import ProbabilityMeasure

        if not isinstance(prob_measure, ProbabilityMeasure) or not isinstance(
            wrt, ProbabilityMeasure
        ):
            raise TypeError(
                "'prob_measure' and 'wrt' must be instances of ProbabilityMeasure."
            )
        if prob_measure.data is None or wrt.data is None:
            raise ValueError(
                "'prob_measure' and 'wrt' must have their 'data' attributes set."
            )
        if prob_measure.sig_alg != wrt.sig_alg:
            raise ValueError(
                "'prob_measure' and 'wrt' must be defined on the same sigma-algebra."
            )
        if not isinstance(tol, float):
            raise TypeError("'tol' must be a float.")
        if tol <= 0:
            raise ValueError("'tol' must be positive.")

        P = wrt
        Q = prob_measure

        if not (~(P.data < tol) | (Q.data < tol)).all():
            raise ValueError(
                "'prob_measure' is not absolutely continuous with respect to the second measure 'wrt'."
            )

        name = f"d{Q.name}_d{P.name}"
        mapping = (Q.data / P.data).fillna(0.0)
        mapping = (
            pd.merge(
                left=P.sig_alg.data, right=mapping, left_on="atom_ID", right_index=True
            )
            .drop(columns=["atom_ID"])
            .squeeze(axis=1)
            .rename(name)
        )

        return cls(
            sample_space=P.sample_space,
            sig_alg=P.sig_alg,
            prob_measure=P,
            mapping=mapping,
            name=name,
        )
