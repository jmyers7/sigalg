"""A class representing a Radon-Nikodym derivative."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .measurable_function import MeasurableFunction

if TYPE_CHECKING:
    from ..measures.measure import Measure


class RadonNikodym(MeasurableFunction):
    r"""A class representing a Radon-Nikodym derivative of one measure with respect to another.

    The `__init__` method is not meant to be used directly. Instead, use the `from_measures` class method to create an instance of this class.

    See the Notes section below for the mathematical details.

    Examples
    --------
    Define a probability measure on the power-set sigma-algebra of a sample space.

    >>> from sigalg.core import (
    ...     Measure,
    ...     ProbabilityMeasure,
    ...     RadonNikodym,
    ...     SampleSpace,
    ... )
    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> P = ProbabilityMeasure(
    ...     domain=Omega,
    ...     mapping={
    ...         0: 0.2,
    ...         1: 0.8,
    ...         2: 0.0,
    ...     },
    ... )

    Compute the probability mass function of the probability measure, which is the Radon-Nikodym derivative of the probability measure with respect to the counting measure.

    >>> C = Measure.counting(domain=Omega)
    >>> dP_dC = RadonNikodym.from_measures(P, C)
    >>> print(dP_dC)  # doctest: +NORMALIZE_WHITESPACE
    Radon-Nikodym derivative 'dP_dC':
            dP_dC
    sample
    0         0.2
    1         0.8
    2         0.0

    Define a random variable and check that its expectation with respect to the probability measure is equal to the integral of the product of the random variable and the Radon-Nikodym derivative with respect to the counting measure. (This is a special case of one of the change-of-measure formulas given in the Notes section.)

    >>> X = MeasurableFunction.from_randnorm(
    ...     domain=Omega,
    ...     name="X",
    ...     random_state=42,
    ... )
    >>> X.integrate(measure=P) == (X * dP_dC).integrate(measure=C)
    True

    Notes
    -----
    Let $(X,\mathcal{F})$ be a finite measurable space, and let $\mu$ and $\nu$ be two measures defined on $\mathcal{F}$. We shall say that $\nu$ is *absolutely continuous* with respect to $\mu$, written $\nu \ll \mu$, provided that $\mu(U) = 0$ implies $\nu(U)=0$, for all $U\in \mathcal{F}$. In this case, the *Radon-Nikodym derivative* of $\nu$ with respect to $\mu$, denoted $d\nu/d\mu$, is the unique $\mathcal{F}$-measurable random variable such that

    $$
    \frac{d\nu}{d\mu}(A) = \begin{cases}
    \frac{\nu(A)}{\mu(A)} & : \mu(A) \neq 0, \\
    0 & : \mu(A) = 0,
    \end{cases}
    $$

    for all atoms $A$ of $\mathcal{F}$. (Since $d\nu/d\mu$ is supposed to be $\mathcal{F}$-measurable, it must be constant on the atoms of $\mathcal{F}$, and the expression on the left-hand side of this equality denotes this constant value.)

    The Radon-Nikodym derivative is characterized up to equality almost everywhere (relative to $\mu$) by the following properties: It is a nonnegative $\mathcal{F}$-measurable function such that

    $$
    \nu(U) = \int_X \frac{d\nu}{d\mu} \, d\mu
    $$

    for all measurable sets $U\in \mathcal{F}$.

    There is a second change-of-measure type formula involving a second function: If $f:X\to \mathbb{R}$ is $\mathcal{F}$-measurable, then

    $$
    \int_X f \, d\nu = \int_X f \frac{d\nu}{d\mu} \, d\mu.
    $$
    """

    _repr_name = "RadonNikodym"
    _str_name = "Radon-Nikodym derivative"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_measures(
        cls,
        measure: Measure,
        base_measure: Measure,
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> RadonNikodym:
        r"""Compute the Radon-Nikodym derivative of one measure with respect to another.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        measure : Measure
            The measure whose Radon-Nikodym derivative is to be computed.
        base_measure : Measure
            The base measure.
        name : Hashable | None, default=None
            The name of the derivative. If `None`, a default will be generated.
        tol : float, default=1e-8
            A tolerance level for checking absolute continuity.

        Raises
        ------
        TypeError
            If `measure` or `base_measure` is not an instance of `Measure`, or if `tol` is not a float.
        ValueError
            If `measure` or `base_measure` does not have its `data` attribute set, or if they are not defined on the same sigma-algebra, or if `measure` is not absolutely continuous with respect to `base_measure`, or if `tol` is not positive.

        Returns
        -------
        derivative : RadonNikodym
            An instance of the `RadonNikodym` class representing the Radon-Nikodym derivative of `measure` with respect to `base_measure`.

        Examples
        --------
        Define a probability measure on the power-set sigma-algebra of a sample space.

        >>> from sigalg.core import (
        ...     Measure,
        ...     ProbabilityMeasure,
        ...     RadonNikodym,
        ...     SampleSpace,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.8,
        ...         2: 0.0,
        ...     },
        ... )

        Compute the probability mass function of the probability measure, which is the Radon-Nikodym derivative of the probability measure with respect to the counting measure.

        >>> C = Measure.counting(domain=Omega)
        >>> dP_dC = RadonNikodym.from_measures(P, C)
        >>> print(dP_dC)  # doctest: +NORMALIZE_WHITESPACE
        Radon-Nikodym derivative 'dP_dC':
                dP_dC
        sample
        0         0.2
        1         0.8
        2         0.0

        Define a random variable and check that its expectation with respect to the probability measure is equal to the integral of the product of the random variable and the Radon-Nikodym derivative with respect to the counting measure. (This is a special case of one of the change-of-measure formulas given in the Notes section.)

        >>> X = MeasurableFunction.from_randnorm(
        ...     domain=Omega,
        ...     name="X",
        ...     random_state=42,
        ... )
        >>> X.integrate(measure=P) == (X * dP_dC).integrate(measure=C)
        True

        Notes
        -----
        Let $(X,\mathcal{F})$ be a finite measurable space, and let $\mu$ and $\nu$ be two measures defined on $\mathcal{F}$. We shall say that $\nu$ is *absolutely continuous* with respect to $\mu$, written $\nu \ll \mu$, provided that $\mu(U) = 0$ implies $\nu(U)=0$, for all $U\in \mathcal{F}$. In this case, the *Radon-Nikodym derivative* of $\nu$ with respect to $\mu$, denoted $d\nu/d\mu$, is the unique $\mathcal{F}$-measurable random variable such that

        $$
        \frac{d\nu}{d\mu}(A) = \begin{cases}
        \frac{\nu(A)}{\mu(A)} & : \mu(A) \neq 0, \\
        0 & : \mu(A) = 0,
        \end{cases}
        $$

        for all atoms $A$ of $\mathcal{F}$. (Since $d\nu/d\mu$ is supposed to be $\mathcal{F}$-measurable, it must be constant on the atoms of $\mathcal{F}$, and the expression on the left-hand side of this equality denotes this constant value.)

        The Radon-Nikodym derivative is characterized up to equality almost everywhere (relative to $\mu$) by the following properties: It is a nonnegative $\mathcal{F}$-measurable function such that

        $$
        \nu(U) = \int_X \frac{d\nu}{d\mu} \, d\mu
        $$

        for all measurable sets $U\in \mathcal{F}$.

        There is a second change-of-measure type formula involving a second function: If $f:X\to \mathbb{R}$ is $\mathcal{F}$-measurable, then

        $$
        \int_X f \, d\nu = \int_X f \frac{d\nu}{d\mu} \, d\mu.
        $$
        """
        from ..measures.measure import Measure
        from .._utils.utils import _to_df

        if not isinstance(measure, Measure) or not isinstance(base_measure, Measure):
            raise TypeError(
                "'measure' and 'base_measure' must be instances of Measure."
            )
        if measure.data is None or base_measure.data is None:
            raise ValueError(
                "'measure' and 'base_measure' must have their 'data' attributes set."
            )
        if measure.sig_alg != base_measure.sig_alg:
            raise ValueError(
                "'measure' and 'base_measure' must be defined on the same sigma-algebra."
            )
        if not isinstance(tol, float):
            raise TypeError("'tol' must be a float.")
        if tol <= 0:
            raise ValueError("'tol' must be positive.")

        if not measure << base_measure:
            raise ValueError(
                "The measure is not absolutely continuous with respect to the base measure."
            )

        if name is None:
            name = f"d{measure.name}_d{base_measure.name}"
        mapping = (measure.data / base_measure.data).fillna(0.0).rename("derivative")

        sig_alg_data = _to_df(base_measure.sig_alg.data).add_suffix("_ID")

        # TODO: check merge logic — possibly change to `on`?
        mapping = pd.merge(
            left=sig_alg_data,
            right=mapping,
            left_on=list(sig_alg_data.columns),
            right_index=True,
        )["derivative"].rename(name)

        return cls(
            domain=base_measure.sig_alg.domain,
            sig_alg=base_measure.sig_alg,
            measure=base_measure,
            mapping=mapping,
            name=mapping.name,
        )
