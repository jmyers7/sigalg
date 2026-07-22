"""Class for operators on random vectors, such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of probability measures."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..base.event import Event
    from ..measures.parametrized_probability_measure import (
        ParametrizedProbabilityMeasure,
    )
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .random_variable import RandomVariable
    from .random_vector import RandomVector


class Operators:
    """Class containing methods such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of probability measures.

    The class does not have an `__init__` method, and all methods are class methods.
    """

    # TODO: refactor to *not* call expectation
    @classmethod
    def integrate(
        cls,
        rv: RandomVector,
        event: Event | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> pd.Series | Real:
        r"""Compute the Lebesgue integral of a random vector with respect to a probability measure over an (optional) event.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector to integrate.
        event: Event | None, default=None
            The optional event over which to integrate. If `None`, the integral will be taken over the entire sample space contained in the `domain` attribute of the random vector.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`, or if `event` is not an `Event` or `None`.
        ValueError
            If `prob_measure` is given and is not defined on the sigma-algebra of the random vector, or if `event` is given and is not an element of the sigma-algebra of the random vector.

        Returns
        -------
        integral : pd.Series | Real
            If `rv` has dimension > 1, returns a pd.Series representing the integral of each component of the random vector. If `rv` has dimension 1, returns a Real representing the integral.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
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
        ...         0: 0.3,
        ...         1: 0.2,
        ...         2: 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> A = prob_space.get_event([0, 1, 2, 3])
        >>> integral = Operators.integrate(rv=X, event=A)
        >>> print(integral)
        index
        0    0.9
        1    1.4
        Name: int_A X dP, dtype: float64
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 3,
        ...         3: 3,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> integral = Operators.integrate(rv=Y, event=A)
        >>> print(integral)
        0.9000000000000001

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}$ be a random variable on a probability space $(\Omega, \mathcal{F}, P)$. Assume $\Omega$ is finite (as it always is, in SigAlg) and let $\{A_i\}_{i\in I}$ be the atoms of $\mathcal{F}$, indexed by some set $I$. Since $X$ is $\mathcal{F}$-measurable, it takes a constant value $x_i$ on each atom $A_i$, i.e., $X(\omega)=x_i$ for each $\omega \in A_i$. Then the *Lebesgue integral* of $X$ is

        $$
        \int_A X \, dP = \sum_{i\in I} x_i P(A_i).
        $$

        If $B$ is an event in $\mathcal{F}$, then we may also integrate over $B$ by setting

        $$
        \int_B X \, dP = \int_A XI_B \, dP,
        $$

        where $I_B$ is the indicator function of $B$.

        If $X:\Omega \to \mathbb{R}^d$ is instead a random vector of dimension $d>1$, with components

        $$
        X = (X_1, X_2, \ldots, X_d),
        $$

        then we define the *Lebesgue integral* of $X$ to be the $d$-dimensional vector whose entries are the separate Lebesgue integrals $\int_A X_j \, dP$, for $j=1,2,\ldots,d$.
        """
        from ..base.event import Event
        from ..measures.probability_measure import ProbabilityMeasure
        from ..random_objects.random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError(
                "If given, prob_measure must be a ProbabilityMeasure instance."
            )
        if event is not None and not isinstance(event, Event):
            raise TypeError("If given, the event must be an Event instance.")
        if prob_measure is not None and prob_measure.sig_alg != rv.sig_alg:
            raise ValueError(
                "If given, prob_measure must be defined on the sigma-algebra of the random vector."
            )
        if event is not None and event not in rv.sig_alg:
            raise ValueError(
                "If given, the event must be an element of the sigma-algebra of the random vector."
            )

        if prob_measure is None:
            prob_measure = rv.prob_measure

        if event is None:
            integral = cls.expectation(rv=rv, prob_measure=prob_measure).item()
            name = f"int {rv.name} d{prob_measure.name}"

        else:
            indicator = RandomVector.indicator_of(
                event=event, dim=rv.dimension
            ).with_probability_measure(rv.prob_measure)
            integral = cls.expectation(
                rv=rv * indicator, prob_measure=prob_measure
            ).item()
            name = f"int_{event.name} {rv.name} d{prob_measure.name}"

        if isinstance(integral, pd.Series):
            integral.name = name

        return integral

    @classmethod
    def expectation(
        cls,
        rv: RandomVector,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the expectation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the expectation.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.
        ValueError
            If `sig_alg` is given and is not a sub-sigma-algebra of the sigma-algebra of the random vector, or if `prob_measure` is given and is not defined on the sigma-algebra of the random vector.

        Returns
        -------
        exp : RandomVector
            The expected value of the random variable.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
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
        ...         0: 0.3,
        ...         1: 0.2,
        ...         2: 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> unconditional_expectation = Operators.expectation(rv=X)
        >>> print(unconditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X)':
        index     0    1
        sample
        0       3.4  4.4
        1       3.4  4.4
        2       3.4  4.4
        3       3.4  4.4
        4       3.4  4.4
        5       3.4  4.4
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ...     name="G",
        ... )
        >>> conditional_expectation = Operators.expectation(rv=X, sig_alg=G)
        >>> print(conditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X|G)':
        index          0         1
        sample
        0       1.000000  2.000000
        1       1.000000  2.000000
        2       4.428571  5.428571
        3       4.428571  5.428571
        4       4.428571  5.428571
        5       4.428571  5.428571
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 4,
        ...         3: 4,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> unconditional_expectation = Operators.expectation(rv=Y)
        >>> print(unconditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(Y)':
                E(Y)
        sample
        0        3.0
        1        3.0
        2        3.0
        3        3.0
        4        3.0
        5        3.0
        >>> conditional_expectation = Operators.expectation(rv=Y, sig_alg=G)
        >>> print(conditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(Y|G)':
                  E(Y|G)
        sample
        0      -1.000000
        1      -1.000000
        2       4.714286
        3       4.714286
        4       4.714286
        5       4.714286

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a probability space $(\Omega, \mathcal{F},P)$ for which $E(X^2) < \infty$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional expectation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $E(X \mid \mathcal{G})$ such that

        $$
        \int_B X \, dP = \int_B E(X\mid \mathcal{G}) \, dP,
        $$

        for all $B\in \mathcal{G}$. Using the inner product

        $$
        \langle X, Y \rangle \stackrel{\text{def}}{=} \int_\Omega XY \, dP
        $$

        in the Hilbert space $L^2(\Omega, \mathcal{F}, P)$, this equality may be rewritten as

        $$
        \langle X - E(X\mid \mathcal{G}), I_B \rangle = 0, \tag{$\ast$}
        $$

        where $I_B$ is the indicator function of $B$. In the case that $\Omega$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and the subspace $L^2(\Omega, \mathcal{G}, P)$ of $L^2(\Omega, \mathcal{F},P)$ has an orthogonal basis given by the indicator functions of the atoms of $\mathcal{G}$ with nonzero probability. Then the equation ($\ast$) shows that $E(X \mid \mathcal{G})$ is the orthogonal projection of $X$ onto $L^2(\Omega, \mathcal{G},P)$. In particular, we have the generalized Fourier expansion

        $$
        E(X\mid \mathcal{G}) = \sum_B \frac{\langle X, I_B \rangle}{\|I_B\|^2} I_B = \sum_B \frac{\int_B X \, dP}{P(B)} I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability. If for each such $B$ we define the conditional probability measure $P_B$ on $B$ with $P_B(C) = P(C)/P(B)$ for $C\subset B$, then $\int_B X \, dP/P(B)$ is the same as the (unconditional) expectation $E(X|_B) = \int_B X|_B \, dP_B$, where $X|_B : B \to \mathbb{R}$ is the restricted random variable. Thus, we have

        $$
        E(X\mid \mathcal{G}) = \sum_B E(X|_B) I_B,
        $$

        which shows that $E(X\mid \mathcal{G})$ takes the constant value $E(X|_B)$ on each atom $B$.

        If $\mathcal{G} = \{\emptyset, \Omega\}$ is the trivial $\sigma$-algebra, then $\mathcal{G}$ has only one atom, namely $\Omega$ itself. Then from above we get

        $$
        E( X \mid \{\emptyset,\Omega\}) = \frac{\int_\Omega X \, dP}{P(\Omega)} I_\Omega = E(X),
        $$

        which shows $E(X \mid \{\emptyset, \Omega\})$ is the constant random variable equal to the unconditional expectation $E(X) = \int_\Omega X \, dP$ everywhere. At the other extreme, if $\mathcal{G}$ is equal to $\mathcal{F}$, then $X$ is already $\mathcal{G}$-measurable, hence it is in $L^2(\Omega, \mathcal{G}, P)$, and hence $E(X\mid \mathcal{G}) = X$.

        Finally, if $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional expectation* to be the $d$-dimensional vector whose entries are the separate conditional expectations $E(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        from ..base.probability_space import ProbabilitySpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        cls._validate_parameters(rv=rv, sig_alg=sig_alg, prob_measure=prob_measure)

        if prob_measure is None:
            prob_measure = rv.prob_measure
            prob_space = rv.prob_space
        else:
            prob_space = ProbabilitySpace(rv.sample_space, rv.sig_alg, prob_measure)
        if sig_alg is None:
            sig_alg = SigmaAlgebra.trivial(sample_space=rv.sample_space)

        combined_sig_alg_atom_data = (
            pd.concat(
                [sig_alg.data.to_frame().add_suffix("_sub"), rv.sig_alg.data],
                axis=1,
            )
            .drop_duplicates()
            .set_index("atom_ID")
        )

        combined_data = pd.concat(
            [
                rv.atom_data,
                combined_sig_alg_atom_data,
                prob_measure.data,
            ],
            axis=1,
        )

        combined_data["normalized_prob"] = combined_data.groupby(
            "atom_ID_sub", sort=False
        )["probability"].transform(lambda x: x / x.sum())

        vector_columns = (
            list(rv.data.columns) if isinstance(rv.data, pd.DataFrame) else rv.data.name
        )

        expectation_data = combined_data.groupby("atom_ID_sub", sort=False).apply(
            lambda g: g[vector_columns].mul(g["normalized_prob"], axis=0).sum()
        )

        if sig_alg.is_trivial:
            expectation_name = f"E({rv.name})"
        else:
            expectation_name = f"E({rv.name}|{sig_alg.name})"

        if isinstance(expectation_data, pd.Series):
            expectation_data.name = expectation_name

        expectation_mapping = (
            pd.merge(
                sig_alg.data.to_frame().add_suffix("_sub"),
                expectation_data,
                on="atom_ID_sub",
            )
            .drop("atom_ID_sub", axis=1)
            .squeeze(axis=1)
        )

        expectation_mapping.index = rv.sample_space.data
        if isinstance(expectation_mapping, pd.DataFrame):
            expectation_mapping.columns = rv.index.data

        return type(rv)(*prob_space, mapping=expectation_mapping, name=expectation_name)

    @classmethod
    def variance(
        cls,
        rv: RandomVector,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the variance of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the variance.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.
        ValueError
            If `sig_alg` is given and is not a sub-sigma-algebra of the sigma-algebra of the random vector, or if `prob_measure` is given and is not defined on the sigma-algebra of the random vector.

        Returns
        -------
        var : RandomVector
            The variance of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
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
        ...         0: 0.3,
        ...         1: 0.2,
        ...         2: 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> unconditional_variance = Operators.variance(rv=X)
        >>> print(unconditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(X)':
        index      0     1
        sample
        0       3.04  3.04
        1       3.04  3.04
        2       3.04  3.04
        3       3.04  3.04
        4       3.04  3.04
        5       3.04  3.04
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ...     name="G",
        ... )
        >>> conditional_variance = Operators.variance(rv=X, sig_alg=G)
        >>> print(conditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(X|G)':
        index          0         1
        sample
        0       0.000000  0.000000
        1       0.000000  0.000000
        2       0.816327  0.816327
        3       0.816327  0.816327
        4       0.816327  0.816327
        5       0.816327  0.816327
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 4,
        ...         3: 4,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> unconditional_variance = Operators.variance(rv=Y)
        >>> print(unconditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(Y)':
                V(Y)
        sample
        0        7.0
        1        7.0
        2        7.0
        3        7.0
        4        7.0
        5        7.0
        >>> conditional_variance = Operators.variance(rv=Y, sig_alg=G)
        >>> print(conditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(Y|G)':
                  V(Y|G)
        sample
        0       0.000000
        1       0.000000
        2       0.204082
        3       0.204082
        4       0.204082
        5       0.204082

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a probability space $(\Omega, \mathcal{F},P)$ for which $E(X^2) < \infty$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional variance* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $V(X \mid \mathcal{G})$ for which

        $$
        V(X\mid \mathcal{G}) = E\left[ (X-E(X\mid \mathcal{G}))^2 \mid \mathcal{G}\right].
        $$

        In the case that $\Omega$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and the space $L^2(\Omega, \mathcal{G}, P)$ has an orthogonal basis given by the indicator functions of the atoms of $\mathcal{G}$ with nonzero probability. Then we have

        $$
        V(X\mid \mathcal{G}) = \sum_B V(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $V(X|_B)$ is the variance of the restricted random variable $X|_B:B\to \mathbb{R}$ on $B$ equipped with the conditional probability measure $P_B$ with $P_B(C) = P(C)/P(B)$ for $C\subset B$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional variance* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional variances $V(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        cls._validate_parameters(rv=rv, sig_alg=sig_alg, prob_measure=prob_measure)

        result = (
            cls.expectation(
                rv=rv**2,
                sig_alg=sig_alg,
                prob_measure=prob_measure,
            )
            - cls.expectation(rv=rv, sig_alg=sig_alg, prob_measure=prob_measure) ** 2
        )

        name = (
            f"V({rv.name}|{sig_alg.name})" if sig_alg is not None else f"V({rv.name})"
        )

        return result.with_name(name)

    @classmethod
    def std(
        cls,
        rv: RandomVector,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the standard deviation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the standard deviation.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.
        ValueError
            If `sig_alg` is given and is not a sub-sigma-algebra of the sigma-algebra of the random vector, or if `prob_measure` is given and is not defined on the sigma-algebra of the random vector.

        Returns
        -------
        std : RandomVector
            The standard deviation of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (0, 1),
        ...         2: (1, 5),
        ...         3: (1, 5),
        ...         4: (3, 2),
        ...         5: (3, 2),
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         (0, 1): 0.3,
        ...         (1, 5): 0.2,
        ...         (3, 2): 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> unconditional_std = Operators.std(rv=X)
        >>> print(unconditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(X)':
        index         0        1
        sample
        0       1.74356  1.74356
        1       1.74356  1.74356
        2       1.74356  1.74356
        3       1.74356  1.74356
        4       1.74356  1.74356
        5       1.74356  1.74356
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (0, -1),
        ...         1: (0, -1),
        ...         2: (1, 1),
        ...         3: (1, 1),
        ...         4: (1, 1),
        ...         5: (1, 1),
        ...     },
        ...     name="G",
        ... )
        >>> conditional_std = Operators.std(rv=X, sig_alg=G)
        >>> print(conditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(X|G)':
        index          0         1
        sample
        0       0.000000  0.000000
        1       0.000000  0.000000
        2       0.903508  0.903508
        3       0.903508  0.903508
        4       0.903508  0.903508
        5       0.903508  0.903508
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 4,
        ...         3: 4,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> unconditional_std = Operators.std(rv=Y)
        >>> print(unconditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(Y)':
                  std(Y)
        sample
        0       2.645751
        1       2.645751
        2       2.645751
        3       2.645751
        4       2.645751
        5       2.645751
        >>> conditional_std = Operators.std(rv=Y, sig_alg=G)
        >>> print(conditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(Y|G)':
                std(Y|G)
        sample
        0      0.000000
        1      0.000000
        2      0.451754
        3      0.451754
        4      0.451754
        5      0.451754

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a probability space $(\Omega, \mathcal{F},P)$ for which $E(X^2) < \infty$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional standard deviation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $\sigma(X \mid \mathcal{G})$ for which

        $$
        \sigma(X\mid \mathcal{G}) = \sqrt{V(X\mid \mathcal{G})}.
        $$

        In the case that $\Omega$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and the space $L^2(\Omega, \mathcal{G}, P)$ has an orthogonal basis given by the indicator functions of the atoms of $\mathcal{G}$ with nonzero probability. Then we have

        $$
        \sigma(X\mid \mathcal{G}) = \sum_B \sigma(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B)$ is the standard deviation of the restricted random variable $X|_B:B\to \mathbb{R}$ on $B$ equipped with the conditional probability measure $P_B$ with $P_B(C) = P(C)/P(B)$ for $C\subset B$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional standard deviation* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional standard deviations $\sigma(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        cls._validate_parameters(rv=rv, sig_alg=sig_alg, prob_measure=prob_measure)

        result = cls.variance(rv, sig_alg, prob_measure) ** 0.5

        name = (
            (f"std({rv.name}|{sig_alg.name})")
            if sig_alg is not None
            else f"std({rv.name})"
        )

        return result.with_name(name)

    @classmethod
    def cov(
        cls,
        rv1: RandomVariable,
        rv2: RandomVariable,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVariable:
        r"""Compute the covariance of two random variables, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv1 : RandomVariable
            The first random vector for which to compute the covariance.
        rv2 : RandomVariable
            The second random vector for which to compute the covariance
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability used to compute the covariance. If `None`, the common probability measure carried by the random variables is used (accessed through their `prob_measure` attribute).

        Raises
        ------
        TypeError
            If `rv1` or `rv2` is not a `RandomVariable`, or if `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.
        ValueError
            If `rv1` and `rv2` do not have the same domain, or if `prob_measure` is not passed and the probability measures on the random variables are not equal, or if `prob_measure` is passed and is not defined on the same sample space as `rv1`.

        Returns
        -------
        cov : RandomVariable
            The covariance of the random variables.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> P = ProbabilityMeasure.from_rand(sample_space=Omega, random_state=rng)
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0          0.320930
        1          0.318334
        2          0.037349
        3          0.311850
        4          0.011538
        >>> prob_space = ProbabilitySpace(sample_space=Omega, prob_measure=P)
        >>> X = RandomVariable.from_randint(*prob_space, low=-20, high=21, random_state=rng)
        >>> Y = RandomVariable.from_randint(
        ...     *prob_space,
        ...     low=-10,
        ...     high=11,
        ...     random_state=rng,
        ...     name="Y",
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                 X
        sample
        0        9
        1       12
        2        1
        3      -15
        4       14
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0      -1
        1       0
        2      -3
        3      -7
        4       9
        >>> unconditional_cov = Operators.cov(X, Y)
        >>> print(unconditional_cov)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'cov(X, Y)':
                cov(X, Y)
        sample
        0        36.79834
        1        36.79834
        2        36.79834
        3        36.79834
        4        36.79834
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )
        >>> conditional_cov = Operators.cov(X, Y, G)
        >>> print(conditional_cov)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'cov(X, Y|G)':
                cov(X, Y|G)
        sample
        0          0.749988
        1          0.749988
        2         19.074692
        3         19.074692
        4         19.074692

        Notes
        -----
        Let $X,Y:\Omega \to \mathbb{R}$ be two random variables on a probability space $(\Omega, \mathcal{F},P)$ for which $E(X^2), E(Y^2) < \infty$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional covariance* of $X$ and $Y$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $\sigma(X, Y \mid \mathcal{G})$ for which

        $$
        \sigma(X,Y\mid \mathcal{G}) = E(XY \mid \mathcal{G}) - E(X\mid \mathcal{G})E(Y\mid \mathcal{G}).
        $$

        In the case that $\Omega$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and the space $L^2(\Omega, \mathcal{G}, P)$ has an orthogonal basis given by the indicator functions of the atoms of $\mathcal{G}$ with nonzero probability. Then we have

        $$
        \sigma(X,Y\mid \mathcal{G}) = \sum_B \sigma(X|_B, Y|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B, Y|_B)$ is the covariance of the restricted random variables $X|_B, Y|_B:B\to \mathbb{R}$ where $B$ is equipped with the conditional probability measure $P_B$ such that $P_B(C) = P(C)/P(B)$ for $C\subset B$.
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable

        if not isinstance(rv1, RandomVariable) or not isinstance(rv2, RandomVariable):
            raise TypeError("rv1 and rv2 must be RandomVariables.")
        if rv1.sample_space != rv2.sample_space:
            raise ValueError("rv1 and rv2 must be defined on the same sample space.")
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra)
            or sig_alg.sample_space != rv1.sample_space
        ):
            raise TypeError(
                "sig_alg must be a SigmaAlgebra or None, and its sample space must match the domain of the random variables."
            )

        if prob_measure is None:
            if rv1.prob_measure != rv2.prob_measure:
                raise ValueError(
                    "If prob_measure is not passed, then the probability measures on the random variables will be used. But they are not equal."
                )
            else:
                prob_measure = rv1.prob_measure
        elif not isinstance(prob_measure, ProbabilityMeasure):
            raise TypeError("prob_measure must be a ProbabilityMeasure or None.")
        elif prob_measure.sample_space != rv1.sample_space:
            raise ValueError(
                "prob_measure must be defined on the same sample space as rv1."
            )

        result = cls.expectation(rv1 * rv2, sig_alg, prob_measure) - cls.expectation(
            rv1, sig_alg, prob_measure
        ) * cls.expectation(rv2, sig_alg, prob_measure)

        name = (
            f"cov({rv1.name}, {rv2.name}|{sig_alg.name})"
            if sig_alg is not None
            else f"cov({rv1.name}, {rv2.name})"
        )

        return result.with_name(name)

    @classmethod
    def corr(
        cls,
        rv1: RandomVector,
        rv2: RandomVector,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVariable:
        r"""Compute the correlation of two random variables, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv1 : RandomVariable
            The first random vector for which to compute the correlation.
        rv2 : RandomVariable
            The second random vector for which to compute the correlation
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability used to compute the correlation. If `None`, the common probability measure carried by the random variables is used (accessed through their `prob_measure` attribute).

        Raises
        ------
        TypeError
            If `rv1` or `rv2` is not a `RandomVariable`, or if `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.
        ValueError
            If `rv1` and `rv2` do not have the same domain, or if `prob_measure` is not passed and the probability measures on the random variables are not equal, or if `prob_measure` is passed and is not defined on the same sample space as `rv1`.

        Returns
        -------
        corr : RandomVariable
            The correlation of the two random variables.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> P = ProbabilityMeasure.from_rand(sample_space=Omega, random_state=rng)
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0          0.320930
        1          0.318334
        2          0.037349
        3          0.311850
        4          0.011538
        >>> prob_space = ProbabilitySpace(sample_space=Omega, prob_measure=P)
        >>> X = RandomVariable.from_randint(*prob_space, low=-20, high=21, random_state=rng)
        >>> Y = RandomVariable.from_randint(
        ...     *prob_space,
        ...     low=-10,
        ...     high=11,
        ...     random_state=rng,
        ...     name="Y",
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        sample
        0        9
        1       12
        2        1
        3      -15
        4       14
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
               Y
        sample
        0      -1
        1       0
        2      -3
        3      -7
        4       9
        >>> unconditional_corr = Operators.corr(X, Y)
        >>> print(unconditional_corr)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'corr(X, Y)':
               corr(X, Y)
        sample
        0         0.959264
        1         0.959264
        2         0.959264
        3         0.959264
        4         0.959264
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )
        >>> conditional_corr = Operators.corr(X, Y, G)
        >>> print(conditional_corr)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'corr(X, Y|G)':
               corr(X, Y|G)
        sample
        0             1.0000
        1             1.0000
        2             0.9308
        3             0.9308
        4             0.9308

        Notes
        -----
        Let $X,Y:\Omega \to \mathbb{R}$ be two random variables on a probability space $(\Omega, \mathcal{F},P)$ for which $E(X^2), E(Y^2) < \infty$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional correlation* of $X$ and $Y$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $\rho(X, Y \mid \mathcal{G})$ for which

        $$
        \rho(X,Y\mid \mathcal{G}) = \frac{\sigma(X,Y \mid \mathcal{G})}{\sigma(X\mid \mathcal{G})\sigma(Y\mid \mathcal{G})},
        $$

        provided that the standard deviations in the denominator are nonzero. In the case that $\Omega$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and the space $L^2(\Omega, \mathcal{G}, P)$ has an orthogonal basis given by the indicator functions of the atoms of $\mathcal{G}$ with nonzero probability. Then we have

        $$
        \rho(X,Y\mid \mathcal{G}) = \sum_B \rho(X|_B, Y|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\rho(X|_B, Y|_B)$ is the correlation of the restricted random variables $X|_B, Y|_B:B\to \mathbb{R}$ where $B$ is equipped with the conditional probability measure $P_B$ such that $P_B(C) = P(C)/P(B)$ for $C\subset B$.

        See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable

        if not isinstance(rv1, RandomVariable) or not isinstance(rv2, RandomVariable):
            raise TypeError("rv1 and rv2 must be RandomVariables.")
        if rv1.sample_space != rv2.sample_space:
            raise ValueError("rv1 and rv2 must be defined on the same sample space.")
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra)
            or sig_alg.sample_space != rv1.sample_space
        ):
            raise TypeError(
                "sig_alg must be a SigmaAlgebra or None, and its sample space must match the domain of the random variables."
            )

        if prob_measure is None:
            if rv1.prob_measure != rv2.prob_measure:
                raise ValueError(
                    "If prob_measure is not passed, then the probability measures on the random variables will be used. But they are not equal."
                )
            else:
                prob_measure = rv1.prob_measure
        elif not isinstance(prob_measure, ProbabilityMeasure):
            raise TypeError("prob_measure must be a ProbabilityMeasure or None.")
        elif prob_measure.sample_space != rv1.sample_space:
            raise ValueError(
                "prob_measure must be defined on the same sample space as rv1."
            )

        result = cls.cov(rv1, rv2, sig_alg, prob_measure) / (
            cls.std(rv1, sig_alg, prob_measure) * cls.std(rv2, sig_alg, prob_measure)
        )

        name = (
            f"corr({rv1.name}, {rv2.name}|{sig_alg.name})"
            if sig_alg is not None
            else f"corr({rv1.name}, {rv2.name})"
        )

        return result.with_name(name)

    @classmethod
    def pushforward(
        cls,
        rv: RandomVector,
        prob_measure: ParametrizedProbabilityMeasure | ProbabilityMeasure | None = None,
    ) -> ParametrizedProbabilityMeasure | ProbabilityMeasure:
        r"""Push forward a (parametrized) probability measure on the domain of a random vector to a probability measure on its range.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            Random vector.
        prob_measure : ParametrizedProbabilityMeasure | ProbabilityMeasure | None, default=None
            (Parametrized) probability measure to push forward. If `None`, the probability measure carried by the random vector is used.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `prob_measure` is not a `ParametrizedProbabilityMeasure` or `ProbabilityMeasure` (if given).
        ValueError
            If `rv` is not defined on the sample space of `prob_measure` (if given), or if the sigma-algebra of `rv` is not the sigma-algebra of `prob_measure` (if given).

        Returns
        -------
        pushforward_measure : ParametrizedProbabilityMeasure | ProbabilityMeasure
            The resulting probability measure `P_X`.

        Examples
        --------
        Define a probability space and a random vector `X`.

        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> from sigalg.processes import RandomWalk
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.35,
        ...         2: 0.4,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         3: (0, 1),
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (1, 2),
        ...     },
        ... )

        Push forward the probability measure `P` on the domain of `X` to a probability measure `P_X` on the range of `X`.

        >>> P_X = Operators.pushforward(rv=X, prob_measure=P)
        >>> print(P_X)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_X':
                 probability
        X_0 X_1
        1   2            0.6
        0   1            0.4

        Define a random walk `Y`.

        >>> Y = RandomWalk.generate(
        ...     mode="enum",
        ...     p=0.7,
        ...     length=2,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'Y':
        time    0  1  2
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2

        Extract the probability measure `Q` from `Y`.

        >>> Q = Y.prob_measure.with_name("Q")
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        sample
        0              0.09
        1              0.21
        2              0.21
        3              0.49

        Condition on the random walk trajectories up to time 1 and obtain a parametrized probability measure.

        >>> Q_conditional = Q.given(Y[0, 1])
        >>> print(Q_conditional)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q(?|Y_0, Y_1)':
                        probability
        Y_0 Y_1 sample
        0   -1  0               0.3
                1               0.7
                2               0.0
                3               0.0
             1  0               0.0
                1               0.0
                2               0.3
                3               0.7

        Push forward the parametrized probability measure `Q_conditional` to a parametrized probability measure on the range of `Y[2]`.

        >>> Q_conditional_Y_2 = Q_conditional >> Y[2]
        >>> print(Q_conditional_Y_2)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q(?|Y_0, Y_1)_Y_2':
                     probability
        Y_0 Y_1 Y_2
        0   -1  -2           0.3
                 0           0.7
                 2           0.0
             1  -2           0.0
                 0           0.3
                 2           0.7

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F},P)$. Then we define a probability measure $P_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $P$, by setting

        $$
        P_X(A) = P\left( \{\omega \in \Omega : X(\omega) \in A\}\right),
        $$

        for all Borel subsets $A\subset \mathbb{R}^d$.

        If $P$ is a parametrized probability measure on $\Omega$ with parameter domain $\Theta$, then we define a parametrized probability measure $P_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $P$, by setting

        $$
        P_X(\theta, A) = P\left(\theta, \{\omega \in \Omega : X(\omega) \in A\}\right),
        $$

        for all $\theta \in \Theta$ and all Borel subsets $A\subset \mathbb{R}^d$.
        """
        from ..base.domain import Domain
        from ..base.sample_space import SampleSpace
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ..measures.probability_measure import ProbabilityMeasure
        from ..random_objects.random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")
        if prob_measure is not None and not isinstance(
            prob_measure, ParametrizedProbabilityMeasure | ProbabilityMeasure
        ):
            raise TypeError(
                "prob_measure must be a ParametrizedProbabilityMeasure or ProbabilityMeasure instance."
            )
        if prob_measure is not None and (
            rv.sample_space != prob_measure.sample_space
            or rv.sig_alg != prob_measure.sig_alg
        ):
            raise ValueError(
                "rv must be defined on the sample space of prob_measure, and its sigma-algebra must match that of prob_measure."
            )

        if prob_measure is None:
            prob_measure = rv.prob_measure

        rv_atom_data = rv.atom_data.copy()
        rv_atom_data.columns = rv.component_names
        rv_atom_data.index = Domain(
            indices=list(rv.atom_data.index), variable_names=rv.sig_alg.variable_names
        ).data

        mapping = pd.merge(
            left=prob_measure.data,
            right=rv_atom_data,
            left_index=True,
            right_index=True,
        )
        domain_variable_names = (
            prob_measure.parameter_names + rv.component_names
            if isinstance(prob_measure, ParametrizedProbabilityMeasure)
            else rv.component_names
        )
        mapping = (
            mapping.reset_index()
            .groupby(domain_variable_names, sort=False)["probability"]
            .sum()
        )

        domain_variable_names = [
            name.replace(".", "_") for name in domain_variable_names
        ]

        name = (
            f"{prob_measure.name}_{rv.name}"
            if (isinstance(prob_measure.name, str) and isinstance(rv.name, str))
            else "pushforward"
        )

        if isinstance(prob_measure, ParametrizedProbabilityMeasure):
            if isinstance(rv.data, pd.DataFrame):
                indices = rv.data.drop_duplicates().apply(tuple, axis=1).to_list()
            else:
                indices = rv.data.drop_duplicates().to_list()
            range_sample_space = SampleSpace(
                indices=indices,
                name=f"{rv.name}_range",
                variable_names=rv.component_names,
            )
            return ParametrizedProbabilityMeasure(
                sample_space=range_sample_space,
                mapping=mapping,
                name=name,
            )
        else:
            range_sample_space = SampleSpace(
                indices=mapping.index,
                name=f"{rv.name}_range",
            )
            return ProbabilityMeasure(
                sample_space=range_sample_space,
                mapping=mapping,
                name=name,
            )

    @staticmethod
    def _validate_parameters(
        rv: RandomVector,
        sig_alg: SigmaAlgebra | None,
        prob_measure: ProbabilityMeasure | None,
    ):
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("If given, sig_alg must be a SigmaAlgebra instance.")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError(
                "If given, prob_measure must be a ProbabilityMeasure instance."
            )
        if sig_alg is not None and not sig_alg <= rv.sig_alg:
            raise ValueError(
                "If given, sig_alg must be a sub-sigma-algebra of the random vector's sigma-algebra."
            )
        if prob_measure is not None and prob_measure.sig_alg != rv.sig_alg:
            raise ValueError(
                "If given, prob_measure must be defined on the sigma-algebra of the random vector."
            )


class OperatorsMethods:
    """Mixin class to add operators as methods to `RandomVector` and `ProbabilityMeasure`."""

    def integrate(
        self,
        *,
        rv: RandomVector | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        event: Event | None = None,
    ) -> pd.Series | Real:
        """Compute the Lebesgue integral of a random vector with respect to a probability measure over an (optional) event.

        Calls `Operators.integrate` with appropriate arguments. See the docstring of `Operators.integrate` for details.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector to integrate. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `self` is a random vector and `prob_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.
        event : Event | None, default=None
            The optional event over which to integrate. If `None`, the integral will be taken over the entire sample space contained in the `domain` attribute of the random vector.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `prob_measure` is not `None` or not equal to `self`.

        Returns
        -------
        integral : pd.Series | Real
            If `rv` has dimension > 1, returns a pd.Series representing the integral of each component of the random vector. If `rv` has dimension 1, returns a Real representing the integral.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
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
        ...         0: 0.3,
        ...         1: 0.2,
        ...         2: 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> A = prob_space.get_event([0, 1, 2, 3])
        >>> integral = X.integrate(event=A)
        >>> print(integral)
        index
        0    0.9
        1    1.4
        Name: int_A X dP, dtype: float64
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 3,
        ...         3: 3,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> integral = Y.integrate(event=A)
        >>> print(integral)
        0.9000000000000001
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling integrate on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.integrate(
                rv=self,
                prob_measure=prob_measure,
                event=event,
            )
        elif isinstance(self, ProbabilityMeasure):
            if prob_measure is not None and prob_measure != self:
                raise ValueError(
                    "prob_measure must be None or equal to self when calling integrate on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.integrate(
                rv=rv,
                prob_measure=self,
                event=event,
            )

    def expectation(
        self,
        *,
        rv: RandomVector | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the expectation of a random vector, optionally conditioned on a sigma algebra.

        Calls `Operators.expectation` with appropriate arguments. See the docstring of `Operators.expectation` for details.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the expectation. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional expectation.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `prob_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `prob_measure` is not `None` or not equal to `self`.

        Returns
        -------
        exp : RandomVector
            The expectation of the random vector, optionally conditioned on the sigma algebra and with respect to the specified probability measure.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
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
        ...         0: 0.3,
        ...         1: 0.2,
        ...         2: 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> unconditional_expectation = X.expectation()
        >>> print(unconditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X)':
        index     0    1
        sample
        0       3.4  4.4
        1       3.4  4.4
        2       3.4  4.4
        3       3.4  4.4
        4       3.4  4.4
        5       3.4  4.4
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ...     name="G",
        ... )
        >>> conditional_expectation = X.expectation(sig_alg=G)
        >>> print(conditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X|G)':
        index          0         1
        sample
        0       1.000000  2.000000
        1       1.000000  2.000000
        2       4.428571  5.428571
        3       4.428571  5.428571
        4       4.428571  5.428571
        5       4.428571  5.428571
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 4,
        ...         3: 4,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> unconditional_expectation = Y.expectation()
        >>> print(unconditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(Y)':
                E(Y)
        sample
        0        3.0
        1        3.0
        2        3.0
        3        3.0
        4        3.0
        5        3.0
        >>> conditional_expectation = Y.expectation(sig_alg=G)
        >>> print(conditional_expectation)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(Y|G)':
                  E(Y|G)
        sample
        0      -1.000000
        1      -1.000000
        2       4.714286
        3       4.714286
        4       4.714286
        5       4.714286
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling expectation on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.expectation(
                rv=self,
                sig_alg=sig_alg,
                prob_measure=prob_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if prob_measure is not None and prob_measure != self:
                raise ValueError(
                    "prob_measure must be None or equal to self when calling expectation on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.expectation(
                rv=rv,
                sig_alg=sig_alg,
                prob_measure=self,
            )

    def variance(
        self,
        *,
        rv: RandomVector | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the variance of a random vector, optionally conditioned on a sigma algebra.

        Calls `Operators.variance` with appropriate arguments. See the docstring of `Operators.variance` for details.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the variance. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional variance.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `prob_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `prob_measure` is not `None` or not equal to `self`.

        Returns
        -------
        var : RandomVector
            The variance of the random vector, optionally conditioned on the sigma algebra and with respect to the specified probability measure.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
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
        ...         0: 0.3,
        ...         1: 0.2,
        ...         2: 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> unconditional_variance = X.variance()
        >>> print(unconditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(X)':
        index      0     1
        sample
        0       3.04  3.04
        1       3.04  3.04
        2       3.04  3.04
        3       3.04  3.04
        4       3.04  3.04
        5       3.04  3.04
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ...     name="G",
        ... )
        >>> conditional_variance = X.variance(sig_alg=G)
        >>> print(conditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(X|G)':
        index          0         1
        sample
        0       0.000000  0.000000
        1       0.000000  0.000000
        2       0.816327  0.816327
        3       0.816327  0.816327
        4       0.816327  0.816327
        5       0.816327  0.816327
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 4,
        ...         3: 4,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> unconditional_variance = Y.variance()
        >>> print(unconditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(Y)':
                V(Y)
        sample
        0        7.0
        1        7.0
        2        7.0
        3        7.0
        4        7.0
        5        7.0
        >>> conditional_variance = Y.variance(sig_alg=G)
        >>> print(conditional_variance)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(Y|G)':
                  V(Y|G)
        sample
        0       0.000000
        1       0.000000
        2       0.204082
        3       0.204082
        4       0.204082
        5       0.204082
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling variance on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.variance(
                rv=self,
                sig_alg=sig_alg,
                prob_measure=prob_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if prob_measure is not None and prob_measure != self:
                raise ValueError(
                    "prob_measure must be None or equal to self when calling variance on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.variance(
                rv=rv,
                sig_alg=sig_alg,
                prob_measure=self,
            )

    def std(
        self,
        *,
        rv: RandomVector | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the standard deviation of a random vector, optionally conditioned on a sigma algebra.

        Calls `Operators.std` with appropriate arguments. See the docstring of `Operators.std` for details.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the standard deviation. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional standard deviation.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `prob_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `prob_measure` is not `None` or not equal to `self`.

        Returns
        -------
        std : RandomVector
            The standard deviation of the random vector, optionally conditioned on the sigma algebra and with respect to the specified probability measure.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (0, 1),
        ...         2: (1, 5),
        ...         3: (1, 5),
        ...         4: (3, 2),
        ...         5: (3, 2),
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         (0, 1): 0.3,
        ...         (1, 5): 0.2,
        ...         (3, 2): 0.5,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...         4: (5, 6),
        ...         5: (5, 6),
        ...     },
        ... )
        >>> unconditional_std = X.std()
        >>> print(unconditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(X)':
        index         0        1
        sample
        0       1.74356  1.74356
        1       1.74356  1.74356
        2       1.74356  1.74356
        3       1.74356  1.74356
        4       1.74356  1.74356
        5       1.74356  1.74356
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (0, -1),
        ...         1: (0, -1),
        ...         2: (1, 1),
        ...         3: (1, 1),
        ...         4: (1, 1),
        ...         5: (1, 1),
        ...     },
        ...     name="G",
        ... )
        >>> conditional_std = X.std(sig_alg=G)
        >>> print(conditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(X|G)':
        index          0         1
        sample
        0       0.000000  0.000000
        1       0.000000  0.000000
        2       0.903508  0.903508
        3       0.903508  0.903508
        4       0.903508  0.903508
        5       0.903508  0.903508
        >>> Y = RandomVariable(
        ...     *prob_space,
        ...     mapping={
        ...         0: -1,
        ...         1: -1,
        ...         2: 4,
        ...         3: 4,
        ...         4: 5,
        ...         5: 5,
        ...     },
        ...     name="Y",
        ... )
        >>> unconditional_std = Y.std()
        >>> print(unconditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(Y)':
                  std(Y)
        sample
        0       2.645751
        1       2.645751
        2       2.645751
        3       2.645751
        4       2.645751
        5       2.645751
        >>> conditional_std = Y.std(sig_alg=G)
        >>> print(conditional_std)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(Y|G)':
                std(Y|G)
        sample
        0      0.000000
        1      0.000000
        2      0.451754
        3      0.451754
        4      0.451754
        5      0.451754
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling std on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.std(
                rv=self,
                sig_alg=sig_alg,
                prob_measure=prob_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if prob_measure is not None and prob_measure != self:
                raise ValueError(
                    "prob_measure must be None or equal to self when calling std on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.std(
                rv=rv,
                sig_alg=sig_alg,
                prob_measure=self,
            )

    def pushforward(
        self,
        *,
        rv: RandomVector | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilityMeasure:
        """Push forward a probability measure on the domain of a random vector to a probability measure on its range.

        Calls `Operators.pushforward` with appropriate arguments. See the docstring of `Operators.pushforward` for details.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector to push forward. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure to push forward. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `prob_measure` is not `None` or not equal to `self`.

        Returns
        -------
        pushforward_measure : ProbabilityMeasure
            The resulting probability measure after pushing forward.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.35,
        ...         2: 0.4,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         3: (0, 1),
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (1, 2),
        ...     },
        ... )
        >>> pushforward = X.pushforward(prob_measure=P)
        >>> print(pushforward)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_X':
                 probability
        X_0 X_1
        1   2            0.6
        0   1            0.4
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling pushforward on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.pushforward(
                rv=self,
                prob_measure=prob_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if prob_measure is not None and prob_measure != self:
                raise ValueError(
                    "prob_measure must be None or equal to self when calling pushforward on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.pushforward(
                rv=rv,
                prob_measure=self,
            )
