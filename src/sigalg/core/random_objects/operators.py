"""Class for operators on random vectors, such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of probability measures."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..base.event import Event
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .random_vector import RandomVector


class Operators:
    """Class containing methods such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of probability measures.

    The class does not have an `__init__` method, and all methods are class methods.
    """

    @classmethod
    def integrate(
        cls,
        rv: RandomVector,
        probability_measure: ProbabilityMeasure | None = None,
        event: Event | None = None,
    ) -> pd.Series | Real:
        r"""Compute the Lebesgue integral of a random vector with respect to a probability measure over an (optional) event.

        Let $X: \Omega \to \mathbb{R}$ be a random variable on a probability space $(\Omega, \mathcal{F},P)$. This method computes the Lebesgue integral

        $$
        \int_A X \, dP,
        $$

        where $A$ is an event in $\mathcal{F}$. If $\Omega$ is finite (as it always is, in SigAlg), then the Lebesgue integral reduces to a finite sum

        $$
        \sum_{\omega \in A} X(\omega) P(\{\omega\}).
        $$

        While in the mathematical theory $A$ is supposed to be an $\mathcal{F}$-measurable subset of $\Omega$, this requirement is not enforced in SigAlg. If the event $A$ is not specified, it defaults to the sample space itself $A = \Omega$. If the measure $P$ is not specified, it defaults to the measure carried by the random variable in its `probability_measure` attribute.

        If $X:\Omega \to \mathbb{R}^d$ is instead a random vector of dimension $d>1$, with components

        $$
        X = (X_1, X_2, \ldots, X_d),
        $$

        then this method returns a `pd.Series` object whose values are the separate Lebesgue integrals $\int_A X_j \, dP$, for $j=1,2,\ldots,d$.

        Parameters
        ----------
        rv : RandomVector
            The random vector to integrate.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).
        event: Event | None, default=None
            The optional event over which to integrate. If `None`, the integral will be taken over the entire sample space contined in the `domain` attribute of the random vector.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `probability_measure` is not a `ProbabilityMeasure` or `None`, or if `event` is not an `Event` or `None`, or if their sample spaces do not match.

        Returns
        -------
        integral : pd.Series | Real
            If `rv` has dimension > 1, returns a pd.Series representing the integral of each component of the random vector. If `rv` has dimension 1, returns a Real representing the integral.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (1, 2), 2: (3, 4)})
        >>> # Integral of a 2-dimensional random vector
        >>> Operators.integrate(rv=X, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        integral
        integral(X_0)    2.0
        integral(X_1)    3.0
        Name: integral(X), dtype: float64
        >>> # Integral of a random variable
        >>> Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 1, 1: 1, 2: 0})
        >>> float(Operators.integrate(rv=Y, probability_measure=P))
        0.5
        """
        from ..base.event import Event
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..random_objects.random_variable import RandomVariable
        from ..random_objects.random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector.")
        if probability_measure is not None and (
            not isinstance(probability_measure, ProbabilityMeasure)
            or probability_measure.sample_space != rv.domain
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure or None, and its sample space must match the domain of the random vector."
            )
        if event is not None and (
            not isinstance(event, Event) or event.sample_space != rv.domain
        ):
            raise TypeError(
                "event must be an Event or None, and its sample space must match the domain of the random vector."
            )

        if event is None:
            event = rv.domain.get_event(list(rv.domain))
        if probability_measure is None:
            probability_measure = rv.probability_measure
        if isinstance(rv, RandomVariable):
            indicator = RandomVariable.indicator_of(
                event=event
            ).with_probability_measure(probability_measure=probability_measure)
        else:
            indicator = RandomVector.indicator_of(
                event=event, dim=rv.dimension
            ).with_probability_measure(probability_measure=probability_measure)

        integrand = rv * indicator
        exp = cls.expectation(rv=integrand, probability_measure=probability_measure)
        integral = exp.data.iloc[0]
        if rv.dimension > 1:
            index_names = [f"integral({idx_name})" for idx_name in rv.index]
            integral.index = pd.Index(index_names, name="integral")
            integral.name = f"integral({rv.name})" if rv.name is not None else None

        return integral

    @classmethod
    def expectation(
        cls,
        rv: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the expectation of a random vector with respect to a probability measure, optionally conditioned on a sigma-algebra.

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

        then this method returns a `RandomVector` whose component random variables are the conditional expectations $E(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the expectation.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability used to compute the expectation. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).

        Raises
        ------
        TypeError
            If `rv` is not a RandomVector, or if `sigma_algebra` is not a `SigmaAlgebra` or `None`, or if `probability_measure` is not a ProbabilityMeausre or `None`.

        Returns
        -------
        exp : RandomVector
            The expected value of the random variable.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.15,
        ...         2: 0.65,
        ...     }
        ... )
        >>> X = RandomVariable(domain=Omega).from_dict(
        ...     {
        ...         0: -1,
        ...         1: 2,
        ...         2: 4,
        ...     }
        ... )
        >>> X.probability_measure = P
        >>> conditional_exp = Operators.expectation(rv=X, sigma_algebra=G)
        >>> print(conditional_exp) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(X|G)':
                E(X|G)
        sample
        0       -1.000
        1        3.625
        2        3.625
        >>> unconditional_exp = Operators.expectation(rv=X)
        >>> print(unconditional_exp) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(X)':
                E(X)
        sample
        0        2.7
        1        2.7
        2        2.7
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (-1, 3),
        ...         2: (4, 0),
        ...     }
        ... )
        >>> Y.probability_measure = P
        >>> conditional_exp = Operators.expectation(rv=Y, sigma_algebra=G)
        >>> print(conditional_exp) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(Y|G)':
        expectation  E(Y_0|G)  E(Y_1|G)
        sample
        0              1.0000    2.0000
        1              3.0625    0.5625
        2              3.0625    0.5625
        >>> unconditional_exp = Operators.expectation(rv=Y)
        >>> print(unconditional_exp) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(Y)':
        expectation  E(Y_0)  E(Y_1)
        sample
        0              2.65    0.85
        1              2.65    0.85
        2              2.65    0.85
        """
        from ..base.index import Index
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector.")
        if sigma_algebra is not None and (
            not isinstance(sigma_algebra, SigmaAlgebra)
            or sigma_algebra.sample_space != rv.domain
        ):
            raise TypeError(
                "sigma_algebra must be a SigmaAlgebra or None, and its sample space must match the domain of the random vector."
            )
        if probability_measure is not None and (
            not isinstance(probability_measure, ProbabilityMeasure)
            or probability_measure.sample_space != rv.domain
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure or None, and its sample space must match the domain of the random vector."
            )

        if probability_measure is None:
            probability_measure = rv.probability_measure

        if sigma_algebra is None:
            probabilities = probability_measure.data
            expectations = rv.data.mul(probabilities, axis=0).sum()
            expectations_name = f"E({rv.name})" if rv.name is not None else None
            if isinstance(expectations, pd.Series):
                result = RandomVector(
                    domain=rv.domain, name=expectations_name
                ).from_dict(dict.fromkeys(rv.domain, tuple(expectations)))
                indices = [f"E({idx_name})" for idx_name in rv.index]
                index = Index(name="index", data_name="expectation").from_list(indices)
                result.index = index
            else:
                result = RandomVariable(
                    domain=rv.domain, name=expectations_name
                ).from_dict(dict.fromkeys(rv.domain, expectations))

        else:
            df = pd.concat(
                [rv.data, sigma_algebra.data, probability_measure.data], axis=1
            )

            df["normalized_prob"] = df.groupby("atom ID")["probability"].transform(
                lambda x: x / x.sum()
            )

            vector_cols = (
                rv.data.columns if isinstance(rv.data, pd.DataFrame) else [rv.data.name]
            )
            expected_df = df.groupby("atom ID", group_keys=False).apply(
                cls._compute_expectation_of_group,
                vector_cols=vector_cols,
                include_groups=False,
            )

            outputs = {idx: tuple(row) for idx, row in expected_df.iterrows()}

            name = (
                f"E({rv.name}|{sigma_algebra.name})"
                if rv.name is not None and sigma_algebra.name is not None
                else None
            )

            if rv.dimension == 1:
                result = RandomVariable(domain=rv.domain, name=name).from_dict(outputs)
            else:
                result = RandomVector(domain=rv.domain, name=name).from_dict(outputs)
                result.data.fillna(0, inplace=True)
                indices = [
                    f"E({idx_name}|{sigma_algebra.name})" for idx_name in rv.index
                ]
                index = Index(name="index", data_name="expectation").from_list(indices)
                result.index = index

        return result.with_probability_measure(
            probability_measure=rv.probability_measure
        )

    @classmethod
    def _compute_expectation_of_group(cls, group, vector_cols):
        weights = group["normalized_prob"].values[:, None]
        expected = (group[vector_cols].values * weights).sum(axis=0)
        return pd.DataFrame(
            [expected] * len(group), index=group.index, columns=vector_cols
        )

    @classmethod
    def variance(
        cls,
        rv: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the variance of a random vector, optionally conditioned on a sigma-algebra.

        Let $X:\Omega \to \mathbb{R}$ be a random variable on a probability space $(\Omega, \mathcal{F},P)$ for which $E(X^2) < \infty$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional variance* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $V(X \mid \mathcal{G})$ for which

        $$
        V(X\mid \mathcal{G}) = E\left( (X - E(X\mid \mathcal{G}))^2 \mid \mathcal{G} \right).
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

        then this method returns a `RandomVector` whose component random variables are the conditional variances $V(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the variance.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability used to compute the variance. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sigma_algebra` is not a `SigmaAlgebra` or `None`, or if `probability_measure` is not a `ProbabilityMeasure` or `None`.

        Returns
        -------
        var : RandomVector
            The variance of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.15,
        ...         2: 0.65,
        ...     }
        ... )
        >>> X = RandomVariable(domain=Omega).from_dict(
        ...     {
        ...         0: -1,
        ...         1: 2,
        ...         2: 4,
        ...     }
        ... )
        >>> X.probability_measure = P
        >>> conditional_var = Operators.variance(rv=X, sigma_algebra=G)
        >>> print(conditional_var) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(X|G)':
                V(X|G)
        sample
        0       0.000000
        1       0.609375
        2       0.609375
        >>> unconditional_var = Operators.variance(rv=X)
        >>> print(unconditional_var) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(X)':
                V(X)
        sample
        0       3.91
        1       3.91
        2       3.91
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (-1, 3),
        ...         2: (4, 0),
        ...     }
        ... )
        >>> Y.probability_measure = P
        >>> conditional_var = Operators.variance(rv=Y, sigma_algebra=G)
        >>> print(conditional_var) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(Y|G)':
        variance  V(Y_0|G)  V(Y_1|G)
        sample
        0         0.000000  0.000000
        1         3.808594  1.371094
        2         3.808594  1.371094
        >>> unconditional_var = Operators.variance(rv=Y)
        >>> print(unconditional_var) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(Y)':
        variance  V(Y_0)  V(Y_1)
        sample
        0         3.7275  1.4275
        1         3.7275  1.4275
        2         3.7275  1.4275
        """
        from ..base.index import Index
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector.")
        if sigma_algebra is not None and (
            not isinstance(sigma_algebra, SigmaAlgebra)
            or sigma_algebra.sample_space != rv.domain
        ):
            raise TypeError(
                "sigma_algebra must be a SigmaAlgebra or None, and its sample space must match the domain of the random vector."
            )
        if probability_measure is not None and (
            not isinstance(probability_measure, ProbabilityMeasure)
            or probability_measure.sample_space != rv.domain
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure or None, and its sample space must match the domain of the random vector."
            )

        exp = cls.expectation(
            rv, sigma_algebra=sigma_algebra, probability_measure=probability_measure
        )
        result = cls.expectation(
            (rv - exp) ** 2,
            sigma_algebra=sigma_algebra,
            probability_measure=probability_measure,
        )

        if sigma_algebra is not None:
            name = (
                f"V({rv.name}|{sigma_algebra.name})"
                if rv.name is not None and sigma_algebra.name is not None
                else None
            )
            if rv.dimension > 1:
                indices = [
                    f"V({idx_name}|{sigma_algebra.name})" for idx_name in rv.index
                ]
                index = Index(name="index", data_name="variance").from_list(indices)
                result.index = index
        else:
            name = f"V({rv.name})" if rv.name is not None else None
            if rv.dimension > 1:
                indices = [f"V({idx_name})" for idx_name in rv.index]
                index = Index(name="index", data_name="variance").from_list(indices)
                result.index = index

        result = result.with_name(name)

        return result

    @classmethod
    def std(
        cls,
        rv: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the standard deviation of a random vector, optionally conditioned on a sigma-algebra.

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

        then this method returns a `RandomVector` whose component random variables are the conditional standard deviations $\sigma(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the standard deviation.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability used to compute the standard deviation. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sigma_algebra` is not a `SigmaAlgebra` or `None`, or if `probability_measure` is not a `ProbabilityMeasure` or `None`.

        Returns
        -------
        std : RandomVector
            The standard deviation of the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.15,
        ...         2: 0.65,
        ...     }
        ... )
        >>> X = RandomVariable(domain=Omega).from_dict(
        ...     {
        ...         0: -1,
        ...         1: 2,
        ...         2: 4,
        ...     }
        ... )
        >>> X.probability_measure = P
        >>> conditional_std = Operators.std(rv=X, sigma_algebra=G)
        >>> print(conditional_std) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(X|G)':
                std(X|G)
        sample
        0       0.000000
        1       0.780625
        2       0.780625
        >>> unconditional_var = Operators.variance(rv=X)
        >>> unconditional_std = Operators.std(rv=X)
        >>> print(unconditional_std) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(X)':
                std(X)
        sample
        0       1.977372
        1       1.977372
        2       1.977372
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (-1, 3),
        ...         2: (4, 0),
        ...     }
        ... )
        >>> Y.probability_measure = P
        >>> conditional_std = Operators.std(rv=Y, sigma_algebra=G)
        >>> print(conditional_std) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(Y|G)':
        std     std(Y_0|G)  std(Y_1|G)
        sample
        0         0.000000    0.000000
        1         1.951562    1.170937
        2         1.951562    1.170937
        >>> unconditional_std = Operators.std(rv=Y)
        >>> print(unconditional_std) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(Y)':
        std     std(Y_0)  std(Y_1)
        sample
        0       1.930673   1.19478
        1       1.930673   1.19478
        2       1.930673   1.19478
        """
        from ..base.index import Index
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector.")
        if sigma_algebra is not None and (
            not isinstance(sigma_algebra, SigmaAlgebra)
            or sigma_algebra.sample_space != rv.domain
        ):
            raise TypeError(
                "sigma_algebra must be a SigmaAlgebra or None, and its sample space must match the domain of the random vector."
            )
        if probability_measure is not None and (
            not isinstance(probability_measure, ProbabilityMeasure)
            or probability_measure.sample_space != rv.domain
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure or None, and its sample space must match the domain of the random vector."
            )

        result = (
            cls.variance(
                rv, sigma_algebra=sigma_algebra, probability_measure=probability_measure
            )
            ** 0.5
        )

        if sigma_algebra is not None:
            name = (
                f"std({rv.name}|{sigma_algebra.name})"
                if rv.name is not None and sigma_algebra.name is not None
                else None
            )
            if rv.dimension > 1:
                indices = [
                    f"std({idx_name}|{sigma_algebra.name})" for idx_name in rv.index
                ]
                index = Index(name="index", data_name="std").from_list(indices)
                result.index = index
        else:
            name = f"std({rv.name})" if rv.name is not None else None
            if rv.dimension > 1:
                indices = [f"std({idx_name})" for idx_name in rv.index]
                index = Index(name="index", data_name="std").from_list(indices)
                result.index = index

        result = result.with_name(name)

        return result

    # TODO: Update docstrings
    @classmethod
    def covariance(
        cls,
        rv1: RandomVector,
        rv2: RandomVector | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> pd.DataFrame | Real:
        """Compute the covariance matrix of one or two random vectors.

        If `rv2` is provided, computes the covariance matrix Cov(rv1, rv2). If `rv2` is `None`, computes the covariance matrix Cov(rv1, rv1). If `probability_measure` is `None`, uses the probability measure carried by `rv1`. If both random vectors have dimension 1, returns a scalar covariance.

        Parameters
        ----------
        rv1 : RandomVector
            The first random vector.
        rv2 : RandomVector | None, default=None
            The second random vector. If `None`, computes Cov(rv1, rv1).
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `None`, uses `rv1.probability_measure`.

        Raises
        ------
        TypeError
            If `rv1` is not a `RandomVector`, or if `rv2` is not a `RandomVector` or `None`, or if `probability_measure` is not a `ProbabilityMeasure` or `None`.
        ValueError
            If `rv1` and `rv2` have different domains or dimensions (when `rv2` is not `None`), or if `probability_measure` is not defined on the same sample space as `rv1` (when `probability_measure` is not `None`).

        Returns
        -------
        cov : pd.DataFrame | Real
            If both random vectors have dimension > 1, returns a pd.DataFrame representing the covariance matrix. If both have dimension 1, returns a Real representing the covariance.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> covariance = Operators.covariance
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> # Covariance of two 2-dimensional random vectors is a 2x2 matrix
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (2, 1), 2: (3, 4)})
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict({0: (3, -2), 1: (1, 5), 2: (6, 8)})
        >>> covariance(X, Y, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        feature   Y_0   Y_1
        feature
        X_0      1.23  2.87
        X_1      2.97  2.93
        >>> # Covariance of two random variables is a scalar
        >>> Z = RandomVariable(domain=Omega, name="Z").from_dict({0: 1, 1: -2, 2: 3})
        >>> W = RandomVariable(domain=Omega, name="W").from_dict({0: 5, 1: 6, 2: 1})
        >>> covariance(Z, W, probability_measure=P)
        -4.73
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if not isinstance(rv1, RandomVector):
            raise TypeError("rv1 must be a RandomVector.")
        if rv2 is not None and not isinstance(rv2, RandomVector):
            raise TypeError("rv2 must be a RandomVector or None.")
        if rv2 is not None and rv1.domain != rv2.domain:
            raise ValueError("rv1 and rv2 must have the same domain.")
        if rv2 is not None and rv1.dimension != rv2.dimension:
            raise ValueError("rv1 and rv2 must have the same dimension.")

        if probability_measure is None:
            probability_measure = rv1.probability_measure
        elif not isinstance(probability_measure, ProbabilityMeasure):
            raise TypeError("probability_measure must be a ProbabilityMeasure or None.")
        elif probability_measure.sample_space != rv1.domain:
            raise ValueError(
                "probability_measure must be defined on the same sample space as rv1."
            )

        if rv2 is None:
            rv2 = rv1

        E_rv1 = cls.expectation(rv1, probability_measure=probability_measure)
        E_rv2 = cls.expectation(rv2, probability_measure=probability_measure)

        centered_rv1 = rv1 - E_rv1
        centered_rv2 = rv2 - E_rv2

        arr1 = (
            centered_rv1.data.values
            if isinstance(centered_rv1.data, pd.DataFrame)
            else centered_rv1.data.values.reshape(-1, 1)
        )
        arr2 = (
            centered_rv2.data.values
            if isinstance(centered_rv2.data, pd.DataFrame)
            else centered_rv2.data.values.reshape(-1, 1)
        )
        probs_arr = probability_measure.data.values.reshape(-1, 1)

        cov_matrix = arr1.T @ (probs_arr * arr2)

        if rv1.dimension == 1 and rv2.dimension == 1:
            return cov_matrix.item()
        else:
            return pd.DataFrame(
                cov_matrix,
                index=rv1.data.columns,
                columns=rv2.data.columns,
            )

    # TODO: Update docstrings
    @classmethod
    def correlation(
        cls,
        rv1: RandomVector,
        rv2: RandomVector,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> pd.DataFrame | Real:
        """Compute the correlation matrix of two random vectors.

        Parameters
        ----------
        rv1 : RandomVector
            The first random vector.
        rv2 : RandomVector
            The second random vector.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `None`, uses `rv1.probability_measure`.

        Returns
        -------
        corr : pd.DataFrame | Real
            If both random vectors have dimension > 1, returns a pd.DataFrame representing the correlation matrix. If both have dimension 1, returns a Real representing the correlation.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> correlation = Operators.correlation
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (2, 1), 2: (3, 4)})
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict({0: (3, -2), 1: (1, 5), 2: (6, 8)})
        >>> # Correlation of two 2-dimensional random vectors is a 2x2 matrix
        >>> correlation(X, Y, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        feature       Y_0       Y_1
        feature
        X_0      0.712173  0.972077
        X_1      0.998304  0.576119
        >>> # Correlation of two random variables is a scalar
        >>> Z = RandomVariable(domain=Omega, name="Z").from_dict({0: -1, 1: 4, 2: 6})
        >>> W = RandomVariable(domain=Omega, name="W").from_dict({0: 2, 1: -3, 2: 5})
        >>> float(correlation(Z, W, probability_measure=P))
        0.3273268353539886
        """
        cov_matrix = cls.covariance(rv1, rv2, probability_measure=probability_measure)
        std_rv1 = cls.std(rv1, probability_measure=probability_measure).data.loc[0]
        std_rv2 = cls.std(rv2, probability_measure=probability_measure).data.loc[0]

        if rv1.dimension == 1 and rv2.dimension == 1:
            return cov_matrix / (std_rv1 * std_rv2)
        else:
            cov_matrix = cov_matrix.values
            std_rv1 = std_rv1.values.reshape(-1, 1)
            std_rv2 = std_rv2.values.reshape(-1, 1)
            corr_matrix = cov_matrix / (std_rv1 @ std_rv2.T)
            return pd.DataFrame(
                corr_matrix,
                index=rv1.data.columns,
                columns=rv2.data.columns,
            )

    @classmethod
    def pushforward(
        cls,
        rv: RandomVector,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilityMeasure:
        """Push forward a probability measure on the domain of a random vector to a probability measure on its range.

        Given a random vector `X: Omega -> S` and a probability measure `P`
        on `Omega`, constructs the probability measure `P_X` on the range `X.range`.

        Parameters
        ----------
        rv : RandomVector
            Random vector.
        probability_measure : ProbabilityMeasure | None, default=None
            Probability measure `P` defining the probabilities on the domain sample space. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `probability_measure` is not a `ProbabilityMeasure` (if given).
        ValueError
            If `rv` is not defined on the sample space of `probability_measure` (if given).

        Returns
        -------
        pushforward_measure : ProbabilityMeasure
            The resulting probability measure `P_X`.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Operators, ProbabilityMeasure, RandomVector, SampleSpace
        >>> pushforward = Operators.pushforward
        >>> domain = SampleSpace.generate_sequence(size=3)
        >>> X = RandomVector(domain=domain).from_dict(
        ...     {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)},
        ... )
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        omega_0    1   2
        omega_1    3   4
        omega_2    3   4
        >>> prob_measure = ProbabilityMeasure(sample_space=domain).from_dict(
        ...     {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3},
        ... )
        >>> P_X = pushforward(probability_measure=prob_measure, rv=X)
        >>> print(P_X)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_X':
            probability
        1 2          0.2
        3 4          0.8
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..random_objects.random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")
        if probability_measure is not None and not isinstance(
            probability_measure, ProbabilityMeasure
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure instance."
            )
        if (
            probability_measure is not None
            and rv.domain != probability_measure.sample_space
        ):
            raise ValueError(
                "rv must be defined on the sample space of probability_measure."
            )

        if probability_measure is None:
            probability_measure = rv.probability_measure

        pushforward_data = pd.concat([rv.data, probability_measure.data], axis=1)
        pushforward_data = (
            pushforward_data.groupby(pushforward_data.columns[: rv.dimension].to_list())
            .sum()
            .squeeze()
        )
        pushforward_data.index = rv.range.sample_space.data

        pushforward_name = (
            f"{probability_measure.name}_{rv.name}"
            if (isinstance(probability_measure.name, str) and isinstance(rv.name, str))
            else "pushforward"
        )
        pushforward = ProbabilityMeasure(
            sample_space=rv.range.sample_space, name=pushforward_name
        ).from_pandas(pushforward_data)

        return pushforward


class OperatorsMethods:
    """Mixin class to add operators as methods to `RandomVector` and `ProbabilityMeasure`."""

    # TODO: Update docstrings
    def integrate(
        self,
        *,
        rv: RandomVector | None = None,
        probability_measure: ProbabilityMeasure | None = None,
        event: Event | None = None,
    ) -> pd.Series | Real:
        """Compute the integral of a `RandomVector` with respect to a `ProbabilityMeasure` over an (optional) event.

        If `self` is a `RandomVector`, computes the integral of `self` with respect to `probability_measure`. In this case, `rv` must be `None` or equal to `self`.

        If `self` is a `ProbabilityMeasure`, computes the integral of the random vector `rv` with respect to `self`. In this case, `probability_measure` must be `None` or equal to `self`.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the integral. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `probability_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.
        event : Event | None, default=None
            The event over which to compute the integral. If `None`, computes the integral over the entire sample space.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `probability_measure` is not `None` or not equal to `self`.

        Returns
        -------
        integral : RandomVector
            The integral of the random vector with respect to the probability measure.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling integrate on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.integrate(
                rv=self,
                probability_measure=probability_measure,
                event=event,
            )
        elif isinstance(self, ProbabilityMeasure):
            if probability_measure is not None and probability_measure != self:
                raise ValueError(
                    "probability_measure must be None or equal to self when calling integrate on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.integrate(
                rv=rv,
                probability_measure=self,
                event=event,
            )

    # TODO: Update docstrings
    def expectation(
        self,
        *,
        rv: RandomVector | None = None,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the expectation of a random vector, optionally conditioned on a sigma algebra and with respect to a specified probability measure.

        If `self` is a `RandomVector`, computes the expectation of `self` with respect to `probability_measure`, optionally conditioned on `sigma_algebra`. In this case, `rv` must be `None` or equal to `self`.

        If `self` is a `ProbabilityMeasure`, computes the expectation of the random vector `rv` with respect to `self`, optionally conditioned on `sigma_algebra`. In this case, `probability_measure` must be `None` or equal to `self`.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the expectation. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional expectation.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `probability_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `probability_measure` is not `None` or not equal to `self`.

        Returns
        -------
        exp : RandomVector
            The expectation of the random vector, optionally conditioned on the sigma algebra and with respect to the specified probability measure.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling expectation on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.expectation(
                rv=self,
                sigma_algebra=sigma_algebra,
                probability_measure=probability_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if probability_measure is not None and probability_measure != self:
                raise ValueError(
                    "probability_measure must be None or equal to self when calling expectation on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.expectation(
                rv=rv,
                sigma_algebra=sigma_algebra,
                probability_measure=self,
            )

    # TODO: Update docstrings
    def variance(
        self,
        *,
        rv: RandomVector | None = None,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the variance of a random vector, optionally conditioned on a sigma algebra and with respect to a specified probability measure.

        If `self` is a `RandomVector`, computes the variance of `self` with respect to `probability_measure`, optionally conditioned on `sigma_algebra`. In this case, `rv` must be `None` or equal to `self`.

        If `self` is a `ProbabilityMeasure`, computes the variance of the random vector `rv` with respect to `self`, optionally conditioned on `sigma_algebra`. In this case, `probability_measure` must be `None` or equal to `self`.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the expectation. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional expectation.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `probability_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `probability_measure` is not `None` or not equal to `self`.

        Returns
        -------
        var : RandomVector
            The variance of the random vector, optionally conditioned on the sigma algebra and with respect to the specified probability measure.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling variance on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.variance(
                rv=self,
                sigma_algebra=sigma_algebra,
                probability_measure=probability_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if probability_measure is not None and probability_measure != self:
                raise ValueError(
                    "probability_measure must be None or equal to self when calling variance on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.variance(
                rv=rv,
                sigma_algebra=sigma_algebra,
                probability_measure=self,
            )

    # TODO: Update docstrings
    def std(
        self,
        *,
        rv: RandomVector | None = None,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the standard deviation of a random vector, optionally conditioned on a sigma algebra and with respect to a specified probability measure.

        If `self` is a `RandomVector`, computes the standard deviation of `self` with respect to `probability_measure`, optionally conditioned on `sigma_algebra`. In this case, `rv` must be `None` or equal to `self`.

        If `self` is a `ProbabilityMeasure`, computes the standard deviation of the random vector `rv` with respect to `self`, optionally conditioned on `sigma_algebra`. In this case, `probability_measure` must be `None` or equal to `self`.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the standard deviation. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional standard deviation.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `probability_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `probability_measure` is not `None` or not equal to `self`.

        Returns
        -------
        std : RandomVector
            The standard deviation of the random vector, optionally conditioned on the sigma algebra and with respect to the specified probability measure.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling std on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.std(
                rv=self,
                sigma_algebra=sigma_algebra,
                probability_measure=probability_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if probability_measure is not None and probability_measure != self:
                raise ValueError(
                    "probability_measure must be None or equal to self when calling std on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.std(
                rv=rv,
                sigma_algebra=sigma_algebra,
                probability_measure=self,
            )

    def pushforward(
        self,
        *,
        rv: RandomVector | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilityMeasure:
        """Push forward a probability measure on the domain of a random vector to a probability measure on its range.

        If `self` is a `RandomVector`, computes the pushforward of `probability_measure` by `self`. In this case, `rv` must be `None` or equal to `self`.

        If `self` is a `ProbabilityMeasure`, computes the pushforward of `self` by the random vector `rv`. In this case, `probability_measure` must be `None` or equal to `self`.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector to push forward. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to push forward. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

        Raises
        ------
        ValueError
            If `self` is a `RandomVector` and `rv` is not `None` or not equal to `self`, or if `self` is a `ProbabilityMeasure` and `probability_measure` is not `None` or not equal to `self`.

        Returns
        -------
        pushforward_measure : ProbabilityMeasure
            The resulting probability measure after pushing forward.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from .random_vector import RandomVector

        if isinstance(self, RandomVector):
            if rv is not None and rv != self:
                raise ValueError(
                    "rv must be None or equal to self when calling pushforward on a RandomVector, as the random vector itself is used as the argument."
                )
            return Operators.pushforward(
                rv=self,
                probability_measure=probability_measure,
            )
        elif isinstance(self, ProbabilityMeasure):
            if probability_measure is not None and probability_measure != self:
                raise ValueError(
                    "probability_measure must be None or equal to self when calling pushforward on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.pushforward(
                rv=rv,
                probability_measure=self,
            )
