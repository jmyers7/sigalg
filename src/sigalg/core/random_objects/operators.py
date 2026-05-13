"""Class for operators on random vectors, such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of probability measures."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..base.event import Event
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .random_variable import RandomVariable
    from .random_vector import RandomVector


class Operators:
    """Class containing methods such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of probability measures.

    The class does not have an `__init__` method, and all methods are class methods.
    """

    @classmethod
    def integrate(
        cls,
        rv: RandomVector,
        prob_measure: ProbabilityMeasure | None = None,
        event: Event | None = None,
    ) -> pd.Series | Real:
        r"""Compute the Lebesgue integral of a random vector with respect to a probability measure over an (optional) event.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector to integrate.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used (accessed through its `prob_measure` attribute).
        event: Event | None, default=None
            The optional event over which to integrate. If `None`, the integral will be taken over the entire sample space contained in the `domain` attribute of the random vector.

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
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (1, 2), 2: (3, 4)})
        >>> # Integral of a 2-dimensional random vector
        >>> Operators.integrate(rv=X, prob_measure=P) # doctest: +NORMALIZE_WHITESPACE
        integral
        integral(X_0)    2.0
        integral(X_1)    3.0
        Name: integral(X), dtype: float64
        >>> # Integral of a random variable
        >>> Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 1, 1: 1, 2: 0})
        >>> print(Operators.integrate(rv=Y, prob_measure=P))
        0.5

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}$ be a random variable on a probability space $(\Omega, \mathcal{F}, P)$. Assume $\Omega$ is finite (as it alawys is, in SigAlg) and let $\{A_i\}_{i\in I}$ be the atoms of $\mathcal{F}$, indexed by some set $I$. Since $X$ is $\mathcal{F}$-measurable, it takes a constant value $x_i$ on each atom $A_i$, i.e., $X(\omega)=x_i$ for each $\omega \in A_i$. Then the *Lebesgue integral* of $X$ is

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

        then we define the *Lebesgue* integral of $X$ to be the $d$-dimensional vector whose entries are the separate Lebesgue integrals $\int_A X_j \, dP$, for $j=1,2,\ldots,d$.
        """
        from ..base.event import Event
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..random_objects.random_variable import RandomVariable
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
        if prob_measure is not None and not prob_measure.sig_alg != rv.sig_alg:
            raise ValueError(
                "If given, prob_measure must be defined on the sigma-algebra of the random vector."
            )
        if event is not None and event not in rv.sig_alg:
            raise ValueError(
                "If given, the event must be an element of the sigma-algebra of the random vector."
            )

        if event is None:
            event = rv.sig_alg.get_event(list(rv.domain))
        else:
            event = rv.sig_alg.get_event(list(event))
        if prob_measure is None:
            prob_measure = rv.prob_measure

        if isinstance(rv, RandomVariable):
            indicator = RandomVariable.indicator_of(
                event=event
            ).with_probability_measure(prob_measure=prob_measure)
        else:
            indicator = RandomVector.indicator_of(
                event=event, dim=rv.dimension
            ).with_probability_measure(prob_measure=prob_measure)

        exp = cls.expectation(rv=rv * indicator, prob_measure=prob_measure)
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
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        r"""Compute the expectation of a random vector with respect to a probability measure, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the expectation.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability used to compute the expectation. If `None`, the probability measure carried by the random vector is used (accessed through its `prob_measure` attribute).

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
        >>> X.prob_measure = P
        >>> conditional_exp = Operators.expectation(rv=X, sig_alg=G)
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
        >>> Y.prob_measure = P
        >>> conditional_exp = Operators.expectation(rv=Y, sig_alg=G)
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

        then this method returns a `RandomVector` whose component random variables are the conditional expectations $E(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable
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

        if prob_measure is None:
            prob_measure = rv.prob_measure
        if sig_alg is None:
            sig_alg = SigmaAlgebra.trivial(sample_space=rv.domain)

        vector_columns = (
            rv.data.columns if isinstance(rv.data, pd.DataFrame) else rv.data.name
        )

        combined_sig_alg_atom_data = (
            pd.concat([sig_alg.data.rename("sub_atom_ID"), rv.sig_alg.data], axis=1)
            .drop_duplicates()
            .set_index("atom ID")
        )

        combined_data = pd.concat(
            [
                rv.atom_data,
                combined_sig_alg_atom_data,
                prob_measure.data,
            ],
            axis=1,
        )

        combined_data["normalized_prob"] = combined_data.groupby("sub_atom_ID")[
            "probability"
        ].transform(lambda x: x / x.sum())

        expectation_data = combined_data.groupby("sub_atom_ID").apply(
            lambda g: g[vector_columns].mul(g["normalized_prob"], axis=0).sum()
        )

        if sig_alg.is_trivial:
            name = f"E({rv.name})" if rv.name is not None else "expectation"
        else:
            name = (
                f"E({rv.name}|{sig_alg.name})"
                if rv.name is not None and sig_alg.name is not None
                else "expectation"
            )

        if isinstance(expectation_data, pd.Series):
            expectation_data.rename(name, inplace=True)
        else:
            if sig_alg.is_trivial:
                expectation_data.columns = (
                    [f"E({component_name})" for component_name in rv.index]
                    if rv.index is not None
                    else expectation_data.columns
                )
            else:
                expectation_data.columns = (
                    [
                        f"E({component_name}|{sig_alg.name})"
                        for component_name in rv.index
                    ]
                    if rv.index is not None
                    else expectation_data.columns
                )

        combined_data = combined_data.join(
            expectation_data, on="sub_atom_ID", rsuffix="_expectation"
        )

        if isinstance(expectation_data, pd.Series):
            data = combined_data[name]
            expectation = RandomVariable(*rv.prob_space, name=name).from_pandas(
                data, type="atom"
            )
        else:
            data = combined_data[expectation_data.columns]
            expectation = RandomVector(*rv.prob_space, name=name).from_pandas(
                data, type="atom"
            )

        return expectation

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
            The probability used to compute the variance. If `None`, the probability measure carried by the random vector is used (accessed through its `prob_measure` attribute).

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.

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
        >>> X.prob_measure = P
        >>> conditional_var = Operators.variance(rv=X, sig_alg=G)
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
        >>> Y.prob_measure = P
        >>> conditional_var = Operators.variance(rv=Y, sig_alg=G)
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

        then this method returns a `RandomVector` whose component random variables are the conditional variances $V(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.

        See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
        """
        from ..base.index import Index
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector.")
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra) or sig_alg.sample_space != rv.domain
        ):
            raise TypeError(
                "sig_alg must be a SigmaAlgebra or None, and its sample space must match the domain of the random vector."
            )
        if prob_measure is not None and (
            not isinstance(prob_measure, ProbabilityMeasure)
            or prob_measure.sample_space != rv.domain
        ):
            raise TypeError(
                "prob_measure must be a ProbabilityMeasure or None, and its sample space must match the domain of the random vector."
            )

        result = (
            cls.expectation(
                rv**2,
                sig_alg,
                prob_measure,
            )
            - cls.expectation(rv, sig_alg, prob_measure) ** 2
        )

        if sig_alg is not None:
            name = (
                f"V({rv.name}|{sig_alg.name})"
                if rv.name is not None and sig_alg.name is not None
                else None
            )
            if rv.dimension > 1:
                indices = [f"V({idx_name}|{sig_alg.name})" for idx_name in rv.index]
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
            The probability used to compute the standard deviation. If `None`, the probability measure carried by the random vector is used (accessed through its `prob_measure` attribute).

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.

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
        >>> X.prob_measure = P
        >>> conditional_std = Operators.std(rv=X, sig_alg=G)
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
        >>> Y.prob_measure = P
        >>> conditional_std = Operators.std(rv=Y, sig_alg=G)
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

        then this method returns a `RandomVector` whose component random variables are the conditional standard deviations $\sigma(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.

        See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
        """
        from ..base.index import Index
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_vector import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector.")
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra) or sig_alg.sample_space != rv.domain
        ):
            raise TypeError(
                "sig_alg must be a SigmaAlgebra or None, and its sample space must match the domain of the random vector."
            )
        if prob_measure is not None and (
            not isinstance(prob_measure, ProbabilityMeasure)
            or prob_measure.sample_space != rv.domain
        ):
            raise TypeError(
                "prob_measure must be a ProbabilityMeasure or None, and its sample space must match the domain of the random vector."
            )

        result = cls.variance(rv, sig_alg, prob_measure) ** 0.5

        if sig_alg is not None:
            name = (
                f"std({rv.name}|{sig_alg.name})"
                if rv.name is not None and sig_alg.name is not None
                else None
            )
            if rv.dimension > 1:
                indices = [f"std({idx_name}|{sig_alg.name})" for idx_name in rv.index]
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
        >>> from sigalg.core import Operators, ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace().from_sequence(size=5)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_rand(random_state=rng)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0          0.320930
        1          0.311850
        2          0.318334
        3          0.037349
        4          0.011538
        >>> X = RandomVariable(domain=Omega).from_randint(low=-20, high=21, random_state=rng)
        >>> Y = RandomVariable(domain=Omega, name="Y").from_randint(
        ...     low=-10, high=11, random_state=rng
        ... )
        >>> X.prob_measure = P
        >>> Y.prob_measure = P
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        sample
        0        1
        1       20
        2       10
        3       11
        4        9
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       6
        1       0
        2      -8
        3       7
        4      -1
        >>> print(Operators.cov(X, Y)) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'cov(X, Y)':
                cov(X, Y)
        sample
        0      -16.962212
        1      -16.962212
        2      -16.962212
        3      -16.962212
        4      -16.962212
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     }
        ... )
        >>> print(Operators.cov(X, Y, G)) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'cov(X, Y|G)':
                cov(X, Y|G)
        sample
        0        -28.494132
        1        -28.494132
        2          1.182969
        3          1.182969
        4          1.182969

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

        See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable

        if not isinstance(rv1, RandomVariable) or not isinstance(rv2, RandomVariable):
            raise TypeError("rv1 and rv2 must be RandomVariables.")
        if rv1.domain != rv2.domain:
            raise ValueError("rv1 and rv2 must have the same domain.")
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra) or sig_alg.sample_space != rv1.domain
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
        elif prob_measure.sample_space != rv1.domain:
            raise ValueError(
                "prob_measure must be defined on the same sample space as rv1."
            )

        result = cls.expectation(rv1 * rv2, sig_alg, prob_measure) - cls.expectation(
            rv1, sig_alg, prob_measure
        ) * cls.expectation(rv2, sig_alg, prob_measure)

        if sig_alg is not None:
            name = (
                f"cov({rv1.name}, {rv2.name}|{sig_alg.name})"
                if rv1.name is not None
                and rv2.name is not None
                and sig_alg.name is not None
                else None
            )
        else:
            name = (
                f"cov({rv1.name}, {rv2.name})"
                if rv1.name is not None and rv2.name is not None
                else None
            )

        result = result.with_name(name)

        return result

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
        >>> from sigalg.core import Operators, ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace().from_sequence(size=5)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_rand(random_state=rng)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0          0.320930
        1          0.311850
        2          0.318334
        3          0.037349
        4          0.011538
        >>> X = RandomVariable(domain=Omega).from_randint(low=-20, high=21, random_state=rng)
        >>> Y = RandomVariable(domain=Omega, name="Y").from_randint(
        ...     low=-10, high=11, random_state=rng
        ... )
        >>> X.prob_measure = P
        >>> Y.prob_measure = P
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        sample
        0        1
        1       20
        2       10
        3       11
        4        9
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       6
        1       0
        2      -8
        3       7
        4      -1
        >>> print(Operators.corr(X, Y)) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'corr(X, Y)':
                corr(X, Y)
        sample
        0        -0.386861
        1        -0.386861
        2        -0.386861
        3        -0.386861
        4        -0.386861
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     }
        ... )
        >>> print(Operators.corr(X, Y, G)) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'corr(X, Y|G)':
                corr(X, Y|G)
        sample
        0           -1.00000
        1           -1.00000
        2            0.71463
        3            0.71463
        4            0.71463

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
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable

        if not isinstance(rv1, RandomVariable) or not isinstance(rv2, RandomVariable):
            raise TypeError("rv1 and rv2 must be RandomVariables.")
        if rv1.domain != rv2.domain:
            raise ValueError("rv1 and rv2 must have the same domain.")
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra) or sig_alg.sample_space != rv1.domain
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
        elif prob_measure.sample_space != rv1.domain:
            raise ValueError(
                "prob_measure must be defined on the same sample space as rv1."
            )

        result = cls.cov(rv1, rv2, sig_alg, prob_measure) / (
            cls.std(rv1, sig_alg, prob_measure) * cls.std(rv2, sig_alg, prob_measure)
        )

        if sig_alg is not None:
            name = (
                f"corr({rv1.name}, {rv2.name}|{sig_alg.name})"
                if rv1.name is not None
                and rv2.name is not None
                and sig_alg.name is not None
                else None
            )
        else:
            name = (
                f"corr({rv1.name}, {rv2.name})"
                if rv1.name is not None and rv2.name is not None
                else None
            )

        result = result.with_name(name)

        return result

    @classmethod
    def pushforward(
        cls,
        rv: RandomVector,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilityMeasure:
        r"""Push forward a probability measure on the domain of a random vector to a probability measure on its range.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            Random vector.
        prob_measure : ProbabilityMeasure | None, default=None
            Probability measure to push forward. If `None`, the probability measure carried by the random vector is used (accessed through its `prob_measure` attribute).

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `prob_measure` is not a `ProbabilityMeasure` (if given).
        ValueError
            If `rv` is not defined on the sample space of `prob_measure` (if given).

        Returns
        -------
        pushforward_measure : ProbabilityMeasure
            The resulting probability measure `P_X`.

        Examples
        --------
        >>> from sigalg.core import Operators, ProbabilityMeasure, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0.15,
        ...         1: 0.35,
        ...         2: 0.1,
        ...         3: 0.4,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, -1),
        ...         3: (0, 1),
        ...     }
        ... )
        >>> pushforward = Operators.pushforward(rv=X, prob_measure=P)
        >>> print(pushforward) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_X':
            probability
        0  1          0.4
        1  2          0.5
        3 -1          0.1

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F},P)$. Then we define a probability measure $P_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $P$ by setting

        $$
        P_X(A) = P\left( \{\omega \in \Omega : X(\omega) \in A\}\right),
        $$

        for all Borel measurable subsets $A\subset \mathbb{R}^d$.

        See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..random_objects.random_vector import RandomVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError("prob_measure must be a ProbabilityMeasure instance.")
        if prob_measure is not None and rv.domain != prob_measure.sample_space:
            raise ValueError("rv must be defined on the sample space of prob_measure.")

        if prob_measure is None:
            prob_measure = rv.prob_measure

        pushforward_data = pd.concat([rv.data, prob_measure.data], axis=1)
        pushforward_data = (
            pushforward_data.groupby(pushforward_data.columns[: rv.dimension].to_list())
            .sum()
            .squeeze()
        )
        pushforward_data.index = rv.range.sample_space.data

        pushforward_name = (
            f"{prob_measure.name}_{rv.name}"
            if (isinstance(prob_measure.name, str) and isinstance(rv.name, str))
            else "pushforward"
        )
        pushforward = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(rv.range.sample_space), name=pushforward_name
        ).from_pandas(pushforward_data)

        return pushforward


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
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (1, 2), 2: (3, 4)})
        >>> X.prob_measure = P
        >>> # Integral of a 2-dimensional random vector using method call
        >>> X.integrate() # doctest: +NORMALIZE_WHITESPACE
        integral
        integral(X_0)    2.0
        integral(X_1)    3.0
        Name: integral(X), dtype: float64
        >>> # Integral of a random variable using method call
        >>> Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 1, 1: 1, 2: 0})
        >>> Y.prob_measure = P
        >>> float(Y.integrate())
        0.5
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
        >>> X.prob_measure = P
        >>> conditional_exp = X.expectation(sig_alg=G)
        >>> print(conditional_exp) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'E(X|G)':
                E(X|G)
        sample
        0       -1.000
        1        3.625
        2        3.625
        >>> unconditional_exp = X.expectation()
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
        >>> Y.prob_measure = P
        >>> conditional_exp = Y.expectation(sig_alg=G)
        >>> print(conditional_exp) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(Y|G)':
        expectation  E(Y_0|G)  E(Y_1|G)
        sample
        0              1.0000    2.0000
        1              3.0625    0.5625
        2              3.0625    0.5625
        >>> unconditional_exp = Y.expectation()
        >>> print(unconditional_exp) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(Y)':
        expectation  E(Y_0)  E(Y_1)
        sample
        0              2.65    0.85
        1              2.65    0.85
        2              2.65    0.85
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
        >>> X.prob_measure = P
        >>> conditional_var = X.variance(sig_alg=G)
        >>> print(conditional_var) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(X|G)':
                V(X|G)
        sample
        0       0.000000
        1       0.609375
        2       0.609375
        >>> unconditional_var = X.variance()
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
        >>> Y.prob_measure = P
        >>> conditional_var = Y.variance(sig_alg=G)
        >>> print(conditional_var) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(Y|G)':
        variance  V(Y_0|G)  V(Y_1|G)
        sample
        0         0.000000  0.000000
        1         3.808594  1.371094
        2         3.808594  1.371094
        >>> unconditional_var = Y.variance()
        >>> print(unconditional_var) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(Y)':
        variance  V(Y_0)  V(Y_1)
        sample
        0         3.7275  1.4275
        1         3.7275  1.4275
        2         3.7275  1.4275
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
        >>> X.prob_measure = P
        >>> conditional_std = X.std(sig_alg=G)
        >>> print(conditional_std) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(X|G)':
                std(X|G)
        sample
        0       0.000000
        1       0.780625
        2       0.780625
        >>> unconditional_std = X.std()
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
        >>> Y.prob_measure = P
        >>> conditional_std = Y.std(sig_alg=G)
        >>> print(conditional_std) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(Y|G)':
        std     std(Y_0|G)  std(Y_1|G)
        sample
        0         0.000000    0.000000
        1         1.951562    1.170937
        2         1.951562    1.170937
        >>> unconditional_std = Y.std()
        >>> print(unconditional_std) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(Y)':
        std     std(Y_0)  std(Y_1)
        sample
        0       1.930673   1.19478
        1       1.930673   1.19478
        2       1.930673   1.19478
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
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0.15,
        ...         1: 0.35,
        ...         2: 0.1,
        ...         3: 0.4,
        ...     }
        ... )
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, -1),
        ...         3: (0, 1),
        ...     }
        ... )
        >>> X.prob_measure = P
        >>> pushforward = X.pushforward()
        >>> print(pushforward) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P_X':
            probability
        0  1          0.4
        1  2          0.5
        3 -1          0.1
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
