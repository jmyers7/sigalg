"""Class for operators on random vectors, such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of measures."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..measures.measure import Measure
    from ..measures.parametrized_measure import (
        ParametrizedMeasure,
    )
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.measurable_set import MeasurableSet
    from .measurable_function import MeasurableFunction
    from .measurable_vector import MeasurableVector
    from .random_vector import RandomVector


class Operators:
    """Class containing methods such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of measures."""

    @classmethod
    def sum(
        cls,
        vec: MeasurableVector,
        name: Hashable | None = None,
    ) -> MeasurableFunction:
        """Compute the sum of the components of a measurable vector.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector whose components are to be summed.
        name : Hashable | None, default=None
            The name of the resulting measurable function. If `None`, a default name will be generated.

        Raises
        ------
        TypeError
            If `vec` is not an instance of `MeasurableVector`.

        Returns
        -------
        summed_vec : MeasurableFunction
            A measurable function representing the sum of the components of the input measurable vector.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector
        >>> D = Domain.from_sequence(size=2, variable_name="flip", name="D")
        >>> X = (D ^ 3).with_name("X")
        >>> f = MeasurableVector.from_identity(domain=X)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index                 0  1  2
        flip_0 flip_1 flip_2
        0      0      0       0  0  0
                      1       0  0  1
               1      0       0  1  0
                      1       0  1  1
        1      0      0       1  0  0
                      1       1  0  1
               1      0       1  1  0
                      1       1  1  1
        >>> g = f.sum(name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                              g
        flip_0 flip_1 flip_2
        0      0      0       0
                      1       1
               1      0       1
                      1       2
        1      0      0       1
                      1       2
               1      0       2
                      1       3
        """
        from ..functions.measurable_vector import MeasurableVector

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be an instance of MeasurableVector.")

        data_trans = vec.data.copy()
        data_trans = data_trans.sum(axis=1)

        if name is None:
            name = f"{vec.name}_sum"

        return MeasurableVector(
            *vec.measurable_space,
            measure=vec.measure,
            mapping=data_trans,
            name=name,
        )

    @classmethod
    def integrate(
        cls,
        function: MeasurableVector,
        measurable_set: MeasurableSet | None = None,
        measure: Measure | None = None,
    ) -> pd.Series | Real:
        r"""Compute the Lebesgue integral of a measurable vector with respect to a measure over an (optional) set.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        function : MeasurableVector
            The measurable vector to integrate.
        measurable_set: MeasurableSet | None, default=None
            The optional set over which to integrate. If `None`, the integral will be taken over the entire domain of the measurable vector.
        measure : Measure | None, default=None
            The measure with respect to which to integrate. If `None`, the measure of the underlying measure space of the measurable vector is used (if it exists).

        Raises
        ------
        TypeError
            If `function` is not a `MeasurableVector`, or if `measure` is given and is not a `Measure` instance, or if `measurable_set` is given and is not a `MeasurableSet` instance, or if the `measure` attribute of `function` is `None` and `measure` is `None`.
        ValueError
            If `measure` is given and is not defined on the sigma-algebra of the measurable vector, or if `measurable_set` is given and is not an element of the sigma-algebra of the measurable vector.

        Returns
        -------
        integral : pd.Series | Real
            If `function` has dimension > 1, returns a pd.Series representing the integral of each component of the measurable vector. If `function` has dimension 1, returns a Real representing the integral.

        Examples
        --------
        Define a measure space and a measurable function.

        >>> import numpy as np
        >>> from sigalg.core import MeasurableFunction, MeasureSpace, Operators
        >>> rng = np.random.default_rng(42)
        >>> measure_space = MeasureSpace.from_rand(
        ...     domain_size=100,
        ...     num_atoms=27,
        ...     num_null_atoms=12,
        ...     random_state=rng,
        ... )
        >>> X, F, mu = measure_space
        >>> f = MeasurableFunction.from_randnorm(
        ...     *measure_space,
        ...     random_state=rng,
        ... )

        Get a measurable set from the sigma-algebra, compute the integral over this set, and check that it agrees with the defining formula for the Lebesgue integral.

        >>> U = F.get_random_set(num_atoms=4, name="U", random_state=rng)
        >>> I_U = U.indicator
        >>> int = Operators.integrate
        >>> print(int(f, U) == sum(I_U(A) * f(A) * mu(A) for A in F))
        True

        Check that the integral over a null set is 0.

        >>> N = measure_space.get_random_set(
        ...     num_atoms=3,
        ...     is_null=True,
        ...     name="N",
        ...     random_state=rng,
        ... )
        >>> I_N = N.indicator
        >>> print(int(f, N))
        0.0

        Notes
        -----
        Let $f: X \to \mathbb{R}$ be a measurable function on a measure space $(X, \mathcal{F}, \mu)$. Assuming $X$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{F}$ is determined by its set $\alpha(\mathcal{F})$ of atoms. Let $U$ be a measurable set in $\mathcal{F}$, and write $I_U$ for its indicator function. Since both $f$ and $I_U$ are $\mathcal{F}$-measurable, they take constant values on each atom $A\in \alpha(\mathcal{F})$ that we write as $f(A)$ and $I_U(A)$, respectively. Then the *Lebesgue integral* of $f$ over $U$ is the number

        $$
        \int_U f \, d\mu = \sum_{A\in \alpha(\mathcal{F})} I_U(A)f(A) \mu(A).
        $$

        If $f:X \to \mathbb{R}^d$ is instead a measurable vector of dimension $d>1$, with components

        $$
        f = (f_1, f_2, \ldots, f_d),
        $$

        then we define the *Lebesgue integral* of $f$ over $U$ to be the $d$-dimensional vector whose entries are the separate Lebesgue integrals $\int_U f_j \, d\mu$, for $j=1,2,\ldots,d$.
        """
        from ..functions.measurable_vector import MeasurableVector
        from ..measures.measure import Measure
        from ..spaces.measurable_set import MeasurableSet

        if not isinstance(function, MeasurableVector):
            raise TypeError("function must be a MeasurableVector instance.")
        if measure is not None and not isinstance(measure, Measure):
            raise TypeError("If given, measure must be a Measure instance.")
        if measurable_set is not None and not isinstance(measurable_set, MeasurableSet):
            raise TypeError(
                "If given, the measurable_set must be a MeasurableSet instance."
            )
        if measure is not None and measure.sig_alg != function.sig_alg:
            raise ValueError(
                "If given, measure must be defined on the sigma-algebra of the measurable vector."
            )
        if measure is None:
            if function.measure is None:
                raise TypeError(
                    "The measure of the measurable vector is None, please pass an explicit value for the measure parameter."
                )
            else:
                measure = function.measure
        if measurable_set is not None and measurable_set not in function.sig_alg:
            raise ValueError(
                "If given, the measurable_set must be an element of the sigma-algebra of the measurable vector."
            )

        if measurable_set is None:
            indicator_atom_data = pd.Series(1, index=function.sig_alg.atom_space.data)
        else:
            if measurable_set.sig_alg != function.sig_alg:
                indicator_atom_data = (
                    pd.concat(
                        [measurable_set.indicator.data, function.sig_alg.data],
                        axis=1,
                    )
                    .drop_duplicates()
                    .set_index("atom_ID")
                    .squeeze(axis=1)
                )
            else:
                indicator_atom_data = measurable_set.indicator.atom_data

        integral = (
            function.atom_data.multiply(measure.data, axis=0)
            .multiply(indicator_atom_data, axis=0)
            .sum()
        )

        if isinstance(integral, pd.Series):
            if measurable_set is not None:
                integral.name = (
                    f"int_{measurable_set.name} {function.name} d{measure.name}"
                )
            else:
                integral.name = f"int {function.name} d{measure.name}"

        if isinstance(integral, pd.Series):
            return integral
        else:
            return integral.astype(Real)

    @classmethod
    def expectation(
        cls,
        rv: MeasurableVector,
        given: SigmaAlgebra | RandomVector | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableVector:
        r"""Compute the expectation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : MeasurableVector
            The random vector for which to compute the expectation.
        given : SigmaAlgebra | RandomVector | None, default=None
            The sigma-algebra or random vector to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        exp : MeasurableVector
            The expected value of the random vector.

        Examples
        --------
        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=2,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Give aliases to the `integrate` and `expectation` methods, and get the constant random variable whose unique value is `1`.

        >>> E = Operators.expectation
        >>> int = Operators.integrate
        >>> trivial = SigmaAlgebra.trivial(Omega)
        >>> one = RandomVector.from_constant(
        ...     domain=Omega,
        ...     sig_alg=trivial,
        ...     measure=P | trivial,
        ...     constant=1,
        ... )

        Check that the unconditional expectation of the random variable `X` is equal to the constant random variable whose unique value is the Lebesgue integral of the random variable.

        >>> print(E(X) == int(X) * one)
        True

        Compute the unconditional expectation of the random vector `Y`, and check that its components are the unconditional expectations of the components of `Y`.

        >>> for E_Y_i, Y_i in zip(E(Y), Y):
        ...     print(E_Y_i == int(Y_i) * one)
        True
        True

        Define a sub-sigma-algebra of `F` for conditional expectations.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional expectation of the random variable `X` is equal to its Fourier expansion.

        >>> print(E(X, G) == sum(int(X, B) / P(B) * B.indicator for B in G if P(B) != 0))
        True

        Check the same for the components of the conditional expectation of the random vector `Y`.

        >>> for E_Y_i_G, Y_i in zip(E(Y, G), Y):
        ...     print(E_Y_i_G == sum(int(Y_i, B) / P(B) * B.indicator for B in G if P(B) != 0))
        True
        True

        Check that passing an explict measure--different from the one carried by the random variable--into the `expectation` method works:

        >>> Q = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=4,
        ...     name="Q",
        ...     random_state=rng,
        ... )
        >>> one = RandomVector.from_constant(
        ...     domain=Omega,
        ...     sig_alg=trivial,
        ...     measure=Q | trivial,
        ...     constant=1,
        ... )
        >>> print(E(X, measure=Q) == int(X, measure=Q) * one)
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional expectation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $E(X\mid \mathcal{G})$ for which

        $$
        \int_V E(X\mid \mathcal{G}) \, dP = \int_V X \, dP,
        $$

        for all $V\in \mathcal{G}$. All such random variables are equal almost surely.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional expectation called a *Fourier expansion*:

        $$
        E(X\mid \mathcal{G}) = \sum_B \frac{\int_B X \, dP}{P(B)} I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability and $I_B$ is the indicator function of $B$.

        The *unconditional expectation* of $X$, denoted $E(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. In this case $E(X)$ is the constant random variable with

        $$
        E(X)(\omega) = \int_\Omega X \, dP,
        $$

        for all $\omega\in \Omega$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional expectation* to be the $d$-dimensional vector whose entries are the separate conditional expectations $E(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

        if isinstance(given, RandomVector):
            given = given.generated_sig_alg

        cls._validate_univariate_parameters(rv=rv, sig_alg=given, measure=measure)

        if measure is None:
            measure = rv.prob_measure
        if given is None:
            given = SigmaAlgebra.trivial(rv.domain)
            mapping = (
                pd.Series(cls.integrate(rv, measure=measure), index=rv.domain.data)
                if isinstance(rv.data, pd.Series)
                else pd.DataFrame(
                    cls.integrate(rv, measure=measure).to_dict(), index=rv.domain.data
                )
            )
            name = f"E({rv.name})"
            return RandomVariable(
                domain=rv.domain,
                sig_alg=given,
                measure=measure | given,
                mapping=mapping,
                name=name,
            )

        rv_cols = [rv.name] if isinstance(rv.data, pd.Series) else list(rv.index)

        rv_times_prob = (
            rv.atom_data.multiply(measure.data, axis=0).rename(rv.name)
            if isinstance(rv.atom_data, pd.Series)
            else rv.atom_data.multiply(measure.data, axis=0)
        )

        sig_alg_data = (
            pd.concat(
                [given.data.to_frame().add_suffix("_sub"), rv.sig_alg.data], axis=1
            )
            .drop_duplicates()
            .set_index("atom_ID")
        )

        combined_data = pd.concat([rv_times_prob, sig_alg_data, measure.data], axis=1)
        grouped = combined_data.groupby("atom_ID_sub")[rv_cols + ["probability"]].sum()
        mapping = grouped[rv_cols].divide(grouped["probability"], axis=0).fillna(0.0)
        mapping = (
            pd.merge(
                left=given.data, right=mapping, left_on="atom_ID", right_index=True
            )
            .drop(columns="atom_ID")
            .squeeze(axis=1)
        )

        name = f"E({rv.name}|{given.name})"

        if isinstance(mapping, pd.Series):
            mapping.name = name

        return RandomVector(
            domain=rv.domain,
            sig_alg=given,
            measure=measure | given,
            mapping=mapping,
            index=rv.index.data if isinstance(rv.data, pd.DataFrame) else None,
            name=name,
        )

    @classmethod
    def variance(
        cls,
        rv: MeasurableVector,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableVector:
        r"""Compute the variance of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : MeasurableVector
            The random vector for which to compute the variance.
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        var : MeasurableVector
            The variance of the random vector.

        Examples
        --------
        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=2,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Give aliases to the `variance` and `expectation` methods.

        >>> V = Operators.variance
        >>> E = Operators.expectation

        Check that the unconditional variance is equal to its definition and that it can be computed using the "shortcut formula."

        >>> print(V(X) == E((X - E(X)) ** 2))
        True
        >>> print(V(X) == E(X**2) - E(X) ** 2)
        True

        Compute the unconditional variance of the random vector `Y`, and check that its components are the unconditional variances of the components of `Y`.

        >>> for V_Y_i, Y_i in zip(V(Y), Y):
        ...     print(V_Y_i == E((Y_i - E(Y_i)) ** 2))
        ...     print(V_Y_i == E(Y_i**2) - E(Y_i) ** 2)
        True
        True
        True
        True

        Define a sub-sigma-algebra of `F` for conditional variances.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional variance of the random variable `X` is equal to a linear combination of indicators weighted by unconditional variances.

        >>> print(V(X, G) == sum(V(X | B).item() * B.indicator for B in G if P(B) > 0))
        True

        Check the same for the random vector `Y`.

        >>> for V_Y_i_G, Y_i in zip(V(Y, G), Y):
        ...     print(V_Y_i_G == sum(V(Y_i | B).item() * B.indicator for B in G if P(B) > 0))
        True
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F}, P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional variance* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        V(X\mid \mathcal{G}) = E\left[ (X-E(X\mid \mathcal{G}))^2 \mid \mathcal{G}\right].
        $$

        The *unconditional variance* of $X$, denoted $V(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional variance is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional variance:

        $$
        V(X\mid \mathcal{G}) = \sum_B V(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $V(X|_B)$ is the unconditional variance of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional variance* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional variances $V(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        cls._validate_univariate_parameters(rv=rv, sig_alg=given, measure=measure)

        result = (
            cls.expectation(
                rv=rv**2,
                given=given,
                measure=measure,
            )
            - cls.expectation(rv=rv, given=given, measure=measure) ** 2
        )

        name = f"V({rv.name}|{given.name})" if given is not None else f"V({rv.name})"

        return result.with_name(name)

    @classmethod
    def std(
        cls,
        rv: MeasurableVector,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableVector:
        r"""Compute the standard deviation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : MeasurableVector
            The random vector for which to compute the standard deviation.
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        std : MeasurableVector
            The standard deviation of the random vector.

        Examples
        --------
        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=2,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Give aliases to the `variance` and `std` methods.

        >>> V = Operators.variance
        >>> std = Operators.std

        Check that the unconditional standard deviation is equal to its definition.

        >>> print(std(X) == V(X) ** 0.5)
        True

        Compute the unconditional standard deviation of the random vector `Y`, and check that its components are the unconditional standard deviations of the components of `Y`.

        >>> for std_Y_i, Y_i in zip(std(Y), Y):
        ...     print(std_Y_i == V(Y_i) ** 0.5)
        True
        True

        Define a sub-sigma-algebra of `F` for conditional standard deviations.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional standard deviation of the random variable `X` is equal to a linear combination of indicators weighted by unconditional standard deviations.

        >>> print(std(X, G) == sum(std(X | B).item() * B.indicator for B in G if P(B) > 0))
        True

        Check the same for the random vector `Y`.

        >>> for std_Y_i_G, Y_i in zip(std(Y, G), Y):
        ...     print(
        ...         np.allclose(
        ...             std_Y_i_G.data,
        ...             sum(std(Y_i | B).item() * B.indicator for B in G if P(B) > 0).data,
        ...             atol=1e-7,
        ...         )
        ...     )
        True
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional standard deviation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $\sigma(X \mid \mathcal{G})$ that is equal almost surely to the random variable

        $$
        \sigma(X\mid \mathcal{G}) = \sqrt{V(X\mid \mathcal{G})}.
        $$

        The *unconditional standard deviation* of $X$, denoted $\sigma(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional standard deviation is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional standard deviation:

        $$
        \sigma(X\mid \mathcal{G}) = \sum_B \sigma(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B)$ is the unconditional standard deviation of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional standard deviation* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional standard deviations $\sigma(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        cls._validate_univariate_parameters(rv=rv, sig_alg=given, measure=measure)

        result = cls.variance(rv, given, measure) ** 0.5
        result._data = result._data.fillna(0.0)

        name = (
            (f"std({rv.name}|{given.name})") if given is not None else f"std({rv.name})"
        )

        return result.with_name(name)

    @classmethod
    def cov(
        cls,
        rv1: MeasurableFunction,
        rv2: MeasurableFunction,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableFunction:
        r"""Compute the covariance of two random variables, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv1 : MeasurableFunction
            The first random variable for which to compute the covariance.
        rv2 : MeasurableFunction
            The second random variable for which to compute the covariance
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability used to compute the covariance. If `None`, the common probability measure carried by the random variables is used (accessed through their `prob_measure` attribute).

        Returns
        -------
        cov : MeasurableFunction
            The covariance of the random variables.

        Examples
        --------
        Define a probability space along with two random variables.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Give aliases to the `expectation` and `cov` methods.

        >>> E = Operators.expectation
        >>> cov = Operators.cov

        Check that the unconditional covariance is equal to its definition.

        >>> print(cov(X, Y) == E(X * Y) - E(X) * E(Y))
        True

        Define a sub-sigma-algebra of `F` for conditional covariance.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional covariance of the random variables is equal to a linear combination of indicators weighted by unconditional covariances.

        >>> print(cov(X, Y, G) == sum(cov(X | B, Y | B).item() * B.indicator for B in G if P(B) > 0))
        True

        Notes
        -----
        Let $X,Y:\Omega \to \mathbb{R}$ be two random variables on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional covariance* of $X$ and $Y$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        \sigma(X,Y\mid \mathcal{G}) = E(XY \mid \mathcal{G}) - E(X\mid \mathcal{G})E(Y\mid \mathcal{G}).
        $$

        The *unconditional covariance* of $X$ and $Y$, denoted $\sigma(X, Y)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional covariance is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional covariance:

        $$
        \sigma(X,Y\mid \mathcal{G}) = \sum_B \sigma(X|_B, Y|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B, Y|_B)$ is the unconditional covariance of the restricted random variables $X|_B, Y|_B:B\to \mathbb{R}$.
        """
        cls._validate_bivariate_parameters(
            rv1=rv1, rv2=rv2, sig_alg=given, measure=measure
        )

        result = cls.expectation(rv1 * rv2, given, measure) - cls.expectation(
            rv1, given, measure
        ) * cls.expectation(rv2, given, measure)

        name = (
            f"cov({rv1.name}, {rv2.name}|{given.name})"
            if given is not None
            else f"cov({rv1.name}, {rv2.name})"
        )

        return result.with_name(name)

    @classmethod
    def corr(
        cls,
        rv1: MeasurableFunction,
        rv2: MeasurableFunction,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableFunction:
        r"""Compute the correlation of two random variables, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv1 : MeasurableFunction
            The first random variable for which to compute the correlation.
        rv2 : MeasurableFunction
            The second random variable for which to compute the correlation
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability used to compute the correlation. If `None`, the common probability measure carried by the random variables is used (accessed through their `prob_measure` attribute).

        Returns
        -------
        corr : MeasurableFunction
            The correlation of the two random variables.

        Examples
        --------
        Define a probability space along with two random variables.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Give aliases to the `std`, `cov`, and `corr` methods.

        >>> std = Operators.std
        >>> cov = Operators.cov
        >>> corr = Operators.corr

        Check that the unconditional correlation is equal to its definition.

        >>> print(corr(X, Y) == cov(X, Y) / (std(X) * std(Y)))
        True

        Define a sub-sigma-algebra of `F` for conditional correlation.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional correlation of the random variables is equal to a linear combination of indicators weighted by unconditional correlations.

        >>> print(corr(X, Y, G) == sum(corr(X | B, Y | B).item() * B.indicator for B in G if P(B) > 0))
        True

        Notes
        -----
        Let $X,Y:\Omega \to \mathbb{R}$ be two random variables on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional correlation* of $X$ and $Y$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        \rho(X,Y\mid \mathcal{G}) = \frac{\sigma(X, Y \mid \mathcal{G})}{\sigma(X \mid \mathcal{G}) \sigma(Y \mid \mathcal{G})}.
        $$

        The *unconditional correlation* of $X$ and $Y$, denoted $\rho(X, Y)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional correlation is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional correlation:

        $$
        \rho(X,Y\mid \mathcal{G}) = \sum_B \rho(X|_B, Y|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\rho(X|_B, Y|_B)$ is the unconditional correlation of the restricted random variables $X|_B, Y|_B:B\to \mathbb{R}$.
        """
        cls._validate_bivariate_parameters(
            rv1=rv1, rv2=rv2, sig_alg=given, measure=measure
        )

        result = cls.cov(rv1, rv2, given, measure) / (
            cls.std(rv1, given, measure) * cls.std(rv2, given, measure)
        )
        result._data = result._data.fillna(0.0)

        name = (
            f"corr({rv1.name}, {rv2.name}|{given.name})"
            if given is not None
            else f"corr({rv1.name}, {rv2.name})"
        )

        return result.with_name(name)

    @classmethod
    def pushforward(
        cls,
        vec: MeasurableVector,
        measure: Measure | ParametrizedMeasure | None = None,
    ) -> Measure | ParametrizedMeasure:
        r"""Push forward a (parametrized) measure on the domain of a measurable vector to a measure on its range.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector along which to push forward the measure.
        measure : Measure | ParametrizedMeasure | None, default=None
            Measure to push forward. If `None`, the measure carried by the measurable vector is used.

        Raises
        ------
        TypeError
            If `vec` is not a MeasurableVector, or if `measure` is not a Measure or ParametrizedMeasure.
        ValueError
            If `measure` is not `None` and does not have the same sigma-algebra as `vec`, or if `measure` is `None` and `vec` does not carry a measure.

        Returns
        -------
        pushforward : Measure | ParametrizedMeasure
            The measure pushed forward along the measurable vector.

        Examples
        --------
        Define a measure space.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableVector,
        ...     Measure,
        ...     Operators,
        ...     ParametrizedProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ... )

        Define a 2-dimensional measurable vector and pushforward the measure `mu`.

        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> mu_f = Operators.pushforward(f, mu)
        >>> print(mu_f)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu_f':
                 measure
        f_0 f_1
        1   2          1
        3   4          5

        Now define a measurable space with a sample space.

        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )

        Define a parametrized probability measure on the sigma-algebra.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> def mapping(*, theta, u):
        ...     if theta == 0:
        ...         if u == 0:
        ...             return 0.1
        ...         elif u == 1:
        ...             return 0.2
        ...         else:
        ...             return 0.7
        ...     if theta == 1:
        ...         if u == 0:
        ...             return 0.4
        ...         elif u == 1:
        ...             return 0.5
        ...         else:
        ...             return 0.1
        >>> P = ParametrizedProbabilityMeasure(
        ...     measure_domain=F, parameter_domain=Theta, mapping=mapping
        ... )

        Define a 2-dimensional random vector and pushforward the parametrized probability measure `P`.

        >>> X = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 1),
        ...         1: (1, 1),
        ...         2: (3, 1),
        ...         3: (3, 1),
        ...     },
        ... )
        >>> P_X = Operators.pushforward(X, P)
        >>> print(P_X)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P_X':
                       probability
        theta X_0 X_1
        0     1   1            0.1
              3   1            0.9
        1     1   1            0.4
              3   1            0.6

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a measurable vector on a measure space $(X, \mathcal{F}, \mu)$. Then we define a measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(A) = \mu\left( \{x \in X : f(x) \in A\}\right),
        $$

        for all Borel subsets $A\subset \mathbb{R}^d$.

        If $\mu$ is a parametrized measure on $X$ with parameter domain $\Theta$, then we define a parametrized measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(\theta, A) = \mu\left(\theta, \{x \in X : f(x) \in A\}\right),
        $$

        for all $\theta \in \Theta$ and all Borel subsets $A\subset \mathbb{R}^d$.
        """
        from ..functions.measurable_vector import MeasurableVector
        from ..measures.measure import Measure
        from ..measures.parametrized_measure import (
            ParametrizedMeasure,
        )
        from ..spaces.domain import Domain

        if not isinstance(vec, MeasurableVector):
            raise TypeError("vec must be a MeasurableVector.")
        if measure is not None and not isinstance(
            measure, Measure | ParametrizedMeasure
        ):
            raise TypeError("measure must be a Measure or ParametrizedMeasure.")
        if measure is not None and vec.sig_alg != measure.sig_alg:
            raise ValueError("vec must have the same sigma-algebra as that of measure.")

        if measure is None:
            if vec.measure is not None:
                measure = vec.measure
            else:
                raise ValueError(
                    "If measure is not given, then the measurable vector must carry a measure."
                )

        atom_id_index = Domain(
            indices=vec.sig_alg.atom_ids, variable_names=vec.sig_alg.variable_names
        )

        vec_atom_data = vec.atom_data.copy()
        vec_atom_data.columns = vec.component_names
        vec_atom_data.index = atom_id_index.data

        if not isinstance(measure, ParametrizedMeasure):
            measure_data = measure.data.copy()
            measure_data.index = atom_id_index.data
        else:
            measure_data = measure.data

        mapping = pd.merge(
            left=measure_data, right=vec_atom_data, left_index=True, right_index=True
        )

        parameter_names = (
            measure.parameter_names if isinstance(measure, ParametrizedMeasure) else []
        )

        mapping = mapping.groupby(parameter_names + vec.component_names)[
            measure.output_name
        ].sum()

        name = (
            f"{measure.name}_{vec.name}"
            if (isinstance(measure.name, str) and isinstance(vec.name, str))
            else "pushforward"
        )

        if isinstance(measure, ParametrizedMeasure):
            return ParametrizedMeasure(
                measure_domain=vec.range.sig_alg,
                mapping=mapping,
                output_name=measure.output_name,
                kind=measure.kind,
                name=name,
            )

        else:
            return Measure(
                domain=vec.range.sig_alg,
                mapping=mapping,
                output_name=measure.output_name,
                kind=measure.kind,
                name=name,
            )

    @staticmethod
    def _validate_univariate_parameters(
        rv: MeasurableVector,
        sig_alg: SigmaAlgebra | None,
        measure: ProbabilityMeasure | None,
    ):
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .measurable_vector import MeasurableVector

        if not isinstance(rv, MeasurableVector):
            raise TypeError("rv must be a MeasurableVector instance.")
        if measure is None and (
            rv.measure is None or not isinstance(rv.measure, ProbabilityMeasure)
        ):
            raise ValueError(
                "If measure is not given, then the random vector must carry a probability measure."
            )
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("If given, sig_alg must be a SigmaAlgebra instance.")
        if measure is not None and not isinstance(measure, ProbabilityMeasure):
            raise TypeError("If given, measure must be a ProbabilityMeasure instance.")
        if sig_alg is not None and not sig_alg <= rv.sig_alg:
            raise ValueError(
                "If given, sig_alg must be a sub-sigma-algebra of the random vector's sigma-algebra."
            )
        if measure is not None and measure.sig_alg != rv.sig_alg:
            raise ValueError(
                "If given, measure must be defined on the sigma-algebra of the random vector."
            )

    @staticmethod
    def _validate_bivariate_parameters(
        rv1: MeasurableFunction,
        rv2: MeasurableFunction,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> None:
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .measurable_function import MeasurableFunction

        if not isinstance(rv1, MeasurableFunction) or not isinstance(
            rv2, MeasurableFunction
        ):
            raise TypeError("rv1 and rv2 must be MeasurableFunctions.")
        if rv1.measurable_space != rv2.measurable_space:
            raise ValueError(
                "rv1 and rv2 must be defined on the same measurable space."
            )
        if sig_alg is not None and (
            not isinstance(sig_alg, SigmaAlgebra) or not sig_alg <= rv1.sig_alg
        ):
            raise TypeError(
                "sig_alg must be a SigmaAlgebra or None, and it must be a sub-sigma-algebra of the sigma-algebra of the random variables."
            )
        if measure is None:
            if (
                rv1.measure != rv2.measure
                or rv1.measure is None
                or not isinstance(rv1.measure, ProbabilityMeasure)
            ):
                raise ValueError(
                    "If measure is not passed, the random variables must have the same probability measures."
                )
            else:
                measure = rv1.measure
        else:
            if not isinstance(measure, ProbabilityMeasure):
                raise TypeError("measure must be a ProbabilityMeasure or None.")
            if measure.sig_alg != rv1.sig_alg:
                raise ValueError(
                    "If measure is passed, it must be defined on the sigma-algebra of the random variables."
                )


class OperatorsMethods:
    """Mixin class to add operators to `MeasurableVector`."""

    def sum(self, name: Hashable | None = None) -> MeasurableFunction:
        """Compute the sum of the components of a measurable vector.

        Calls `Operators.sum` with appropriate arguments.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the resulting measurable function. If `None`, a default name will be generated.

        Returns
        -------
        summed_vec : MeasurableFunction
            The measurable function representing the sum of the components of the measurable vector.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector
        >>> D = Domain.from_sequence(size=2, variable_name="flip", name="D")
        >>> X = (D ^ 3).with_name("X")
        >>> f = MeasurableVector.from_identity(domain=X)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        index                 0  1  2
        flip_0 flip_1 flip_2
        0      0      0       0  0  0
                      1       0  0  1
               1      0       0  1  0
                      1       0  1  1
        1      0      0       1  0  0
                      1       1  0  1
               1      0       1  1  0
                      1       1  1  1
        >>> g = f.sum(name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                              g
        flip_0 flip_1 flip_2
        0      0      0       0
                      1       1
               1      0       1
                      1       2
        1      0      0       1
                      1       2
               1      0       2
                      1       3
        """
        return Operators.sum(vec=self, name=name)

    def integrate(
        self,
        measurable_set: MeasurableSet | None = None,
        measure: Measure | None = None,
    ) -> pd.Series | Real:
        r"""Compute the Lebesgue integral of a measurable vector with respect to a measure over an (optional) set.

        See the Notes section below for the mathematical details.

        Calls `Operators.integrate` with appropriate arguments.

        Parameters
        ----------
        measurable_set: MeasurableSet | None, default=None
            The optional set over which to integrate. If `None`, the integral will be taken over the entire domain of the measurable vector.
        measure : Measure | None, default=None
            The measure with respect to which to integrate. If `None`, the measure of the underlying measure space of the measurable vector is used (if it exists).

        Returns
        -------
        integral : pd.Series | Real
            If the measurable vector has dimension > 1, returns a pd.Series representing the integral of each component of the measurable vector. If the measurable vector has dimension 1, returns a Real representing the integral.

        Examples
        --------
        Define a measure space and a measurable function.

        >>> import numpy as np
        >>> from sigalg.core import MeasurableFunction, MeasureSpace, Operators
        >>> rng = np.random.default_rng(42)
        >>> measure_space = MeasureSpace.from_rand(
        ...     domain_size=100,
        ...     num_atoms=27,
        ...     num_null_atoms=12,
        ...     random_state=rng,
        ... )
        >>> X, F, mu = measure_space
        >>> f = MeasurableFunction.from_randnorm(
        ...     *measure_space,
        ...     random_state=rng,
        ... )

        Get a measurable set from the sigma-algebra, compute the integral over this set, and check that it agrees with the defining formula for the Lebesgue integral.

        >>> U = F.get_random_set(num_atoms=4, name="U", random_state=rng)
        >>> I_U = U.indicator
        >>> print(f.integrate(U) == sum(I_U(A) * f(A) * mu(A) for A in F))
        True

        Check that the integral over a null set is 0.

        >>> N = measure_space.get_random_set(
        ...     num_atoms=3,
        ...     is_null=True,
        ...     name="N",
        ...     random_state=rng,
        ... )
        >>> I_N = N.indicator
        >>> print(f.integrate(N))
        0.0

        Notes
        -----
        Let $f: X \to \mathbb{R}$ be a measurable function on a measure space $(X, \mathcal{F}, \mu)$. Assuming $X$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{F}$ is determined by its set $\alpha(\mathcal{F})$ of atoms. Let $U$ be a measurable set in $\mathcal{F}$, and write $I_U$ for its indicator function. Since both $f$ and $I_U$ are $\mathcal{F}$-measurable, they take constant values on each atom $A\in \alpha(\mathcal{F})$ that we write as $f(A)$ and $I_U(A)$, respectively. Then the *Lebesgue integral* of $f$ over $U$ is the number

        $$
        \int_U f \, d\mu = \sum_{A\in \alpha(\mathcal{F})} I_U(A)f(A) \mu(A).
        $$

        If $f:X \to \mathbb{R}^d$ is instead a measurable vector of dimension $d>1$, with components

        $$
        f = (f_1, f_2, \ldots, f_d),
        $$

        then we define the *Lebesgue integral* of $f$ over $U$ to be the $d$-dimensional vector whose entries are the separate Lebesgue integrals $\int_U f_j \, d\mu$, for $j=1,2,\ldots,d$.
        """
        return Operators.integrate(
            function=self,
            measurable_set=measurable_set,
            measure=measure,
        )

    def expectation(
        self,
        given: SigmaAlgebra | RandomVector | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableVector:
        r"""Compute the expectation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Calls `Operators.expectation` with appropriate arguments.

        Parameters
        ----------
        given : SigmaAlgebra | RandomVector | None, default=None
            The sigma-algebra or random vector to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        exp : MeasurableVector
            The expectation of the random vector.

        Examples
        --------
        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=2,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Get the constant random variable whose unique value is `1`.

        >>> trivial = SigmaAlgebra.trivial(Omega)
        >>> one = RandomVector.from_constant(
        ...     domain=Omega,
        ...     sig_alg=trivial,
        ...     measure=P | trivial,
        ...     constant=1,
        ... )

        Check that the unconditional expectation of the random variable `X` is equal to the constant random variable whose unique value is the Lebesgue integral of the random variable.

        >>> print(X.expectation() == X.integrate() * one)
        True

        Compute the unconditional expectation of the random vector `Y`, and check that its components are the unconditional expectations of the components of `Y`.

        >>> for E_Y_i, Y_i in zip(Y.expectation(), Y):
        ...     print(E_Y_i == Y_i.integrate() * one)
        True
        True

        Define a sub-sigma-algebra of `F` for conditional expectations.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional expectation of the random variable `X` is equal to its Fourier expansion.

        >>> print(X.expectation(given=G) == sum(X.integrate(B) / P(B) * B.indicator for B in G if P(B) != 0))
        True

        Check the same for the components of the conditional expectation of the random vector `Y`.

        >>> for E_Y_i_G, Y_i in zip(Y.expectation(given=G), Y):
        ...     print(E_Y_i_G == sum(Y_i.integrate(B) / P(B) * B.indicator for B in G if P(B) != 0))
        True
        True

        Check that passing an explict measure--different from the one carried by the random variable--into the `expectation` method works:

        >>> Q = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=4,
        ...     name="Q",
        ...     random_state=rng,
        ... )
        >>> one = RandomVector.from_constant(
        ...     domain=Omega,
        ...     sig_alg=trivial,
        ...     measure=Q | trivial,
        ...     constant=1,
        ... )
        >>> print(X.expectation(measure=Q) == X.integrate(measure=Q) * one)
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional expectation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $E(X\mid \mathcal{G})$ for which

        $$
        \int_V E(X\mid \mathcal{G}) \, dP = \int_V X \, dP,
        $$

        for all $V\in \mathcal{G}$. All such random variables are equal almost surely.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional expectation called a *Fourier expansion*:

        $$
        E(X\mid \mathcal{G}) = \sum_B \frac{\int_B X \, dP}{P(B)} I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability and $I_B$ is the indicator function of $B$.

        The *unconditional expectation* of $X$, denoted $E(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. In this case $E(X)$ is the constant random variable with

        $$
        E(X)(\omega) = \int_\Omega X \, dP,
        $$

        for all $\omega\in \Omega$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional expectation* to be the $d$-dimensional vector whose entries are the separate conditional expectations $E(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        return Operators.expectation(
            rv=self,
            given=given,
            measure=measure,
        )

    def variance(
        self,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableVector:
        r"""Compute the variance of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Calls `Operators.variance` with appropriate arguments.

        Parameters
        ----------
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        var : MeasurableVector
            The variance of the random vector.

        Examples
        --------
        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=2,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Check that the unconditional variance is equal to its definition and that it can be computed using the "shortcut formula."

        >>> print(X.variance() == ((X - X.expectation()) ** 2).expectation())
        True
        >>> print(X.variance() == (X**2).expectation() - X.expectation() ** 2)
        True

        Compute the unconditional variance of the random vector `Y`, and check that its components are the unconditional variances of the components of `Y`.

        >>> for V_Y_i, Y_i in zip(Y.variance(), Y):
        ...     print(V_Y_i == ((Y_i - Y_i.expectation()) ** 2).expectation())
        ...     print(V_Y_i == (Y_i**2).expectation() - Y_i.expectation() ** 2)
        True
        True
        True
        True

        Define a sub-sigma-algebra of `F` for conditional variances.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional variance of the random variable `X` is equal to a linear combination of indicators weighted by unconditional variances.

        >>> print(X.variance(given=G) == sum((X | B).variance().item() * B.indicator for B in G if P(B) > 0))
        True

        Check the same for the random vector `Y`.

        >>> for V_Y_i_G, Y_i in zip(Y.variance(given=G), Y):
        ...     print(V_Y_i_G == sum((Y_i | B).variance().item() * B.indicator for B in G if P(B) > 0))
        True
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F}, P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional variance* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable that is equal almost surely to the random variable

        $$
        V(X\mid \mathcal{G}) = E\left[ (X-E(X\mid \mathcal{G}))^2 \mid \mathcal{G}\right].
        $$

        The *unconditional variance* of $X$, denoted $V(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional variance is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional variance:

        $$
        V(X\mid \mathcal{G}) = \sum_B V(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $V(X|_B)$ is the unconditional variance of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional variance* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional variances $V(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        return Operators.variance(
            rv=self,
            given=given,
            measure=measure,
        )

    def std(
        self,
        given: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
    ) -> MeasurableVector:
        r"""Compute the standard deviation of a random vector, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Calls `Operators.std` with appropriate arguments.

        Parameters
        ----------
        given : SigmaAlgebra | None, default=None
            The sigma-algebra to condition on. If `None`, the trivial sigma-algebra is used.
        measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure of the underlying probability space of the random vector is used.

        Returns
        -------
        std : MeasurableVector
            The standard deviation of the random vector.

        Examples
        --------
        Define a probability space along with a 1-dimensinonal random variable and a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=100)
        >>> F = SigmaAlgebra.from_rand(domain=Omega, num_atoms=13, random_state=rng)
        >>> P = ProbabilityMeasure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> X = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=1,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randnorm(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     dim=2,
        ...     name="Y",
        ...     random_state=rng,
        ... )

        Check that the unconditional standard deviation is equal to its definition.

        >>> print(X.std() == X.variance() ** 0.5)
        True

        Compute the unconditional standard deviation of the random vector `Y`, and check that its components are the unconditional standard deviations of the components of `Y`.

        >>> for std_Y_i, Y_i in zip(Y.std(), Y):
        ...     print(std_Y_i == Y_i.variance() ** 0.5)
        True
        True

        Define a sub-sigma-algebra of `F` for conditional standard deviations.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=8,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional standard deviation of the random variable `X` is equal to a linear combination of indicators weighted by unconditional standard deviations.

        >>> print(X.std(given=G) == sum((X | B).std().item() * B.indicator for B in G if P(B) > 0))
        True

        Check the same for the random vector `Y`.

        >>> for std_Y_i_G, Y_i in zip(Y.std(given=G), Y):
        ...     print(
        ...         np.allclose(
        ...             std_Y_i_G.data,
        ...             sum((Y_i | B).std().item() * B.indicator for B in G if P(B) > 0).data,
        ...             atol=1e-7,
        ...         )
        ...     )
        True
        True

        Notes
        -----
        Let $X:\Omega \to \mathbb{R}$ be a random variable on a finite probability space $(\Omega, \mathcal{F},P)$, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional standard deviation* of $X$ with respect to $\mathcal{G}$ is any $\mathcal{G}$-measurable random variable $\sigma(X \mid \mathcal{G})$ that is equal almost surely to the random variable

        $$
        \sigma(X\mid \mathcal{G}) = \sqrt{V(X\mid \mathcal{G})}.
        $$

        The *unconditional standard deviation* of $X$, denoted $\sigma(X)$, is the case when $\mathcal{G}$ is the trivial $\sigma$-algebra with $\Omega$ as its only atom. The unconditional standard deviation is a constant random variable.

        The $\sigma$-algebra $\mathcal{G}$ is determined by its (finitely many) atoms, and we have the following formula for a conditional standard deviation:

        $$
        \sigma(X\mid \mathcal{G}) = \sum_B \sigma(X|_B) I_B,
        $$

        where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability, and where $\sigma(X|_B)$ is the unconditional standard deviation of the restricted random variable $X|_B:B\to \mathbb{R}$.

        If $X : \Omega \to \mathbb{R}^d$ is a random vector of dimension $d>1$, with components

        $$
        X = (X_1,X_2,\ldots,X_d),
        $$

        then we define the *conditional standard deviation* of $X$ to be the $d$-dimensional vector whose entries are the separate conditional standard deviations $\sigma(X_j \mid \mathcal{G})$, for $j=1,2,\ldots,d$.
        """
        return Operators.std(
            rv=self,
            given=given,
            measure=measure,
        )

    def pushforward(
        self,
        measure: Measure | ParametrizedMeasure | None = None,
    ) -> Measure | ParametrizedMeasure:
        r"""Push forward a (parametrized) measure on the domain of a measurable vector to a measure on its range.

        See the Notes section below for the mathematical details.

        Calls `Operators.pushforward` with appropriate arguments.

        Parameters
        ----------
        measure : Measure | ParametrizedMeasure | None, default=None
            Measure to push forward. If `None`, the measure carried by the measurable vector is used.

        Returns
        -------
        pushforward : Measure | ParametrizedMeasure
            The measure pushed forward along the measurable vector.

        Examples
        --------
        Define a measure space.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableVector,
        ...     Measure,
        ...     Operators,
        ...     ParametrizedProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ... )

        Define a 2-dimensional measurable vector and pushforward the measure `mu`.

        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> mu_f = f.pushforward(mu)
        >>> print(mu_f)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu_f':
                    measure
        f_0 f_1
        1   2          1
        3   4          5

        Now define a measurable space with a sample space.

        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )

        Define a parametrized probability measure on the sigma-algebra.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> def mapping(*, theta, u):
        ...     if theta == 0:
        ...         if u == 0:
        ...             return 0.1
        ...         elif u == 1:
        ...             return 0.2
        ...         else:
        ...             return 0.7
        ...     if theta == 1:
        ...         if u == 0:
        ...             return 0.4
        ...         elif u == 1:
        ...             return 0.5
        ...         else:
        ...             return 0.1
        >>> P = ParametrizedProbabilityMeasure(
        ...     measure_domain=F, parameter_domain=Theta, mapping=mapping
        ... )

        Define a 2-dimensional random vector and pushforward the parametrized probability measure `P`.

        >>> X = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 1),
        ...         1: (1, 1),
        ...         2: (3, 1),
        ...         3: (3, 1),
        ...     },
        ... )
        >>> P_X = X.pushforward(P)
        >>> print(P_X)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P_X':
                        probability
        theta X_0 X_1
        0     1   1            0.1
              3   1            0.9
        1     1   1            0.4
              3   1            0.6

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a measurable vector on a measure space $(X, \mathcal{F}, \mu)$. Then we define a measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(A) = \mu\left( \{x \in X : f(x) \in A\}\right),
        $$

        for all Borel subsets $A\subset \mathbb{R}^d$.

        If $\mu$ is a parametrized measure on $X$ with parameter domain $\Theta$, then we define a parametrized measure $\mu_X$ on $\mathbb{R}^d$, called the *pushforward* (or *image*) *measure* of $\mu$ along $f$, by setting

        $$
        \mu_X(\theta, A) = \mu\left(\theta, \{x \in X : f(x) \in A\}\right),
        $$

        for all $\theta \in \Theta$ and all Borel subsets $A\subset \mathbb{R}^d$.
        """
        return Operators.pushforward(
            vec=self,
            measure=measure,
        )
