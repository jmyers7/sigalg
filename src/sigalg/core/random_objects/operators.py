from __future__ import annotations  # noqa: D100

from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .random_vector import RandomVector


class Operators:
    """Class containing operators on random vectors, such as integration, expectation, variance, standard deviation, covariance, correlation, and pushforward of probability measures."""

    @classmethod
    def integrate(
        cls,
        rv: RandomVector,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> pd.Series | Real:
        """Compute the integral of a `RandomVector` with respect to a `ProbabilityMeasure`.

        Parameters
        ----------
        rv : RandomVector
            The random vector to integrate.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which to integrate. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).

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
        >>> integrate = Operators.integrate
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (1, 2), 2: (3, 4)})
        >>> # Integral of a 2-dimensional random vector
        >>> integrate(rv=X, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        feature
        X_0    2.0
        X_1    3.0
        Name: integral(X), dtype: float64
        >>> # Integral of a random variable
        >>> Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 1, 1: 1, 2: 0})
        >>> float(integrate(rv=Y, probability_measure=P))
        0.5
        """
        exp = cls.expectation(rv=rv, probability_measure=probability_measure)
        integral = exp.data.iloc[0]
        if rv.dimension > 1:
            integral.index = rv.data.columns
            integral.name = f"integral({rv.name})" if rv.name is not None else None

        return integral

    @classmethod
    def expectation(
        cls,
        rv: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the expectation of a `RandomVector` with respect to a `ProbabilityMeasure`, optionally conditioned on a `SigmaAlgebra`.

        The conditional expectation of a random variable is another random variable that is constant on each atom of the sigma algebra, its value on an atom being the mean value of the original random variable on that atom. This mean value is computed with respect to the conditional probabilities of the atom. If an atom has probability 0, the expected value is defined to be 0 on this atom.

        The unconditional expectation is the same as the conditional expectation with respect to the trivial sigma algebra (with a single atom equal to the entire sample space), so this description applies to the unconditional expectation too. In particular, the unconditional expectation of a random variable is a constant random variable equal to the mean value of the original random variable with respect to the probability measure.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the expectation.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional expectation.
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
        >>> from sigalg.core import Operators, RandomVector, SampleSpace, SigmaAlgebra
        >>> expectation = Operators.expectation
        >>> domain = SampleSpace().from_sequence(size=3, prefix="omega")
        >>> outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        >>> probabilities = {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3}
        >>> X = RandomVector(domain).from_dict(outputs).with_probability_measure(probabilities)
        >>> # Compute unconditional expectation
        >>> expectation(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X)':
        expectation   E(X)_0  E(X)_1
        sample
        omega_0          3.2     4.2
        omega_1          3.2     4.2
        omega_2          3.2     4.2
        >>> # Compute conditional expectation given a sigma algebra
        >>> F = SigmaAlgebra(domain).from_dict({"omega_0": 0, "omega_1": 0, "omega_2": 1})
        >>> expectation(X, F) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X|F)':
        expectation   E(X|F)_0  E(X|F)_1
        sample
        omega_0       2.428571  3.428571
        omega_1       2.428571  3.428571
        omega_2       5.000000  6.000000
        """
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
                result.index.data.name = "expectation"
                result.data.columns.name = "expectation"
                return result
            else:
                return RandomVariable(
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

            expectations = RandomVector(domain=rv.domain, name=name).from_dict(outputs)
            expectations.data.fillna(0, inplace=True)

            if expectations.dimension > 1:
                expectations.index.data.name = "expectation"
                expectations.data.columns.name = "expectation"

            return expectations

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
        """Compute the variance of a random vector, optionally conditioned on a sigma algebra.

        The conditional variance of a random variable is another random variable that is constant on each atom of the sigma algebra, its value on an atom being the variance of the original random variable on that atom. This variance is computed with respect to the conditional probabilities of the atom.

        The unconditional variance is the same as the conditional variance with respect to the trivial sigma algebra (with a single atom equal to the entire sample space), so this description applies to the unconditional variance too. In particular, the unconditional variance of a random variable is a constant random variable equal to the variance of the original random variable with respect to the probability measure.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the variance.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma algebra with respect to which to compute the variance. If `None`, computes the variance without conditioning.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `None`, uses `rv.probability_measure`.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sigma_algebra` is not a `SigmaAlgebra` or `None`, or if `probability_measure` is not a `ProbabilityMeasure` or `None`.

        Returns
        -------
        var : RandomVector
            The variance of the random vector, optionally conditioned on the sigma algebra.

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
        >>> variance = Operators.variance
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (2, 1), 2: (3, 4)})
        >>> # Unconditional variance of a 2-dimensional random vector
        >>> variance(X, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(X)':
        variance  V(X)_0  V(X)_1
        sample
        0           0.61    1.81
        1           0.61    1.81
        2           0.61    1.81
        >>> # Conditional variance of a 2-dimensional random vector
        >>> F = SigmaAlgebra(sample_space=Omega, name="F").from_dict({0: 0, 1: 0, 2: 1})
        >>> variance(X, sigma_algebra=F, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(X|F)':
        variance  V(X|F)_0  V(X|F)_1
        sample
        0             0.24      0.24
        1             0.24      0.24
        2             0.00      0.00
        >>> # Unconditional variance of a random variable
        >>> Z = RandomVariable(domain=Omega, name="Z").from_dict({0: 1, 1: -2, 2: 3})
        >>> variance(Z, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'V(Z)':
                V(Z)
        sample
        0       4.69
        1       4.69
        2       4.69
        >>> # Conditional variance of a random variable
        >>> variance(Z, sigma_algebra=F, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'V(Z|F)':
                V(Z|F)
        sample
        0         2.16
        1         2.16
        2         0.00
        """
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
        var = cls.expectation(
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
        else:
            name = f"V({rv.name})" if rv.name is not None else None

        var = var.with_name(name, modify_index=True)

        if var.dimension > 1:
            var.index.data.name = "variance"

        return var

    @classmethod
    def std(
        cls,
        rv: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Compute the standard deviation of a random vector.

        The conditional standard deviation of a random variable is another random variable that is constant on each atom of the sigma algebra, its value on an atom being the standard deviation of the original random variable on that atom. This standard deviation is computed with respect to the conditional probabilities of the atom.

        The unconditional standard deviation is the same as the conditional standard deviation with respect to the trivial sigma algebra (with a single atom equal to the entire sample space), so this description applies to the unconditional standard deviation too. In particular, the unconditional standard deviation of a random variable is a constant random variable equal to the standard deviation of the original random variable with respect to the probability measure.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the standard deviation.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional standard deviation.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `None`, uses `rv.probability_measure`.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector`, or if `sigma_algebra` is not a `SigmaAlgebra` or `None`, or if `probability_measure` is not a `ProbabilityMeasure` or `None`.

        Returns
        -------
        std : RandomVector
            The standard deviation of the random vector, optionally conditioned on the sigma algebra.

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
        >>> std = Operators.std
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.3, 2: 0.5})
        >>> X = RandomVector(domain=Omega, name="X").from_dict({0: (1, 2), 1: (2, 1), 2: (3, 4)})
        >>> # Unconditional standard deviation of a 2-dimensional random vector
        >>> std(X, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(X)':
        std  std(X)_0  std(X)_1
        sample
        0        0.781025  1.345362
        1        0.781025  1.345362
        2        0.781025  1.345362
        >>> # Conditional standard deviation of a 2-dimensional random vector
        >>> F = SigmaAlgebra(sample_space=Omega, name="F").from_dict({0: 0, 1: 0, 2: 1})
        >>> std(X, sigma_algebra=F, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'std(X|F)':
        std  std(X|F)_0  std(X|F)_1
        sample
        0          0.489898    0.489898
        1          0.489898    0.489898
        2          0.000000    0.000000
        >>> # Unconditional standard deviation of a random variable
        >>> Z = RandomVariable(domain=Omega, name="Z").from_dict({0: 1, 1: -2, 2: 3})
        >>> std(Z, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(Z)':
            std(Z)
        sample
        0       2.165641
        1       2.165641
        2       2.165641
        >>> # Conditional standard deviation of a random variable
        >>> std(Z, sigma_algebra=F, probability_measure=P) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'std(Z|F)':
                std(Z|F)
        sample
        0       1.469694
        1       1.469694
        2       0.000000
        """
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

        std = (
            cls.variance(
                rv, sigma_algebra=sigma_algebra, probability_measure=probability_measure
            )
            ** 0.5
        )

        if sigma_algebra is None:
            std = std.with_name(
                f"std({rv.name})" if rv.name is not None else None, modify_index=True
            )
        else:
            std = std.with_name(
                (
                    f"std({rv.name}|{sigma_algebra.name})"
                    if rv.name is not None and sigma_algebra.name is not None
                    else None
                ),
                modify_index=True,
            )

        if std.dimension > 1:
            std.index.data.name = "std"
            std.data.columns.name = "std"

        return std

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
        >>> X_range = X.range
        >>> print(pd.concat([X_range.data, P_X.data], axis=1)) # doctest: +NORMALIZE_WHITESPACE
                X_0  X_1  probability
        output
        x_0       1   2          0.2
        x_1       3   4          0.8
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

        if rv.dimension == 1:
            rv_cols = [rv.data.name]
        else:
            rv_cols = rv.data.columns.tolist()
        pushforward_probs = (
            pd.concat([rv.data, probability_measure.data], axis=1)
            .groupby(rv_cols)
            .sum()
        )
        pushforward_probs.index = rv.range.data.index
        measure_name = f"P_{rv.name}" if rv.name is not None else None
        pushforward_measure = ProbabilityMeasure(name=measure_name).from_pandas(
            pushforward_probs.iloc[:, -1]
        )

        return pushforward_measure


class OperatorsMethods:
    """Mixin class to add operators as methods to `RandomVector` and `ProbabilityMeasure`."""

    def integrate(
        self,
        *,
        rv: RandomVector | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> pd.Series | Real:
        """Compute the integral of a `RandomVector` with respect to a `ProbabilityMeasure`.

        If `self` is a `RandomVector`, computes the integral of `self` with respect to `probability_measure`. In this case, `rv` must be `None` or equal to `self`.

        If `self` is a `ProbabilityMeasure`, computes the integral of the random vector `rv` with respect to `self`. In this case, `probability_measure` must be `None` or equal to `self`.

        Parameters
        ----------
        rv : RandomVector | None, default=None
            The random vector for which to compute the integral. Must be `None` or equal to `self` if `self` is a `RandomVector`.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to use. If `self` is a random vector and `probability_measure` is `None`, uses the probability measure associated with the random vector. Must be `None` or equal to `self` if `self` is a `ProbabilityMeasure`.

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
            )
        elif isinstance(self, ProbabilityMeasure):
            if probability_measure is not None and probability_measure != self:
                raise ValueError(
                    "probability_measure must be None or equal to self when calling integrate on a ProbabilityMeasure, as the probability measure itself is used as the argument."
                )
            return Operators.integrate(
                rv=rv,
                probability_measure=self,
            )

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
