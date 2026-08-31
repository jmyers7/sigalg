"""A class representing a parametrized probability measure."""

from __future__ import annotations

import inspect
from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

from .parametrized_measure import ParametrizedMeasure

if TYPE_CHECKING:
    from scipy.stats import rv_discrete

    from ..functions.parametrized_measurable_function import (
        ParametrizedMeasurableFunction,
    )
    from ..measures.measure import Measure
    from ..spaces.domain import Domain


class ParametrizedProbabilityMeasure(ParametrizedMeasure):
    r"""A class representing a parametrized probability measure.

    The `__init__` constructor is not meant to be used directly. Instead, the user should use the `from_domains` class method.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from math import comb
    >>> from sigalg.core import (
    ...     Domain,
    ...     ParametrizedProbabilityMeasure,
    ...     SampleSpace,
    ... )

    Define a 1-dimensional parameter domain and sample space.

    >>> Theta = Domain([0.0, 0.25, 0.75, 1.0], name="Theta", variable_names=["theta"])
    >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")

    Define a binomial probability distribution Bin(n=2, theta), parametrized by theta.

    >>> def mapping(*, theta, omega):
    ...     omega = int(omega)
    ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
    >>> P = ParametrizedProbabilityMeasure.from_domains(
    ...     measure_domain=Omega, parameter_domain=Theta, mapping=mapping
    ... )
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized probability measure 'P':
    theta  0.00    0.25    0.75  1.00
    omega
    0       1.0  0.5625  0.0625   0.0
    1       0.0  0.3750  0.3750   0.0
    2       0.0  0.0625  0.5625   1.0

    Evaluate at a parameter to obtain a probability measure.

    >>> print(P(theta=0.25))  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P(theta=0.25)':
           P(theta=0.25)
    omega
    0             0.5625
    1             0.3750
    2             0.0625

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be a measurable space and $\Theta$ a nonempty set. A *parametrized probability measure* is a function

    $$
    P : \Theta \times \mathcal{F} \to \mathbb{R}
    $$

    such that, for each fixed $\theta \in \Theta$, the partial function

    $$
    P(\theta, -): \mathcal{F} \to \mathbb{R}, \quad U \mapsto P(\theta,U),
    $$

    is a probability measure on the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain* and elements $\theta\in \Theta$ are called *parameters*.
    """

    _repr_name = "ParametrizedProbabilityMeasure"
    _str_name = "Parametrized probability measure"
    _default_name = "P"

    # --------------------- constructors --------------------- #

    # TODO: change to from_validated
    @classmethod
    def from_scipy(
        cls,
        dist: rv_discrete,
        support: tuple[Hashable, list],
        parameter_domain: Domain,
        name: Hashable = "P",
    ) -> ParametrizedProbabilityMeasure:
        """Initialize the parametrized probability measure from a discrete SciPy probability distribution.

        Parameters
        ----------
        dist : rv_discrete
            A discrete SciPy probability distribution.
        support : tuple[Hashable, list]
            A tuple containing the name of the support variable and a list of its possible values.
        parameter_domain : Domain
            The domain of the parameters for the parametrized probability measure.
        name : Hashable, default="P"
            The name of the parametrized probability measure.

        Returns
        -------
        param_prob_measure : ParametrizedProbabilityMeasure
            The constructed parametrized probability measure.

        Examples
        --------
        >>> from scipy.stats import binom, hypergeom
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedProbabilityMeasure,
        ... )

        We will build a parametrized binomial probability measure using the `binom` class from SciPy. First, we inspect the signature of the `pmf` method on the SciPy website and notice that the measure is parametrized by `n` and `p`. We build a parameter domain for these.

        >>> Theta = Domain(
        ...     [(2, 0.25), (3, 0.75)],
        ...     variable_names=["n", "p"],
        ...     name="Theta",
        ... )

        The support variable of the `pmf` is `k`. We choose it to have support `[0, 1, 2, 3]`.

        >>> P = ParametrizedProbabilityMeasure.from_scipy(
        ...     dist=binom,
        ...     support=("k", [0, 1, 2, 3]),
        ...     parameter_domain=Theta,
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
        n       2         3
        p    0.25      0.75
        k
        0  0.5625  0.015625
        1  0.3750  0.140625
        2  0.0625  0.421875
        3  0.0000  0.421875

        For a second example, we build a parametrized hypergeometric distribution. Again, we inspect the signature of the `pmf` method on the SciPy website and note that it is parametrized by `M`, `n`, and `N`.

        >>> Theta = Domain(
        ...     [(5, 3, 3), (10, 5, 5)],
        ...     variable_names=["M", "n", "N"],
        ...     name="Theta",
        ... )

        The support variable of the `pmf` is again `k`.

        >>> Q = ParametrizedProbabilityMeasure.from_scipy(
        ...     dist=hypergeom,
        ...     support=("k", [0, 1, 2, 3, 4, 5]),
        ...     parameter_domain=Theta,
        ...     name="Q",
        ... )
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q':
        M    5        10
        n    3         5
        N    3         5
        k
        0  0.0  0.003968
        1  0.3  0.099206
        2  0.6  0.396825
        3  0.1  0.396825
        4  0.0  0.099206
        5  0.0  0.003968
        """
        from scipy.stats import rv_discrete

        from ..spaces.domain import Domain
        from ..spaces.sample_space import SampleSpace

        if not isinstance(parameter_domain, Domain):
            raise TypeError("parameter_domain must an instance of Domain")
        if not isinstance(dist, rv_discrete):
            raise TypeError("dist must be a discrete scipy distribution (rv_discrete)")
        if not isinstance(support, tuple) or len(support) != 2:
            raise ValueError("support must be a 2-tuple (name, values)")
        if not isinstance(support[0], Hashable):
            raise TypeError("support[0] must be hashable")
        if not isinstance(support[1], list):
            raise TypeError("support[1] must be a list")

        sample_space = SampleSpace(support[1], variable_names=[support[0]])

        parameters = parameter_domain.variable_names
        parameter_names = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in parameters
        ] + [inspect.Parameter(support[0], inspect.Parameter.KEYWORD_ONLY)]
        sig = inspect.Signature(parameter_names)

        def mapping(**kwargs):
            bound = sig.bind(**kwargs)
            return dist.pmf(**bound.arguments)

        mapping.__signature__ = sig

        return cls.from_domains(
            measure_domain=sample_space,
            parameter_domain=parameter_domain,
            mapping=mapping,
            output_name="probability",
            name=name,
        )

    # --------------------- probability methods --------------------- #

    def derivative(
        self,
        base_measure: Measure | None = None,
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> ParametrizedMeasurableFunction:
        r"""Compute the Radon-Nikodym derivative with respect to a base measure.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        base_measure : Measure | None, default=None
            The base measure with respect to which the Radon-Nikodym derivative is computed.
            If None, the counting measure on the domain of the sigma-algebra is used.
        name : Hashable | None, default=None
            The name of the resulting Radon-Nikodym derivative. If None, a default name
            is generated based on the names of the current measure and the base measure.

        Returns
        -------
        derivative : ParametrizedMeasurableFunction
            The Radon-Nikodym derivative of the current measure with respect to the base measure.

        Examples
        --------
        >>> from sigalg import (
        ...     Domain,
        ...     Measure,
        ...     Operators,
        ...     ParametrizedProbabilityMeasure,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a parametrized probability measure on a measurable space.

        >>> Omega = SampleSpace.from_sequence(size=6, variable_name="omega")
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,
        ...         5: 2,
        ...     },
        ... )
        >>> mapping = dict(zip(Theta @ F.atom_space, [0.3, 0.5, 0.2, 0.0, 0.1, 0.9]))
        >>> P = ParametrizedProbabilityMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
        theta    0    1
        F
        0      0.3  0.0
        1      0.5  0.1
        2      0.2  0.9

        Define a base measure.

        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 4,
        ...     },
        ... )

        In SigAlg, the derivative `P.derivative(mu)` is an instance of `ParametrizedMeasurableFunction`. Conceptually, it is convenient to think of a parametrized probability measure as a family of probability measures, one for each parameter value. Likewise, a parametrized measurable function should be thought of as a family, too. Then the derivative `P.derivative(mu)` is just the family of Radon-Nikodym derivatives of the component measures of the parametrized measure, one for each parameter.

        >>> P.derivative(mu)
        ParametrizedMeasurableFunction(parameters=(theta), measurable_vars=(omega), domain=Omega, sig_alg=F, measure=mu, name=dP_dmu)

        Print the derivative for inspection.

        >>> print(P.derivative(mu))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'dP_dmu':
        theta     0      1
        omega
        0      0.30  0.000
        1      0.30  0.000
        2      0.25  0.050
        3      0.25  0.050
        4      0.05  0.225
        5      0.05  0.225

        Check that the Radon-Nikodym derivative `P.derivative(mu)` has the defining property of a Radon-Nikodym derivative.

        >>> integrate = Operators.integrate
        >>> all(P(A) == integrate(P.derivative(mu), A) for A in F)
        True

        The `derivative` method of `ParametrizedProbabilityMeasure` allows us to recover the familiar conditional probability mass functions from elementary probability theory. To demonstrate, let's define a new measure on the power-set sigma-algebra of the sample space above.

        >>> Q = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.0,
        ...         1: 0.05,
        ...         2: 0.1,
        ...         3: 0.2,
        ...         4: 0.2,
        ...         5: 0.45,
        ...     },
        ...     name="Q",
        ... )

        Define a pair of random variables.

        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     measure=Q,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...         4: 3,
        ...         5: 3,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     domain=Omega,
        ...     measure=Q,
        ...     mapping={
        ...         0: 2,
        ...         1: 3,
        ...         2: 4,
        ...         3: 7,
        ...         4: 2,
        ...         5: 5,
        ...     },
        ...     name="Y"
        ... )

        Condition the measure on `X`, creating an instance of `ParametrizedProbabilityMeasure`.

        >>> print(Q.conditional(X))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q(-|X)':
        X        1         2         3
        omega
        0      0.0  0.000000  0.000000
        1      1.0  0.000000  0.000000
        2      0.0  0.333333  0.000000
        3      0.0  0.666667  0.000000
        4      0.0  0.000000  0.307692
        5      0.0  0.000000  0.692308

        Notice that the parameters of this measure are the values of `X`. Notice also that this is a parametrized probability measure on the original sample space. We push it forward along the random variable `Y`.

        >>> print(Q.conditional(X) >> Y)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q(-|X)_Y':
        X    1         2         3
        Y
        2  0.0  0.000000  0.307692
        3  1.0  0.000000  0.000000
        4  0.0  0.333333  0.000000
        5  0.0  0.000000  0.692308
        7  0.0  0.666667  0.000000

        Notice now that it is a parametrized probability measure on the range of `Y`. If we apply the `derivative` method without an explict `measure` parameter, it defaults to the counting measure and we obtain the conditional probability mass function of `Y` given `X`.

        >>> print((Q.conditional(X) >> Y).derivative())  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'dQ(-|X)_Y_dC':
        X    1         2         3
        Y
        2  0.0  0.000000  0.307692
        3  1.0  0.000000  0.000000
        4  0.0  0.333333  0.000000
        5  0.0  0.000000  0.692308
        7  0.0  0.666667  0.000000

        Notice the subtle (but meaningful) difference between this last printout and the one before: One is an instance of `ParametrizedProbabilityMeasure`, while the other is an instance of `ParametrizedMeasurableFunction`. These are different things!

        Finally, we verify that integrating the conditional probability mass function yields conditional probability.

        >>> all(
        ...     (Q.conditional(X) >> Y)([y]) == integrate((Q.conditional(X) >> Y).derivative(), [y])
        ...     for y in Y.range
        ... )
        True


        Notes
        -----
        By definition, a *parametrized probability measure* on a measurable space $(\Omega, \mathcal{F})$, with parameter domain $\Theta$, is a function

        $$
        P: \Theta \times \mathcal{F} \to [0,1]
        $$

        for which each partial function

        $$
        P(\theta, -) : \mathcal{F} \to [0,1], \quad U \mapsto P(\theta, U),
        $$

        is a probability measure, for each $\theta \in \Theta$. If $\mu$ is a measure on $\mathcal{F}$ with the property that $P(\theta, -) \ll \mu$ for each $\theta$, then the *Radon-Nikodym derivative* of $P$ with respect to $\mu$ is the function

        $$
        \frac{dP}{d\mu} : \Theta \times \Omega \to \mathbb{R}
        $$

        for which each partial function

        $$
        \frac{dP}{d\mu}(\theta, -) : \Omega \to \mathbb{R}, \quad \omega \mapsto \frac{dP}{d\mu}(\theta, \omega)
        $$

        is a Radon-Nikodym derivative of $P(\theta, -)$ with respect to $\mu$, for each $\theta\in \Theta$. The measure $\mu$ is called the *base measure*.
        """
        from .._utils.measure_helpers import compute_radon_nikodym
        from ..functions.parametrized_measurable_function import (
            ParametrizedMeasurableFunction,
        )
        from ..measures.measure import Measure

        if base_measure is None:
            base_measure = Measure.counting(self.sig_alg.domain)

        if name is None:
            name = f"d{self.name}_d{base_measure.name}"

        data = compute_radon_nikodym(
            self_data=self.data,
            base_measure_data=base_measure.data,
            sig_alg_data=self.sig_alg.data,
            parameter_names=self.parameter_names,
        )

        return ParametrizedMeasurableFunction._from_validated(
            data=data,
            sig_alg=self.sig_alg,
            measure=base_measure,
            complete_domain_name=f"{self.parameter_domain_name} x {self.sig_alg.domain.name}",
            parameter_domain_name=self.parameter_domain_name,
            parameter_names=self.parameter_names,
            name=name,
        )

    def surprisal(
        self,
        base_measure: Measure | None = None,
        base: Literal["e", "2", "10"] = "e",
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> ParametrizedMeasurableFunction:
        """Pass."""
        from .._utils.measure_helpers import compute_surprisal
        from ..functions.parametrized_measurable_function import (
            ParametrizedMeasurableFunction,
        )
        from ..measures.measure import Measure

        if base_measure is None:
            base_measure = Measure.counting(self.sig_alg.domain)

        if name is None:
            name = f"s({self.name}; {base_measure.name})"

        data = compute_surprisal(
            self_data=self.data,
            base_measure_data=base_measure.data,
            sig_alg_data=self.sig_alg.data,
            parameter_names=self.parameter_names,
            base=base,
        )

        return ParametrizedMeasurableFunction._from_validated(
            data=data,
            sig_alg=self.sig_alg,
            measure=None,
            complete_domain_name=f"{self.parameter_domain_name} x {self.sig_alg.domain.name}",
            parameter_domain_name=self.parameter_domain_name,
            parameter_names=self.parameter_names,
            name=name,
        )
