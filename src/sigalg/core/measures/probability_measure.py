"""A class representing a probability measure on a sigma-algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .measure import Measure

if TYPE_CHECKING:
    from collections.abc import Hashable
    from numbers import Real

    from ...typing.mapping_like import MappingLike
    from ...typing.measure_domain import MeasureDomain
    from ..functions.measurable_function import MeasurableFunction
    from ..functions.measurable_vector import MeasurableVector
    from ..functions.parametrized_measurable_function import (
        ParametrizedMeasurableFunction,
    )
    from ..functions.radon_nikodym import RadonNikodym
    from ..functions.random_variable import RandomVariable
    from ..functions.random_vector import RandomVector
    from ..measures.parametrized_probability_measure import (
        ParametrizedProbabilityMeasure,
    )
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.measure_space import MeasureSpace
    from ..spaces.set import Set


class ProbabilityMeasure(Measure):
    r"""A class representing a probability measure on a sigma-algebra.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : MeasureDomain | None, default=None
        The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`; in the latter case the domain will be set to the-power set of the `Domain` instance.
    mapping : MappingLike | None, default=None
        A mapping from the domain to the probability values.
    output_name : str, default="probability"
        The name of the output variable of the measure.
    name : Hashable, default="P"
        The name of the measure.
    **kwargs
        Keyword arguments to catch unexpected parameters.

    Examples
    --------
    Define a probability measure on a sigma-algebra with two atoms.

    >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> F = SigmaAlgebra(
    ...    domain=Omega,
    ...    mapping={
    ...        0: 0,
    ...        1: 0,
    ...        2: 1,
    ...    },
    ... )
    >>> P = ProbabilityMeasure(domain=F, mapping={0: 0.2, 1: 0.8})
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
             P
    F
    0      0.2
    1      0.8

    Define a probability measure directly on a sample space, which will use the power-set sigma-algebra by default.

    >>> Q = ProbabilityMeasure(
    ...     domain=Omega,
    ...     mapping={
    ...         0: 0.1,
    ...         1: 0.3,
    ...         2: 0.6,
    ...     },
    ...     name="Q",
    ... )
    >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'Q':
            Q
    s
    0     0.1
    1     0.3
    2     0.6
    >>> print(Q.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
            R
    s
    0       0
    1       1
    2       2

    Define the same probability measure directly on a `list` of points.

    >>> Q = ProbabilityMeasure(
    ...     domain=[0, 1, 2],
    ...     mapping={
    ...         0: 0.1,
    ...         1: 0.3,
    ...         2: 0.6,
    ...     },
    ...     name="Q",
    ... )
    >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'Q':
            Q
    x
    0     0.1
    1     0.3
    2     0.6
    >>> print(Q.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
            R
    x
    0       0
    1       1
    2       2

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be a measurable space consisting of a $\sigma$-algebra $\mathcal{F}$ on a nonempty set $\Omega$. A *probability measure* $P$ is a countably additive function $P: \mathcal{F} \to [0,\infty)$ such that $P(\Omega) = 1$. Here, *countable additivity* means that

    $$
    P \left( \bigcup_{k=1}^\infty A_k \right) = \sum_{k=1}^\infty P(A_k)
    $$

    for all collections $\{A_k\}_{k=1}^\infty$ of pairwise disjoint measurable sets. If $\Omega$ is finite (as it always is, in SigAlg), then $P$ needs only to be finitely additive in order to be countably additive.
    """

    _default_name = "P"
    _repr_name = "ProbabilityMeasure"
    _str_name = "Probability measure"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: MeasureDomain | None = None,
        mapping: MappingLike | None = None,
        output_name: Hashable | None = None,
        name: Hashable = "P",
    ) -> None:

        super().__init__(
            domain=domain,
            mapping=mapping,
            kind="probability",
            domain_kind="SampleSpace",
            output_name=output_name,
            name=name,
        )

    @classmethod
    def uniform(
        cls,
        domain: MeasureDomain,
        output_name: Hashable | None = None,
        name: Hashable = "U",
    ) -> ProbabilityMeasure:
        r"""Create a uniform probability measure on a sigma-algebra or sample space.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        domain : MeasureDomain
            The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`; in the latter case the domain will be set to the power set of the `Domain` instance.
        name : Hashable, default="U"
            A name for the measure.

        Returns
        -------
        prob_measure: ProbabilityMeasure
            A uniform ProbabilityMeasure instance on the provided sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> U = ProbabilityMeasure.uniform(domain=F)
        >>> print(U)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'U':
                  U
        F
        0   0.333333
        1   0.333333
        2   0.333333
        >>> V = ProbabilityMeasure.uniform(domain=Omega, name="V")
        >>> print(V)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'V':
                  V
        s
        0       0.25
        1       0.25
        2       0.25
        3       0.25

        Notes
        -----
        Let $(\Omega,\mathcal{F})$ be a measurable space where $\Omega$ is finite, and suppose that $\mathcal{F}$ has $n$ atoms. The *uniform probability measure* on $\mathcal{F}$ is the unique probability measure $P$ such that

        $$
        P(A) = \frac{1}{n}
        $$

        for all atoms $A\in \mathcal{F}$.
        """
        import pandas as pd

        from ...validation.measure_domain_normalizer import MeasureDomainNormalizer

        if output_name is None:
            output_name = name

        v = MeasureDomainNormalizer(measure_domain=domain, kind="probability")

        n = len(v.domain)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on sigma-algebra with no atoms."
            )
        data = pd.Series(1.0 / n, index=v.domain.data, name=output_name)

        return cls._from_validated(
            data=data,
            kind="probability",
            sig_alg=v.sig_alg,
            name=name,
        )

    # --------------------- probability methods --------------------- #

    def sample(
        self,
        size: int = 1,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> MeasureSpace:
        """Generate random samples from this probability measure.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate. Must be positive.
        name : Hashable | None, default=None
            A name for the random sample. If `None`, a default will be generatd.
        random_state : int | np.random.Generator | None, default=None
            Random seed or generator for reproducibility.

        Returns
        -------
        sample : MeasureSpace
            An instance of `MeasureSpace` whose domain consists of the random samples and whose measure is a counting measure giving the number of each sample produced.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace
        >>> rng = np.random.default_rng(42)

        Define a probability measure on the power set of a sample space containing three sample points.

        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.45,
        ...         2: 0.3,
        ...     },
        ... )

        Draw a random sample from the probability measure of size `150`.

        >>> P_sample = P.sample(size=150, random_state=rng)

        Print the measure of the sample.

        >>> print(P_sample.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'C':
               C
        s
        1     74
        2     39
        0     37

        Divide the counting measure by the sample size to obtain a probability measure. Note that it closely matches the original probability measure.

        >>> C = P_sample.measure
        >>> print(C / 150)  # doctest: +NORMALIZE_WHITESPACE
        Function '(C / 150)':
           (C / 150)
        s
        1   0.493333
        2   0.260000
        0   0.246667
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain
        from ..spaces.measure_space import MeasureSpace
        from .measure import Measure

        if not isinstance(size, int):
            raise TypeError("size must be an integer.")
        if size < 1:
            raise ValueError("size must be positive.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        samples = rng.choice(list(self.domain), size=size, p=list(self.data))
        samples = pd.DataFrame(samples, columns=self.domain.variable_names)

        if name is None:
            name = f"{self.name}_sample"

        data = samples.value_counts().rename("C")
        domain = Domain._from_validated(
            data=data.index
            if self.domain.dimension > 1
            else data.index.get_level_values(0),
            name=name,
        )
        sig_alg = SigmaAlgebra.power_set(domain)

        measure = Measure._from_validated(
            data=data,
            kind="measure",
            sig_alg=sig_alg,
            name="C",
        )

        return MeasureSpace._from_validated(measure=measure)

    def conditional(
        self,
        event: Set,
        given: SigmaAlgebra | Set | RandomVector,
    ) -> RandomVariable:
        r"""Compute a conditional probability.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : MeasurableSet
            The event of which to compute the conditional probability.
        given : SigmaAlgebra | MeasurableSet | RandomVector
            The given condition, which can be a sigma-algebra, an event, or a random vector.

        Raises
        ------
        TypeError
            If `event` is not an `MeasurableSet` instance, or if `given` is not a `SigmaAlgebra`, `MeasurableSet`, or `RandomVector` instance.

        Returns
        -------
        cond_prob : RandomVariable
            A random variable representing the conditional probability of `event` given `given`.

        Examples
        --------
        Define a probability space and a sub-sigma-algebra.

        >>> import numpy as np
        >>> import pytest
        >>> from sigalg.core import ProbabilitySpace, SigmaAlgebra
        >>> rng = np.random.default_rng(seed=42)
        >>> Omega, F, P = ProbabilitySpace.from_rand(
        ...     domain_size=25,
        ...     num_atoms=12,
        ...     random_state=rng,
        ... )
        >>> G = SigmaAlgebra.from_rand(
        ...     super=F,
        ...     num_atoms=7,
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional probability has its defining property through 15 random trials. (See the Notes section.)

        >>> is_consistent = []
        >>> for _ in range(15):
        ...     U = F.get_random_set(
        ...         num_atoms=int(rng.integers(0, F.num_atoms, endpoint=True)),
        ...         random_state=rng,
        ...         name="U",
        ...     )
        ...     V = G.get_random_set(
        ...         num_atoms=int(rng.integers(0, G.num_atoms, endpoint=True)),
        ...         random_state=rng,
        ...         name="V",
        ...     )
        ...     is_consistent.append(
        ...         P(U & V) == pytest.approx(P.conditional(event=U, given=G).integrate(V))
        ...     )
        >>> print(all(is_consistent))
        True

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space, let $U \in \mathcal{F}$ be an event, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional probability* of $U$ given $\mathcal{G}$ is a $\mathcal{G}$-measurable random variable, denoted $P(U|\mathcal{G})$, for which

        $$
        P(U \cap V) = \int_V P(U|\mathcal{G}) \, dP,
        $$

        for all $V \in \mathcal{G}$. The conditional probability is unique up to almost sure equality.
        """
        from ..functions.random_vector import RandomVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.set import Set

        if not isinstance(given, SigmaAlgebra | Set | RandomVector):
            raise TypeError(
                "given must be a SigmaAlgebra, MeasurableSet, or RandomVector instance."
            )
        if not isinstance(event, Set):
            raise TypeError("event must be an MeasurableSet instance.")

        if isinstance(given, Set):
            sig_alg = SigmaAlgebra.from_set(given)
        elif isinstance(given, RandomVector):
            sig_alg = SigmaAlgebra.from_function(given)
        else:
            sig_alg = given

        return event.indicator.expectation(given=sig_alg, measure=self).with_name(
            f"P({event.name}|{sig_alg.name})"
        )

    def given(
        self,
        condition: SigmaAlgebra | Set | MeasurableVector,
        conditioning_suffix: str = "_g",
        name: Hashable | None = None,
    ) -> ParametrizedProbabilityMeasure:
        r"""Compute a conditional probability measure.

        Parameters
        ----------
        condition : SigmaAlgebra | MeasurableSet | MeasurableVector
            The given condition, which can be a sigma-algebra, an event, or a random vector.
        name : Hashable | None, default=None
            The name of the resulting parametrized probability measure.

        Raises
        ------
        TypeError
            If `condition` is not a `SigmaAlgebra`, `MeasurableSet`, or `MeasurableVector` instance.

        Returns
        -------
        cond_prob_measure : ParametrizedProbabilityMeasure
            A parametrized probability measure representing the conditional probability given `condition`.

        Examples
        --------
        Define a probability space, a random variable, and a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import ProbabilitySpace, RandomVariable, RandomVector
        >>> rng = np.random.default_rng(seed=42)
        >>> prob_space = ProbabilitySpace.from_rand(
        ...     domain_size=100,
        ...     num_atoms=42,
        ...     random_state=rng,
        ... )
        >>> X = RandomVariable.from_randint(
        ...     *prob_space,
        ...     diff_values=2,
        ...     random_state=rng,
        ... )
        >>> Y = RandomVector.from_randint(
        ...     *prob_space,
        ...     diff_values=3,
        ...     dim=2,
        ...     random_state=rng,
        ...     name="Y",
        ... )

        Extract the measure from the probability space. Verify that the conditional expectation of the random variable given the random vector may be computed as an integral of the variable against the conditional probability distribution.

        >>> P = prob_space.measure
        >>> X.expectation(Y)(Y == (0, 0)) == X.integrate(measure=P.given(Y)(Y_0=0, Y_1=0))
        True


        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space, let $U \in \mathcal{F}$ be an event, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional probability* of $U$ given $\mathcal{G}$ is a $\mathcal{G}$-measurable random variable, denoted $P(U|\mathcal{G})$, for which

        $$
        P(U \cap V) = \int_V P(U|\mathcal{G}) \, dP,
        $$

        for all $V \in \mathcal{G}$. The conditional probability is unique up to almost sure equality.

        """
        from .._utils.utils import to_df
        from ..functions.measurable_vector import MeasurableVector
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain
        from ..spaces.set import Set

        if not isinstance(condition, SigmaAlgebra | Set | MeasurableVector):
            raise TypeError(
                "'condition' must be a SigmaAlgebra, MeasurableSet, or MeasurableVector instance."
            )

        if isinstance(condition, Set):
            condition = SigmaAlgebra.from_set(condition)
        elif isinstance(condition, MeasurableVector):
            condition = SigmaAlgebra.from_function(condition)
        super = self.sig_alg

        if not condition <= super:
            raise ValueError(
                "The 'condition' sigma-algebra must be a sub-sigma-algebra of the probability measure's sigma-algebra."
            )

        super_data = to_df(super.data, "_super")
        sub_data = to_df(condition.data, "_sub")

        mapping = pd.concat([super_data, sub_data], axis=1).drop_duplicates(
            list(super_data.columns)
        )

        # TODO: check merge logic — possibly change to `on`?
        mapping = pd.merge(
            left=mapping,
            right=self.data,
            left_on=list(super_data.columns),
            right_index=True,
        )

        mapping["super_atom_probs"] = mapping.groupby(
            by=list(super_data.columns), sort=False
        )["probability"].transform(sum)
        mapping["sub_atom_probs"] = mapping.groupby(
            by=list(sub_data.columns), sort=False
        )["probability"].transform(sum)

        mapping["probability"] = mapping["super_atom_probs"] / mapping["sub_atom_probs"]

        mapping = mapping.drop_duplicates(list(super_data.columns))[
            list(super_data.columns) + list(sub_data.columns) + ["probability"]
        ].set_index(list(sub_data.columns) + list(super_data.columns))

        sub_variable_names = [
            (name + conditioning_suffix if name in super.variable_names else name)
            for name in condition.variable_names
        ]
        mapping.index.names = sub_variable_names + super.variable_names

        sub_atom_space_copy = condition.atom_space.copy()
        sub_atom_space_copy.variable_names = sub_variable_names
        domain = Domain.cartesian_product([sub_atom_space_copy, super.atom_space]).sort(
            ascending=True
        )

        mapping = (
            mapping.reindex(domain.data)
            .squeeze(axis=1)
            .fillna(0.0)
            .rename("probability")
        )

        non_null_sub_atom_IDs = mapping.groupby(sub_variable_names).apply(
            lambda grp: grp.sum() > 1e-8
        )
        mapping = mapping[non_null_sub_atom_IDs.reindex(mapping.index)]

        parameter_domain = Domain(
            non_null_sub_atom_IDs[non_null_sub_atom_IDs].index, name=condition.name
        )

        if name is None:
            if condition.name.startswith("sigma(") and condition.name.endswith(")"):
                name = f"{self.name}(?|{condition.name[6:-1]})"
            else:
                name = f"{self.name}(?|{condition.name})"

        return ParametrizedProbabilityMeasure.from_domains(
            measure_domain=super,
            parameter_domain=parameter_domain,
            mapping=mapping,
            name=name,
        )

    def derivative(
        self,
        base_measure: Measure | None = None,
        given: SigmaAlgebra | MeasurableVector | None = None,
        name: Hashable | None = None,
    ) -> RadonNikodym | ParametrizedMeasurableFunction:
        r"""Compute the Radon-Nikodym derivative with respect to a base measure, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        base_measure : Measure | None, default=None
            The base measure with respect to which the derivative is computed. If `None`, the counting measure is used.
        given : SigmaAlgebra | MeasurableVector | None, default=None
            The sigma-algebra or random vector on which to condition the derivative. If `None`, the unconditional probability mass function is computed.
        name : Hashable | None, default=None
            The name of the resulting measurable function representing the derivative. If `None`, a default name is generated.

        Returns
        -------
        derivative : RadonNikodym | ParametrizedMeasurableFunction
            Either an instance of `RadonNikodym` if the derivative is unconditional, or an instance of `ParametrizedMeasurableFunction`.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Measure,
        ...     Operators,
        ...     ProbabilitySpace,
        ...     RandomVariable,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)

        Define a probability space.

        >>> prob_space = ProbabilitySpace.from_rand(
        ...     domain_size=45,
        ...     domain_dim=1,
        ...     domain_variable_names=["omega"],
        ...     num_atoms=23,
        ...     sig_alg_dim=1,
        ...     sig_alg_variable_names=["x"],
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> Omega, F, P = prob_space

        Define a base measure for Radon-Nikodym derivatives.

        >>> mu = Measure.from_rand(
        ...     domain=F,
        ...     random_state=rng,
        ... )

        Extract a measurable set from the sigma-algebra and check that the Radon-Nikodym derivative has its defining property. (See the Notes section below.)

        >>> integrate = Operators.integrate
        >>> U = F.get_random_set(num_atoms=12, random_state=rng, name="U")
        >>> np.allclose(P(U), integrate(P.derivative(mu), U, measure=mu))
        True

        Define a sub-sigma-algebra for conditioning. Note that the atom identifiers use the variable name `u`.

        >>> G = SigmaAlgebra.from_rand(
        ...     super=F,
        ...     num_atoms=8,
        ...     dim=1,
        ...     variable_names=["u"],
        ...     name="G",
        ...     random_state=rng,
        ... )

        Check that the conditional Radon-Nikodym derivative has its defining property. (See the Notes section below.)

        >>> np.allclose(P.given(G)(U), integrate(P.derivative(mu, G), U, mu))
        True

        Define a random variable to demonstrate the connnection with expectations. (See the Notes section below.) Alias the expectation operator.

        >>> X = RandomVariable.from_randnorm(
        ...     *prob_space,
        ...     diff_values=5,
        ...     random_state=rng,
        ... )

        In SigAlg, the conditional Radon-Nikodym derivative `P.derivative(mu, G)` is an instance of `ParametrizedMeasurableFunction`, where the parameters are the atom identifiers of `G`. Conceptually, one should think of the conditional derivative as a *family* of derivatives, one for each atom identifier `u`.

        >>> P.derivative(mu, G)
        ParametrizedMeasurableFunction(parameters=(u), measurable_vars=(omega), domain=Omega, sig_alg=F, measure=None, name=dP_dmu)

        Multiplication of a `RandomVariable` instance against an instance of `ParametrizedMeasurableFunction` is defined in SigAlg as long as both are defined on the same sample space. The multiplication is the usual function multiplication, parameter by parameter, and yields an instance of `ParametrizedRandomVariable` since the `measure` attribute of the product is inherited from `X`.

        >>> X * P.derivative(mu, G)
        ParametrizedRandomVariable(parameters=(u), measurable_vars=(omega), domain=Omega, sig_alg=F, measure=P, name=(X*dP_dmu))

        Think of a `ParametrizedRandomVariable` as a family of random variables, one for each parameter. It is possible to pass such an instance into the `integrate` operator. The result is a `Function` whose values are the integrals of the random variables, parameter by parameter.

        >>> integrate(X * P.derivative(mu, G), measure=mu)
        Function(parameters=(u), domain=G, output_name=integral, name=int (X*dP_dmu) dmu)

        This is a function on the atom identifiers of the sigma-algebra `G`. It naturally defines a measurable function on the sample space by sending each sample point to the atom containing it, and then mapping the atom identifier according to the function. SigAlg has a method to produce measurable functions from functions defined on atom identifiers.

        >>> measurable = integrate(X * P.derivative(mu, G), measure=mu).to_measurable_function(sig_alg=G)
        >>> measurable
        MeasurableFunction(domain=Omega, sig_alg=G, name=int (X*dP_dmu) dmu)

        To recap: `measurable` represents a function on the sample space whose value on a sample point is the integral of `X` against the probability measure obtained from `P.derivative(mu, G)` evaluated at the atom identifier containing that sample point. If the reader now looks at the Notes section below, they'll see that `measure` is supposed to be the conditional expectation of `X` given `G` (at least up to a `P`-null set). We check this:

        >>> E = Operators.expectation
        >>> np.allclose(E(X, G), measurable)
        True

        Notes
        -----
        Let $P$ be a probability measure on a measurable space $(\Omega, \mathcal{F})$ that is absolutely continuous with respect to a second measure $\mu$. A *Radon-Nikodym derivative* of $P$ with respect to $\mu$ is an $\mathcal{F}$-measurable function $dP/d\mu$ for which

        $$
        P(U) = \int_\Omega \frac{dP}{d\mu} \, d\mu
        $$

        for all $U\in \mathcal{F}$.

        Now suppose $\mathcal{G}$ is a sub-$\sigma$-algebra of $\mathcal{F}$, let $U\in \mathcal{F}$, and let $P(U \mid \mathcal{G})$ be the conditional probability distribution (which is a $\mathcal{G}$-measurable function). Provided that $\Omega$ is "nice" (as it always is in SigAlg, because it is finite). then for each $\omega\in \Omega$ the function

        $$
        P({?}\mid \mathcal{G})(\omega):\mathcal{F} \to [0,1], \quad U \mapsto P(U \mid \mathcal{G})(\omega),
        $$

        is a probability measure on $\mathcal{F}$. A *conditional Radon-Nikodym derivative* of $P$ given $\mathcal{G}$ is an $\mathcal{F}$-measurable function for which

        $$
        P(U\mid\mathcal{G})(\omega) = \int_\Omega \frac{dP({?}\mid\mathcal{G})(\omega)}{d\mu} \, d\mu
        $$

        for all $U\in \mathcal{F}$.

        If $X$ is a random variable, it follows from standard properties of Radon-Nikodym derivatives that

        $$
        \int_\Omega X \, dP({?}\mid \mathcal{G})(\omega) = \int_\Omega X \frac{dP({?}\mid \mathcal{G})(\omega)}{d\mu} \, d\mu
        $$

        for every $\omega\in \Omega$. But the integral on the left is exactly the conditional expectation of $X$ given $\mathcal{G}$, and so we have

        $$
        E(X \mid \mathcal{G})(\omega) = \int_\Omega X \frac{dP({?}\mid \mathcal{G})(\omega)}{d\mu} \, d\mu
        $$

        as well.
        """
        from .._utils.utils import to_df
        from ..functions.measurable_vector import MeasurableVector
        from ..functions.parametrized_measurable_function import (
            ParametrizedMeasurableFunction,
        )
        from ..functions.radon_nikodym import RadonNikodym
        from ..spaces.domain import Domain

        if name is None:
            name = name = f"d{self.name}_d{base_measure.name}"

        if given is None:
            return RadonNikodym.from_measures(
                measure=self, base_measure=base_measure, name=name
            )

        else:
            if isinstance(given, MeasurableVector):
                given = given.generated_sig_alg
            super = self.sig_alg
            sub = given

            super_data = to_df(super.data, "_super", subscript_index_flag=True)
            sub_data = to_df(sub.data, "_sub", subscript_index_flag=True)
            domain_data = super.domain.data.to_frame().add_suffix("_d")

            # TODO: check merge logic — possibly change to `on`?
            prob_data = pd.merge(
                left=super_data,
                right=self.data.rename("super_atom_prob"),
                left_on=list(super_data.columns),
                right_index=True,
            )
            base_prob_data = pd.merge(
                left=super_data,
                right=base_measure.data.rename("super_atom_base_prob"),
                left_on=list(super_data.columns),
                right_index=True,
            )

            data = pd.concat(
                [sub_data, prob_data, base_prob_data["super_atom_base_prob"]],
                axis=1,
            ).drop_duplicates(list(super_data.columns))
            data["sub_atom_prob"] = data.groupby(list(sub_data.columns))[
                "super_atom_prob"
            ].transform(sum)

            null_rows = data[data["sub_atom_prob"] < 1e-8]
            null_sub_atom_ids = (
                null_rows[list(sub_data.columns)]
                .drop_duplicates()
                .set_index(list(sub_data.columns))
                .index
            )
            null_sub_atom_ids.names = sub.variable_names

            data["output"] = data["super_atom_prob"] / (
                data["super_atom_base_prob"] * data["sub_atom_prob"]
            )

            # TODO: check merge logic — possibly change to `on`?
            data = pd.merge(
                left=super_data.reset_index(),
                right=data,
                left_on=list(super_data.columns),
                right_on=list(super_data.columns),
            )

            parameter_idx = (
                sub.atom_space.data.difference(null_sub_atom_ids)
                .to_frame()
                .add_suffix("_sub")
            )

            cross = pd.merge(left=parameter_idx, right=domain_data, how="cross")

            # TODO: check merge logic — possibly change to `on`?
            mapping = pd.merge(
                left=cross,
                right=data.dropna(),
                left_on=list(sub_data.columns) + list(domain_data.columns),
                right_on=list(sub_data.columns) + list(domain_data.columns),
                how="outer",
            ).fillna(0)

            mapping = mapping.set_index(
                list(sub_data.columns) + list(domain_data.columns)
            )["output"].rename(name)
            mapping.index.names = sub.variable_names + sub.domain.variable_names

            domain = Domain(
                mapping.index, name=f"{sub.name} x {self.sig_alg.domain.name}"
            )

            return ParametrizedMeasurableFunction.from_domains(
                measurable_domain=self.sig_alg.domain,
                sig_alg=self.sig_alg,
                measure=None,
                complete_domain=domain,
                mapping=mapping,
                name=name,
                parameter_domain_name=given.name,
            )

    def surprisal(
        self,
        base_measure: Measure,
        given: SigmaAlgebra | RandomVector | None = None,
        tol: float = 1e-8,
        name: Hashable | None = None,
    ) -> MeasurableFunction:
        """Compute the surprisal of the probability measure with respect to a base measure, optionally conditioned on a sigma-algebra or random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        base_measure : Measure
            The base measure with respect to which the surprisal is computed.
        given : SigmaAlgebra | RandomVector | None, default=None
            The sigma-algebra or random vector on which to condition the surprisal. If `None`, the surprisal is unconditional.
        tol : float, default=1e-8
            A tolerance level for checking absolute continuity.
        name : Hashable | None, default=None
            The name of the resulting measurable function representing the surprisal. If `None`, a default name is generated.

        Returns
        -------
        surprisal : MeasurableFunction
            A measurable function representing the surprisal of the probability measure with respect to the base measure, optionally conditioned on the given sigma-algebra or random vector.
        """
        from .._utils.utils import to_df
        from ..functions.measurable_function import MeasurableFunction
        from ..measures.measure import Measure

        if not isinstance(base_measure, Measure):
            raise TypeError("The base measure must be an instance of Measure.")
        if self.data is None or base_measure.data is None:
            raise ValueError(
                "The probability measure and the base measure must have their 'data' attributes set."
            )
        if self.sig_alg != base_measure.sig_alg:
            raise ValueError(
                "The probability measure and the base measure must be defined on the same sigma-algebra."
            )
        if not isinstance(tol, float):
            raise TypeError("'tol' must be a float.")
        if tol <= 0:
            raise ValueError("'tol' must be positive.")

        if not ((base_measure.data >= tol) | (self.data < tol)).all():
            raise ValueError(
                "The probability measure is not absolutely continuous with respect to the base measure."
            )

        rn_der = (self.data / base_measure.data).fillna(0.0).rename("derivative")

        with np.errstate(divide="ignore"):
            s = -np.log(rn_der)
        s = s.mask(np.isinf(s), 0).rename("surprisal")

        sig_alg_data = to_df(base_measure.sig_alg.data)

        # TODO: check merge logic — possibly change to `on`?
        mapping = pd.merge(
            left=base_measure.sig_alg.data,
            right=s,
            left_on=list(sig_alg_data.columns),
            right_index=True,
        )["surprisal"]

        if name is None:
            name = f"s({self.name}, {base_measure.name})"

        return MeasurableFunction(
            measure=self,
            mapping=mapping.rename(name),
            name=name,
        )

    @staticmethod
    def _to_tuple(x):
        if isinstance(x, tuple):
            return x
        else:
            return (x,)

    def are_independent(
        self,
        given1: Set | MeasurableVector | SigmaAlgebra,
        given2: Set | MeasurableVector | SigmaAlgebra,
        tol: Real = 1e-8,
    ) -> bool:
        r"""Check if two events, two random vectors, or two sigma-algebras are independent.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        given1 : MeasurableSet | MeasurableVector | SigmaAlgebra
            The first given to test for independence.
        given2 : MeasurableSet | MeasurableVector | SigmaAlgebra
            The second given to test for independence.
        tol : Real, default=1e-10
            The numerical tolerance for checking independence.

        Raises
        ------
        ValueError
            If one of the two givens is not measurable with respect to the probability measure's sigma-algebra.
        TypeError
            If the provided objects are not of the correct type.

        Returns
        -------
        is_independent : bool
            `True` if the events, random vectors, or sigma-algebras are independent, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> Omega = SampleSpace.cartesian_power([0, 1], n=2, name="Omega")
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping=lambda *, s_0, s_1: 0.75 ** (s_0 + s_1) * 0.25 ** (2 - s_0 - s_1),
        ... )
        >>> prob_space = ProbabilitySpace(domain=Omega, measure=P)
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, R, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         s_0  s_1
           0    0
           0    1
           1    0
           1    1
        <BLANKLINE>
        * Sigma algebra 'R':
        i        s_0  s_1
        s_0 s_1
        0   0      0    0
            1      0    1
        1   0      1    0
            1      1    1
        <BLANKLINE>
        * Probability measure 'P':
                      P
        s_0 s_1
        0   0    0.0625
            1    0.1875
        1   0    0.1875
            1    0.5625
        >>> A = prob_space.get_set(
        ...     [(0, 0), (0, 1)],
        ...     name="A",
        ... )
        >>> B = prob_space.get_set(
        ...     [(0, 1), (1, 1)],
        ...     name="B",
        ... )
        >>> P.are_independent(A, B)
        True
        >>> X = RandomVector.from_identity(domain=Omega, measure=P, index=[1, 2])
        >>> X_1, X_2 = X
        >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_1':
                    X_1
        s_0 s_1
        0   0         0
            1         0
        1   0         1
            1         1
        >>> print(X_2)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_2':
                    X_2
        s_0 s_1
        0   0         0
            1         1
        1   0         0
            1         1
        >>> P.are_independent(X_1, X_2)
        True
        >>> Y = (X_1 + X_2).with_name("Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                      Y
        s_0 s_1
        0   0         0
            1         1
        1   0         1
            1         2
        >>> P.are_independent(X_1, Y)
        False

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space, and let $\mathcal{G}$ and $\mathcal{H}$ be two sub-$\sigma$-algebras of $\mathcal{F}$. We say that $\mathcal{G}$ and $\mathcal{H}$ are *independent* if for every $G \in \mathcal{G}$ and $H \in \mathcal{H}$, we have

        $$
        P(G \cap H) = P(G) P(H).
        $$

        In the special case where $\mathcal{G} = \sigma(A)$ and $\mathcal{H} = \sigma(B)$ are the $\sigma$-algebras generated by two events $A$ and $B$ in $\mathcal{F}$, this reduces to the condition that

        $$
        P(A \cap B) = P(A) P(B),
        $$

        and we say that the events $A$ and $B$ are *independent*.
        """
        from ..functions.measurable_vector import MeasurableVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.set import Set

        if not isinstance(
            given1, Set | MeasurableVector | SigmaAlgebra
        ) or not isinstance(given2, Set | MeasurableVector | SigmaAlgebra):
            raise TypeError(
                "The givens must be instances of MeasurableSet, RandomVector, or SigmaAlgebra."
            )

        givens = []
        for given in [given1, given2]:
            if isinstance(given, Set | MeasurableVector):
                givens.append(given.generated_sig_alg)
            else:
                givens.append(given)
        given1, given2 = givens

        if not (given1 <= self.sig_alg and given2 <= self.sig_alg):
            raise ValueError(
                "One of the two givens is not measurable with respect to the probability measure's sigma-algebra."
            )

        for atom1 in given1:
            for atom2 in given2:
                event1 = self.sig_alg.get_set(list(atom1), name=atom1.name)
                event2 = self.sig_alg.get_set(list(atom2), name=atom2.name)
                if abs(self(event1 & event2) - self(event1) * self(event2)) >= tol:
                    return False

        return True

    def equal_almost_surely(
        self,
        first: MeasurableVector,
        second: MeasurableVector,
        tol: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two random vectors are equal almost surely.

        See the Notes section below for the mathematical details.

        This method calls the `Measure.equal_almost_everywhere` method.

        Parameters
        ----------
        first : MeasurableVector
            The first random vector.
        second : MeasurableVector
            The second random vector.
        tol : float, default=1e-8
            The tolerance below which a probability is considered to be zero for the purposes of this comparison.
        rtol : float, default=1e-5
            The relative tolerance for `np.isclose` when comparing the random vectors.
        atol : float, default=1e-8
            The absolute tolerance for `np.isclose` when comparing the random vectors.

        Returns
        -------
        equal_as : bool
            True if the random vectors are equal almost surely; False otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.4,
        ...         1: 0.6,
        ...         2: 0.0,
        ...     },
        ... )
        >>> X = RandomVariable.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 3.0,
        ...     },
        ... )
        >>> Y = RandomVariable.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 4.0,
        ...     },
        ...     name="Y",
        ... )
        >>> Z = RandomVariable.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 3.0,
        ...         2: 3.0,
        ...     },
        ...     name="Z",
        ... )
        >>> print(P.equal_almost_surely(X, Y))
        True
        >>> print(P.equal_almost_surely(X, Z))
        False
        >>> U = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 2),
        ...     },
        ...     name="U",
        ... )
        >>> V = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (-1, 4),
        ...     },
        ...     name="V",
        ... )
        >>> W = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (-1, 1),
        ...         2: (3, 2),
        ...     },
        ...     name="W",
        ... )
        >>> print(P.equal_almost_surely(U, V))
        True
        >>> print(P.equal_almost_surely(U, W))
        False

        Notes
        -----
        Two random vectors $X,Y:\Omega \to \mathbb{R}^d$ defined on a probability space $(\Omega, \mathcal{F}, P)$ are *equal almost surely* if

        $$
        P \left( \{\omega \in \Omega : X(\omega) \neq Y(\omega)\} \right) = 0.
        $$
        """
        return self.equal_almost_everywhere(
            first=first, second=second, tol=tol, rtol=rtol, atol=atol
        )
