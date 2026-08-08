"""A class representing a probability measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .measure import Measure

if TYPE_CHECKING:
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
    from ..spaces.measurable_set import MeasurableSet


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
             probability
    atom_ID
    0                0.2
    1                0.8

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
            probability
    sample
    0               0.1
    1               0.3
    2               0.6
    >>> print(Q.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'power_set':
            sample
    sample
    0            0
    1            1
    2            2

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
            probability
    point
    0               0.1
    1               0.3
    2               0.6
    >>> print(Q.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'power_set':
            point
    point
    0           0
    1           1
    2           2

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
        output_name: str = "probability",
        name: Hashable = "P",
        **kwargs,
    ) -> None:

        super().__init__(
            domain=domain,
            mapping=mapping,
            output_name=output_name,
            name=name,
            kind="probability",
        )

    @classmethod
    def uniform(
        cls,
        domain: MeasureDomain,
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
        >>> Omega = SampleSpace().from_sequence(size=4)
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
                 probability
        atom_ID
        0           0.333333
        1           0.333333
        2           0.333333
        >>> V = ProbabilityMeasure.uniform(domain=Omega, name="V")
        >>> print(V)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'V':
              probability
        sample
        0            0.25
        1            0.25
        2            0.25
        3            0.25

        Notes
        -----
        Let $(\Omega,\mathcal{F})$ be a measurable space where $\Omega$ is finite, and suppose that $\mathcal{F}$ has $n$ atoms. The *uniform probability measure* on $\mathcal{F}$ is the unique probability measure $P$ such that

        $$
        P(A) = \frac{1}{n}
        $$

        for all atoms $A\in \mathcal{F}$.
        """
        from ...validation.measure_domain_validator import MeasureDomainValidator

        v = MeasureDomainValidator(measure_domain=domain, kind="probability")

        n = len(v.domain)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on sigma-algebra with no atoms."
            )
        probs = dict.fromkeys(v.domain, 1.0 / n)

        return cls(domain=v.sig_alg, mapping=probs, name=name)

    # --------------------- probability methods --------------------- #

    def sample(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> pd.Series | pd.DataFrame:
        """Generate random samples from this probability measure.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate. Must be positive.
        random_state : int | np.random.Generator | None, default=None
            Random seed or generator for reproducibility.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `random_state` is not an integer, `np.random.Generator`, or `None`.

        Returns
        -------
        sample : pd.Series | pd.DataFrame
            If the domain of the probability measure is 1-dimensional, then a `pd.Series` is returned containing the random sample. Otherwise, if the domain is multi-dimensional, a `pd.DataFrame` is returned whose rows contain the random sample and has columns indexed by the variable names of the domain.

        Examples
        --------
        Define a sigma-algebra with 1-dimensional atom IDs with variable name `x`.

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
        ...     variable_names=["x"],
        ... )

        Define a probability measure on the sigma-algebra and sample from it. Notice the output is a `pd.Series`.

        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.45,
        ...         2: 0.3,
        ...     },
        ... )
        >>> P_sample = P.sample(size=5, random_state=42)
        >>> print(P_sample)  # doctest: +NORMALIZE_WHITESPACE
        0    2
        1    1
        2    2
        3    1
        4    0
        Name: x, dtype: int64

        Define a sigma-algebra with 2-dimensional atom IDs with variable names `x` and `y`.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (0, 1),
        ...         2: (2, 3),
        ...         3: (3, 4),
        ...     },
        ...     name="G",
        ...     variable_names=["x", "y"],
        ... )

        Define a probability measure on the new sigma-algebra and sample from it. Notice the output is a `pd.DataFrame`.

        >>> Q = ProbabilityMeasure(
        ...     domain=G,
        ...     mapping={
        ...         (0, 1): 0.25,
        ...         (2, 3): 0.45,
        ...         (3, 4): 0.3,
        ...     },
        ...     name="Q",
        ... )
        >>> Q_sample = Q.sample(size=5, random_state=42)
        >>> print(Q_sample)  # doctest: +NORMALIZE_WHITESPACE
        x  y
        0  3  4
        1  2  3
        2  3  4
        3  2  3
        4  0  1
        """
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

        return pd.DataFrame(samples, columns=self.domain.variable_names).squeeze(axis=1)

    def conditional(
        self,
        event: MeasurableSet,
        given: SigmaAlgebra | MeasurableSet | RandomVector,
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
        from ..spaces.measurable_set import MeasurableSet

        if not isinstance(given, SigmaAlgebra | MeasurableSet | RandomVector):
            raise TypeError(
                "given must be a SigmaAlgebra, MeasurableSet, or RandomVector instance."
            )
        if not isinstance(event, MeasurableSet):
            raise TypeError("event must be an MeasurableSet instance.")

        if isinstance(given, MeasurableSet):
            sig_alg = SigmaAlgebra.from_set(given)
        elif isinstance(given, RandomVector):
            sig_alg = SigmaAlgebra.from_measurable_vector(given)
        else:
            sig_alg = given

        return event.indicator.expectation(given=sig_alg, measure=self).with_name(
            f"P({event.name}|{sig_alg.name})"
        )

    def given(
        self,
        condition: SigmaAlgebra | MeasurableSet | MeasurableVector,
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
        from ..functions.measurable_vector import MeasurableVector
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain
        from ..spaces.measurable_set import MeasurableSet

        if not isinstance(condition, SigmaAlgebra | MeasurableSet | MeasurableVector):
            raise TypeError(
                "'condition' must be a SigmaAlgebra, MeasurableSet, or MeasurableVector instance."
            )

        if isinstance(condition, MeasurableSet):
            condition = SigmaAlgebra.from_set(condition)
        elif isinstance(condition, MeasurableVector):
            condition = SigmaAlgebra.from_measurable_vector(condition)
        super = self.sig_alg

        if not condition <= super:
            raise ValueError(
                "The 'condition' sigma-algebra must be a sub-sigma-algebra of the probability measure's sigma-algebra."
            )

        super_data = self._to_df(super.data, "_super")
        sub_data = self._to_df(condition.data, "_sub")

        mapping = pd.concat([super_data, sub_data], axis=1).drop_duplicates(
            list(super_data.columns)
        )

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
        >>> from sigalg.core import Measure, ProbabilitySpace, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Define a random probability space, a sub-sigma-algebra, and a base measure.

        >>> prob_space = ProbabilitySpace.from_rand(
        ...     domain_size=45,
        ...     domain_dim=1,
        ...     domain_variable_names=["omega"],
        ...     num_atoms=23,
        ...     sig_alg_dim=1,
        ...     sig_alg_variable_names=["u"],
        ...     num_null_atoms=5,
        ...     random_state=rng,
        ... )
        >>> Omega, F, P = prob_space
        >>> G = SigmaAlgebra.from_rand(
        ...     super=F,
        ...     num_atoms=8,
        ...     dim=1,
        ...     variable_names=["x"],
        ...     name="G",
        ...     random_state=rng,
        ... )
        >>> mu = Measure.from_rand(
        ...     domain=F,
        ...     random_state=rng,
        ... )

        Get a random set from the sigma-algebra.

        >>> U = F.get_random_set(num_atoms=12, random_state=rng, name="U")

        Check that the conditional Radon-Nikodym derivative as its defining property. (See the Notes section below.)

        >>> P.given(G)(U) == P.derivative(mu, G).integrate(U, mu)
        True

        Notes
        -----
        Let $P$ be a probability measure on a measurable space $(\Omega, \mathcal{F})$ that is absolutely continuous with respect to a second measure $\mu$. A *Radon-Nikodym derivative* of $P$ with respect to $\mu$ is an $\mathcal{F}$-measurable function $dP/d\mu$ for which

        $$
        P(U) = \int_\Omega \frac{dP}{d\mu} \, d\mu
        $$

        for all $U\in \mathcal{F}$.

        Now suppose $\mathcal{G}$ is a sub-$\sigma$-algebra of $\mathcal{F}$, let $U\in \mathcal{F}$, and let $P(U \mid \mathcal{G})$ be the conditional probability distribution, which is a $\mathcal{G}$-measurable function. Provided that $\Omega$ is "nice" (as it always is in SigAlg, because it is finite). then for each $\omega\in \Omega$ the function

        $$
        P({?}\mid \mathcal{G})(\omega):\mathcal{F} \to [0,1], \quad U \mapsto P(U \mid \mathcal{G})(\omega),
        $$

        is a probability measure on $\mathcal{F}$. A *conditional Radon-Nikodym derivative* of $P$ given $\mathcal{G}$ is an $\mathcal{F}$-measurable function for which

        $$
        P(U\mid\mathcal{G})(\omega) = \int_\Omega \frac{dP({?}\mid\mathcal{G})(\omega)}{d\mu} \, d\mu
        $$

        for all $U\in \mathcal{F}$.
        """
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

            super_data = self._to_df(super.data, "_super", subscript_index_flag=True)
            sub_data = self._to_df(sub.data, "_sub", subscript_index_flag=True)
            domain_data = super.domain.data.to_frame().add_suffix("_d")

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
            parameter_domain_data = (
                mapping.index.to_frame()[sub.variable_names]
                .drop_duplicates()
                .set_index(sub.variable_names)
                .index
            )
            parameter_domain = Domain(parameter_domain_data, name=sub.name)

            derivative = ParametrizedMeasurableFunction(
                domain=domain,
                mapping=mapping,
                output_name=name,
                name=name,
            )
            derivative._init_measurable_attrs(
                measurable_domain=self.sig_alg.domain,
                parameter_domain=parameter_domain,
                sig_alg=self.sig_alg,
                measure=None,
            )

            return derivative

    @staticmethod
    def _to_df(
        data: pd.Series | pd.DataFrame,
        suffix: str | None = None,
        subscript_index_flag: bool = False,
    ) -> pd.DataFrame:
        if suffix is None:
            suffix = ""
        if isinstance(data, pd.DataFrame):
            result = data.add_suffix(suffix)
        else:
            result = data.to_frame().add_suffix(suffix)

        if subscript_index_flag:
            result.index.names = [f"{name}_d" for name in result.index.names]
        return result

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
        from ..functions.measurable_function import MeasurableFunction
        from ..measures.measure import Measure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

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

        sig_alg_data = SigmaAlgebra._to_df(base_measure.sig_alg.data)

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
        given1: MeasurableSet | MeasurableVector | SigmaAlgebra,
        given2: MeasurableSet | MeasurableVector | SigmaAlgebra,
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
        >>> Omega = SampleSpace.cartesian_power(
        ...     [0, 1], n=2, variable_names=["flip_1", "flip_2"], name="Omega"
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=Omega,
        ...     mapping=lambda *, flip_1, flip_2: (
        ...         0.75 ** (flip_1 + flip_2) * 0.25 ** (2 - flip_1 - flip_2)
        ...     ),
        ... )
        >>> prob_space = ProbabilitySpace(domain=Omega, measure=P)
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, power_set, P)
        =======================================
        <BLANKLINE>
        * Sample space 'Omega':
         flip_1  flip_2
             0       0
             0       1
             1       0
             1       1
        <BLANKLINE>
        * Sigma algebra 'power_set':
                       flip_1  flip_2
        flip_1 flip_2
        0      0            0       0
               1            0       1
        1      0            1       0
               1            1       1
        <BLANKLINE>
        * Probability measure 'P':
                       probability
        flip_1 flip_2
        0      0            0.0625
               1            0.1875
        1      0            0.1875
               1            0.5625
        >>> A = prob_space.get_set(
        ...     [(0, 0), (0, 1)],
        ...     name="A",
        ... )
        >>> B = prob_space.get_set(
        ...     [(0, 1), (1, 1)],
        ...     name="B",
        ... )
        >>> print(P.are_independent(A, B))
        True
        >>> X = RandomVector.from_identity(*prob_space, index=[1, 2])
        >>> X_1, X_2 = X
        >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_1':
                       X_1
        flip_1 flip_2
        0      0         0
               1         0
        1      0         1
               1         1
        >>> print(X_2)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_2':
                       X_2
        flip_1 flip_2
        0      0         0
               1         1
        1      0         0
               1         1
        >>> print(P.are_independent(X_1, X_2))
        True
        >>> Y = (X_1 + X_2).with_name("Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                       Y
        flip_1 flip_2
        0      0       0
               1       1
        1      0       1
               1       2
        >>> print(P.are_independent(X_1, Y))
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
        from ..spaces.measurable_set import MeasurableSet

        if not isinstance(
            given1, MeasurableSet | MeasurableVector | SigmaAlgebra
        ) or not isinstance(given2, MeasurableSet | MeasurableVector | SigmaAlgebra):
            raise TypeError(
                "The givens must be instances of MeasurableSet, RandomVector, or SigmaAlgebra."
            )

        givens = []
        for given in [given1, given2]:
            if isinstance(given, MeasurableSet):
                givens.append(SigmaAlgebra.from_set(given))
            elif isinstance(given, MeasurableVector):
                givens.append(SigmaAlgebra.from_measurable_vector(given))
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
