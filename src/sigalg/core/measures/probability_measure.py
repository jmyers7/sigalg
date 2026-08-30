"""A class representing a probability measure on a sigma-algebra."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .measure import Measure

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...typing.mapping_like import MappingLike
    from ...typing.measure_domain import MeasureDomain
    from ..functions.measurable_function import MeasurableFunction
    from ..functions.measurable_vector import MeasurableVector
    from ..functions.parametrized_measurable_function import (
        ParametrizedMeasurableFunction,
    )
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
        The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`. In the latter case, the domain will be set to the power set of the domain.
    mapping : MappingLike | None, default=None
        A mapping from the domain to the probability values.
    output_name : Hashable | None, default=None
        The name of the output variable of the measure. If `None`, a default will be generated.
    name : Hashable | None, default=None
        The name of the measure. If `None`, a default will be generated.

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra

    Define a probability measure on a sigma-algebra with two atoms.

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
    omega
    0     0.1
    1     0.3
    2     0.6
    >>> print(Q.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
            R
    omega
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
    Let $(\Omega, \mathcal{F})$ be a measurable space. A *probability measure* $P$ is a countably additive function $P: \mathcal{F} \to [0,\infty)$ such that $P(\Omega) = 1$. Here, *countable additivity* means that

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
        name: Hashable | None = None,
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
            The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`; in the latter case, the domain will be set to the power set of the `Domain` instance.
        output_name : Hashable | None, default=None
            The name of the output variable of the measure. If `None`, a default will be generated.
        name : Hashable, default="U"
            A name for the measure.

        Returns
        -------
        prob_measure: ProbabilityMeasure
            A uniform ProbabilityMeasure instance on the provided sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra

        Define a measurable space with a sigma-algebra having three atoms.

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

        Define a uniform probability measure on the sigma-algebra. Each atom is given equal probability `1/3`.

        >>> U = ProbabilityMeasure.uniform(domain=F)
        >>> print(U)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'U':
                  U
        F
        0   0.333333
        1   0.333333
        2   0.333333

        Define a uniform probability measure directly on the sample space. This will create the power-set sigma-algebra by default, and each outcome is given equal probability `1/4`.

        >>> V = ProbabilityMeasure.uniform(domain=Omega, name="V")
        >>> print(V)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'V':
                  V
        omega
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

    # --------------------- dunder operators --------------------- #

    def __matmul__(self, other):
        """Return the tensor product of this probability measure with another.

        Internally calls the `ProbabilityMeasure.tensor_product` method. See the docstring there for more details.
        """
        return ProbabilityMeasure.tensor_product([self, other])

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
        omega
        1     74
        2     39
        0     37

        Divide the counting measure by the sample size to obtain a probability measure. Note that it closely matches the original probability measure.

        >>> C = P_sample.measure
        >>> print(C / 150)  # doctest: +NORMALIZE_WHITESPACE
        Function '(C / 150)':
                (C / 150)
        omega
        1        0.493333
        2        0.260000
        0        0.246667
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
        given: SigmaAlgebra | Set | MeasurableVector,
        subset: Set | None = None,
        base_measure: Measure | None = None,
        name: Hashable | None = None,
    ) -> RandomVariable | ParametrizedProbabilityMeasure:
        r"""Compute a conditional probability.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        given : SigmaAlgebra | Set | MeasurableVector
            The given condition, which can be a sigma-algebra, an event, or a random vector.
        subset : Set | None, default=None
            This method will either return a `RandomVariable` or a `ParametrizedProbabilityMeasure` depending on whether this parameter is `None` or not. See the Notes and Examples section below.
        base_measure : Measure | None, default=None
            If the given sigma-algebra contains null atoms, a default probability measure will be generated for those atoms. This parameter is an optional measure with respect to which these default distributions will be absolutely continuous. See the Notes and Examples section below.
        name : Hashable | None, default=None
            The name of the resulting parametrized probability measure. If `None`, a default will be generated.

        Returns
        -------
        cond_prob_measure : RandomVariable | ParametrizedProbabilityMeasure
            A random variable or a parametrized probability measure representing the conditional probability.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Measure,
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     Set,
        ...     SigmaAlgebra,
        ... )

        Define a probability space. Notice that the sigma-algebra has two null atoms with identifiers `3` and `4`.

        >>> Omega = SampleSpace.from_sequence(size=7)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...         4: 3,  # null atom
        ...         5: 4,  # null atom
        ...         6: 4,  # null atom
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.6,
        ...         2: 0.2,
        ...         3: 0.0,
        ...         4: 0.0,
        ...     },
        ... )

        Define a sub-sigma-algebra of `F`. Notice that this sigma-algebra also has a null atom, with identifier `2`.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,  # null atom
        ...         5: 2,  # null atom
        ...         6: 2,  # null atom
        ...     },
        ...     name="G",
        ... )

        We first compute the conditional probability with an explit `subset` parameter, returning an instance of `RandomVariable`.

        >>> U = Set([1, 2, 3], domain=Omega, name="U")
        >>> P.conditional(G, subset=U)
        RandomVariable(parameters=(omega), domain=Omega, sig_alg=G, measure=P|G, name=P(U|G))
        >>> print(P.conditional(G, subset=U))  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'P(U|G)':
               P(U|G)
        omega
        0        0.75
        1        0.75
        2        1.00
        3        1.00
        4        0.00
        5        0.00
        6        0.00

        Notice the random variable takes the constant value `0.00` on the null atom of `G`. (This is the default SigAlg behavior.)

        If the `subset` parameter of the `conditional` method is left as its default value `None`, then the method will return an instance of `ParametrizedProbabilityMeasure`.

        >>> P.conditional(G)
        ParametrizedProbabilityMeasure(parameters=(G), domain_vars=(F), sig_alg=F, name=P(-|G))
        >>> print(P.conditional(G))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P(-|G)':
        G     0    1    2
        F
        0  0.25  0.0  0.2
        1  0.75  0.0  0.2
        2  0.00  1.0  0.2
        3  0.00  0.0  0.2
        4  0.00  0.0  0.2

        Notice that the parameter space for the measure is the set of atom identifiers of the sub-sigma-algebra `G`. The null atom of `G` has identifier `2` (the third column). The mathematical theory does not place any restrictions on the probability distribution given a null atom, just as long as it is *some* valid distribution. The default in SigAlg is to create the uniform one.

        However, sometimes it is useful for this distribution to be chosen absolutely continuous with respect to a base measure. We create a base measure.

        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 3,
        ...         4: 4,
        ...     }
        ... )

        Notice that `mu` vanishes on the atom with identifier `0`. The default uniform measure `P.conditional(G)` when `G=2` is not absolutely continuous with respect to `mu`, but we can generate one that *is* by passing `mu` as the `base_measure` parameter into the `conditional` method.

        >>> print(P.conditional(G, base_measure=mu))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P(-|G)':
        G     0    1     2
        F
        0  0.25  0.0  0.00
        1  0.75  0.0  0.25
        2  0.00  1.0  0.25
        3  0.00  0.0  0.25
        4  0.00  0.0  0.25

        In any case, we may call either of these parametrized measures on the subset `U` from above.

        >>> print(P.conditional(G)(U))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(-|G)(U)':
           P(-|G)(U)
        G
        0       0.75
        1       1.00
        2       0.40

        Notice the difference between `P.conditional(G, subset=U)` from above and this last printout of `P.conditional(G)(U)`. The former is a random variable defined on the sample space `Omega`, while the latter is a function defined on the set of atom identifiers of `G`. We may "ascend" the latter to a function on the sample space by calling the `ascend` method. (Essentially, we are "broadcasting" the values of the function from the atom identifiers to the entire sample space).

        >>> print(P.conditional(G)(U).ascend(G))  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'P(-|G)(U)':
               P(-|G)(U)
        omega
        0           0.75
        1           0.75
        2           1.00
        3           1.00
        4           0.40
        5           0.40
        6           0.40

        We check that these two ways of creating a random variable are equal `P`-a.s.

        >>> P.equal_almost_surely(P.conditional(G)(U).ascend(G), P.conditional(G, subset=U))
        True

        Finally, we check the law of total probability. (See the Notes section below.)

        >>> integrate = Operators.integrate
        >>> all(P(U & B) == integrate(P.conditional(G, U), B, measure=P) for B in G)
        True

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space, and let $\mathcal{G}$ be a sub-$\sigma$-algebra of $\mathcal{F}$. The *conditional probability* of an event $U\in \mathcal{G}$, given $\mathcal{G}$, is any $\mathcal{G}$-measurable random variable $P(U\mid \mathcal{G})$ such that

        $$
        P(U \cap V) = \int_V P(U \mid \mathcal{G}) \, dP,
        $$

        for all $V\in \mathcal{G}$. This equation is called the *law of total probability*. In particular, we may take

        $$
        P(U\mid \mathcal{G}) = E(I_U \mid \mathcal{G}),
        $$

        where $I_U$ is the indicator function of $U$.

        Suppose now that $\Omega$ is finite. From the Fourier expansion of the conditional expectation, we see that

        $$
        P(U \mid \mathcal{G}) = \sum_B \frac{P(U\cap B)}{P(B)} I_B,
        $$

        $P$-almost surely, where the sum extends over all atoms $B$ of $\mathcal{G}$ with nonzero probability. Using this formula, one may show that for each $\omega\in \Omega$ not contained in a $P$-null atom of $\mathcal{G}$, the function

        $$
        P(- \mid \mathcal{G})(\omega): \mathcal{F} \to [0,1], \quad U \mapsto P(U\mid \mathcal{G})(\omega),
        $$

        is a probability measure on $\mathcal{F}$. If $\omega$ is in a $P$-null atom, then we suppose that some fixed probability measure is assigned to $P(- \mid \mathcal{G})(\omega)$. It may be arbitrarily assigned — it just needs to be a valid measure. In applications, it is sometimes convenient to select these measures to be absolutely continuous with respect to a given base measure.
        """
        from .._utils.function_helpers import compute_expectation
        from .._utils.measure_helpers import compute_conditional_prob_measure
        from ..functions.random_variable import RandomVariable
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ..spaces.set import Set

        given_name = given.name
        given = self._normalize_given_and_base_measure(
            given=given, base_measure=base_measure
        )

        if subset is not None:
            if not isinstance(subset, Set):
                raise TypeError("If given, subset must be an Set instance.")
            if subset.domain != self.sig_alg.domain:
                raise ValueError(
                    "If given, the domain of the subset must match the domain of the sigma-algebra of the probability meaasure."
                )

        if subset is not None:
            restricted_measure = self | given

            data = compute_expectation(
                rv_atom_data=subset.indicator_atom_data(self.sig_alg),
                given_data=given.data,
                given_variable_names=given.variable_names,
                atom_data=given.up_lattice.get_atom_data(self.sig_alg),
                measure_data=self.data,
                measure_data_on_given=restricted_measure.data,
            )

            if name is None:
                name = f"P({subset.name}|{given_name})"

            return RandomVariable._from_validated(
                data=data.rename(name),
                sig_alg=given,
                measure=restricted_measure,
                index_kind="Index",
                index_name=None,
                name=name,
            )

        else:
            if name is None:
                name = f"{self.name}(-|{given_name})"

            data = compute_conditional_prob_measure(
                self_data=self.data,
                restricted_self_data=(self | given).data,
                atom_data=self.sig_alg.down_lattice.get_atom_data(given),
                given_variable_names=given.variable_names,
                base_measure_data=getattr(base_measure, "data", None),
                return_raw_data=False,
            )

            return ParametrizedProbabilityMeasure._from_validated(
                data=data.rename(name),
                sig_alg=self.sig_alg,
                kind="param_probability",
                complete_domain_name=f"{given_name} x {self.sig_alg.name}",
                parameter_domain_name=given_name,
                parameter_names=given.variable_names,
                name=name,
            )

    def derivative(
        self,
        base_measure: Measure | None = None,
        given: SigmaAlgebra | MeasurableVector | None = None,
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> MeasurableFunction | ParametrizedMeasurableFunction:
        r"""Compute the Radon-Nikodym derivative with respect to a base measure, optionally conditioned on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        base_measure : Measure | None, default=None
            The base measure with respect to which the derivative is computed. If `None`, the counting measure is used.
        given : SigmaAlgebra | MeasurableVector | None, default=None
            The optional sigma-algebra or random vector on which to condition the derivative.
        name : Hashable | None, default=None
            The name of the derivative. If `None`, a default name is generated.

        Returns
        -------
        derivative : MeasurableFunction | ParametrizedMeasurableFunction
            Either an instance of `MeasurableFunction` if the derivative is unconditional, or an instance of `ParametrizedMeasurableFunction`.

        Examples
        --------
        >>> from itertools import product
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Measure,
        ...     Operators,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     SampleSpace,
        ...     Set,
        ...     SigmaAlgebra,
        ... )

        Define a probability space and a base measure for Radon-Nikodym derivatives. Notice that the sigma-algebra has null atoms.

        >>> Omega = SampleSpace.from_sequence(size=7)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...         4: 3,  # P-null atom
        ...         5: 4,  # P- and mu-null atom
        ...         6: 4,  # P- and mu-null atom
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.6,
        ...         2: 0.2,
        ...         3: 0.0,
        ...         4: 0.0,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...         3: 4,
        ...         4: 0,
        ...     },
        ... )

        Compute the Radon-Nikodym derivative of `P` with respect to `mu`.

        >>> P.derivative(mu)
        MeasurableFunction(parameters=(omega), domain=Omega, sig_alg=F, measure=mu, name=dP_dmu)
        >>> print(P.derivative(mu))  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'dP_dmu':
               dP_dmu
        omega
        0    0.200000
        1    0.300000
        2    0.066667
        3    0.066667
        4    0.000000
        5    0.000000
        6    0.000000

        Notice that the derivative is `0.0` on the `mu`-null atom. This is the default SigAlg behavior.

        Extract a set from the sigma-algebra and check that the derivative has its defining property.

        >>> integrate = Operators.integrate
        >>> U = Set([1, 2, 3], domain=Omega, name="U")
        >>> P(U) == integrate(P.derivative(mu), U)
        True

        Define a sub-sigma-algebra for conditional derivatives. Notice the null atom.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 2,  # P-null atom
        ...         5: 2,  # P-null atom
        ...         6: 2,  # P-null atom
        ...     },
        ...     name="G",
        ... )

        In SigALg, conditional derivatives are instances of `ParametrizedMeasurableFunction`.

        >>> P.derivative(mu, G)
        ParametrizedMeasurableFunction(parameters=(G), measurable_vars=(omega), domain=Omega, sig_alg=F, measure=mu, name=dP(-|G)_dmu)
        >>> print(P.derivative(mu, G))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'dP(-|G)_dmu':
        G          0         1         2
        omega
        0      0.250  0.000000  0.250000
        1      0.375  0.000000  0.125000
        2      0.000  0.333333  0.083333
        3      0.000  0.333333  0.083333
        4      0.000  0.000000  0.062500
        5      0.000  0.000000  0.000000
        6      0.000  0.000000  0.000000

        Note that the parameter space for the function is the set of atom identifiers of the conditioning sigma-algebra `G`.

        The integral of the conditional Radon-Nikodym derivative is supposed to coincide (`P`-almost surely) with the conditional probability distribution. We check this, using the set `U` from above.

        >>> integrate(P.derivative(mu, G), U, measure=mu)
        Function(parameters=(G), domain=G, name=int_U dP(-|G)_dmu dmu)
        >>> print(integrate(P.derivative(mu, G), U, measure=mu))  # doctest: +NORMALIZE_WHITESPACE
        Function 'int_U dP(-|G)_dmu dmu':
           int_U dP(-|G)_dmu dmu
        G
        0                   0.75
        1                   1.00
        2                   0.50

        This is a `Function` instance defined on the atom identifiers of `G`. We may "ascend" to a function on the entire sample space by calling the `ascend` method. (Essentially, broadcasting the values of the function from the atom identifers to the entire sample space.) We check that this function coincides with the conditional distribution.

        >>> P.equal_almost_surely(integrate(P.derivative(mu, G), U, measure=mu).ascend(G), P.conditional(G, subset=U))
        True

        Now define a random variable to demonstrate the connnection with expectations.

        >>> X = RandomVariable(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: -2,
        ...         2: 4,
        ...         3: 4,
        ...         4: 5,
        ...         5: 3,
        ...         6: 3,
        ...     },
        ... )

        As we saw above, in SigAlg, the conditional derivative is an instance of `ParametrizedMeasurableFunction`, where the parameters are the atom identifiers of `G`. Conceptually, one should think of the conditional derivative as a family of derivatives, one for each atom identifier.

        Multiplication of a `RandomVariable` instance against an instance of `ParametrizedMeasurableFunction` is defined in SigAlg as long as both are defined on the same sample space. The multiplication is the usual function multiplication, parameter by parameter, and yields an instance of `ParametrizedMeasurableFunction`.

        >>> X * P.derivative(mu, G)
        ParametrizedMeasurableFunction(parameters=(G), measurable_vars=(omega), domain=Omega, sig_alg=F, measure=None, name=(X * dP(-|G)_dmu))

        Think of a `ParametrizedMeasurableFunction` as a family of measurable functions, one for each parameter. It is possible to pass such an instance into the `integrate` operator. The result is a `Function` whose values are the integrals of the random variables, parameter by parameter.

        >>> integrate(X * P.derivative(mu, G), measure=mu)
        Function(parameters=(G), domain=G, name=int (X * dP(-|G)_dmu) dmu)

        This is a function on the atom identifiers of the sigma-algebra `G`. As above, it naturally "ascends" to a function on the sample space, which we obtain using the `ascend` method. The mathematical theory says that this function is equal to the conditional expectation (at least `P`-almost surely).

        >>> E = Operators.expectation
        >>> P.equal_almost_surely(E(X, G), integrate(X * P.derivative(mu, G), measure=mu).ascend(G))
        True

        Notes
        -----
        Let $P$ be a probability measure on a finite measurable space $(\Omega, \mathcal{F})$ that is absolutely continuous with respect to a second measure $\mu$. A *Radon-Nikodym derivative* of $P$ with respect to $\mu$ is an $\mathcal{F}$-measurable function $h:\Omega \to \mathbb{R}$ for which

        $$
        P(U) = \int_\Omega h \, d\mu
        $$

        for all $U\in \mathcal{F}$. We often write $dP/d\mu$ for $h$.

        Now suppose that $\mathcal{G}$ is a sub-$\sigma$-algebra of $\mathcal{F}$. A *conditional Radon-Nikodym derivative* of $P$ given $\mathcal{G}$ is any $(\mathcal{F} \otimes \mathcal{G})$-measurable function $h: \Omega \times \Omega \to \mathbb{R}$ for which

        $$
        P(U\cap V) = \int_{U\times V} h(\omega, \omega') \, d(\mu \otimes P)(\omega, \omega')
        $$

        for all $U\in \mathcal{F}$ and $V\in \mathcal{G}$. By Fubini's theorem, we may write this latter equation as

        $$
        P(U\cap V) = \int_V \left[ \int_U h(\omega, \omega') \, d\mu(\omega) \right] \, dP(\omega'),
        $$

        from which it follows that

        $$
        P(U\mid \mathcal{G})(\omega') = \int_U h(\omega, \omega') \, d\mu(\omega),
        $$

        where the equality holds for all $\omega'$ $P$-almost surely. It thus follows that for all such $\omega'$, the partial function

        $$
        h(-, \omega') : \Omega \to \mathbb{R}, \quad \omega \mapsto h(\omega, \omega'),
        $$

        is a Radon-Nikodym derivative of the probability measure $P(- \mid \mathcal{G})(\omega')$ with respect to $\mu$.
        """
        from .._utils.measure_helpers import compute_radon_nikodym
        from ..functions.measurable_function import MeasurableFunction
        from ..functions.parametrized_measurable_function import (
            ParametrizedMeasurableFunction,
        )

        if given is not None:
            given_name = given.name
        given = self._normalize_given_and_base_measure(
            given=given, base_measure=base_measure
        )

        if base_measure is None:
            base_measure = Measure.counting(self.sig_alg)

        if not self.is_absolutely_continuous(base_measure, tol=tol):
            raise ValueError(
                "The given measure is not absolutely continuous with respect to the base measure."
            )

        if given is not None and name is None:
            name = f"d{self.name}(-|{given_name})_d{base_measure.name}"

        elif name is None:
            name = f"d{self.name}_d{base_measure.name}"

        data = compute_radon_nikodym(
            self_data=self.data,
            base_measure_data=base_measure.data,
            sig_alg_data=self.sig_alg.data,
            given_data=given.data if given is not None else None,
            given_variable_names=given.variable_names if given is not None else None,
            atom_data=self.sig_alg.down_lattice.get_atom_data(given)
            if given is not None
            else None,
            restricted_self_data=(self | given).data if given is not None else None,
        )

        if given is None:
            return MeasurableFunction._from_validated(
                data=data.rename(name),
                sig_alg=self.sig_alg,
                measure=base_measure,
                index_kind="Index",
                index_name=None,
                name=name,
            )

        else:
            return ParametrizedMeasurableFunction._from_validated(
                data=data,
                sig_alg=self.sig_alg,
                measure=base_measure,
                complete_domain_name=f"{given.name} x {self.sig_alg.domain.name}",
                parameter_domain_name=given.name,
                parameter_names=given.variable_names,
                name=name,
            )

    def surprisal(
        self,
        base_measure: Measure | None = None,
        given: SigmaAlgebra | RandomVector | None = None,
        base: Literal["e", "2", "10"] = "e",
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> MeasurableFunction | ParametrizedMeasurableFunction:
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
        from .._utils.measure_helpers import compute_surprisal
        from ..functions.measurable_function import MeasurableFunction
        from ..functions.parametrized_measurable_function import (
            ParametrizedMeasurableFunction,
        )
        from ..measures.measure import Measure

        if given is not None:
            given_name = given.name
        given = self._normalize_given_and_base_measure(
            given=given, base_measure=base_measure
        )

        if base_measure is None:
            base_measure = Measure.counting(self.sig_alg)

        if not self.is_absolutely_continuous(base_measure, tol=tol):
            raise ValueError(
                "The given measure is not absolutely continuous with respect to the base measure."
            )

        if given is not None and name is None:
            name = f"s({self.name}|{given_name}; {base_measure.name})"

        elif name is None:
            name = f"s({self.name}; {base_measure.name})"

        data = compute_surprisal(
            self_data=self.data,
            base_measure_data=base_measure.data,
            sig_alg_data=self.sig_alg.data,
            given_data=given.data if given is not None else None,
            given_variable_names=given.variable_names if given is not None else None,
            atom_data=self.sig_alg.down_lattice.get_atom_data(given)
            if given is not None
            else None,
            restricted_self_data=(self | given).data if given is not None else None,
            base=base,
        )

        if given is None:
            return MeasurableFunction._from_validated(
                data=data,
                sig_alg=self.sig_alg,
                measure=self,
                index_kind="Index",
                index_name=None,
                name=name,
            )

        else:
            return ParametrizedMeasurableFunction._from_validated(
                data=data,
                sig_alg=self.sig_alg,
                measure=self,
                complete_domain_name=f"{given_name} x {self.sig_alg.domain.name}",
                parameter_domain_name=given_name,
                parameter_names=given.variable_names,
                name=name,
            )

    def entropy(
        self,
        base_measure: Measure | None = None,
        given: SigmaAlgebra | RandomVector | None = None,
        base: Literal["e", "2", "10"] = "e",
        name: Hashable | None = None,
        tol: float = 1e-8,
    ) -> Real | MeasurableFunction:
        """Pass."""
        import pandas as pd

        from .._utils.function_helpers import ascend_from_atom_space
        from .._utils.measure_helpers import compute_entropy
        from ..functions.measurable_function import MeasurableFunction

        if given is not None:
            given_name = given.name
        given = self._normalize_given_and_base_measure(
            given=given, base_measure=base_measure
        )

        if base_measure is None:
            base_measure = Measure.counting(self.sig_alg)

        if not self.is_absolutely_continuous(base_measure, tol=tol):
            raise ValueError(
                "The given measure is not absolutely continuous with respect to the base measure."
            )

        if given is not None and name is None:
            name = f"H({self.name}|{given_name}; {base_measure.name})"

        elif name is None:
            name = f"H({self.name}; {base_measure.name})"

        data = compute_entropy(
            self_data=self.data,
            base_measure_data=base_measure.data,
            sig_alg_data=self.sig_alg.data,
            given_data=given.data if given is not None else None,
            given_variable_names=given.variable_names if given is not None else None,
            atom_data=self.sig_alg.down_lattice.get_atom_data(given)
            if given is not None
            else None,
            restricted_self_data=(self | given).data if given is not None else None,
            base=base,
        )

        if isinstance(data, pd.Series):
            data = ascend_from_atom_space(self_data=data, sig_alg_data=given.data)

            return MeasurableFunction._from_validated(
                data=data,
                sig_alg=given,
                measure=self | given,
                index_kind="Index",
                index_name=None,
                name=name,
            )

        else:
            return data.astype(Real)

    def divergence(
        self,
        other: ProbabilityMeasure,
        base: Literal["e", "2", "10"] = "e",
        tol: float = 1e-8,
    ) -> Real:
        """Pass."""
        return -self.entropy(base_measure=other, base=base, tol=tol)

    def mutual_info(
        self,
        first_variables: list[Hashable] | None = None,
        second_variables: list[Hashable] | None = None,
        base: Literal["e", "2", "10"] = "e",
    ) -> Real:
        """Pass."""
        import pandas as pd

        from .._utils.measure_helpers import compute_entropy

        if (first_variables is None) != (second_variables is None):
            raise TypeError(
                "Either both first_variables and second_variables must "
                "be given, or neither given (in which case the "
                "probability measure must be defined on a 2-dimesional "
                "domain)."
            )
        if first_variables is None:
            first_variables = [self.variable_names[0]]
            second_variables = [self.variable_names[1]]
        if set(first_variables) & set(second_variables):
            raise ValueError("The first and second sets of variables must be disjoint.")
        if set(first_variables) | set(second_variables) != set(self.variable_names):
            raise ValueError(
                "The union of the first and second sets of variable names "
                "must equal the set of all variable names of the probability measure."
            )
        if not self.sig_alg.is_power_set:
            raise ValueError(
                "The mutual_info method is only defined for probability"
                " measures defined on power-set sigma-algebras."
            )

        left_marginal = (
            self.data.groupby(level=first_variables).sum().rename("left").reset_index()
        )
        right_marginal = (
            self.data.groupby(level=second_variables)
            .sum()
            .rename("right")
            .reset_index()
        )

        crossed_data = (
            pd.merge(left=left_marginal, right=right_marginal, how="cross")
            .set_index(self.variable_names)
            .sort_index()
        ).astype(float)

        product = (crossed_data["left"] * crossed_data["right"]).rename("product")

        self_data = self.data.reindex(product.index, fill_value=0.0)

        data = -compute_entropy(
            self_data=self_data,
            base_measure_data=product,
            sig_alg_data=self.sig_alg.data,
            base=base,
        )

        return float(data) + 0.0

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
        >>> Omega = SampleSpace.cartesian_power([0, 1], n=2, variable_names=["s_0", "s_1"], name="Omega")
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
            first=first, second=second, rtol=rtol, atol=atol
        )

    # --------------------- utils --------------------- #

    def _normalize_given_and_base_measure(
        self,
        given: SigmaAlgebra | Set | MeasurableVector | None = None,
        base_measure: Measure | None = None,
    ) -> SigmaAlgebra:
        from ..functions.measurable_vector import MeasurableVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.set import Set

        if given is not None:
            if not isinstance(given, SigmaAlgebra | Set | MeasurableVector):
                raise TypeError(
                    "If given, the 'given' must be a SigmaAlgebra, Set, or MeasurableVector instance."
                )

            if isinstance(given, Set | MeasurableVector):
                given = given.generated_sig_alg

            if not given <= self.sig_alg:
                raise ValueError(
                    "The conditioning sigma-algebra must be a sub-sigma-algebra of the sigma-algebra of the probability measure."
                )

            if set(given.variable_names) & set(self.sig_alg.variable_names):
                raise ValueError(
                    "The variable names of the underlying sigma-algebra and the given sigma-algebra must be disjoint."
                )

        if base_measure is not None and self.sig_alg != base_measure.sig_alg:
            raise TypeError(
                "If given, the base measure must have the same sigma-algebra as the probability measure."
            )

        return given
