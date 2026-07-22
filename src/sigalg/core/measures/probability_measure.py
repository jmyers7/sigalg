"""A class representing a probability measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import dirichlet

from .measure import Measure

if TYPE_CHECKING:
    from ...validation.mapping_validator import MappingLike
    from ..spaces.domain import Domain
    from ..spaces.measurable_set import MeasurableSet
    from ..spaces.sample_space import SampleSpace
    from ..measures.parametrized_probability_measure import (
        ParametrizedProbabilityMeasure,
    )
    from ..functions.random_variable import RandomVariable
    from ..functions.random_vector import RandomVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class ProbabilityMeasure(Measure):
    r"""A class representing a probability measure on a sigma-algebra.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra on which the probability measure is defined.
    sample_space : SampleSpace | IndexLike | None, default=None
        The sample space on which the probability measure is defined.
    domain : Domain | None, default=None
        The domain of the probability measure. This parameter is not intended to be set by the user.
    mapping : MappingLike | Callable | None, default=None
        A mapping from the domain to the probability measure values.
    kind : Literal["any", "probabilities"], default="probabilities"
        The kind of measure. If `probabilities`, the outputs of the measure must sum to 1. If `any`, the outputs of the measure can take any non-negative values, and the measure will be an instance of `Measure` rather than `ProbabilityMeasure`.
    output_name : str, default="probability"
        The name of the output variable of the probability measure.
    name : Hashable | None, default=None
        A name for the probability measure. If `None`, the default name `P` is used.

    Examples
    --------
    Define a probability measure on a sigma-algebra with two atoms.

    >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> F = SigmaAlgebra(
    ...    sample_space=Omega,
    ...    mapping={
    ...        0: 0,
    ...        1: 0,
    ...        2: 1,
    ...    },
    ... )
    >>> P = ProbabilityMeasure(sig_alg=F, mapping={0: 0.2, 1: 0.8})
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
             probability
    atom_ID
    0                0.2
    1                0.8

    Define a probability measure directly on a sample space, which will use the power-set sigma-algebra by default.

    >>> Q = ProbabilityMeasure(
    ...     sample_space=Omega,
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

    Define a general measure by using the `ProbabilityMeasure` constructor with the parameter `kind` set to `any`. This will create an instance of `Measure`.

    >>> mu = ProbabilityMeasure(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 1,
    ...         1: 2,
    ...         2: 3,
    ...     },
    ...     kind="any",
    ...     name="mu",
    ... )
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'mu':
             measure
    sample
    0              1
    1              2
    2              3
    >>> print(type(mu))
    <class 'sigalg.core.measures.measure.Measure'>

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be a measurable space consisting of a $\sigma$-algebra $\mathcal{F}$ on a nonempty set $\Omega$. A *probability measure* $P$ is a countably additive function $P: \mathcal{F} \to [0,\infty)$ such that $P(\Omega) = 1$. Here, *countable additivity* means that

    $$
    P \left( \bigcup_{k=1}^\infty A_k \right) = \sum_{k=1}^\infty P(A_k)
    $$

    for all collections $\{A_k\}_{k=1}^\infty$ of pairwise disjoint measurable sets. If $\Omega$ is finite (as it always is, in SigAlg), then $P$ needs only to be finitely additive in order to be countably additive.
    """

    _default_name = "P"
    _repr_name = "Probability measure"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        domain: Domain | None = None,
        mapping: MappingLike | Callable | None = None,
        kind: Literal["any", "probabilities"] = "probabilities",
        output_name: str = "probability",
        name: Hashable | None = None,
        **kwargs,
    ) -> None:
        output_name = "measure" if kind == "any" else output_name

        super().__init__(
            sig_alg=sig_alg,
            sample_space=sample_space,
            domain=domain,
            mapping=mapping,
            output_name=output_name,
            name=name,
            kind=kind,
        )

        if kind == "any":
            self.__class__ = Measure

    @classmethod
    def uniform(
        cls,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        name: Hashable = "U",
    ) -> ProbabilityMeasure:
        r"""Create a uniform probability measure on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra on which to define the uniform probability measure.
        name : Hashable, default="U"
            A name for the probability measure.

        Raises
        ------
        ValueError
            If the sigma-algebra has no atoms.
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance, or if `name` is not hashable.

        Returns
        -------
        prob_measure: ProbabilityMeasure
            A uniform ProbabilityMeasure instance on the provided sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> U = ProbabilityMeasure.uniform(sig_alg=F)
        >>> print(U)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'U':
                 probability
        atom_ID
        0           0.333333
        1           0.333333
        2           0.333333
        >>> V = ProbabilityMeasure.uniform(sample_space=Omega, name="V")
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
        Let $(\Omega,\mathcal{F})$ be an event space where $\Omega$ is finite, and suppose that $\mathcal{F}$ has $n$ atoms. The *uniform probability measure* on $\mathcal{F}$ is the probability measure $P$ defined by

        $$
        P(A) = \frac{1}{n},
        $$

        for all atoms $A\in \mathcal{F}$.
        """
        from ..spaces.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance, if given.")
        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance, if given.")
        if (sig_alg is not None and sample_space is not None) and (
            sig_alg.sample_space != sample_space
        ):
            raise ValueError(
                "The sample space of the sigma-algebra does not match the given sample space."
            )
        if sig_alg is None and sample_space is None:
            raise ValueError("At least one of sig_alg or sample_space must be given.")

        space = sig_alg.atom_ids if sig_alg is not None else sample_space

        n = len(space)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on sigma-algebra with no atoms."
            )
        probs = dict.fromkeys(space, 1.0 / n)

        return cls(sig_alg=sig_alg, sample_space=sample_space, mapping=probs, name=name)

    @classmethod
    def from_rand(
        cls,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        num_null_atoms: int = 0,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "P",
    ) -> ProbabilityMeasure:
        """Generate a random probability measure.

        This method generates a random probability measure on the sample space by sampling from a Dirichlet distribution with all concentration parameters equal to 1.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra on which this probability measure is defined.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.
        name : Hashable, default="P"
            The name of this probability measure.

        Raises
        ------
        TypeError
            If `random_state` is not an integer, Generator, or `None`.

        Returns
        -------
        self : ProbabilityMeasure
            A probability measure with randomly generated probabilities.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(seed=42)
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure.from_rand(sig_alg=F, random_state=rng)
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                 probability
        atom_ID
        0           0.492826
        1           0.507174
        >>> Q = ProbabilityMeasure.from_rand(sample_space=Omega, random_state=rng, name="Q")
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        sample
        0          0.153827
        1          0.047522
        2          0.798651
        """
        from ..spaces.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance, if given.")
        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance, if given.")
        if (sig_alg is not None and sample_space is not None) and (
            sig_alg.sample_space != sample_space
        ):
            raise ValueError(
                "The sample space of the sigma-algebra does not match the given sample space."
            )
        if sig_alg is None and sample_space is None:
            raise ValueError("At least one of sig_alg or sample_space must be given.")
        if not isinstance(num_null_atoms, int):
            raise TypeError("num_null_atoms must be an integer.")
        if num_null_atoms < 0:
            raise ValueError("num_null_atoms must be non-negative.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        space = sig_alg.atom_ids if sig_alg is not None else sample_space

        if num_null_atoms >= len(space):
            raise ValueError(
                "num_null_atoms must be less than either the number of atoms of sig_alg (if given) or the size of the sample space (if given)."
            )

        probs_arr = (
            dirichlet.rvs(
                alpha=[
                    1,
                ]
                * (len(space) - num_null_atoms),
                random_state=rng,
            )
            .squeeze()
            .tolist()
        )
        probs_arr = [probs_arr] if not isinstance(probs_arr, list) else probs_arr
        probs_arr = probs_arr + [0.0] * num_null_atoms
        rng.shuffle(probs_arr)
        mapping = dict(zip(space, probs_arr))

        return cls(
            sig_alg=sig_alg, sample_space=sample_space, mapping=mapping, name=name
        )

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
            Random seed or generator for reproducibility. If `None`, the random state is not set.

        Returns
        -------
        sample : pd.Series | pd.DataFrame
            If the domain of the probability measure is 1-dimensional, then a `pd.Series` is returned containing the random sample. Otherwise, if the domain is multi-dimensional, a `pd.DataFrame` is returned whose rows contain the random sample and has columns indexed by the variable names of the domain.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `random_state` is not an integer, `np.random.Generator`, or `None`.

        Examples
        --------
        Define a sigma-algebra with 1-dimensional atom IDs with variable name `x`.

        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
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
        ...     sig_alg=F,
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
        ...     sample_space=Omega,
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
        ...     sig_alg=G,
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

        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=7)
        >>> F = SigmaAlgebra(
        ...    sample_space=Omega,
        ...    mapping={
        ...        0: 0,
        ...        1: 1,
        ...        2: 1,
        ...        3: 2,
        ...        4: 2,
        ...        5: 3,
        ...        6: 3,
        ...    },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.1,
        ...         1: 0.25,
        ...         2: 0.25,
        ...         3: 0.4,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...         5: 2,
        ...         6: 2,
        ...     },
        ...     name="G",
        ... )

        Extract an event `A` from the sigma-algebra `F` and the three atoms from `G`.

        >>> A = F.get_event([1, 2, 3, 4], name="A")
        >>> B_0, B_1, B_2 = G.to_atoms

        Compute the conditional probability and check that its values match the familiar formula for conditional probability.

        >>> cond_prob = P.conditional(event=A, given=G)
        >>> for atom in G.to_atoms:
        ...     print(cond_prob(atom) == P(atom & A) / P(atom))
        True
        True
        True

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space, and let $A \in \mathcal{F}$ be an event and $\mathcal{G} \subseteq \mathcal{F}$ be a sub-$\sigma$-algebra. The *conditional probability* of $A$ given $\mathcal{G}$ is a $\mathcal{G}$-measurable random variable, denoted $P(A|\mathcal{G})$, for which

        $$
        P(A \cap B) = \int_B P(A|\mathcal{G}) \, dP,
        $$

        for all $B \in \mathcal{G}$. The conditional probability is unique up to almost sure equality.

        In the case that $\mathcal{G} = \sigma(B)$ is the $\sigma$-algebra generated by an event $B\in \mathcal{F}$, we obtain the familiar formula

        $$
        P(A \mid \sigma(B))(\omega) = \frac{P(A\cap B)}{P(B)}
        $$

        for all $\omega \in B$, provided $P(B) > 0$.
        """
        from ..spaces.measurable_set import MeasurableSet
        from ..functions.random_variable import RandomVariable
        from ..functions.random_vector import RandomVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(given, SigmaAlgebra | MeasurableSet | RandomVector):
            raise TypeError(
                "given must be a SigmaAlgebra, MeasurableSet, or RandomVector instance."
            )
        if not isinstance(event, MeasurableSet):
            raise TypeError("event must be an MeasurableSet instance.")

        if isinstance(given, MeasurableSet):
            sig_alg = SigmaAlgebra.from_event(given)
        elif isinstance(given, RandomVector):
            sig_alg = SigmaAlgebra.from_random_vector(given)
        else:
            sig_alg = given

        I = RandomVariable.indicator_of(event)

        return self.expectation(rv=I, sig_alg=sig_alg).with_name(
            f"P({event.name}|{sig_alg.name})"
        )

    def given(
        self,
        sub: SigmaAlgebra | MeasurableSet | RandomVector,
        /,
        conditioning_suffix: str = "_g",
        name: Hashable | None = None,
    ) -> ParametrizedProbabilityMeasure:
        r"""Compute a conditional probability measure.

        Parameters
        ----------
        given : SigmaAlgebra | MeasurableSet | RandomVector
            The given condition, which can be a sigma-algebra, an event, or a random vector.
        name : Hashable | None, default=None
            The name of the resulting parametrized probability measure.

        Raises
        ------
        TypeError
            If `given` is not a `SigmaAlgebra`, `MeasurableSet`, or `RandomVector` instance.

        Returns
        -------
        cond_prob_measure : ParametrizedProbabilityMeasure
            A parametrized probability measure representing the conditional probability given `given`.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=7)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...         5: 3,
        ...         6: 3,
        ...     },
        ...     variable_names=["A"],
        ... )
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...         5: 2,
        ...         6: 2,
        ...     },
        ...     name="G",
        ...     variable_names=["B"],
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.1,
        ...         1: 0.25,
        ...         2: 0.25,
        ...         3: 0.4,
        ...     },
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
           probability
        A
        0         0.10
        1         0.25
        2         0.25
        3         0.40
        >>> P_given_G = P.given(G)
        >>> print(P_given_G)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P(?|G)':
             probability
        B A
        0 0     1.000000
          1     0.000000
          2     0.000000
          3     0.000000
        1 0     0.000000
          1     1.000000
          2     0.000000
          3     0.000000
        2 0     0.000000
          1     0.000000
          2     0.384615
          3     0.615385
        >>> prob_measure = P_given_G(B=2)
        >>> print(prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P(?|G)(B=2)':
           probability
        A
        0     0.000000
        1     0.000000
        2     0.384615
        3     0.615385
        """
        from ..spaces.domain import Domain
        from ..spaces.measurable_set import MeasurableSet
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ..functions.random_vector import RandomVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sub, SigmaAlgebra | MeasurableSet | RandomVector):
            raise TypeError(
                "'given' must be a SigmaAlgebra, MeasurableSet, or RandomVector instance."
            )

        if isinstance(sub, MeasurableSet):
            sub = SigmaAlgebra.from_event(sub)
        elif isinstance(sub, RandomVector):
            sub = SigmaAlgebra.from_random_vector(sub)
        super = self.sig_alg

        if not sub <= super:
            raise ValueError(
                "The 'given' sigma-algebra must be a sub-sigma-algebra of the probability measure's sigma-algebra."
            )

        mapping = (
            pd.concat(
                [super.data.rename("super_ID"), sub.data.rename("sub_ID")], axis=1
            )
            .drop_duplicates("super_ID")
            .reset_index(drop=True)
        )
        mapping = pd.merge(mapping, self.data, left_on="super_ID", right_index=True)

        mapping["super_atom_probs"] = mapping.groupby(by="super_ID", sort=False)[
            "probability"
        ].transform(sum)
        mapping["sub_atom_probs"] = mapping.groupby(by="sub_ID", sort=False)[
            "probability"
        ].transform(sum)
        mapping["probability"] = mapping["super_atom_probs"] / mapping["sub_atom_probs"]
        mapping = mapping.drop_duplicates(subset="super_ID")[
            ["super_ID", "sub_ID", "probability"]
        ].set_index(["sub_ID", "super_ID"])

        sub_variable_names = [
            (name + conditioning_suffix if name in super.variable_names else name)
            for name in sub.variable_names
        ]
        super_variable_names = super.variable_names

        mapping.index = pd.MultiIndex.from_tuples(
            [self._to_tuple(sub) + self._to_tuple(sup) for sub, sup in mapping.index],
            names=sub_variable_names + super_variable_names,
        )
        sub_atom_space_copy = sub.atom_space.copy()
        sub_atom_space_copy.variable_names = sub_variable_names
        domain = Domain.cartesian_product([sub_atom_space_copy, super.atom_space])
        mapping = (
            mapping.reindex(domain.data, fill_value=0.0).squeeze(axis=1).fillna(0.0)
        )

        if name is None:
            if sub.name.startswith("sigma(") and sub.name.endswith(")"):
                name = f"{self.name}(?|{sub.name[6:-1]})"
            else:
                name = f"{self.name}(?|{sub.name})"

        return ParametrizedProbabilityMeasure(
            sig_alg=super, domain=domain, mapping=mapping, name=name
        )

    @staticmethod
    def _to_tuple(x):
        if isinstance(x, tuple):
            return x
        else:
            return (x,)

    def are_independent(
        self,
        event1: MeasurableSet | None = None,
        event2: MeasurableSet | None = None,
        rv1: RandomVector | None = None,
        rv2: RandomVector | None = None,
        algebra1: SigmaAlgebra | None = None,
        algebra2: SigmaAlgebra | None = None,
        tol: Real = 1e-8,
    ) -> bool:
        r"""Check if two events, two random vectors, or two sigma-algebras are independent.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event1 : MeasurableSet | None, default=None
            The first event.
        event2 : MeasurableSet | None, default=None
            The second event.
        rv1: RandomVector | None, default=None
            The first random vector.
        rv2: RandomVector | None, default=None
            The second random vector.
        algebra1 : SigmaAlgebra | None, default=None
            The first sigma-algebra.
        algebra2 : SigmaAlgebra | None, default=None
            The second sigma-algebra.
        tol : Real, default=1e-10
            The numerical tolerance for checking independence.

        Raises
        ------
        ValueError
            If neither events, random vectors, nor sigma-algebras are provided, or if two of these types are provided, or if the provided objects are from a different sample space.
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
        ...     sample_space=Omega,
        ...     mapping=lambda *, flip_1, flip_2: (
        ...         0.75 ** (flip_1 + flip_2) * 0.25 ** (2 - flip_1 - flip_2)
        ...     ),
        ... )
        >>> prob_space = ProbabilitySpace(sample_space=Omega, prob_measure=P)
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
                      atom_ID
        flip_1 flip_2
        0      0       (0, 0)
               1       (0, 1)
        1      0       (1, 0)
               1       (1, 1)
        <BLANKLINE>
        * Probability measure 'P':
                       probability
        flip_1 flip_2
        0      0            0.0625
               1            0.1875
        1      0            0.1875
               1            0.5625
        >>> A = prob_space.get_event(
        ...     [(0, 0), (0, 1)],
        ...     name="A",
        ... )
        >>> B = prob_space.get_event(
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
        >>> print(P.are_independent(rv1=X_1, rv2=X_2))
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
        >>> print(P.are_independent(rv1=X_1, rv2=Y))
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
        from ..spaces.measurable_set import MeasurableSet
        from ..functions.random_vector import RandomVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        events_provided = event1 is not None and event2 is not None
        rvs_provided = rv1 is not None and rv2 is not None
        algebras_provided = algebra1 is not None and algebra2 is not None

        if sum((events_provided, rvs_provided, algebras_provided)) != 1:
            raise ValueError(
                "Must provide exactly one of the following pairs of arguments: (event1, event2), (rv1, rv2), or (algebra1, algebra2)."
            )

        if events_provided:
            if not isinstance(event1, MeasurableSet) or not isinstance(event2, MeasurableSet):
                raise TypeError("event1 and event2 must be MeasurableSet instances.")

            if not self.sig_alg.is_power_set:
                for event in (event1, event2):
                    if event.sig_alg != self.sig_alg:
                        raise ValueError(
                            "MeasurableSet is not measurable with respect to this probability measure's sigma-algebra"
                        )

            if abs(self(event1 & event2) - self(event1) * self(event2)) < tol:
                return True
            else:
                return False

        if rvs_provided or algebras_provided:
            if rvs_provided:
                if not isinstance(rv1, RandomVector) or not isinstance(
                    rv2, RandomVector
                ):
                    raise TypeError("rv1 and rv2 must be RandomVector instances.")
                if (
                    rv1.sample_space != self.sig_alg.sample_space
                    or rv2.sample_space != self.sig_alg.sample_space
                ):
                    raise ValueError(
                        "Random vectors must be from this probability measure's sample space."
                    )

                algebra1 = SigmaAlgebra.from_random_vector(rv1)
                algebra2 = SigmaAlgebra.from_random_vector(rv2)

            if not isinstance(algebra1, SigmaAlgebra) or not isinstance(
                algebra2, SigmaAlgebra
            ):
                raise TypeError("algebra1 and algebra2 must be SigmaAlgebra instances.")
            if not (algebra1 <= self.sig_alg and algebra2 <= self.sig_alg):
                raise ValueError(
                    "Both sigma-algebras must be sub-algebras of the probability measure's sigma-algebra"
                )

            for atom1 in algebra1.to_atoms:
                for atom2 in algebra2.to_atoms:
                    event1 = self.sig_alg.get_event(list(atom1), name=atom1.name)
                    event2 = self.sig_alg.get_event(list(atom2), name=atom2.name)
                    if not self.are_independent(event1=event1, event2=event2, tol=tol):
                        return False
            return True

    def equal_almost_surely(
        self,
        first: RandomVector,
        second: RandomVector,
        tol: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two random vectors are equal almost surely.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        first : RandomVector
            The first random vector.
        second : RandomVector
            The second random vector.
        tol : float, default=1e-8
            The tolerance below which a probability is considered to be zero for the purposes of this comparison.
        rtol : float, default=1e-5
            The relative tolerance for `np.isclose` when comparing the random vectors.
        atol : float, default=1e-8
            The absolute tolerance for `np.isclose` when comparing the random vectors.

        Raises
        ------
        TypeError
            If `first` or `second` are not `RandomVector` instances.
        ValueError
            If `first` or `second` are from a different sample space than this probability measure's sample space, or if they have different dimensions.

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
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.4,
        ...         1: 0.6,
        ...         2: 0.0,
        ...     },
        ... )
        >>> # Test on random variables
        >>> X = RandomVariable(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 3.0,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 4.0,
        ...     },
        ...     name="Y",
        ... )
        >>> Z = RandomVariable(
        ...     sample_space=Omega,
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
        >>> # Test on random vectors of dimension > 1
        >>> U = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 2),
        ...     },
        ...     name="U",
        ... )
        >>> V = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (-1, 4),
        ...     },
        ...     name="V",
        ... )
        >>> W = RandomVector(
        ...     sample_space=Omega,
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


class ProbabilityMeasureMethods:
    """Mixin class providing probability measure methods to other classes."""

    def sample(
        self, size: int = 1, random_state: int | np.random.Generator | None = None
    ) -> pd.Series | pd.DataFrame:
        """Generate random samples from this probability measure.

        Calls `ProbabilityMeasure.sample`. See the docstring of `ProbabilityMeasure.sample` for details.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate. Must be positive.
        random_state : int | np.random.Generator | None, default=None
            Random seed or generator for reproducibility. If `None`, the random state is not set.

        Returns
        -------
        sample : pd.Series | pd.DataFrame
            If the domain of the probability measure is 1-dimensional, then a `pd.Series` is returned containing the random sample. Otherwise, if the domain is multi-dimensional, a `pd.DataFrame` is returned whose rows contain the random sample and has columns indexed by the variable names of the domain.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `random_state` is not an integer, `np.random.Generator`, or `None`.
        """
        return self.prob_measure.sample(size, random_state)

    def conditional_probability(self, event: MeasurableSet, given: MeasurableSet) -> Real:
        """Compute the conditional probability P(A|B).

        Calls `ProbabilityMeasure.conditional_probability`. See the docstring of `ProbabilityMeasure.conditional_probability` for details.

        Parameters
        ----------
        event : MeasurableSet
            The event A.
        given : MeasurableSet
            The event B.

        Returns
        -------
        conditional_prob : Real
            The conditional probability P(A|B).
        """
        return self.prob_measure.conditional_probability(event, given)

    def are_independent(
        self,
        event1: MeasurableSet | None = None,
        event2: MeasurableSet | None = None,
        rv1: RandomVector | None = None,
        rv2: RandomVector | None = None,
        algebra1: SigmaAlgebra | None = None,
        algebra2: SigmaAlgebra | None = None,
        tol: Real = 1e-8,
    ) -> bool:
        """Check if two events, two random vectors, or two sigma-algebras are independent.

        Calls `ProbabilityMeasure.are_independent`. See the docstring of `ProbabilityMeasure.are_independent` for details.

        Parameters
        ----------
        event1 : MeasurableSet | None, default=None
            The first event.
        event2 : MeasurableSet | None, default=None
            The second event.
        rv1: RandomVector | None, default=None
            The first random vector.
        rv2: RandomVector | None, default=None
            The second random vector.
        algebra1 : SigmaAlgebra | None, default=None
            The first sigma-algebra.
        algebra2 : SigmaAlgebra | None, default=None
            The second sigma-algebra.
        tol : Real, default=1e-10
            The numerical tolerance for checking independence.

        Returns
        -------
        is_independent : bool
            `True` if the events, random vectors, or sigma-algebras are independent, `False` otherwise.
        """
        return self.prob_measure.are_independent(
            event1=event1,
            event2=event2,
            rv1=rv1,
            rv2=rv2,
            algebra1=algebra1,
            algebra2=algebra2,
            tol=tol,
        )

    def almost_surely_equal(
        self,
        first: RandomVector,
        second: RandomVector,
        tol: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two random vectors are equal almost surely.

        Calls `ProbabilityMeasure.almost_surely_equal`. See the docstring of `ProbabilityMeasure.almost_surely_equal` for details.

        Parameters
        ----------
        first : RandomVector
            The first random vector.
        second : RandomVector
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
        """
        return self.prob_measure.almost_surely_equal(
            first=first, second=second, tol=tol, rtol=rtol, atol=atol
        )
