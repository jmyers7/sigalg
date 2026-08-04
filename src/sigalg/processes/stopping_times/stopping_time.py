"""A class representing a stopping time."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from math import inf
from typing import TYPE_CHECKING

from ...core.functions.random_variable import RandomVariable
from ...core.sigma_algebras.filtration import Filtration

if TYPE_CHECKING:
    from ...typing.mapping_like import MappingLike
    from ..base.stochastic_process import StochasticProcess


# TODO: update all docstrings
class StoppingTime(RandomVariable):
    r"""A class representing a stopping time.

    The constructor is not meant to be used directly. Instead, the user should call the `from_filtration` class method.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
        The sample space of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space.
    name : Hashable, default="X"
        The name of the stopping time.

    Examples
    --------
    A gambler begins with 10 units of currency in the bank and plays a game with unit stakes. The house has probability of 0.6 of winning, in which case the gambler loses one unit, and so the gambler has probability 0.4 of winning one unit. We suppose that the money in the gambler's bank is modeled by a random walk stochastic process, and that the gambler plays ten games.

    In SigAlg, we set up this random walk as follows, and generate eight random trajectories:

    >>> from math import inf
    >>> from sigalg.core import Time
    >>> from sigalg.processes import RandomWalk, StoppingTime
    >>> T = Time.discrete(start=1, stop=10)
    >>> S = RandomWalk.generate(
    ...     mode="sim",
    ...     p=0.4,
    ...     initial_state=10,
    ...     index=T,
    ...     n_trajectories=8,
    ...     random_state=42,
    ...     name="S",
    ... )
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Random walk 'S':
    time    1   2   3   4   5   6   7   8   9   10
    sample
    0       10  11  10  11  12  11  12  13  14  13
    1       10   9   8   9  10  11  10   9   8   7
    2       10  11  12  13  12  13  14  15  14  13
    3       10   9   8   9  10  11  10   9   8   7
    4       10   9   8   7   8   7   8   9   8   9
    5       10  11  10   9  10   9   8   7   8   9
    6       10  11  12  11  10   9   8   9   8   7
    7       10  11  12  11  10   9   8   7   6   5

    The gambler decides that they will stop playing the game when their holdings equal 8 units, or they suffer a 20% loss compared to their initial holdings of 10 units. So, using the printout of `S` above, we see that price trajectory `0` will never stop, price trajectory `1` should stop at time `3`, price trajectory `2` will never stop, price trajectory `3` should stop at time `3`, and so on. These values define a stopping time, which we implement as follows:

    >>> tau = StoppingTime.from_filtration(
    ...     process=S,
    ...     mapping={
    ...         0: inf,  # play will never stop
    ...         1: 3,
    ...         2: inf,  # play will never stop
    ...         3: 3,
    ...         4: 3,
    ...         5: 7,
    ...         6: 7,
    ...         7: 7,
    ...     },
    ... )
    >>> print(tau)  # doctest: +NORMALIZE_WHITESPACE
    Stopping time 'tau':
            tau
    sample
    0       inf
    1       3.0
    2       inf
    3       3.0
    4       3.0
    5       7.0
    6       7.0
    7       7.0

    Notes
    -----
    Let $(\Omega, \mathcal{F}, P)$ be a probability space and $\{\mathcal{F}_t\}_{t\in T}$ a filtration of $\mathcal{F}$, index by a linearly ordered set $T$. A random variable $\tau: \Omega \to T$ is called a *stopping time* if, for each $t\in T$, we have

    $$
    \tau^{-t}(t) \in \mathcal{F}_t.
    $$
    """

    _repr_name = "StoppingTime"
    _str_name = "Stopping time"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_filtration(
        cls,
        process: StochasticProcess | None = None,
        filtration: Filtration | None = None,
        mapping: MappingLike | None = None,
        name: Hashable = "tau",
    ) -> StoppingTime:
        """Pass."""
        from ...core.sigma_algebras.filtration import Filtration
        from ..base.stochastic_process import StochasticProcess

        if process is not None and not isinstance(process, StochasticProcess):
            raise TypeError("process must be a StochasticProcess, if given.")
        if filtration is not None and not isinstance(filtration, Filtration):
            raise TypeError("filtration must be a Filtration, if given.")
        if (process is None) == (filtration is None):
            raise ValueError(
                "One or the other of process or filtration must be given, but not both."
            )

        if filtration is None:
            filtration = process.natural_filtration

        stopping_time = cls(
            *process.prob_space,
            mapping=mapping,
            name=name,
        )
        stopping_time.time = filtration.index

        # HACK: Remember, the constructor for the ultimate parent class MeasurableVector will change class membership. Without this line, the stopping time becomes a RandomVariable
        stopping_time.__class__ = StoppingTime

        if not set(stopping_time.data.values) - {inf} <= set(stopping_time.time.data):
            raise ValueError(
                "The range of the stopping time must be in the time index of the stochastic process."
            )

        for t, event in stopping_time.generated_sig_alg.atom_id_to_atom.items():
            if t == inf:
                check_alg = filtration.finest
            else:
                check_alg = filtration[t]
            if event not in check_alg:
                raise TypeError(
                    "One of the level sets of the stopping time is not measurable with respect to the appropriate sigma-algebra in the filtration."
                )

        return stopping_time
