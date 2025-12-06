import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ..base.process_factory_methods import ProcessFactoryMethods
from ..base.stochastic_process import StochasticProcess
from ..base.time import Time


class IIDProcess(StochasticProcess, ProcessFactoryMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        *,
        rv: rv_frozen,
        time: Time,
        max_trajectories: int = 1000,
        name: str = "X",
        support: list | None = None,
        random_state: int | None = None,
        enumerate: bool = False,
    ) -> None:
        self._validate_parameters(
            rv=rv,
            time=time,
            max_trajectories=max_trajectories,
            name=name,
            support=support,
            random_state=random_state,
            enumerate=enumerate,
        )
        self._rv = rv
        self._time = time
        self._name = name
        self._max_trajectories = max_trajectories
        self._support = support
        self._length = len(time)
        self._random_state = random_state
        self._enumerate = enumerate
        fps = self._generate_fps()
        super().__init__(fps=fps)

    # --------------------- properties --------------------- #

    @property
    def rv(self) -> rv_frozen:
        return self._rv

    @property
    def max_trajectories(self) -> int:
        return self._max_trajectories

    @property
    def support(self) -> list | None:
        return self._support

    @property
    def random_state(self) -> int | None:
        return self._random_state

    @property
    def enumerate(self) -> bool:
        return self._enumerate

    @property
    def time(self) -> Time:
        return self._time

    @property
    def n_support(self) -> int | None:
        if self._support is not None:
            return len(self._support)
        return None

    @property
    def length(self) -> int:
        return self._length

    # --------------------- trajectories logic --------------------- #

    def _simulate_raw_trajectories(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._random_state)
        raw_trajectories = self._rv.rvs(
            size=(self._max_trajectories, self._length),
            random_state=rng,
        )
        return pd.DataFrame(data=raw_trajectories)

    def _compute_exact_probabilities(self, raw_trajectories: pd.DataFrame) -> pd.Series:
        element_wise_probabilities = self._rv.pmf(raw_trajectories.values)
        probabilities = np.prod(element_wise_probabilities, axis=1)
        return pd.Series(probabilities)

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated IID" if self._enumerate else "IID"
        return f"{prefix} {self._rv.dist.name.capitalize()} Process {self._name}"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        rv: rv_frozen,
        time: Time,
        max_trajectories: int,
        name: str,
        support: list | None,
        random_state: int | None,
        enumerate: bool,
    ) -> None:
        from ..base.time import Time

        if not isinstance(rv, rv_frozen):
            raise TypeError(
                "rv must be an instance of scipy.stats._distn_infrastructure.rv_frozen."
            )
        if not isinstance(time, Time):
            raise TypeError("time must be a Time object.")
        if not isinstance(max_trajectories, int) or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if support is not None and not isinstance(support, list):
            raise TypeError("support must be a list or None.")
        if random_state is not None:
            if not isinstance(random_state, int) or random_state < 0:
                raise TypeError("random_state must be a non-negative integer or None.")
        if not isinstance(enumerate, bool):
            raise TypeError("enumerate must be a boolean.")

    def _decide_if_enumeration_feasible(self) -> None:
        if self._support is None:
            raise ValueError(
                "Cannot enumerate trajectories without explicit support. "
                "Please provide the 'support' parameter."
            )

        n_trajectories = self.n_support**self.length

        if n_trajectories > 1_000_000:
            raise ValueError(
                "The number of possible trajectories is too large to enumerate."
            )
        if n_trajectories > self._max_trajectories:
            raise ValueError(
                f"The number of possible trajectories {n_trajectories} is greater than max_trajectories {self._max_trajectories}. "
            )
