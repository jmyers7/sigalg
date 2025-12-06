import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ..base.stochastic_process import StochasticProcess
from ..base.time import Time


class IIDProcess(StochasticProcess):

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
        self._validate_parameters(rv=rv)
        self._rv = rv
        self._support = support

        super().__init__(
            time=time,
            max_trajectories=max_trajectories,
            name=name,
            support=support,
            random_state=random_state,
            enumerate=enumerate,
        )

    # --------------------- properties --------------------- #

    @property
    def rv(self) -> rv_frozen:
        return self._rv

    def __len__(self) -> int:
        return len(self._time)

    # --------------------- trajectories logic --------------------- #

    def _simulate_raw_trajectories(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._random_state)
        raw_trajectories = self._rv.rvs(
            size=(self._max_trajectories, len(self._time.values)),
            random_state=rng,
        )
        return pd.DataFrame(data=raw_trajectories)

    def _compute_exact_probabilities(self, raw_trajectories: pd.DataFrame) -> pd.Series:
        element_wise_probabilities = self._rv.pmf(raw_trajectories.values)
        probabilities = np.prod(element_wise_probabilities, axis=1)
        return pd.Series(probabilities, index=raw_trajectories.index)

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated IID" if self._enumerate else "IID"
        return f"{prefix} {self._rv.dist.name.capitalize()} Process {self._name}"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(rv: rv_frozen) -> None:
        if not isinstance(rv, rv_frozen):
            raise TypeError(
                "rv must be an instance of scipy.stats._distn_infrastructure.rv_frozen."
            )

    def _decide_if_enumeration_feasible(self) -> None:
        if self._support is None:
            raise ValueError(
                "Cannot enumerate trajectories without explicit support. "
                "Please provide the 'support' parameter."
            )

        n_trajectories = self.n_support ** len(self)

        if n_trajectories > 1_000_000:
            raise ValueError(
                "The number of possible trajectories is too large to enumerate."
            )
        if n_trajectories > self._max_trajectories:
            raise ValueError(
                f"The number of possible trajectories {n_trajectories} is greater than max_trajectories {self._max_trajectories}. "
            )
