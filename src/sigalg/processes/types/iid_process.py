import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ..base.stochastic_process import StochasticProcess


class IIDProcess(StochasticProcess):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        rv: rv_frozen,
        max_trajectories: int = 1000,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        rv_type: str | None = None,
        random_state: int | None = None,
    ):
        self._validate_parameters(
            rv=rv,
            max_trajectories=max_trajectories,
            length=length,
            initial_time=initial_time,
            name=name,
            random_state=random_state,
        )
        self._rv = rv
        self._max_trajectories = max_trajectories
        self._length = length
        self._initial_time = initial_time
        self._name = name
        self._rv_type = rv_type
        self._random_state = random_state
        super().__init__()

    # --------------------- properties --------------------- #

    @property
    def rv(self) -> rv_frozen:
        return self._rv

    @property
    def max_trajectories(self) -> int:
        return self._max_trajectories

    @property
    def length(self) -> int:
        return self._length

    @property
    def initial_time(self) -> int:
        return self._initial_time

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    @property
    def rv_type(self) -> str | None:
        return self._rv_type

    @rv_type.setter
    def rv_type(self, rv_type: str | None) -> None:
        if rv_type is not None and not isinstance(rv_type, str):
            raise TypeError("rv_type must be a string or None.")
        self._rv_type = rv_type

    @property
    def random_state(self) -> int | None:
        return self._random_state

    # --------------------- simulation logic --------------------- #

    def _simulate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._random_state)
        simulated_trajectories = self._rv.rvs(
            size=(self._max_trajectories, self._length),
            random_state=rng,
        )
        time_index = list(range(self._initial_time, self._length + self._initial_time))
        return pd.DataFrame(data=simulated_trajectories, columns=time_index)

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        if self._rv_type is not None:
            return f"IID {self._rv_type} Process"
        else:
            return "IID Process"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        rv: rv_frozen,
        max_trajectories: int,
        length: int,
        initial_time: int,
        name: str,
        random_state: int | None,
    ) -> None:
        if not isinstance(rv, rv_frozen):
            raise TypeError(
                "rv must be an instance of scipy.stats._distn_infrastructure.rv_frozen."
            )
        if not isinstance(max_trajectories, int) or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer.")
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer.")
        if not isinstance(initial_time, int):
            raise TypeError("initial_time must be an integer.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if random_state is not None and (
            not isinstance(random_state, int) or random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None.")
