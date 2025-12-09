from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ...processes.base.stochastic_process import StochasticProcess
    from ..base.sample_space import SampleSpace
    from ..base.time import Time
    from .sig_alg_comparator import SigAlgComparator
    from .sigma_algebra import SigmaAlgebra


class Filtration:

    # --------------------- constructor --------------------- #

    def __init__(
        self, sigma_algebras: list[SigmaAlgebra], time: Time, name: str
    ) -> None:
        from .sig_alg_comparator import SigAlgComparator

        self._validate_parameters(sigma_algebras=sigma_algebras, time=time, name=name)
        self._sigma_algebras = sigma_algebras
        self._time = time
        self._name = name
        self._comparator = SigAlgComparator(sigma_algebras=sigma_algebras)
        self._time_to_pos_idx = {t: idx for idx, t in enumerate(self._time)}

    # --------------------- coarsest --------------------- #

    @property
    def sigma_algebras(self) -> list[SigmaAlgebra]:
        return self._sigma_algebras.copy()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    @property
    def time(self) -> Time:
        return self._time

    @property
    def comparator(self) -> SigAlgComparator:
        return self._comparator

    @property
    def values(self) -> pd.DataFrame:
        return self._comparator.df_combined.copy()

    @property
    def sample_space(self) -> SampleSpace:
        return self._sigma_algebras[0].sample_space

    @property
    def coarsest(self) -> SigmaAlgebra:
        return self._sigma_algebras[0]

    @property
    def finest(self) -> SigmaAlgebra:
        return self._sigma_algebras[-1]

    # --------------------- data access methods --------------------- #

    @property
    def at(self) -> Filtration._FiltrationIndexer:
        return Filtration._FiltrationIndexer(self)

    class _FiltrationIndexer:
        def __init__(self, filtration):
            self.filtration = filtration

        def __getitem__(self, time) -> SigmaAlgebra:
            if time not in self.filtration.time:
                raise ValueError(f"Time {time} not in filtration time index")
            pos_idx = self.filtration._time_to_pos_idx[time]
            return self.filtration.sigma_algebras[pos_idx]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self._sigma_algebras)

    def __iter__(self):
        yield from self._sigma_algebras

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_process(
        cls,
        process: StochasticProcess,
    ) -> Filtration:
        pass

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Filtration(name='{self._name}', length={len(self)})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sigma_algebras: list[SigmaAlgebra], time: Time, name: str
    ) -> None:
        from ..base.time import Time
        from .sig_alg_comparator import is_subalgebra
        from .sigma_algebra import SigmaAlgebra

        if not isinstance(sigma_algebras, list) or len(sigma_algebras) == 0:
            raise ValueError("sigma_algebras must be a non-empty list.")
        for alg in sigma_algebras:
            if not isinstance(alg, SigmaAlgebra):
                raise ValueError(
                    "All sigma algebras need to be instances of SigmaAlgebra."
                )
        if not isinstance(time, Time):
            raise TypeError("time must be a Time object.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if len(sigma_algebras) != len(time):
            raise ValueError(
                "The number of sigma algebras must match the length of the time index."
            )
        if len(sigma_algebras) >= 2:
            sample_space = sigma_algebras[0].sample_space
            for alg in sigma_algebras[1:]:
                if alg.sample_space != sample_space:
                    raise ValueError(
                        "All sigma algebras must have the same sample space"
                    )
            for sub_algebra, super_algebra in zip(
                sigma_algebras[:-1], sigma_algebras[1:]
            ):
                if not is_subalgebra(sub_algebra, super_algebra):
                    raise ValueError(
                        "The provided sigma algebras do not form a valid filtration."
                    )
