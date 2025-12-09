from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...processes.base.stochastic_process import StochasticProcess
    from ..base.time import Time
    from .sig_alg_comparator import SigAlgComparator
    from .sigma_algebra import SigmaAlgebra


class Filtration:

    # --------------------- constructor --------------------- #

    def __init__(
        self, sigma_algebras: list[SigmaAlgebra], time: Time, name: str
    ) -> None:
        pass

    # --------------------- coarsest --------------------- #

    @property
    def coarsest(self) -> SigmaAlgebra:
        pass

    @property
    def finest(self) -> SigmaAlgebra:
        pass

    @property
    def is_discrete_time(self) -> bool:
        pass

    # --------------------- data access methods --------------------- #

    def __getitem__(self, index: int) -> SigmaAlgebra:
        pass

    def stage_at(self, label: str) -> SigmaAlgebra:
        pass

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        pass

    def __iter__(self):
        pass

    # --------------------- comparison methods --------------------- #

    def pairwise_comparator(self, s: Real, t: Real) -> SigAlgComparator:
        pass

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_process(
        cls,
        process: StochasticProcess,
    ) -> Filtration:
        pass

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        pass

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sigma_algebras: list[SigmaAlgebra], time: Time, name: str
    ) -> None:
        pass
