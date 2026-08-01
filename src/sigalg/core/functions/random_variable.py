"""Marker class for a 1-dimensional random vector."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .measurable_function import MeasurableFunction
from .random_vector import RandomVector

if TYPE_CHECKING:
    from ...validation.index_validator import IndexLike
    from ...validation.mapping_validator import MappingLike
    from ..indices.index import Index
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.sample_space import SampleSpace


class RandomVariable(RandomVector, MeasurableFunction):
    """Marker class for a 1-dimensional random vector."""

    _repr_name = "RandomVariable"
    _str_name = "Random variable"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: SampleSpace | IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
        mapping: MappingLike | Callable | None = None,
        index: Index | IndexLike | None = None,
        name: Hashable = "X",
    ) -> None:
        super().__init__(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

        if self.dimension > 1:
            self.__class__ = RandomVector
        else:
            self._data = (
                self._data.squeeze(axis=1)
                if isinstance(self._data, pd.DataFrame)
                else self._data
            )
            self._data.name = self._name
            self._index = None
