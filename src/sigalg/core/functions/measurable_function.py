"""Marker class for a measurable function."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from ...validation.index_validator import IndexLike
    from ...validation.mapping_validator import MappingLike
    from ..indices.index import Index
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain


class MeasurableFunction(MeasurableVector):
    """Marker class for a measurable function."""

    _repr_name = "MeasurableFunction"
    _str_name = "Measurable function"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: Domain | IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
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
            self.__class__ = MeasurableVector
        else:
            self._data = (
                self._data.squeeze(axis=1)
                if isinstance(self._data, pd.DataFrame)
                else self._data
            )
            self._data.name = self._name
            self._index = None
