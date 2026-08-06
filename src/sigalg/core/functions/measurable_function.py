"""Marker class for a measurable function."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.measurable_set import MeasurableSet


class MeasurableFunction(MeasurableVector):
    """Marker class for a measurable function."""

    _repr_name = "MeasurableFunction"
    _str_name = "Measurable function"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        mapping: MappingLike | None = None,
        index: IndexLike | None = None,
        name: Hashable = "f",
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

    # --------------------- data methods --------------------- #

    def __call__(
        self, key: Hashable | MeasurableSet | MeasurableFunction
    ) -> Hashable | pd.Series:
        """Pass."""
        if isinstance(key, MeasurableFunction):
            if not set(key.range.domain) <= set(self.domain):
                raise ValueError(
                    "The range of the given measurable function is not a subset of the range of this measurable function. Cannot compose the functions."
                )
            if key.data is None or self.data is None:
                raise ValueError("Cannot compose measurable functions without data.")
            mapping = key.data.apply(lambda x: self.data.get(x)).rename(None)
            return type(self)(
                *key.measurable_space,
                measure=key.measure,
                mapping=mapping,
                name=f"{self.name}({key.name})",
            )
        else:
            return super().__call__(key)
