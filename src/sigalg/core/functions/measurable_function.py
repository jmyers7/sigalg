"""Marker class for a measurable function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from collections.abc import Hashable

    import pandas as pd

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.set import Set


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
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> None:
        if index is not None or index_name is not None:
            raise TypeError(
                "An explict index was passed into the MeasurableFunction/RandomVariable constructor. Please do not do this."
            )

        super().__init__(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            domain_kind=domain_kind,
            domain_name=domain_name,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            index_name=None,
            name=name,
        )

    # --------------------- data methods --------------------- #

    def __call__(
        self, key: Hashable | Set | MeasurableFunction
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
