"""Marker class for a measurable function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


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
                "Cannot pass an index into the constructor of a MeasurableFunction."
            )

        super().__init__(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            domain_kind=domain_kind,
            domain_name=domain_name,
            output_name=output_name,
            index=None,
            index_kind=index_kind,
            index_name=None,
            name=name,
        )

        if self.dimension > 1:
            raise TypeError("A MeasurableFunction must have 1-dimensional outputs.")
