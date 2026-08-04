from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from ..core.sigma_algebras.sigma_algebra import SigmaAlgebra
from ..core.spaces.domain import Domain
from ..typing.measure_domain import MeasureDomain


class MeasureDomainValidator(BaseModel):
    """Validate input data for instances of `sa.Measure`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    measure_domain: MeasureDomain | None = None
    kind: Literal["measure", "probability"] = "measure"

    @property
    def domain(self) -> Domain | None:  # noqa: D102
        return self.measure_domain.atom_space if self.measure_domain else None

    @property
    def sig_alg(self) -> SigmaAlgebra | None:  # noqa: D102
        return self.measure_domain
