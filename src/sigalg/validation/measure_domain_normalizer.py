from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from ..typing.measure_domain import MeasureDomain  # noqa: TC001

if TYPE_CHECKING:
    from ..core.sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..core.spaces.domain import Domain


class MeasureDomainNormalizer(BaseModel):
    """Validate input data for SigAlg objects built with a `MeasureDomain` object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    measure_domain: MeasureDomain
    kind: Literal["measure", "probability"] = "measure"

    @property
    def domain(self) -> Domain | None:  # noqa: D102
        return self.measure_domain.atom_space

    @property
    def sig_alg(self) -> SigmaAlgebra | None:  # noqa: D102
        return self.measure_domain
