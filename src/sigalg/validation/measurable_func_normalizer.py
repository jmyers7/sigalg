from __future__ import annotations

from typing import TYPE_CHECKING, Literal  # noqa: F401

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from ..core.measures.measure import Measure  # noqa: TC001
from ..core.sigma_algebras.sigma_algebra import SigmaAlgebra  # noqa: TC001
from ..core.spaces.domain import Domain  # noqa: TC001


class MeasurableFuncNormalizer(BaseModel):
    """Validate input data for SigAlg objects that are built with domains, sigma-algebras, and measures.

    Rules:

    1. If `measure` is given:

    * Overwrite (or set) `sig_alg = measure.sig_alg` (assuming they are equal up to `__eq__`).

    * Overwrite (or set) `domain = measure.sig_alg.domain`.

    2. If `measure` is not given, but `sig_alg` is:

    * Overwrite (or set) `domain = sig_alg.domain` (assuming they are equal up to `__eq__`).

    * Leave `measure` as `None`.

    3. If both `measure` and `sig_alg` are not given, but `domain` is:

    * Set `sig_alg = SigmaAlgebra.power_set(domain)`.

    * Leave `measure` as `None`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: Domain | None = None
    sig_alg: SigmaAlgebra | None = None
    measure: Measure | None = None

    @model_validator(mode="after")
    def normalize_data(self) -> MeasurableFuncNormalizer:  # noqa: D102
        from ..core.measures.measure import Measure
        from ..core.sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..core.spaces.domain import Domain

        if self.domain is not None and not isinstance(self.domain, Domain):
            raise TypeError(
                "If given, the domain parameter must be an instance of Domain."
            )
        if self.sig_alg is not None and not isinstance(self.sig_alg, SigmaAlgebra):
            raise TypeError(
                "If given, the sig_alg parameter must be an instance of SigmaAlgebra."
            )
        if self.measure is not None and not isinstance(self.measure, Measure):
            raise TypeError(
                "If given, the measure parameter must be an instance of Measure."
            )

        if self.measure is not None:
            if self.sig_alg is None or self.measure.sig_alg == self.sig_alg:
                self.sig_alg = self.measure.sig_alg
            else:
                raise ValueError(
                    "If both the measure and sigma-algebra are given, the sigma-algebra of the former must equal the latter."
                )

            if self.domain is not None and self.sig_alg.domain != self.domain:
                raise ValueError(
                    "If the measure is given and the function has a domain, the domain of the sigma-algebra of the former must equal the latter."
                )

        elif self.sig_alg is not None:
            if self.domain is not None and self.sig_alg.domain != self.domain:
                raise ValueError(
                    "If both the sigma-algebra and the domain are given, the domain of the former must equal the latter."
                )

        elif self.domain is not None:
            self.sig_alg = SigmaAlgebra.power_set(self.domain)

        return self
