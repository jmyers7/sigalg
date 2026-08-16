from __future__ import annotations

from typing import Annotated, Any

from pydantic import GetCoreSchemaHandler  # noqa: TC002
from pydantic_core import core_schema

from ..core.sigma_algebras.sigma_algebra import SigmaAlgebra
from ..core.spaces.domain import Domain


class _MeasureDomainTypeValidator:
    """A validator for the MeasureDomain type.

    Rules:

    1. If the input is a `SigmaAlgebra`, return it as is.

    2. If the input is a `Domain` or and `IndexLike` object that can be coerced into a `Domain`, create a `SigmaAlgebra` using the power set of the `Domain` and return it.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v: Any) -> SigmaAlgebra:
        if not isinstance(v, SigmaAlgebra | Domain):
            try:
                sig_alg = SigmaAlgebra.power_set(Domain(v))
            except Exception as e:
                raise ValueError(f"Cannot coerce {v} into a Domain") from e

        elif isinstance(v, Domain):
            sig_alg = SigmaAlgebra.power_set(v)

        elif isinstance(v, SigmaAlgebra):
            sig_alg = v

        return sig_alg


MeasureDomain = Annotated[SigmaAlgebra, _MeasureDomainTypeValidator]
