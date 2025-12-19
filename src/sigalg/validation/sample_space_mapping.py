from __future__ import annotations  # noqa: D100

from collections.abc import Hashable, Mapping
from numbers import Real
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..core.base.sample_space import SampleSpace


class SampleSpaceMappingIn(BaseModel):  # noqa: D101

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mapping: Mapping[Hashable, Any]
    sample_space: SampleSpace
    name: Hashable
    kind: Literal["any", "probabilities"] = "any"

    @field_validator("mapping", mode="before")
    @classmethod
    def _validate_mapping_type(
        cls, v: Mapping[Hashable, Any]
    ) -> Mapping[Hashable, Any]:
        if not isinstance(v, Mapping):
            raise TypeError(
                "The mapping must be a mapping from sample indices to values."
            )
        for key in v.keys():
            if not isinstance(key, Hashable):
                raise TypeError("All keys in the mapping must be Hashable.")
        return v

    @field_validator("name", mode="before")
    @classmethod
    def _name_must_be_hashable(cls, v: Hashable) -> Hashable:
        if not isinstance(v, Hashable):
            raise TypeError("name must be hashable.")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> SampleSpaceMappingIn:
        sample_ids = set(self.sample_space.data)
        mapping_ids = set(self.mapping.keys())

        if mapping_ids != sample_ids:
            raise ValueError(
                "mapping must contain an entry for every sample index in sample_space."
            )

        if self.kind == "probabilities":
            for value in self.mapping.values():
                if not isinstance(value, Real):
                    raise TypeError("All values in the mapping must be numeric.")
                if value < 0:
                    raise ValueError("All values in the mapping must be non-negative.")
            total = sum(self.mapping.values())
            if not abs(total - 1.0) < 1e-8:
                raise ValueError("The values in the mapping must sum to 1.")

        return self
