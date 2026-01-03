from __future__ import annotations  # noqa: D100

from collections.abc import Hashable

from pydantic import BaseModel, ConfigDict, field_validator


class IndexIn(BaseModel):  # noqa: D101

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="forbid",
    )

    indices: list[Hashable] | None

    @field_validator("indices")
    @classmethod
    def _indices_must_be_list_of_hashables_unique(
        cls, v: list[Hashable] | None
    ) -> list[Hashable] | None:
        if v is None:
            return v
        if not isinstance(v, list):
            raise TypeError("indices must be a list of Hashable items.")
        for item in v:
            if not isinstance(item, Hashable):
                raise TypeError("All items in 'indices' must be Hashable.")
        if len(v) != len(set(v)):
            raise ValueError("All items in 'indices' must be unique.")
        return v
