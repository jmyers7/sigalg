from __future__ import annotations  # noqa: D100

from collections.abc import Hashable

from pydantic import BaseModel, ConfigDict, field_validator


class IndexIn(BaseModel):  # noqa: D101

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="forbid",
    )

    indices: list[Hashable]
    name: Hashable | None = None
    data_name: Hashable | None = None

    @field_validator("indices")
    @classmethod
    def _indices_must_be_list_of_hashables_unique(
        cls, v: list[Hashable]
    ) -> list[Hashable]:
        if not isinstance(v, list):
            raise TypeError("indices must be a list of Hashable items.")
        for item in v:
            if not isinstance(item, Hashable):
                raise TypeError("All items in 'indices' must be Hashable.")
        if len(v) != len(set(v)):
            raise ValueError("All items in 'indices' must be unique.")
        return v

    @field_validator("name", "data_name")
    @classmethod
    def _names_must_be_hashable(cls, v: Hashable | None) -> Hashable | None:
        if v is not None and not isinstance(v, Hashable):
            raise TypeError("name/data_name must be hashable.")
        return v
