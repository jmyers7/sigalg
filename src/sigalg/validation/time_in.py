from __future__ import annotations  # noqa: D100

from numbers import Integral, Real

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TimeIn(BaseModel):  # noqa: D101

    model_config = ConfigDict(arbitrary_types_allowed=True)

    indices: list[Real]
    is_discrete: bool = True

    @field_validator("indices")
    @classmethod
    def _validate_indices(cls, v: list[Real]) -> list[Real]:
        if not isinstance(v, list) or not all(isinstance(x, Real) for x in v):
            raise TypeError("indices must be a list of real numbers.")
        if len(v) == 0:
            raise ValueError("indices cannot be empty.")
        for curr_time, next_time in zip(v[:-1], v[1:]):
            if curr_time > next_time:
                raise ValueError("indices must be in ascending order.")
        return v

    @model_validator(mode="after")
    def _validate_discreteness(self) -> TimeIn:
        if self.is_discrete:
            if not all(
                isinstance(x, Integral) and not isinstance(x, bool)
                for x in self.indices
            ):
                raise ValueError("indices must be integers when is_discrete=True.")
        return self
