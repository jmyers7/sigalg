from __future__ import annotations

from collections.abc import Hashable
from typing import Annotated, Any

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema


class _IndexLikeValidator:
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v: Any) -> pd.Index:
        if isinstance(v, (pd.MultiIndex, pd.Index)):
            if v.nunique() != len(v):
                raise ValueError("index must not contain duplicate values")
            return v.copy()

        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple):
            if not all(isinstance(x, tuple) for x in v):
                raise ValueError(
                    "If the list contains tuples, all elements must be tuples"
                )
            lengths = {len(x) for x in v}
            if len(lengths) != 1:
                raise ValueError("All tuples must have the same length")
            if len(set(v)) != len(v):
                raise ValueError("index must not contain duplicate values")
            return pd.MultiIndex.from_tuples(v)

        elif isinstance(v, list):
            if not all(isinstance(x, Hashable) for x in v):
                raise ValueError("All elements in the index must be Hashable")
            if len(set(v)) != len(v):
                raise ValueError("index must not contain duplicate values")
            return pd.Index(v)

        else:
            raise ValueError("Expected list[Hashable], list[tuple], or pd.Index")


IndexLike = Annotated[pd.Index, _IndexLikeValidator]


class IndexIn(BaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    indices: IndexLike | None = None
    name: Hashable
    variable_names: list[Hashable] | None = None

    @field_validator("variable_names", mode="before")
    @classmethod
    def validate_variable_names(cls, v: list[Hashable] | None) -> list[Hashable] | None:  # noqa: D102
        if v is not None:
            if not isinstance(v, list):
                raise TypeError("variable_names must be a list of Hashable or `None`.")
            if not all(isinstance(x, Hashable) for x in v):
                raise TypeError("All elements in variable_names must be Hashable.")
            if len(set(v)) != len(v):
                raise ValueError("variable_names must not contain duplicate values.")
        return v

    @model_validator(mode="after")
    def generate_variable_names(self) -> IndexIn:  # noqa: D102
        if self.indices is not None:
            if isinstance(self.indices, pd.MultiIndex):
                if (
                    set(self.indices.names) != {None}
                    and self.variable_names is not None
                ):
                    if list(self.indices.names) != self.variable_names:
                        raise ValueError(
                            "The variable names must match the names of the index dimensions."
                        )

                if set(self.indices.names) != {None} and self.variable_names is None:
                    self.variable_names = list(self.indices.names)

                if (
                    set(self.indices.names) == {None}
                    and self.variable_names is not None
                ):
                    if len(self.variable_names) != self.indices.nlevels:
                        raise ValueError(
                            "The number of variable names must match the number of dimensions in the index."
                        )
                    self.indices.names = self.variable_names

                if set(self.indices.names) == {None} and self.variable_names is None:
                    self.variable_names = [
                        f"{self.name}_{i}" for i in range(self.indices.nlevels)
                    ]
                    self.indices.names = self.variable_names

            else:
                if self.indices.name is not None and self.variable_names is not None:
                    if self.indices.name != self.variable_names[0]:
                        raise ValueError(
                            "The variable name must match the name of underlying pd.Index."
                        )

                if self.indices.name is not None and self.variable_names is None:
                    self.variable_names = [self.indices.name]

                if self.indices.name is None and self.variable_names is not None:
                    if len(self.variable_names) != 1:
                        raise ValueError(
                            "There must be exactly one variable name for a non-pd.MultiIndex."
                        )
                    self.indices.name = self.variable_names[0]

                if self.indices.name is None and self.variable_names is None:
                    self.variable_names = [self.name]
                    self.indices.name = self.name

        return self
