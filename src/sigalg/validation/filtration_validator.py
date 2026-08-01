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

from ..core.indices.time import Time
from ..core.sigma_algebras.sigma_algebra import SigmaAlgebra


class _FiltrationLikeValidator:
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v: Any) -> pd.DataFrame:
        if isinstance(v, list):
            if not all(isinstance(sig_alg, SigmaAlgebra) for sig_alg in v):
                raise ValueError(
                    "All elements of the list must be instances of SigmaAlgebra."
                )

            domain = v[0].domain
            if not all(domain == sig_alg.domain for sig_alg in v):
                raise ValueError(
                    "All sigma-algebras must be defined on the same sample space."
                )

            names = [sig_alg.name for sig_alg in v]
            if len(names) != len(set(names)):
                raise ValueError("Cannot have duplicate names for the sigma-algebras.")

            df = pd.concat([sig_alg.data.rename(sig_alg.name) for sig_alg in v], axis=1)
            df.columns = pd.RangeIndex(start=0, stop=df.shape[1], name="index")

            return df

        elif isinstance(v, pd.DataFrame):
            return v.copy()

        else:
            raise ValueError("Expected list[SigmaAlgebra] or pd.DataFrame")


FiltrationLike = Annotated[pd.DataFrame, _FiltrationLikeValidator]


class FiltrationValidator(BaseModel):
    """Pass."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sig_algs: FiltrationLike | None = None
    index: Time | None = None
    name: Hashable | None = None

    @field_validator("sig_algs")
    @classmethod
    def validate_sig_algs(cls, v: FiltrationLike | None) -> FiltrationLike | None:  # noqa: D102
        if v is not None:
            for curr_alg, next_alg in zip(v.columns[:-1], v.columns[1:]):
                if v.groupby(next_alg)[curr_alg].nunique().max() > 1:
                    raise ValueError(
                        "The provided sigma-algebras do not represent a valid filtration."
                    )

        return v

    @model_validator(mode="after")
    def validate_index(self) -> FiltrationValidator:  # noqa: D102
        if self.sig_algs is not None and self.index is not None:
            if not isinstance(self.index, Time):
                raise TypeError(
                    "For a filtration, the index must be an instance of Time."
                )

            default_columns = pd.RangeIndex(start=0, stop=self.sig_algs.shape[1])

            if self.sig_algs.columns.equals(default_columns):
                if len(self.index) == self.sig_algs.shape[1]:
                    self.sig_algs.columns = self.index.data
                else:
                    raise ValueError(
                        "The length of the index does not match the number of sigma-algebras."
                    )

            elif not self.sig_algs.columns.equals(self.index.data):
                raise ValueError(
                    "The index does not match the columns of the data frame."
                )

        return self

    @model_validator(mode="after")
    def generate_index(self) -> FiltrationValidator:  # noqa: D102
        if self.sig_algs is not None and self.index is None:
            self.sig_algs.columns.name = "time"
            self.index = Time(indices=self.sig_algs.columns)

        return self
