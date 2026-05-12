from __future__ import annotations  # noqa: D100

from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from ..core.base.sample_space import SampleSpace

if TYPE_CHECKING:
    from collections.abc import dict_keys, dict_values

    import numpy as np


class SampleSpaceMappingIn(BaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mapping: Mapping[Hashable, Any] | pd.DataFrame | pd.Series
    sample_space: SampleSpace | None
    kind: Literal["any", "probabilities"] = "any"

    def _mapping_keys(self) -> pd.Index | dict_keys:
        if isinstance(self.mapping, pd.DataFrame | pd.Series):
            return self.mapping.index
        else:
            return self.mapping.keys()

    def _mapping_1d_values(self) -> np.ndarray | dict_values:
        if isinstance(self.mapping, pd.Series):
            return self.mapping.values
        else:
            return self.mapping.values()

    def _sort_mapping(self) -> pd.DataFrame | pd.Series | dict:
        if isinstance(self.mapping, pd.DataFrame | pd.Series):
            return self.mapping.reindex(self.sample_space.data)
        else:
            return {key: self.mapping[key] for key in self.sample_space}

    @model_validator(mode="after")
    def _validate_consistency(self) -> SampleSpaceMappingIn:
        mapping_keys = set(self._mapping_keys())
        if self.sample_space is not None:
            sample_space_set = set(self.sample_space)
            if mapping_keys != sample_space_set:
                raise ValueError(
                    "mapping must contain an entry for every sample index in sample_space."
                )
            self.mapping = self._sort_mapping()

        if self.kind == "probabilities" and isinstance(self.mapping, pd.DataFrame):
            raise ValueError(
                "DataFrame mappings are not supported when kind='probabilities'."
            )

        if self.kind == "probabilities":
            mapping_values = self._mapping_1d_values()
            for value in mapping_values:
                if not isinstance(value, Real):
                    raise TypeError("All values in the mapping must be numeric.")
                if value < 0:
                    raise ValueError("All values in the mapping must be non-negative.")

            total = sum(mapping_values)
            if not abs(total - 1.0) < 1e-8:
                raise ValueError("The values in the mapping must sum to 1.")

        return self
