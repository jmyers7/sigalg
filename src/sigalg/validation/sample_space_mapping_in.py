from __future__ import annotations  # noqa: D100

from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from ..core.base.index import Index
from ..core.base.sample_space import SampleSpace

if TYPE_CHECKING:
    from collections.abc import dict_keys, dict_values

    import numpy as np


class SampleSpaceMappingIn(BaseModel):  # noqa: D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mapping: Mapping[Hashable, Any] | pd.DataFrame | pd.Series
    sample_space: SampleSpace | None = None
    index: Index | None = None
    kind: Literal["any", "probabilities"] = "any"

    def _validate_sample_space(self) -> None:
        if self.sample_space is not None:
            sample_space_set = set(self.sample_space)
            if set(self._mapping_keys()) != sample_space_set:
                raise ValueError(
                    "mapping must contain an entry for every sample index in sample_space."
                )
            if isinstance(self.mapping, pd.DataFrame | pd.Series):
                if isinstance(self.mapping.index, pd.MultiIndex):
                    if self.mapping.index.names != self.sample_space.variable_names:
                        raise ValueError(
                            "If the mapping index is a MultiIndex, its names must match the variable names of the sample space."
                        )

    def _mapping_keys(self) -> pd.Index | dict_keys:
        if isinstance(self.mapping, pd.DataFrame | pd.Series):
            return self.mapping.index
        else:
            return self.mapping.keys()

    # def _mapping_1d_values(self) -> np.ndarray | dict_values:
    #     if isinstance(self.mapping, pd.Series):
    #         return self.mapping.values
    #     else:
    #         return self.mapping.values()

    # def _sort_and_validate_sample_space(self) -> pd.DataFrame | pd.Series | dict:
    #     if self.sample_space is not None:
    #         sample_space_set = set(self.sample_space)
    #         if set(self._mapping_keys()) != sample_space_set:
    #             raise ValueError(
    #                 "mapping must contain an entry for every sample index in sample_space."
    #             )
    #         if isinstance(self.mapping, pd.DataFrame | pd.Series):
    #             return self.mapping.reindex(self.sample_space.data)
    #         else:
    #             return {key: self.mapping[key] for key in self.sample_space}
    #     else:
    #         return self.mapping

    # def _sort_and_validate_index(self) -> pd.DataFrame | pd.Series | dict:
    #     if self.index is not None and isinstance(self.mapping, pd.DataFrame):
    #         index_set = set(self.index)
    #         if set(self.mapping.columns) != index_set:
    #             raise ValueError(
    #                 "If the mapping is a data frame, its columns must match the provided index."
    #             )
    #         return self.mapping.reindex(columns=self.index.data)
    #     else:
    #         return self.mapping

    # def _generate_sample_space(self) -> SampleSpace:
    #     if self.sample_space is None:
    #         if isinstance(self.mapping, pd.Series | pd.DataFrame):
    #             name = (
    #                 self.mapping.index.name
    #                 if self.mapping.index.name is not None
    #                 else "Omega"
    #             )
    #             sample_space = SampleSpace(name=name).from_pandas(self.mapping.index)
    #             self.mapping.index = sample_space.data
    #             return sample_space
    #         else:
    #             return SampleSpace().from_list(list(self.mapping.keys()))
    #     else:
    #         return self.sample_space

    # def _generate_index(self) -> Index:
    #     if self.index is None:
    #         if isinstance(self.mapping, pd.DataFrame):
    #             name = (
    #                 self.mapping.columns.name
    #                 if self.mapping.columns.name is not None
    #                 else "I"
    #             )
    #             index = Index(name=name).from_pandas(self.mapping.columns)
    #             self.mapping.columns = index.data
    #             return index
    #         else:
    #             return None
    #     else:
    #         return self.index

    # @model_validator(mode="after")
    # def _validate_dict_mapping(self):
    #     if isinstance(self.mapping, dict):
    #         self.mapping = self._sort_and_validate_sample_space()
    #         self.mapping = self._sort_and_validate_index()
    #         self.sample_space = self._generate_sample_space()
    #         self.index = self._generate_index()

    #     return self

    # @model_validator(mode="after")
    # def _validate_series_mapping(self):
    #     if isinstance(self.mapping, pd.Series):
    #         self.mapping = self._sort_and_validate_sample_space()
    #         self.mapping = self._sort_and_validate_index()
    #         self.sample_space = self._generate_sample_space()
    #         self.index = self._generate_index()

    #     return self

    # @model_validator(mode="after")
    # def _validate_dataframe_mapping(self):
    #     if isinstance(self.mapping, pd.DataFrame):
    #         self.mapping = self._sort_and_validate_sample_space()
    #         self.mapping = self._sort_and_validate_index()
    #         self.sample_space = self._generate_sample_space()
    #         self.index = self._generate_index()

    #     return self

    # @model_validator(mode="after")
    # def _validate_consistency(self):

    #     if self.kind == "probabilities" and isinstance(self.mapping, pd.DataFrame):
    #         raise ValueError(
    #             "DataFrame mappings are not supported when kind='probabilities'."
    #         )

    #     if self.kind == "probabilities":
    #         mapping_values = self._mapping_1d_values()
    #         for value in mapping_values:
    #             if not isinstance(value, Real):
    #                 raise TypeError("All values in the mapping must be numeric.")
    #             if value < 0:
    #                 raise ValueError("All values in the mapping must be non-negative.")

    #         total = sum(mapping_values)
    #         if not abs(total - 1.0) < 1e-8:
    #             raise ValueError("The values in the mapping must sum to 1.")

    #     return self
