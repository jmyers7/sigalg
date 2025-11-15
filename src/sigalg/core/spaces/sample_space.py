import pandas as pd
from collections.abc import Sequence


class SampleSpace(Sequence):
    def __init__(self, indices):
        from ..sigma_algebras import SigmaAlgebra

        self._validate_indices(indices)
        self._index = pd.Index(indices)
        self._sigma_algebra = SigmaAlgebra(
            sample_space=self,
            atom_ids={index: idx for idx, index in enumerate(self._index)},
        )

    @property
    def index(self):
        return self._index

    @property
    def sigma_algebra(self):
        return self._sigma_algebra

    def set_sigma_algebra(self, sigma_algebra):
        from ..sigma_algebras import SigmaAlgebra

        if not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
        if sigma_algebra._sample_space != self:
            raise ValueError(
                "sigma_algebra must have the same sample_space as this SampleSpace."
            )
        self._sigma_algebra = sigma_algebra

    @property
    def sample_space(self):
        return self

    def __len__(self):
        return len(self._index)

    def __getitem__(self, key):
        if isinstance(key, list):
            from .event import Event

            return Event(sample_space=self, event_indices=key)
        return self._index[key]

    def __iter__(self):
        return iter(self._index)

    def __repr__(self):
        return f"SampleSpace({list(self._index)})"

    def __hash__(self):
        if not hasattr(self, "_cached_hash"):
            self._cached_hash = hash(tuple(self._index))
        return self._cached_hash

    def __eq__(self, other):
        return isinstance(other, SampleSpace) and self._index.equals(other._index)

    def add_probability_measure(self, prob_measure):
        from .probability_space import ProbabilitySpace

        return ProbabilitySpace(self.index, prob_measure.probabilities.to_dict())

    @staticmethod
    def _validate_indices(indices):
        if len(indices) != len(set(indices)):
            raise ValueError(
                "Sample space indices must be unique (no duplicates allowed)."
            )
