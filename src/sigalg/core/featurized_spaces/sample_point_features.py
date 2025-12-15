"""Feature vectors for individual sample points.

This module provides the `SamplePointFeatures` class, which represents the feature
vector for a single sample point. Given a feature embedding function `X: Omega -> S`,
a `SamplePointFeatures` object represents `X(omega)` for a specific `omega` in the
sample space.

Classes
-------
SamplePointFeatures
    Represents the feature vector for a single sample point.

Examples
--------
>>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
>>> domain = SampleSpace(["s0", "s1"])
>>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="X")
>>> Y = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=domain, name="Y")
>>> embedding = FeatureEmbedding(random_variables=[X, Y])
>>> features = embedding.get_sample_features("s0")
>>> features.values.tolist()
[1, 2]
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .feature_embedding import FeatureEmbedding


class SamplePointFeatures:
    """Feature vector for a single sample point.

    A `SamplePointFeatures` object represents the feature values for one sample
    point from a feature embedding. Given a feature embedding function `X: Omega -> S`,
    this represents the output `X(omega)` for a specific sample point `omega`.

    Parameters
    ----------
    values : pd.Series
        Series of feature values for the sample point. The series name must
        match the `name` parameter.
    name : Hashable
        Identifier for the sample point.

    Raises
    ------
    TypeError
        If `values` is not a `pd.Series` or `name` is not hashable.
    ValueError
        If `values.name` does not match the provided `name`.

    Examples
    --------
    >>> from sigalg.core import SamplePointFeatures
    >>> import pandas as pd
    >>> values = pd.Series([1, 2, 3], index=["X", "Y", "Z"], name="s0")
    >>> features = SamplePointFeatures(values=values, name="s0")
    >>> len(features)
    3
    >>> features.sum()
    6
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        values: pd.Series,
        name: Hashable,
    ) -> None:
        """Initialize sample point features.

        Parameters
        ----------
        values : pd.Series
            Series of feature values.
        name : Hashable
            Sample point identifier.
        """
        self._validate_parameters(values=values, name=name)
        self.values = values
        self._name = name
        self.feature_embedding = None

    # --------------------- properties --------------------- #

    @property
    def name(self) -> Hashable:
        """Get the sample point identifier.

        Returns
        -------
        name : Hashable
            The identifier for this sample point.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the sample point identifier.

        Parameters
        ----------
        name : Hashable
            New identifier for this sample point.

        Raises
        ------
        TypeError
            If `name` is not hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        self._name = name
        self.values.name = name

    # --------------------- access & iter methods --------------------- #

    @property
    def feature_at(self) -> _iLocIndexer:
        """Get indexer for positional access to features.

        Returns
        -------
        indexer : _iLocIndexer
            Indexer for accessing features by integer position.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> domain = SampleSpace(["s0"])
        >>> X = RandomVariable(outputs={"s0": 1}, domain=domain, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2}, domain=domain, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> features = embedding.get_sample_features("s0")
        >>> features.feature_at[0]
        1
        """
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]):
            return self.parent.values.iloc[key]

    def __iter__(self) -> iter:
        """Iterate over feature values.

        Yields
        ------
        value : Any
            Each feature value in order.
        """
        return iter(self.values)

    def __len__(self) -> int:
        """Return the number of features.

        Returns
        -------
        length : int
            The number of features for this sample point.
        """
        return len(self.values)

    def sum(self) -> Any:
        """Return the sum of all feature values.

        Returns
        -------
        total : Any
            The sum of all feature values.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> domain = SampleSpace(["s0"])
        >>> X = RandomVariable(outputs={"s0": 1}, domain=domain, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2}, domain=domain, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> features = embedding.get_sample_features("s0")
        >>> features.sum()
        3
        """
        return self.values.sum()

    # --------------------- class methods --------------------- #

    @classmethod
    def from_feature_embedding(
        cls,
        sample_index: Hashable,
        feature_embedding: FeatureEmbedding,
    ) -> SamplePointFeatures:
        """Create sample point features from a feature embedding.

        Extracts the feature vector for a specific sample point from a feature
        embedding. This represents evaluating the feature embedding function at
        a single point: `X(sample_index)`.

        Parameters
        ----------
        sample_index : Hashable
            Index of the sample point.
        feature_embedding : FeatureEmbedding
            The feature embedding to extract features from.

        Returns
        -------
        features : SamplePointFeatures
            Feature values for the specified sample point.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> from sigalg.core import SamplePointFeatures
        >>> domain = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="X")
        >>> embedding = FeatureEmbedding(random_variables=[X])
        >>> features = SamplePointFeatures.from_feature_embedding("s0", embedding)
        >>> features.values[0]
        1
        """
        values = feature_embedding.values.loc[sample_index]
        spf = cls(values=values, name=sample_index)
        spf.feature_embedding = feature_embedding
        # spf.values.index.name = "feature"
        return spf

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the sample point features.

        Returns
        -------
        repr_str : str
            A string showing the sample point name and feature values.
        """
        return f"Sample features of '{self.name}':\n{self.values.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        """Check equality with another sample point features object.

        Two sample point features are equal if they have the same name and values.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `SamplePointFeatures` with identical
            name and values, `False` otherwise.
        """
        if not isinstance(other, SamplePointFeatures):
            return False
        return self.name == other.name and self.values.equals(other.values)

    # --------------------- validation methods --------------------- #

    def _validate_parameters(
        self,
        values: pd.Series,
        name: Hashable,
    ) -> None:
        """Validate sample point features construction parameters.

        Parameters
        ----------
        values : pd.Series
            Series to validate.
        name : Hashable
            Name to validate.

        Raises
        ------
        TypeError
            If `values` is not a `pd.Series` or `name` is not hashable.
        ValueError
            If `values.name` does not match the provided `name`.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        if not isinstance(values, pd.Series):
            raise TypeError("values must be a pandas Series.")
        if values.name != name:
            raise ValueError("values.name must match the given name.")
