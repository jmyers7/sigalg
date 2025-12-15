"""Feature embeddings of sample spaces.

This module provides the `FeatureEmbedding` class, which represents a feature embedding `X: Omega -> S`, where `Omega` is a sample space (domain) and `S` is a feature space (codomain).

Classes
-------
FeatureEmbedding
    Represents a feature embedding function mapping sample points to feature vectors.
FeatureEmbeddingMethods
    Mixin providing feature embedding methods to other classes.

Examples
--------
>>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
>>> Omega = SampleSpace(["s0", "s1", "s2"])
>>> X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
>>> Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
>>> embedding = FeatureEmbedding(random_variables=[X, Y])
>>> embedding.shape
(3, 2)
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base.sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.index import Index
    from ..random_objects.random_variable import RandomVariable
    from .featurized_probability_space import FeaturizedProbabilitySpace
    from .sample_point_features import SamplePointFeatures


class FeatureEmbedding(SampleSpaceMethods):
    """A feature embedding function mapping sample points to feature vectors.

    A `FeatureEmbedding` represents a function `X: Omega -> S` where `Omega` is a
    sample space (the domain) and `S` is a feature space (the codomain). Each
    sample point in `Omega` is mapped to a vector of feature values. The feature
    embedding can be constructed from a `list[RandomVariable]` (where each random
    variable represents one feature dimension) or from a `pd.DataFrame` directly.

    Parameters
    ----------
    random_variables : list[RandomVariable], optional
        `list[RandomVariable]` where each random variable represents one feature.
        All random variables must share the same domain. Mutually exclusive with `values`.
    feature_index : Index, optional
        Index of feature names. If `None` and `random_variables` is provided,
        feature names are taken from the random variable names.
    values : pd.DataFrame, optional
        DataFrame where rows are sample points and columns are features.
        Mutually exclusive with `random_variables`.
    domain_name : str, optional
        Name for the domain sample space. If `None`, defaults to "Omega" when
        constructing from `values`, or uses the domain name from `random_variables`.
    name : str, default="X"
        Name identifier for the feature embedding.

    Raises
    ------
    ValueError
        If both `random_variables` and `values` are provided, or if neither is provided.
        If `feature_index` length does not match `random_variables` length.
    TypeError
        If `random_variables` is not a `list[RandomVariable]`, `feature_index` is not
        an `Index`, `values` is not a `pd.DataFrame`, or `name` is not a string.

    Examples
    --------
    >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
    >>> # Construction from random variables
    >>> Omega = SampleSpace(["s0", "s1", "s2"])
    >>> X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
    >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
    >>> embedding = FeatureEmbedding(random_variables=[X, Y])
    >>> embedding.shape
    (3, 2)
    >>> # Construction from DataFrame
    >>> import pandas as pd
    >>> values = pd.DataFrame([[1, 2], [3, 4]], columns=["X", "Y"])
    >>> embedding2 = FeatureEmbedding(values=values)
    >>> len(embedding2)
    2
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        random_variables: list[RandomVariable] | None = None,
        feature_index: Index | None = None,
        values: pd.DataFrame | None = None,
        domain_name: str | None = None,
        name: str = "X",
    ) -> None:
        from ..base.feature_index import FeatureIndex
        from ..base.sample_space import SampleSpace

        self._validate_parameters(
            random_variables=random_variables,
            feature_index=feature_index,
            values=values,
            domain_name=domain_name,
            name=name,
        )

        if values is not None:
            self._values = values
            self._random_variables = None
            if domain_name is None:
                domain_name = "Omega"
            self.domain_name = domain_name
            self.domain = SampleSpace(
                indices=self.values.index.to_list(),
                name=domain_name,
                values_name=self.values.index.name,
            )
            self.feature_index = FeatureIndex(
                indices=values.columns.to_list(), values_name=values.columns.name
            )
        elif random_variables is not None:
            self._values = None
            self._random_variables = random_variables
            self.domain = random_variables[0].domain
            if domain_name is None:
                self.domain_name = self.domain.name
            else:
                self.domain_name = domain_name
                self.domain.name = domain_name
            if feature_index is None:
                indices = [rv.name for rv in random_variables]
                if len(indices) != len(set(indices)):
                    raise ValueError("The names of the RVs must be unique.")
                self.feature_index = FeatureIndex(
                    indices=[rv.name for rv in random_variables]
                )
            else:
                self.feature_index = feature_index
                for pos, rv in enumerate(self.random_variables):
                    rv.name = str(self.feature_index.values[pos])
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.DataFrame:
        """Get the feature values as a `pd.DataFrame`.

        Returns a `pd.DataFrame` where rows correspond to sample points and columns
        correspond to features. If the embedding was constructed from random
        variables, the `pd.DataFrame` is computed on first access.

        Returns
        -------
        values : pd.DataFrame
            `pd.DataFrame` of feature values with sample points as rows and features as columns.
        """
        if self._values is None:
            self._values = pd.concat(
                [rv.values for rv in self.random_variables], axis=1
            )
            self._values.columns = self.feature_index
            self._values.columns.name = self.feature_index.values.name
        return self._values

    @property
    def random_variables(self) -> list[RandomVariable]:
        """Get the list of random variables representing each feature.

        Returns a `list[RandomVariable]` where each random variable corresponds to
        one feature (column) in the feature embedding. If the embedding was
        constructed from a `pd.DataFrame`, the random variables are computed on first access.

        Returns
        -------
        random_variables : list[RandomVariable]
            `list[RandomVariable]` representing each feature dimension.
        """
        from ..random_objects.random_variable import RandomVariable

        if self._random_variables is None:
            self._random_variables = [
                RandomVariable(
                    outputs=self.values[col].to_dict(),
                    domain=self.domain,
                    name=str(col),
                )
                for col in self.values.columns
            ]
        return self._random_variables

    @property
    def name(self) -> str:
        """Get the name identifier for this feature embedding.

        Returns
        -------
        name : str
            The name of this feature embedding.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set the name identifier for this feature embedding.

        Parameters
        ----------
        name : str
            New name for this feature embedding.

        Raises
        ------
        TypeError
            If `name` is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    # --------------------- array methods --------------------- #

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the feature embedding.

        Returns
        -------
        shape : tuple[int, int]
            A tuple `(n_samples, n_features)` giving the dimensions of the embedding.
        """
        return self.values.shape

    def __len__(self) -> int:
        """Return the number of sample points in the feature embedding.

        Returns
        -------
        length : int
            The number of sample points (rows) in the embedding.
        """
        return len(self.values)

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_numpy(cls, array: np.ndarray, name: str = "X") -> FeatureEmbedding:
        """Create a feature embedding from a NumPy array.

        Converts a NumPy array into a feature embedding by wrapping it in a `pd.DataFrame`. Row indices become sample point identifiers and column indices become feature names.

        Parameters
        ----------
        array : np.ndarray
            NumPy array where rows are sample points and columns are features.
        name : str, default="X"
            Name for the feature embedding.

        Returns
        -------
        embedding : FeatureEmbedding
            A feature embedding constructed from the array.

        Raises
        ------
        TypeError
            If `array` is not a NumPy ndarray.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding
        >>> import numpy as np
        >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
        >>> embedding = FeatureEmbedding.from_numpy(arr)
        >>> embedding.shape
        (3, 2)
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray.")
        values = pd.DataFrame(array)
        values.index.name = "sample"
        values.columns.name = "feature"
        return cls(values=values, name=name)

    # --------------------- data access methods --------------------- #

    def get_sample_features(self, sample_index: Hashable) -> SamplePointFeatures:
        """Get the feature vector for a specific sample point.

        Returns a `SamplePointFeatures` object containing the feature values for
        the specified sample point. This represents the evaluation of the feature
        embedding function at a single point: `X(sample_index)`.

        Parameters
        ----------
        sample_index : Hashable
            Index of the sample point in the domain.

        Returns
        -------
        features : SamplePointFeatures
            Feature values for the specified sample point.

        Raises
        ------
        ValueError
            If `sample_index` is not found in the domain.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=Omega, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> features = embedding.get_sample_features("s0")
        >>> list(features)
        [1, 2]
        """
        from .sample_point_features import SamplePointFeatures

        if sample_index not in self.domain:
            raise ValueError(f"Sample index {sample_index} not found in domain.")
        return SamplePointFeatures.from_feature_embedding(
            sample_index=sample_index,
            feature_embedding=self,
        )

    @property
    def get_sample_features_at(self):
        """Get an indexer for sample point features.

        Returns an indexer allowing access to `SamplePointFeatures` by position.

        Returns
        -------
        feature_indexer : _SampleFeaturesIndexer
            Indexer for accessing `SamplePointFeatures` by position.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=Omega, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> features = embedding.get_sample_features_at[0]
        >>> list(features)
        [1, 2]
        """
        return self._SampleFeaturesIndexer(self)

    class _SampleFeaturesIndexer:
        def __init__(self, feature_embedding) -> None:
            self.feature_embedding = feature_embedding

        def __getitem__(self, key: int) -> SamplePointFeatures:
            from .sample_point_features import SamplePointFeatures

            features = self.feature_embedding.values.iloc[key]
            return SamplePointFeatures.from_feature_embedding(
                sample_index=features.name, feature_embedding=self.feature_embedding
            )

    def iter_sample_features(self):
        """Iterate over sample points and their feature vectors.

        Yields tuples of `(sample_index, SamplePointFeatures)` for each sample
        point in the domain, allowing iteration over the feature embedding function's
        entire domain.

        Yields
        ------
        sample_index : Hashable
            Index of the sample point.
        features : SamplePointFeatures
            Feature values for the sample point.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
        >>> embedding = FeatureEmbedding(random_variables=[X])
        >>> for idx, features in embedding.iter_sample_features():
        ...     print(idx, features.feature_at[0])
        s0 1
        s1 3
        """
        for sample_index in self.values.index:
            yield sample_index, self.get_sample_features(sample_index)

    def get_event_features(
        self, event_indices: list[Hashable], name: str = "A"
    ) -> FeatureEmbedding:
        """Get features restricted to a specific event (subset of sample points).

        Creates a new feature embedding containing only the sample points specified
        in `event_indices`. This represents the restriction of the feature embedding
        function to a subset of the domain.

        Parameters
        ----------
        event_indices : list[Hashable]
            `list[Hashable]` of sample point indices defining the event.
        name : str, default="A"
            Name for the restricted domain.

        Returns
        -------
        embedding : FeatureEmbedding
            A new feature embedding restricted to the specified event.

        Raises
        ------
        ValueError
            If any index in `event_indices` is not found in the domain.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1", "s2"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
        >>> embedding = FeatureEmbedding(random_variables=[X])
        >>> event_embed = embedding.get_event_features(["s0", "s1"])
        >>> len(event_embed)
        2
        """
        for idx in event_indices:
            if idx not in self.domain:
                raise ValueError(f"Sample index {idx} not found in sample_space.")

        event_features = FeatureEmbedding(
            values=self.values.loc[event_indices], name=self.name
        )
        event_features.domain.name = name
        return event_features

    @property
    def get_event_features_at(self):
        """Get an indexer for event features.

        Returns an indexer allowing access to `FeatureEmbedding` indexed by position. Includes an optional name for the event.

        Returns
        -------
        event_indexer : _EventIndexer

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1", "s2"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> # Index with a list of positions and a name
        >>> event_embedding1 = embedding.get_event_features_at[[0, 1], "MyEvent"]
        >>> event_embedding1.shape
        (2, 2)
        >>> event_embedding1.domain.name
        'MyEvent'
        >>> # Index with a slice and keep default name
        >>> event_embedding2 = embedding.get_event_features_at[:2]
        >>> event_embedding2.shape
        (2, 2)
        >>> event_embedding2.domain.name
        'A'
        """
        return self._EventIndexer(self)

    class _EventIndexer:
        def __init__(self, feature_embedding) -> None:
            self.feature_embedding = feature_embedding

        def __getitem__(self, key) -> FeatureEmbedding:
            # TODO: Need to check proper validation of key
            if isinstance(key, tuple) and len(key) == 2:
                index_key, name = key
            else:
                index_key = key
                name = "A"

            event = self.feature_embedding.domain[index_key, name]
            event_indices = event.values.to_list()
            return self.feature_embedding.get_event_features(
                event_indices=event_indices,
                name=name,
            )

    def get_feature_rv(self, key: Hashable) -> RandomVariable:
        """Get a random variable corresponding to a specific feature.

        Returns the random variable representing one feature (column) of the
        feature embedding. Each feature can be viewed as a random variable
        defined on the domain sample space.

        Parameters
        ----------
        key : Hashable
            Feature index (column name) to retrieve.

        Returns
        -------
        rv : RandomVariable
            The random variable representing the specified feature.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=Omega, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> X_from_embedding = embedding.get_feature_rv("X")
        >>> X_from_embedding == X
        True
        """
        idx_pos = self.feature_index.values.get_loc(key)
        return self.random_variables[idx_pos]

    def get_sub_embedding(self, feature_indices: list[Hashable]) -> FeatureEmbedding:
        """Get a subembedding from the feature embedding.

        Creates a new subembedding containing only the specified features,
        while keeping all sample points from the domain.

        Parameters
        ----------
        feature_indices : list[Hashable]
            `list[Hashable]` of feature indices to include.

        Returns
        -------
        embedding : FeatureEmbedding
            A new feature embedding with only the specified features.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=Omega, name="Y")
        >>> Z = RandomVariable(outputs={"s0": 0, "s1": 1}, domain=Omega, name="Z")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y, Z])
        >>> sub_embed = embedding.get_sub_embedding(["X", "Z"])
        >>> sub_embed.shape
        (2, 2)
        """
        values = self.values[feature_indices]
        return FeatureEmbedding(values=values, name=self.name + "_sub")

    # --------------------- call methods --------------------- #

    def __call__(self, key: Hashable | list[Hashable] | Event) -> FeatureEmbedding:
        """Evaluate the feature embedding at sample points or events.

        Depending on the type of `key`, this method either retrieves the feature
        vector for a specific sample point, or returns a new feature embedding
        for an event in the domain sample space. In the latter case, the underlying event is assigned the default name `A`. If the user wishes a different name, use `get_event_features` instead.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Event
            If `key` is a `Hashable`, returns the feature vector for that sample point.
            If `key` is a `list[Hashable]` or an `Event`, returns a new feature embedding restricted to those sample points.

        Raises
        ------
        ValueError
            If none of the indices is found in the domain of the embedding.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, RandomVariable, SampleSpace
        >>> Omega = SampleSpace(["s0", "s1", "s2"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> # Call on a single sample index
        >>> list(embedding("s0"))
        [1, 2]
        >>> # Call on a list of sample indices
        >>> embedding(["s0", "s1"]).shape
        (2, 2)
        >>> # Call on an event
        >>> A = Omega.get_event(["s0", "s2"])
        >>> embedding(A).shape
        (2, 2)
        """
        from ..base.event import Event

        # TODO: Need to check proper validation of key
        if isinstance(key, list):
            return self.get_event_features(event_indices=key)
        elif isinstance(key, Event):
            event_indices = list(key)
            return self.get_event_features(event_indices=event_indices, name=key.name)
        else:
            return self.get_sample_features(sample_index=key)

    # --------------------- apply methods --------------------- #

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        """Apply a function to the features of each sample point.

        Applies the given function to each sample point's feature vector,
        returning a `pd.Series` of results indexed by sample points.

        Parameters
        ----------
        function : Callable[[SamplePointFeatures], any]
            Function that takes a `SamplePointFeatures` object and returns a value.

        Returns
        -------
        results : pd.Series
            Series of function results indexed by sample points.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
        >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=Omega, name="Y")
        >>> embedding = FeatureEmbedding(random_variables=[X, Y])
        >>> sums = embedding.apply_to_features(lambda f: f.sum())
        >>> list(sums)
        [3, 7]
        """
        from .sample_point_features import SamplePointFeatures

        def wrapper(row):
            sp = SamplePointFeatures(name=row.name, values=row)
            return function(sp)

        return self.values.apply(wrapper, axis=1)

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        """Check equality with another feature embedding.

        Two feature embeddings are equal if they have the same domain, values,
        and name.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `FeatureEmbedding` with identical
            domain, values, and name, `False` otherwise.
        """
        if not isinstance(other, FeatureEmbedding):
            return False
        return (
            self.domain == other.domain
            and self.values.equals(other.values)
            and self.name == other.name
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the feature embedding.

        Returns
        -------
        repr_str : str
            A string showing the feature embedding name and DataFrame representation.
        """
        return f"Feature embedding {self.name}:\n{self.values}"

    # --------------------- probability methods --------------------- #

    def add_probability_measure_from_features(
        self, pmf: Callable[[SamplePointFeatures], Real]
    ) -> FeaturizedProbabilitySpace:
        """Create a featurized probability space with a probability measure defined by features.

        Creates a `FeaturizedProbabilitySpace` `(Omega, F, P, X)` by defining a probability measure `P` using a function of the features. The function `pmf` takes the features of a sample point and returns its probability.

        Parameters
        ----------
        pmf : Callable[[SamplePointFeatures], Real]
            Function mapping sample point features to probability values.
            Must return non-negative values that sum to 1.

        Returns
        -------
        featurized_space : FeaturizedProbabilitySpace
            A featurized probability space `(Omega, F, P, X)` with the specified
            probability measure.

        Examples
        --------
        >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
        >>> embedding = FeatureEmbedding(random_variables=[X])
        >>> fps = embedding.add_probability_measure_from_features(lambda f: 0.5)
        >>> fps.P("s0")
        0.5
        """
        from ..base import ProbabilitySpace
        from ..probability_measures import ProbabilityMeasure
        from .featurized_probability_space import FeaturizedProbabilitySpace

        probabilities = {
            sample_index: pmf(sample_features)
            for sample_index, sample_features in self.iter_sample_features()
        }
        probability_measure = ProbabilityMeasure(
            sample_space=self.domain, probabilities=probabilities
        )
        probability_space = ProbabilitySpace(
            sample_space=self.domain,
            probability_measure=probability_measure,
        )
        return FeaturizedProbabilitySpace(
            sample_space=self.domain,
            sigma_algebra=probability_space.sigma_algebra,
            probability_measure=probability_measure,
            feature_embedding=self,
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        random_variables: list[RandomVariable],
        feature_index: Index | None,
        values: pd.DataFrame | None = None,
        domain_name: str | None = None,
        name: str | None = None,
    ) -> None:
        """Validate feature embedding construction parameters.

        Parameters
        ----------
        random_variables : list[RandomVariable]
            `list[RandomVariable]` to validate.
        feature_index : Index, optional
            Feature index to validate.
        values : pd.DataFrame, optional
            Values DataFrame to validate.
        domain_name : str, optional
            Domain name to validate.
        name : str, optional
            Name to validate.

        Raises
        ------
        ValueError
            If both `random_variables` and `values` are provided, or if neither is provided.
            If `feature_index` length does not match `random_variables` length.
        TypeError
            If `random_variables` is not a `list[RandomVariable]`, `feature_index` is not
            an `Index`, `values` is not a `pd.DataFrame`, `name` is not a string, or
            `domain_name` is not a string.
        """
        from ..base.index import Index
        from ..random_objects.random_variable import RandomVariable

        if (
            random_variables is not None or feature_index is not None
        ) and values is not None:
            raise ValueError(
                "Cannot specify both random_variables/feature_index and values."
            )
        if random_variables is None and values is None:
            raise ValueError("Must specify either random_variables or values.")
        if feature_index is not None and not isinstance(feature_index, Index):
            raise TypeError("feature_index must be an Index instance.")
        if random_variables is not None:
            if not isinstance(random_variables, list):
                raise TypeError(
                    "random_variables must be a list of RandomVariable instances."
                )
            if not all(isinstance(rv, RandomVariable) for rv in random_variables):
                raise TypeError(
                    "All elements in random_variables must be instances of RandomVariable."
                )
            if feature_index is not None and len(feature_index) != len(
                random_variables
            ):
                raise ValueError(
                    "feature_index and random_variables must have the same length."
                )
        if values is not None and not isinstance(values, pd.DataFrame):
            raise TypeError("values must be a pandas DataFrame.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if domain_name is not None and not isinstance(domain_name, str):
            raise TypeError("domain_name must be a string.")


class FeatureEmbeddingMethods:
    """Mixin class providing feature embedding methods to other classes.

    This mixin provides convenience methods for classes that have a `feature_embedding`
    attribute, allowing them to delegate feature embedding operations to that attribute.
    The class assumes the implementing class has a `feature_embedding` attribute that
    is a `FeatureEmbedding` instance.

    Examples
    --------
    >>> from sigalg.core import FeatureEmbedding, SampleSpace, RandomVariable
    >>> class MyClass(FeatureEmbeddingMethods):
    ...     def __init__(self, embedding):
    ...         self.feature_embedding = embedding
    >>> Omega = SampleSpace(["s0", "s1"])
    >>> X = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=Omega, name="X")
    >>> embedding = FeatureEmbedding(random_variables=[X])
    >>> obj = MyClass(embedding)
    >>> features = obj.get_sample_features("s0")
    >>> int(features.feature_at[0])
    1
    """

    def get_sample_features(self, sample_index: Hashable) -> SamplePointFeatures:
        """Get the feature vector for a specific sample point.

        Delegates to `feature_embedding.get_sample_features`.

        Parameters
        ----------
        sample_index : Hashable
            Index of the sample point in the domain.

        Returns
        -------
        features : SamplePointFeatures
            Feature values for the specified sample point.
        """
        return self.feature_embedding.get_sample_features(sample_index)

    def get_event_features(self, event_indices: list[Hashable]) -> FeatureEmbedding:
        """Get features restricted to a specific event.

        Delegates to `feature_embedding.get_event_features`.

        Parameters
        ----------
        event_indices : list[Hashable]
            `list[Hashable]` of sample point indices defining the event.

        Returns
        -------
        embedding : FeatureEmbedding
            A new feature embedding restricted to the specified event.
        """
        return self.feature_embedding.get_event_features(event_indices)

    @property
    def get_sample_features_at(self):
        """Get indexer for accessing sample features by position.

        Returns
        -------
        indexer : _SampleFeaturesIndexer
            Indexer for positional access to sample features.
        """
        return self.feature_embedding._SampleFeaturesIndexer(self.feature_embedding)

    @property
    def get_event_features_at(self):
        """Get indexer for accessing event features by position.

        Returns
        -------
        indexer : _EventIndexer
            Indexer for positional access to event features.
        """
        return self.feature_embedding._EventIndexer(self.feature_embedding)

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        """Get a random variable corresponding to a specific feature.

        Delegates to `feature_embedding.get_feature_rv`.

        Parameters
        ----------
        feature_index : Hashable
            Feature index to retrieve.

        Returns
        -------
        rv : RandomVariable
            The random variable representing the specified feature.
        """
        return self.feature_embedding.get_feature_rv(feature_index)

    def get_sub_embedding(self, feature_indices: list[Hashable]):
        """Get a subembedding from the feature embedding.

        Delegates to `feature_embedding.get_sub_embedding`.

        Parameters
        ----------
        feature_indices : list[Hashable]
            `list[Hashable]` of feature indices to include.

        Returns
        -------
        embedding : FeatureEmbedding
            A new feature embedding with only the specified features.
        """
        return self.feature_embedding.get_sub_embedding(feature_indices)

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        """Apply a function to the features of each sample point.

        Delegates to `feature_embedding.apply_to_features`.

        Parameters
        ----------
        function : Callable[[SamplePointFeatures], any]
            Function that takes a `SamplePointFeatures` object and returns a value.

        Returns
        -------
        results : pd.Series
            Series of function results indexed by sample points.
        """
        return self.feature_embedding.apply_to_features(function)

    def iter_sample_features(self):
        """Iterate over sample points and their feature vectors.

        Delegates to `feature_embedding.iter_sample_features`.

        Yields
        ------
        sample_index : Hashable
            Index of the sample point.
        features : SamplePointFeatures
            Feature values for the sample point.
        """
        return self.feature_embedding.iter_sample_features()
