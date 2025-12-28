"""Random vector module.

This module defines the `RandomVector` class, which represents a random vector `X: Omega -> S` between two sample spaces.

Classes
-------
RandomVector
    Represents a random vector mapping between two sample spaces.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.index import Index
    from ..base.sample_space import SampleSpace
    from ..featurized_spaces.feature_vector import FeatureVector
    from ..featurized_spaces.featurized_probability_space import (
        FeaturizedProbabilitySpace,
    )
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class RandomVector:
    """A random vector.

    An instance of `RandomVector` represents a mapping `X: Omega -> S` from a sample space `Omega` to a feature space `S`. This means that the image `X(omega)` of a sample point `omega` is a tuple of features drawn from the component spaces, called the feature vector of `omega`. The number of component spaces (i.e., the length of the feature vector) is called the dimension of the random vector.

    Instances of `RandomVector` can be constructed directly from a `domain` sample space and a dictionary of `outputs`, whose keys are the sample points in the domain and whose values are the corresponding feature vectors (as tuples). Alternatively, factory methods are provided to construct a `RandomVector` from a `pd.DataFrame` or a `np.ndarray`.

    Parameters
    ----------
    outputs : Mapping[Hashable, Hashable]
        A mapping from sample points in the domain to their corresponding output vectors (e.g., tuples of feature values).
    domain : SampleSpace
        The sample space over which the random vector is defined.
    name : Hashable | None, default="X"
        The name of the random vector.

    Raises
    ------
    TypeError
        If `outputs` is not a mapping from hashable types to hashable types, or if `name` is not hashable.
    ValueError
        If `outputs` does not contain an entry for every sample ID in `domain`.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, RandomVector
    >>> domain = SampleSpace.generate_default(size=3, prefix="s", name="S")
    >>> outputs = {"s0": (0.1, 0.2), "s1": (0.3, 0.4), "s2": (0.5, 0.6)}
    >>> # Generate a 2-dimensional random vector from outputs dict
    >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
    >>> tuple(X("s0"))
    (0.1, 0.2)
    >>> X.dimension
    2
    >>> # Generate a 1-dimensional random vector from a pd.Series
    >>> import pandas as pd
    >>> data = pd.Series([10, 20, 30], index=pd.Index(["s0", "s1", "s2"], name="S"))
    >>> Y = RandomVector.from_pandas(data, name="Y")
    >>> Y # doctest: +NORMALIZE_WHITESPACE
    Random vector 'Y':
           Y
    S
    s0     10
    s1     20
    s2     30
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: Mapping[Hashable, Hashable],
        domain: SampleSpace,
        name: Hashable | None = "X",
    ) -> None:
        v = SampleSpaceMappingIn(mapping=outputs, sample_space=domain, name=name)

        self.outputs = v.mapping
        self.domain = v.sample_space
        self._name = v.name

        # caches for properties
        self._range: RandomVector | None = None
        self._range_counts: pd.Series | None = None
        self._data: pd.Series | pd.DataFrame | None = None
        self._feature_index: Index | None = None
        self._sigma_algebra: SigmaAlgebra | None = None

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series | pd.DataFrame:
        """Get the underlying pandas data structure of a random vector.

        If the random vector is of dimension 2 or greater, returns the underlying `pd.DataFrame`; otherwise, returns the underlying `pd.Series` for a random vector of dimension 1.

        The private attribute `_data` is initialized to `None` on construction and is computed lazily (and returned) via this property on first access as the `data` property. The reason for this lazy computation is so that the `from_pandas` class method can set the `data` property directly. Otherwise, if the constructor eagerly constructed the `data` property from the `outputs` dict, then a `pd.DataFrame` or `pd.Series` would be passed into `from_pandas`, converted to an `outputs` dict that is passed into the constructor, and then the constructor would build the `data` property from the dict to obtain a copy of the original `pd.DataFrame` or `pd.Series`.

        Returns
        -------
        data: pd.Series | pd.DataFrame
            The underlying `pd.Series` or `pd.DataFrame` representing the random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> outputs_2d = {"s0": (1, 2), "s1": (3, 4)}
        >>> X = RandomVector(outputs=outputs_2d, domain=Omega, name="X")
        >>> # Dataframes underlie random vectors of dimension 2 or greater
        >>> X.data # doctest: +NORMALIZE_WHITESPACE
        feature  X0  X1
        sample
        s0        1   2
        s1        3   4
        >>> outputs_1d = {"s0": 10, "s1": 20}
        >>> Y = RandomVector(outputs=outputs_1d, domain=Omega, name="Y")
        >>> # Series underlie random vectors of dimension 1
        >>> Y.data # doctest: +NORMALIZE_WHITESPACE
        sample
        s0     10
        s1     20
        Name: Y, dtype: int64
        """
        from ..base.index import Index

        if self._data is None:
            data = pd.DataFrame.from_dict(self.outputs, orient="index")
            dimension = data.shape[1]

            if dimension == 1:
                data = data.iloc[:, 0]
                data.index = self.domain.data
                data.name = self.name
                self._data = data
                self._feature_index = None
            else:
                feature_index = Index.generate_default(
                    size=dimension,
                    prefix=self.name,
                    name="feature_index",
                    data_name="feature",
                )
                data.columns = feature_index.data
                data.index.name = self.domain.data.name
                self._data = data
                self._feature_index = feature_index

        return self._data

    @data.setter
    def data(self, data: pd.Series | pd.DataFrame) -> None:
        """Set the underlying pandas data structure.

        The `data` property is not meant to be set directly by the user. This setter is provided so that the `from_pandas` factory method can set the property.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            New `pd.Series` or `pd.DataFrame` to set.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series` or `pd.DataFrame`.
        """
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("data must be a pd.Series or pd.DataFrame.")
        self._data = data

    @property
    def name(self) -> Hashable:
        """Get the name of the random vector.

        Returns
        -------
        name : Hashable
            The name of the random vector.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        self._name = name

    @property
    def feature_index(self) -> Index | None:
        """Get the feature index of a random vector of dimension 2 or greater.

        Returns
        -------
        feature_index : Index | None
            The feature index of the random vector. If the vector is 1-dimensional, returns `None`.
        """
        if self._data is None:
            _ = self.data  # triggers computation of `data` and `_feature_index`
        return self._feature_index

    @feature_index.setter
    def feature_index(self, feature_index: Index) -> None:
        from ..base.index import Index

        if self.dimension == 1:  # accessing `dimension` triggers computation of `data`
            raise ValueError(
                "Cannot set feature_index for a 1-dimensional RandomVector."
            )

        if not isinstance(feature_index, Index):
            raise TypeError("feature_index must be an Index.")
        if len(feature_index) != self.dimension:
            raise ValueError(
                "feature_index size must match the dimension of the RandomVector."
            )
        self._feature_index = feature_index
        self.data.columns = feature_index.data

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        """Get the sigma-algebra induced by the random vector.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra induced by the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> domain = SampleSpace(["s0", "s1", "s2"])
        >>> X = RandomVector(
        ...     outputs={"s0": (1, 2), "s1": (3, 4), "s2": (3, 4)},
        ...     domain=domain,
        ... )
        >>> sigma_algebra = SigmaAlgebra.from_random_vector(X)
        >>> sigma_algebra # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(X)':
               atom ID
        sample
        s0      (1, 2)
        s1      (3, 4)
        s2      (3, 4)
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self._sigma_algebra is None:
            self._sigma_algebra = SigmaAlgebra.from_random_vector(self)
        return self._sigma_algebra

    @property
    def range(self) -> RandomVector:
        """Get the range of the random vector.

        Mathematically, the range of a random vector `X:Omega -> S` is the set of all vectors `X(omega)`, as `omega` varies over the sample space `Omega`. In this implementation, the range is represented as another `RandomVector`, where the domain is a `SampleSpace` that indexes the unique output vectors of the original random vector, and the outputs are these unique vectors themselves.

        If the random vector has a string name (e.g., `X`), the range random vector is named `range(X)`, the domain of `range(X)` has indices `x0`, `x1`, etc., and the feature indices of `range(X)` match those of `X` itself. Otherwise, numerical indices are used.

        Returns
        -------
        range : RandomVector
            A `RandomVector` representing the range of the original random vector.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, RandomVector
        >>> import pandas as pd
        >>> outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)}
        >>> domain = SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega")
        >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
        >>> pd.concat([X.range.data, X.range_counts.rename("counts")], axis=1) # doctest: +NORMALIZE_WHITESPACE
                X0  X1  counts
        output
        x0       1   2       1
        x1       3   4       2
        """
        from ..base import SampleSpace

        if self._range is None:

            outputs_and_counts = self.data.value_counts(sort=False).reset_index(
                name="count"
            )

            range_name = f"range({self.name})" if isinstance(self.name, str) else None
            prefix = self.name.lower() if isinstance(self.name, str) else None
            range_sample_space = SampleSpace.generate_default(
                size=len(outputs_and_counts),
                prefix=prefix,
                name=range_name,
                data_name="output",
            )

            self._range_counts = pd.Series(
                data=outputs_and_counts["count"].values,
                index=range_sample_space.data,
                name="count",
            )

            range_data = outputs_and_counts.drop(columns=["count"])
            if range_data.shape[1] == 1:
                range_data = range_data.iloc[:, 0].rename(self.name)
            range_data.index = range_sample_space.data
            if isinstance(self.data, pd.DataFrame):
                range_data.columns = self.data.columns
                range_data.columns.name = "feature"

            rv_range_name = f"{self.name}_range" if isinstance(self.name, str) else None
            self._range = RandomVector.from_pandas(data=range_data, name=rv_range_name)
            self._range.domain.name = range_name

        return self._range

    @property
    def range_counts(self) -> pd.Series:
        """Get the counts of each unique output in the range.

        This property pairs with the `range` property to identify and provide the frequency of each unique output vector in the random vector's mapping. The dataframe `range.data` contains the unique output vectors, while `range_counts` provides the corresponding counts as an index-aligned `pd.Series`.

        Returns
        -------
        range_counts : pd.Series
            A `pd.Series` where the index identifies the unique output vectors in the range, and the values represent the counts of each output vector in the original random vector.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, RandomVector
        >>> import pandas as pd
        >>> outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)}
        >>> domain = SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega")
        >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
        >>> pd.concat([X.range.data, X.range_counts.rename("counts")], axis=1) # doctest: +NORMALIZE_WHITESPACE
                X0  X1  counts
        output
        x0       1   2       1
        x1       3   4       2
        """
        if self._range_counts is None:
            _ = self.range  # triggers computation of range and counts
        return self._range_counts

    @property
    def dimension(self) -> int:
        """Get the dimension of the random vector.

        Returns
        -------
        dimension : int
            The dimension of the random vector.
        """
        if isinstance(self.data, pd.Series):  # triggers computation of data
            return 1
        else:
            return self.data.shape[1]

    def iter_features(self):
        r"""Iterate over sample points and their feature vectors.

        Yields tuples of `(sample_index, FeatureVector)` for each sample point in the domain, allowing iteration over the random vector's entire domain.

        Yields
        ------
        sample_index : Hashable
            Index of the sample point.
        features : FeatureVector
            Feature vector of the sample point.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVector(outputs={"s0": (1, 2), "s1": (3, 4)}, domain=Omega, name="X")
        >>> for _, features in X.iter_features():
        ...     print(features) # doctest: +NORMALIZE_WHITESPACE
        Feature vector of 's0':
                 s0
        feature
        X0        1
        X1        2
        Feature vector of 's1':
                 s1
        feature
        X0        3
        X1        4
        >>> Y = RandomVector(outputs={"s0": 1, "s1": 2}, domain=Omega, name="Y")
        >>> for idx, features in Y.iter_features():
        ...     print(f"Feature of {idx}: ", features)
        Feature of s0:  1
        Feature of s1:  2
        """
        for sample_index in self.data.index:
            yield sample_index, self(sample_index)

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_pandas(
        cls, data: pd.Series | pd.DataFrame, name: Hashable | None = "X"
    ) -> RandomVector:
        """Create a `RandomVector` from a  `pd.Series` or `pd.DataFrame`.

        A domain `SampleSpace` is automatically generated from the index of the provided `pd.DataFrame`. Its name defaults to `Omega`, which may be reset through the `domain.name` property after construction. If the random vector has dimension greater than 1, a feature index (i.e., an instance of `Index`) is also automatically generated based on the columns of the `pd.DataFrame`.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            A `pd.Series` or `pd.DataFrame` where each row corresponds to a sample point. If `data` is a `pd.Series`, the random vector is 1-dimensional; if `data` is a `pd.DataFrame`, the random vector's dimension equals the number of columns.
        name : Hashable | None, default="X"
            The name of the random vector. In the case that `data` is a `pd.Series`, the `name` attribute of the `pd.Series` is set to this value unless it is already set.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series` or `pd.DataFrame`.

        Returns
        -------
        rv : RandomVector
            The constructed `RandomVector` instance.

        Examples
        --------
        >>> from sigalg.core import RandomVector
        >>> import pandas as pd
        >>> # Create a 2-dimensional random vector
        >>> data = pd.DataFrame(
        ...     [[1, 2], [3, 4], [5, 6]],
        ...     index=pd.Index([0, 1, 2], name="numbers"),
        ...     columns=pd.Index(["feature1", "feature2"], name="features"),
        ... )
        >>> X = RandomVector.from_pandas(data, name="X")
        >>> X # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        features  feature1  feature2
        numbers
        0              1         2
        1              3         4
        2              5         6
        >>> # Create a 1-dimensional random variable from a series
        >>> data = pd.Series(
        ...     [10, 20, 30],
        ...     index=pd.Index([0, 1, 2], name="numbers"),
        ... )
        >>> Y = RandomVector.from_pandas(data, name="Y")
        >>> Y # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
               Y
        numbers
        0     10
        1     20
        2     30
        >>> # Create a 1-dimensional random variable from a single-column dataframe
        >>> data = pd.DataFrame([1, 2, 3], index=pd.Index([0, 1, 2], name="numbers"))
        >>> Z = RandomVector.from_pandas(data, name="Z")
        >>> Z # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Z':
               Z
        numbers
        0     1
        1     2
        2     3
        """
        from ..base.index import Index
        from ..base.sample_space import SampleSpace

        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("data must be a pd.Series or pd.DataFrame.")

        if isinstance(data, pd.Series):
            outputs = data.to_dict()
            if data.name is None:
                data.name = name
        elif data.shape[1] == 1:
            outputs = data.iloc[:, 0].to_dict()
            data = data.iloc[:, 0]
            if data.name is None:
                data.name = name
        else:
            outputs = data.apply(lambda row: tuple(row), axis=1).to_dict()

        domain = SampleSpace.from_pandas(data=data.index)
        rv = cls(outputs=outputs, domain=domain, name=name)
        feature_index = (
            Index.from_pandas(data=data.columns, name="feature_index")
            if isinstance(data, pd.DataFrame)
            else None
        )
        if isinstance(data, pd.DataFrame):
            rv.feature_index = feature_index
        rv.data = data
        return rv

    @classmethod
    def from_numpy(cls, array: np.ndarray, name: Hashable | None = "X") -> RandomVector:
        """Create a random vector from a NumPy array.

        Parameters
        ----------
        array : np.ndarray
            NumPy array where rows are sample points and columns are features.
        name : Hashable | None, default="X"
            Name for the random vector.

        Returns
        -------
        rv : RandomVector
            A random vector constructed from the array.

        Raises
        ------
        TypeError
            If `array` is not a NumPy ndarray.

        Examples
        --------
        >>> from sigalg.core import RandomVector
        >>> import numpy as np
        >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X = RandomVector.from_numpy(arr)
        >>> X # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
           0  1
        0  1  2
        1  3  4
        2  5  6
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray.")
        data = pd.DataFrame(array)
        return cls.from_pandas(data=data, name=name)

    # --------------------- probability methods --------------------- #

    def pushforward(
        self,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> FeaturizedProbabilitySpace:
        """Create a featurized probability space from the range of the random vector and the pushforward of a probability measure along the random vector.

        Given a random vector `X: Omega -> S` and a probability measure `P`
        on `Omega`, constructs the featurized probability space `(range(X), F, P_X, X_range)`, where `range(X)` is the range of `X`, `F` is the power-set sigma-algebra on `range(X)`, `P_X` is the pushforward measure of `P` under `X`, and `X_range` is the feature embedding mapping each index in `range(X)` to a feature vector in the range of `X`.

        Parameters
        ----------
        probability_measure : ProbabilityMeasure | None, default=None
            Probability measure `P` defining the probabilities on the sample space. If `None`, the uniform probability measure on the domain is used.

        Returns
        -------
        fps : FeaturizedProbabilitySpace
            The resulting featurized probability space `(range(X), F, P_X, X_range)`.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, pushforward
        >>> domain = SampleSpace.generate_default(size=3)
        >>> X = RandomVector(
        ...     outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
        ...     domain=domain,
        ...     name="X",
        ... )
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X0  X1
        sample
        omega0    1   2
        omega1    3   4
        omega2    3   4
        >>> prob_measure = ProbabilityMeasure(
        ...     probabilities={"omega0": 0.2, "omega1": 0.5, "omega2": 0.3},
        ...     name="P",
        ...     sample_space=domain,
        ... )
        >>> print(X.pushforward(probability_measure=prob_measure)) # doctest: +NORMALIZE_WHITESPACE
        Featurized probability space (range(X), power_set, P_X, X_range)
        ================================================================
        <BLANKLINE>
        * Sample space 'range(X)':
        ['x0', 'x1']
        <BLANKLINE>
        * Sigma algebra 'power_set':
                atom ID
        output
        x0            0
        x1            1
        <BLANKLINE>
        * Probability measure 'P_X':
                probability
        output
        x0              0.2
        x1              0.8
        <BLANKLINE>
        * Random vector 'X_range':
        feature  X0  X1
        output
        x0        1   2
        x1        3   4
        """
        from .pushforward import pushforward

        return pushforward(rv=self, probability_measure=probability_measure)

    def add_probability_measure_to_domain(
        self, pmf: Callable[[FeatureVector | Hashable], Real]
    ) -> FeaturizedProbabilitySpace:
        """Add a probability measure on the domain of the random vector using a function of the features.

        If the random vector is `X: Omega -> S`, this method constructs a `FeaturizedProbabilitySpace` `(Omega, F, P, X)` by defining a probability measure `P` on the sample space `Omega` using a function of the features (i.e., a probability mass function on the features).

        Parameters
        ----------
        pmf : Callable[[FeatureVector | Hashable], Real]
            Function mapping feature vectors (in dimension > 1) or hashable values (in dimension 1) to probability values. Must return non-negative values that sum to 1.

        Returns
        -------
        featurized_space : FeaturizedProbabilitySpace
            A featurized probability space `(Omega, F, P, X)` with the specified
            probability measure.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> domain = SampleSpace.generate_default(size=4)
        >>> outputs = {
        ...     "omega0": (0, 0),
        ...     "omega1": (0, 1),
        ...     "omega2": (1, 0),
        ...     "omega3": (1, 1),
        ... }
        >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
        >>> def pmf(feature_vector):
        ...     v0, v1 = feature_vector
        ...     return 0.75**v0 * 0.25 ** (1 - v0) * 0.6**v1 * 0.4 ** (1 - v1)
        >>> fps = X.add_probability_measure_to_domain(pmf=pmf)
        >>> fps.probability_measure # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                   P
        sample
        omega0  0.10
        omega1  0.15
        omega2  0.30
        omega3  0.45
        """
        from ..base import ProbabilitySpace
        from ..featurized_spaces.featurized_probability_space import (
            FeaturizedProbabilitySpace,
        )
        from ..probability_measures import ProbabilityMeasure

        probabilities = {
            sample_index: pmf(sample_features)
            for sample_index, sample_features in self.iter_features()
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

    # --------------------- data access --------------------- #

    def __call__(
        self, key: Hashable | list[Hashable] | Event
    ) -> Hashable | FeatureVector | RandomVector:
        """Call a `RandomVector` on a sample point to get features, or call on multiple sample points to get the restrition of the `RandomVector`.

        As a function `X:Omega -> S`, a `RandomVector` can be called on a sample point `omega` in its domain `Omega` to get the corresponding feature vector `X(omega)`. If called on a list of sample points or an `Event` instance `A`, it returns a new `RandomVector` representing the restriction `X|A:A -> S`.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Event
            A sample point in the domain, a list of sample points, or an `Event` instance.

        Raises
        ------
        TypeError
            If `key` is not a `Hashable`, list of `Hashable`, or `Event`.
        KeyError
            If any sample point in `key` is not found in the domain.
        ValueError
            If `key` is an `Event` whose sample space does not match the `RandomVector`'s domain.

        Returns
        -------
        features : Hashable | FeatureVector | RandomVector
            If `key` is a single sample point, returns the corresponding feature vector as a `Hashable` or `FeatureVector`. If `key` is a list of sample points or an `Event`, returns a new `RandomVector` restricted to those sample points.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, RandomVector
        >>> domain = SampleSpace(indices=["s0", "s1", "s2"], name="Omega")
        >>> outputs = {"s0": (1, 2), "s1": (3, 4), "s2": (5, 6)}
        >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
        >>> # Get features for a single sample point
        >>> X("s0") # doctest: +NORMALIZE_WHITESPACE
        Feature vector of 's0':
                s0
        feature
        X0        1
        X1        2
        >>> # Get the restriction of X to an event
        >>> A = domain.get_event(["s0", "s2"])
        >>> X_A = X(A)
        >>> X_A # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        feature  X0  X1
        sample
        s0        1   2
        s2        5   6
        """
        from ..base.event import Event
        from ..featurized_spaces.feature_vector import FeatureVector

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("key must be a Hashable, list, or Event.")
        if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
            if key not in self.domain:
                raise KeyError(f"Sample '{key}' not found in domain.")
            result = self.data.loc[key]
            if not isinstance(result, pd.Series):
                return result
            else:
                return FeatureVector(data=result)
        if isinstance(key, list):
            invalid_indices = [k for k in key if k not in self.domain.data]
            if invalid_indices:
                raise KeyError(f"Samples {invalid_indices} not found in domain.")
            return RandomVector.from_pandas(
                data=self.data.loc[key],
                name=f"{self.name}|event" if self.name is not None else None,
            )
        if isinstance(key, Event):
            if key.sample_space != self.domain:
                raise ValueError(
                    "Event's sample_space must match RandomVector's domain."
                )
            return RandomVector.from_pandas(
                data=self.data.loc[key.indices],
                name=(
                    f"{self.name}|{key.name}"
                    if (self.name is not None and key.name is not None)
                    else None
                ),
            )

    # --------------------- apply methods --------------------- #

    def apply_to_features(
        self, function: Callable[[FeatureVector | Hashable], any]
    ) -> pd.Series:
        """Apply a function to the feature vector of each sample point.

        Applies the given function to each sample point's feature vector,
        returning a `pd.Series` of results indexed by sample points.

        Parameters
        ----------
        function : Callable[[FeatureVector | Hashable], any]
            Function that takes a `FeatureVector` object (in dimension > 1) or a `Hashable` (in dimension 1) and returns a value.

        Returns
        -------
        results : pd.Series
            Series of function results indexed by sample points.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace(["s0", "s1"])
        >>> X = RandomVector(outputs={"s0": (1, 2), "s1": (3, 4)}, domain=Omega, name="X")
        >>> X.apply_to_features(lambda f: f.sum() + 2) # doctest: +NORMALIZE_WHITESPACE
        sample
        s0    5
        s1    9
        dtype: int64
        >>> Y = RandomVector(outputs={"s0": 5, "s1": 10}, domain=Omega, name="Y")
        >>> Y.apply_to_features(lambda x: x * 2) # doctest: +NORMALIZE_WHITESPACE
        sample
        s0    10
        s1    20
        Name: Y, dtype: int64
        """
        from ..featurized_spaces.feature_vector import FeatureVector

        if self.dimension > 1:

            def wrapper(row):
                sp = FeatureVector(data=row)
                return function(sp)

            return self.data.apply(wrapper, axis=1)
        else:
            return self.data.apply(function)

    # --------------------- equality --------------------- #

    def __eq__(self, other: RandomVector) -> bool:
        """Check equality with another random vector.

        Two random vectors are equal if they have the same domain, feature index, and underlying data.

        Parameters
        ----------
        other : RandomVector
            Another random vector to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `RandomVector` with the same domain, feature index, and data.
        """
        if not isinstance(other, RandomVector):
            return False
        if not self.domain == other.domain:
            return False
        if not self.feature_index == other.feature_index:
            return False
        return self.data.equals(other.data)

    # --------------------- Representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the random vector.

        Returns
        -------
        repr_str : str
            The string representation of the random vector.
        """
        if self.dimension == 1:
            data = self.data.to_frame()
            data.columns = [self.name]
        else:
            data = self.data
        return f"Random vector '{self.name}':\n{data}"

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector.

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to add, or a scalar value to add to each feature.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If adding two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the sum.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = self.data + other
            new_name = f"({self.name}+{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data + other.data
            new_name = f"({self.name}+{other.name})"
        else:
            raise TypeError("Can only add RandomVector or scalar to RandomVector.")

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __radd__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector (right-hand side).

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to add, or a scalar value to add to each feature.

        Returns
        -------
        result : RandomVector
            A new random vector representing the sum.
        """
        return self.__add__(other)

    def __sub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract another random vector or a scalar from this random vector.

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to subtract, or a scalar value to subtract from each feature.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If subtracting two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the difference.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = self.data - other
            new_name = f"({self.name}-{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data - other.data
            new_name = f"({self.name}-{other.name})"
        else:
            raise TypeError(
                "Can only subtract RandomVector or scalar from RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rsub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract this random vector from another random vector or a scalar (right-hand side).

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to subtract from, or a scalar value to subtract from each feature.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If subtracting two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the difference.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = other - self.data
            new_name = f"({other}-{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = other.data - self.data
            new_name = f"({other.name}-{self.name})"
        else:
            raise TypeError(
                "Can only subtract RandomVector or scalar from RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __mul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply this random vector by another random vector or a scalar.

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to multiply, or a scalar value to multiply each feature by.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If multiplying two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the product.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = self.data * other
            new_name = f"({self.name}*{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data * other.data
            new_name = f"({self.name}*{other.name})"
        else:
            raise TypeError(
                "Can only multiply RandomVector or scalar with RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rmul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply another random vector or a scalar by this random vector (right-hand side).

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to multiply, or a scalar value to multiply each feature by.

        Returns
        -------
        result : RandomVector
            A new random vector representing the product.
        """
        return self.__mul__(other)

    def __truediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide this random vector by another random vector or a scalar.

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to divide by, or a scalar value to divide each feature by.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If dividing two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the quotient.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = self.data / other
            new_name = f"({self.name}/{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot divide RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data / other.data
            new_name = f"({self.name}/{other.name})"
        else:
            raise TypeError("Can only divide RandomVector or scalar with RandomVector.")

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rtruediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide another random vector or a scalar by this random vector (right-hand side).

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector to divide by, or a scalar value to divide each feature by.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If dividing two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the quotient.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = other / self.data
            new_name = f"({other}/{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot divide RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = other.data / self.data
            new_name = f"({other.name}/{self.name})"
        else:
            raise TypeError("Can only divide RandomVector or scalar with RandomVector.")

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __pow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate this random vector by another random vector or a scalar.

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector as the exponent, or a scalar value as the exponent.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If exponentiating two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the exponentiation.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = self.data**other
            new_name = f"({self.name}**{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data**other.data
            new_name = f"({self.name}**{other.name})"
        else:
            raise TypeError(
                "Can only exponentiate RandomVector or scalar with RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rpow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate another random vector or a scalar by this random vector (right-hand side).

        Parameters
        ----------
        other : RandomVector | Real
            Another random vector as the base, or a scalar value as the base.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If exponentiating two `RandomVector` instances with different domains or dimensions.

        Returns
        -------
        result : RandomVector
            A new random vector representing the exponentiation.
        """
        from ..base.index import Index

        if isinstance(other, Real):
            new_values = other**self.data
            new_name = f"({other}**{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = other.data**self.data
            new_name = f"({other.name}**{self.name})"
        else:
            raise TypeError(
                "Can only exponentiate RandomVector or scalar with RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result
