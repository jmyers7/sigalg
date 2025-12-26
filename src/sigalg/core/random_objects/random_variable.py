"""Random variable module.

This module defines the `RandomVariable` class, which represents a random variable as a mapping from a sample space to a feature space. It includes methods for constructing random variables, accessing their properties, and performing arithmetic operations.

Classes
-------
RandomVariable
    Represents a random variable mapping from a sample space to a feature space.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace
    from .random_variable import RandomVariable
    from .random_vector import RandomVector


class RandomVariable:
    """A random variable.

    An instance of `RandomVariable` represents a mapping `X: Omega -> S` from a sample space `Omega` to a feature space `S`. The image `X(omega)` of a sample point `omega` is called the feature of `omega`.

    Instances of `RandomVariable` can be constructed directly from a `domain` sample space and a dictionary of `outputs`, whose keys are the sample points in the domain and whose values are the features (as hashables). Alternatively, factory methods are provided to construct a `RandomVariable` from a `pd.Series` or a `np.ndarray`.

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
        self._range_counts: pd.Series | None = None
        self._data: pd.DataFrame | None = None

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series:
        """Get the underlying `pd.Series`.

        Returns
        -------
        data : pd.Series
            The underlying `pd.Series` representing the random variable.
        """
        if self._data is None:
            data = pd.Series(self.outputs, name=self._name, index=self.domain.data)
            data.index.name = self.domain.data.name
            self._data = data
        return self._data

    @data.setter
    def data(self, data: pd.Series) -> None:
        """Set the underlying `pd.Series`.

        The `data` property is not meant to be set directly by the user. This setter is provided so that the `from_pandas` factory method can set the property.

        Parameters
        ----------
        data : pd.Series
            New `pd.Series` to set.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.
        """
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pd.Series.")
        self._data = data

    @property
    def name(self) -> Hashable:
        """Get the name of the random variable.

        Returns
        -------
        name : Hashable
            The name of the random variable.
        """
        return self._name

    @property
    def range(self) -> RandomVariable:
        """Get the range of the random variable.

        Mathematically, the range of a random variable `X:Omega -> S` is the set of all values `X(omega)`, as `omega` varies over the sample space `Omega`. In this implementation, the range is represented as another `RandomVariable`, where the domain is a `SampleSpace` that indexes the unique output values of the original random variable, and the outputs are these unique values themselves.

        If the random variable has a string name (e.g., `X`), the range random variable is named `range(X)`, and the domain of `range(X)` has indices `x0`, `x1`, etc. Otherwise, numerical indices are used.

        Returns
        -------
        range : RandomVariable
            A `RandomVariable` representing the range of the original random variable.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> outputs = {"omega0": "a", "omega1": "b", "omega2": "b"}
        >>> domain = SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega")
        >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
        >>> pd.concat([X.range.data, X.range_counts.rename("counts")], axis=1) # doctest: +NORMALIZE_WHITESPACE
                X  counts
        output
        x0      b       2
        x1      a       1
        """
        from ..base import SampleSpace

        range_data = self.data.value_counts()
        range_name = f"range({self.name})" if isinstance(self.name, str) else None
        prefix = self.name.lower() if isinstance(self.name, str) else None
        range_sample_space = SampleSpace.generate_default(
            size=len(range_data),
            prefix=prefix,
            name=range_name,
            data_name="output",
        )
        self._range_counts = pd.Series(
            range_data.values, index=range_sample_space.data, name="count"
        )
        range_data = pd.Series(
            range_data.index, index=range_sample_space.data, name=self.name
        )
        return RandomVariable.from_pandas(data=range_data, name=range_name)

    @property
    def range_counts(self) -> pd.Series:
        """Get the counts of each unique output in the range.

        This property pairs with the `range` property to identify and provide the frequency of each unique output feature in the random variable's mapping. The series `range.data` contains the unique output features, while `range_counts` provides the corresponding counts as an index-aligned series.

        Returns
        -------
        range_counts : pd.Series
            A `pd.Series` where the index identifies the unique output features in the range, and the values represent the counts of each output feature in the original random variable.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> outputs = {"omega0": "a", "omega1": "b", "omega2": "b"}
        >>> domain = SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega")
        >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
        >>> pd.concat([X.range.data, X.range_counts.rename("counts")], axis=1) # doctest: +NORMALIZE_WHITESPACE
                X  counts
        output
        x0      b       2
        x1      a       1
        """
        if self._range_counts is None:
            _ = self.range  # triggers computation of range and counts
        return self._range_counts

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_pandas(
        cls, data: pd.Series, name: Hashable | None = "X"
    ) -> RandomVariable:
        """Create a `RandomVariable` from a `pd.Series`.

        A domain `SampleSpace` is automatically generated from the index of the provided `pd.Series`. Its name defaults to `Omega`, which may be reset through the `domain.name` property after construction.

        Parameters
        ----------
        data : pd.Series
            A `pd.Series` where each element corresponds to a sample point.
        name : Hashable | None, default="X"
            The name of the random variable.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.

        Returns
        -------
        rv : RandomVariable
            The constructed `RandomVariable` instance.

        Examples
        --------
        >>> from sigalg.core import RandomVariable
        >>> import pandas as pd
        >>> data = pd.Series(
        ...     [1, 2, 3],
        ...     index=pd.Index([0, 1, 2], name="numbers"),
        ... )
        >>> X = RandomVariable.from_pandas(data, name="X")
        >>> X # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
        numbers
        0    1
        1    2
        2    3
        dtype: int64
        """
        from ..base.sample_space import SampleSpace

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pd.Series.")

        outputs = data.to_dict()
        domain = SampleSpace.from_pandas(data=data.index)
        rv = cls(outputs=outputs, domain=domain, name=name)
        rv.data = data
        return rv

    # --------------------- conversion methods --------------------- #

    def to_random_vector(self) -> RandomVector:
        """Convert this `RandomVariable` to a 1-dimensional `RandomVector`.

        Returns
        -------
        rv : RandomVector
            A `RandomVector` instance with the same data as this `RandomVariable`.

        Examples
        --------
        >>> from sigalg.core import RandomVariable, SampleSpace
        >>> domain = SampleSpace(indices=["s0", "s1", "s2"], name="Omega")
        >>> outputs = {"s0": 1, "s1": 2, "s2": 3}
        >>> X = RandomVariable(outputs=outputs, domain=domain, name="X")
        >>> X # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
        sample
        s0    1
        s1    2
        s2    3
        Name: X, dtype: int64
        >>> X.to_random_vector() # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        X
        sample
        s0      1
        s1      2
        s2      3
        """
        from .random_vector import RandomVector

        rv = RandomVector.from_pandas(data=self.data.to_frame(), name=self.name)
        rv.domain.name = self.domain.name
        return rv

    # --------------------- data access --------------------- #

    def __call__(
        self, key: Hashable | list[Hashable] | Event
    ) -> Hashable | RandomVariable:
        """Call a `RandomVariable` on a sample point to get features, or call on multiple sample points to get the restrition of the `RandomVariable`.

        As a function `X:Omega -> S`, a `RandomVariable` can be called on a sample point `omega` in its domain `Omega` to get the corresponding feature vector `X(omega)`. If called on a list of sample points or an `Event` instance `A`, it returns a new `RandomVariable` representing the restriction `X|A:A -> S`.

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
        features : Hashable | RandomVariable
            If `key` is a single sample point, returns the corresponding feature vector as a `Hashable`. If `key` is a list of sample points or an `Event`, returns a new `RandomVariable` restricted to those sample points.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, RandomVariable
        >>> domain = SampleSpace(indices=["s0", "s1", "s2"], name="Omega")
        >>> outputs = {"s0": 1, "s1": 2, "s2": 3}
        >>> X = RandomVariable(outputs=outputs, domain=domain, name="X")
        >>> # Get features for a single sample point
        >>> int(X("s0"))
        1
        >>> # Get the restriction of X to an event
        >>> A = domain.get_event(["s0", "s2"])
        >>> X_A = X(A)
        >>> X_A # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X|A':
        sample
        s0    1
        s2    3
        Name: X, dtype: int64
        """
        from ..base.event import Event

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("key must be a Hashable, list, or Event.")
        if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
            if key not in self.domain:
                raise KeyError(f"Sample '{key}' not found in domain.")
            return self.data.loc[key]
        if isinstance(key, list):
            invalid_indices = [k for k in key if k not in self.domain.data]
            if invalid_indices:
                raise KeyError(f"Samples {invalid_indices} not found in domain.")
            return RandomVariable.from_pandas(
                data=self.data.loc[key], name=f"{self.name}|event"
            )
        if isinstance(key, Event):
            if key.sample_space != self.domain:
                raise ValueError(
                    "Event's sample_space must match RandomVariable's domain."
                )
            return RandomVariable.from_pandas(
                data=self.data.loc[key.indices],
                name=f"{self.name}|{key.name}",
            )

    # --------------------- equality --------------------- #

    def __eq__(self, other: RandomVariable) -> bool:
        """Check equality with another random variable.

        Two random variables are equal if they have the same domain, feature index, and underlying data.

        Parameters
        ----------
        other : RandomVariable
            Another random variable to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `RandomVariable` with the same domain, feature index, and data.
        """
        if not isinstance(other, RandomVariable):
            return False
        if not self.domain == other.domain:
            return False
        return self.data.equals(other.data)

    # --------------------- Representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the random variable.

        Returns
        -------
        repr_str : str
            The string representation of the random variable.
        """
        return f"Random variable '{self.name}':\n{self.data}"

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other: RandomVariable | Real) -> RandomVariable:
        """Add another random variable or a scalar to this random variable.

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to add, or a scalar value to add to each feature.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If adding two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the sum.
        """
        if isinstance(other, Real):
            new_name = f"({self.name}+{other})"
            new_values = self.data + other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVariables with different domains.")
            new_name = f"({self.name}+{other.name})"
            new_values = self.data + other.data
        else:
            raise TypeError("Can only add RandomVariable or scalar to RandomVariable.")
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result

    def __radd__(self, other: RandomVariable | Real) -> RandomVariable:
        """Add another random variable or a scalar to this random variable (right-hand side).

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to add, or a scalar value to add to each feature.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the sum.
        """
        return self.__add__(other)

    def __sub__(self, other: RandomVariable | Real) -> RandomVariable:
        """Subtract another random variable or a scalar from this random variable.

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to subtract, or a scalar value to subtract from each feature.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If subtracting two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the difference.
        """
        if isinstance(other, Real):
            new_name = f"({self.name}-{other})"
            new_values = self.data - other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_name = f"({self.name}-{other.name})"
            new_values = self.data - other.data
        else:
            raise TypeError(
                "Can only subtract RandomVariable or scalar from RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result

    def __rsub__(self, other: RandomVariable | Real) -> RandomVariable:
        """Subtract this random variable from another random variable or a scalar (right-hand side).

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to subtract from, or a scalar value to subtract from each feature.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If subtracting two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the difference.
        """
        if isinstance(other, Real):
            new_name = f"{other}-({self.name})"
            new_values = other - self.data
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_name = f"({other.name}-{self.name})"
            new_values = other.data - self.data
        else:
            raise TypeError(
                "Can only subtract RandomVariable or scalar from RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result

    def __mul__(self, other: RandomVariable | Real) -> RandomVariable:
        """Multiply this random variable by another random variable or a scalar.

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to multiply, or a scalar value to multiply each feature by.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If multiplying two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the product.
        """
        if isinstance(other, Real):
            new_name = f"({self.name}*{other})"
            new_values = self.data * other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVariables with different domains."
                )
            new_name = f"({self.name}*{other.name})"
            new_values = self.data * other.data
        else:
            raise TypeError(
                "Can only multiply RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result

    def __rmul__(self, other: RandomVariable | Real) -> RandomVariable:
        """Multiply another random variable or a scalar by this random variable (right-hand side).

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to multiply, or a scalar value to multiply each feature by.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the product.
        """
        return self.__mul__(other)

    def __truediv__(self, other: RandomVariable | Real) -> RandomVariable:
        """Divide this random variable by another random variable or a scalar.

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to divide by, or a scalar value to divide each feature by.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If dividing two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the quotient.
        """
        if isinstance(other, Real):
            new_name = f"({self.name}/{other})"
            new_values = self.data / other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_name = f"({self.name}/{other.name})"
            new_values = self.data / other.data
        else:
            raise TypeError(
                "Can only divide RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result

    def __rtruediv__(self, other: RandomVariable | Real) -> RandomVariable:
        """Divide another random variable or a scalar by this random variable (right-hand side).

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable to divide by, or a scalar value to divide each feature by.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If dividing two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the quotient.
        """
        if isinstance(other, Real):
            new_name = f"{other}/({self.name})"
            new_values = other / self.data
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_name = f"({other.name}/{self.name})"
            new_values = other.data / self.data
        else:
            raise TypeError(
                "Can only divide RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result

    def __pow__(self, other: RandomVariable | Real) -> RandomVariable:
        """Exponentiate this random variable by another random variable or a scalar.

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable as the exponent, or a scalar value as the exponent.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If exponentiating two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the exponentiation.
        """
        if isinstance(other, Real):
            new_name = f"({self.name}**{other})"
            new_values = self.data**other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_name = f"({self.name}**{other.name})"
            new_values = self.data**other.data
        else:
            raise TypeError(
                "Can only exponentiate RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result

    def __rpow__(self, other: RandomVariable | Real) -> RandomVariable:
        """Exponentiate another random variable or a scalar by this random variable (right-hand side).

        Parameters
        ----------
        other : RandomVariable | Real
            Another random variable as the base, or a scalar value as the base.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVariable` or a scalar.
        ValueError
            If exponentiating two `RandomVariable` instances with different domains.

        Returns
        -------
        result : RandomVariable
            A new random variable representing the exponentiation.
        """
        if isinstance(other, Real):
            new_name = f"{other}**({self.name})"
            new_values = other**self.data
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_name = f"({other.name}**{self.name})"
            new_values = other.data**self.data
        else:
            raise TypeError(
                "Can only exponentiate RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_pandas(data=new_values, name=new_name)
        return result
