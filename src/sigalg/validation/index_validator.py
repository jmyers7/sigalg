from __future__ import annotations

from collections.abc import Hashable  # noqa: TC003
from numbers import Real
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from ..typing.index_like import IndexLike  # noqa: TC001

if TYPE_CHECKING:
    from ..core.indices.index import Index


class IndexValidator(BaseModel):
    """Validate input data for SigAlg objects that are constructed from `IndexLike` objects.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        The indices to validate.
    variable_names : list[Hashable] | None, default=None
        The names of the variables in the index.
    variable_names_prefix : Hashable | None, default=None
        If `variable_names` is `None`, default variable names will be generated using this prefix.
    kind : Literal["Index", "Time"] = "Index"
        The type of index.
    name : Hashable | None, default=None
        The name of the object.

    Examples
    --------
    Validate a list of hashable items.

    >>> import pandas as pd
    >>> from sigalg.validation.index_validator import IndexValidator
    >>> indices = [1, 2]
    >>> v = IndexValidator(indices=indices, name="I", variable_names_prefix="index")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2], dtype='int64', name='index')

    Validate a list of ordered pairs.

    >>> indices = [(1, "a"), (2, "b")]
    >>> v = IndexValidator(indices=indices, name="J", variable_names_prefix="index")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['index_0', 'index_1'])

    Validate a list of ordered pairs with a custom `variable_names` parameter.

    >>> indices = [(1, "a"), (2, "b")]
    >>> v = IndexValidator(indices=indices, name="K", variable_names=["num", "letter"])
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['num', 'letter'])

    Validate a `pd.MultiIndex` with no custom level names.

    >>> indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    >>> v = IndexValidator(indices=indices, name="N", variable_names_prefix="index")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['index_0', 'index_1'])

    Validate a `pd.MultiIndex` with pre-existing level names.

    >>> indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["num", "letter"])
    >>> v = IndexValidator(indices=indices, name="O")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['num', 'letter'])

    Validate a `pd.MultiIndex` with a custom `variable_names` parameter.

    >>> indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    >>> v = IndexValidator(indices=indices, name="P", variable_names=["num", "letter"])
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['num', 'letter'])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    indices: IndexLike | None = None
    variable_names: list[Hashable] | None = None
    variable_names_prefix: Hashable | None = None
    kind: Literal["Index", "Time"] = "Index"
    name: Hashable | None = None

    @model_validator(mode="after")
    def generate_variable_names(self) -> IndexValidator:
        """Generate variable names if `indices` is a `pd.Index` and does not come with level names."""
        from ..core.indices.index import Index

        if isinstance(self.data, Index):
            return self

        elif self.data is not None and set(self.data.names) == {None}:
            if self.variable_names is None and self.variable_names_prefix is None:
                raise TypeError(
                    "At least one of variable_names or variable_names_prefix must not be None."
                )
            if self.variable_names is None:
                if self.data.nlevels > 1:
                    self.variable_names = [
                        f"{self.variable_names_prefix}_{i}"
                        for i in range(self.data.nlevels)
                    ]
                else:
                    self.variable_names = [self.variable_names_prefix]

            self.data.names = self.variable_names

        return self

    @model_validator(mode="after")
    def validate_time(self) -> IndexValidator:
        """Validate a time index."""
        if self.data is not None and self.kind == "Time":
            if isinstance(self.indices, pd.MultiIndex):
                raise ValueError(
                    "The underlying data of a time index cannot be a pd.MultiIndex"
                )
            if not all(isinstance(x, Real) for x in self.indices):
                raise ValueError("All elements in the time index must be real numbers")
            if not self.indices.is_monotonic_increasing:
                raise ValueError("Time index must be in ascending order")

        return self

    @property
    def data(self) -> pd.Index | Index | None:  # noqa: D102
        from ..core.indices.index import Index

        if isinstance(self.indices, Index):
            return self.indices.data
        else:
            return self.indices
