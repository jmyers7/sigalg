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
    """Validator for index-like objects.

    Will coerce a list of tuples into a `pd.MultiIndex`, and other lists of hashable objects into a `pd.Index`. Will preserve a `pd.Index` as is. Checks for duplicate values.

    Raises
    ------
    ValueError
        If the `IndexLike` object contains duplicate values, if it contains a mix of tuples and non-tuple items, if it contains tuples of inconsistent lengths, if it is a list that includes non-hashable items, or if it is not a list of hashable items, a list of tuples, or a `pd.Index` object.

    Examples
    --------
    Coerce a list of ordered pairs into a `pd.MultiIndex`.

    >>> import pandas as pd
    >>> from sigalg.validation.index_validator import _IndexLikeValidator
    >>> indices = [1, 2]
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2], dtype='int64')

    Coerce a list of ordered pairs into a `pd.MultiIndex`.

    >>> indices = [(1, 2), (3, 4)]
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 2),
                (3, 4)],
               )

    Preserve a `pd.Index`.

    >>> indices = pd.Index([1, 2, 3])
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2, 3], dtype='int64')

    Preserve a `pd.MultiIndex`.

    >>> indices = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 2),
                (3, 4)],
               )
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v: Any) -> pd.Index:
        if isinstance(v, (pd.MultiIndex, pd.Index)):
            if v.nunique() != len(v):
                raise ValueError("index must not contain duplicate values")
            return v.copy()
        elif (
            isinstance(v, list) and len(v) > 0 and any(isinstance(x, tuple) for x in v)
        ):
            if not all(isinstance(x, tuple) for x in v):
                raise ValueError(
                    "If the list contains tuples, all elements must be tuples"
                )

            lengths = {len(x) for x in v}
            if len(lengths) != 1:
                raise ValueError("All tuples must have the same length")
            if len(set(v)) != len(v):
                raise ValueError("index must not contain duplicate values")
            if lengths == {1}:
                return pd.Index([x[0] for x in v])
            else:
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


class IndexValidator(BaseModel):
    """Validate SigAlg objects that have an underlying `IndexLike` data structure.

    Parameters
    ----------
    indices : IndexLike | None
        The index to validate. Must not contain duplicate values.
    name : Hashable
        The name of the SigAlg object with this `IndexLike` data structure.
    variable_names : list[Hashable] | None
        The names of the variables in the index. In the case that the underlying `IndexLike` object is a `pd.MultiIndex`, these names correspond to the level names. For an underlying `pd.Index` that is not a multi-index, this is a list consisting of the single name of the index. See the Examples section below for usage.

    Raises
    ------
    TypeError
        If `variable_names` is not a list of hashable items (if given).
    ValueError
        If `variable_names` contains duplicate values (if given), if `variable_names` does not match the names of the levels in the underlying `pd.MultiIndex` object (if applicable), if the number of variable names does not match the number of levels in the underlying `pd.MultiIndex` object (if applicable), if the single name in `variable_name` does not match the name of the underlying `pd.Index` object (if applicable), if there are more than one variable names for a non-multi-index `pd.Index` object (if applicable).

    Examples
    --------
    Validate a list of hashable items.

    >>> import pandas as pd
    >>> from sigalg.validation.index_validator import IndexValidator
    >>> indices = [1, 2]
    >>> v = IndexValidator(indices=indices, name="I", variable_names_prefix="index")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2], dtype='int64', name='index')
    >>> print(v.name)
    I
    >>> print(v.variable_names)
    ['index']

    Validate a list of ordered pairs.

    >>> indices = [(1, "a"), (2, "b")]
    >>> v = IndexValidator(indices=indices, name="J", variable_names_prefix="index")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['index_0', 'index_1'])
    >>> print(v.name)
    J
    >>> print(v.variable_names)
    ['index_0', 'index_1']

    Validate a list of ordered pairs with a custom `variable_names` parameter.

    >>> indices = [(1, "a"), (2, "b")]
    >>> v = IndexValidator(indices=indices, name="K", variable_names=["num", "letter"])
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['num', 'letter'])
    >>> print(v.name)
    K
    >>> print(v.variable_names)
    ['num', 'letter']

    Validate `pd.Index` with a pre-existing `name`.

    >>> indices = pd.Index([1, 2], name="num")
    >>> v = IndexValidator(indices=indices, name="M")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2], dtype='int64', name='num')
    >>> print(v.name)
    M
    >>> print(v.variable_names)
    ['num']

    Validate a `pd.MultiIndex` with no custom level names.

    >>> indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    >>> v = IndexValidator(indices=indices, name="N", variable_names_prefix="index")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['index_0', 'index_1'])
    >>> print(v.name)
    N
    >>> print(v.variable_names)
    ['index_0', 'index_1']

    Validate a `pd.MultiIndex` with pre-existing level names.

    >>> indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["num", "letter"])
    >>> v = IndexValidator(indices=indices, name="O")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['num', 'letter'])
    >>> print(v.name)
    O
    >>> print(v.variable_names)
    ['num', 'letter']

    Validate a `pd.MultiIndex` with a custom `variable_names` parameter. The custom variable names are set to the names of the levels in the `pd.MultiIndex`.

    >>> indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    >>> v = IndexValidator(indices=indices, name="P", variable_names=["num", "letter"])
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 'a'),
                (2, 'b')],
               names=['num', 'letter'])
    >>> print(v.name)
    P
    >>> print(v.variable_names)
    ['num', 'letter']
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    indices: IndexLike | None = None
    name: Hashable
    variable_names: list[Hashable] | None = None
    variable_names_prefix: Hashable | None = None

    @field_validator("variable_names", mode="before")
    @classmethod
    def validate_variable_names(cls, v: list[Hashable] | None) -> list[Hashable] | None:
        """Validate that variable_names is a list of Hashable and contains no duplicates."""
        if v is not None:
            if not isinstance(v, list):
                raise TypeError("variable_names must be a list of Hashable or `None`.")
            if not all(isinstance(x, Hashable) for x in v):
                raise TypeError("All elements in variable_names must be Hashable.")
            if len(set(v)) != len(v):
                raise ValueError("variable_names must not contain duplicate values.")
        return v

    @model_validator(mode="after")
    def generate_variable_names(self) -> IndexValidator:  # noqa: D102
        if self.indices is not None:
            if isinstance(self.indices, pd.MultiIndex):
                if (
                    set(self.indices.names) != {None}
                    and self.variable_names is not None
                ):
                    if list(self.indices.names) != self.variable_names:
                        raise ValueError(
                            "The variable names must match the level names of the underlying pd.MultiIndex."
                        )

                if set(self.indices.names) != {None} and self.variable_names is None:
                    self.variable_names = list(self.indices.names)

                if (
                    set(self.indices.names) == {None}
                    and self.variable_names is not None
                ):
                    if len(self.variable_names) != self.indices.nlevels:
                        raise ValueError(
                            "The number of variable names must match the number of levels in the underlying pd.MultiIndex."
                        )
                    self.indices.names = self.variable_names

                if set(self.indices.names) == {None} and self.variable_names is None:
                    if self.variable_names_prefix is None:
                        raise ValueError(
                            "If variable_names is None, then variable_names_prefix must be passed."
                        )

                    self.variable_names = [
                        f"{self.variable_names_prefix}_{i}"
                        for i in range(self.indices.nlevels)
                    ]
                    self.indices.names = self.variable_names

            else:
                if self.variable_names is not None and len(self.variable_names) != 1:
                    raise ValueError(
                        "There must be exactly one variable name for a non-pd.MultiIndex."
                    )

                if self.indices.name is not None and self.variable_names is not None:
                    if self.indices.name != self.variable_names[0]:
                        raise ValueError(
                            "The variable name must match the name of underlying pd.Index."
                        )

                if self.indices.name is not None and self.variable_names is None:
                    self.variable_names = [self.indices.name]

                if self.indices.name is None and self.variable_names is not None:
                    self.indices.name = self.variable_names[0]

                if self.indices.name is None and self.variable_names is None:
                    if self.variable_names_prefix is None:
                        raise ValueError(
                            "If variable_names is None, then variable_names_prefix must be passed."
                        )

                    self.variable_names = [self.variable_names_prefix]
                    self.indices.name = self.variable_names_prefix

        return self
