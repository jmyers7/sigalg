from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from ..typing.index_like import IndexLike


class IndexValidator(BaseModel):
    """Validate input data for instances of `sa.Index`.

    Parameters
    ----------
    indices : IndexLike | None
        The index to validate.
    name : Hashable
        The name of the object.
    variable_names : list[Hashable] | None
        The names of the variables in the index.

    Examples
    --------
    Validate a list of hashable items.

    >>> import pandas as pd
    >>> from sigalg.validation.index_validator import IndexValidator
    >>> indices = [1, 2]
    >>> v = IndexValidator(indices=indices, name="I", variable_names_prefix="index", default_name="I")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2], dtype='int64', name='index')
    >>> print(v.name)
    I
    >>> print(v.variable_names)
    ['index']

    Validate a list of ordered pairs.

    >>> indices = [(1, "a"), (2, "b")]
    >>> v = IndexValidator(indices=indices, name="J", variable_names_prefix="index", default_name="J")
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
    >>> v = IndexValidator(indices=indices, name="K", variable_names=["num", "letter"], default_name="K")
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
    >>> v = IndexValidator(indices=indices, name="M", default_name="M")
    >>> print(v.indices)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2], dtype='int64', name='num')
    >>> print(v.name)
    M
    >>> print(v.variable_names)
    ['num']

    Validate a `pd.MultiIndex` with no custom level names.

    >>> indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    >>> v = IndexValidator(indices=indices, name="N", variable_names_prefix="index", default_name="N")
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
    >>> v = IndexValidator(indices=indices, name="O", default_name="O")
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
    >>> v = IndexValidator(indices=indices, name="P", variable_names=["num", "letter"], default_name="P")
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
    name: Hashable | None
    variable_names: list[Hashable] | None = None
    variable_names_prefix: Hashable | None = None
    default_name: str
    from_sa_index: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def extract_sa_index_metadata(cls, data: Any) -> Any:
        """Extract metadata from an `sa.Index` object if it is passed into the constructor of `IndexValidator`.

        Rules:

        1. If `variable_names` is `None`, then the `variable_names` of the `sa.Index` object will be used. If it is not `None`, then the `variable_names` of the underlying `pd.Index` or `pd.MultiIndex` object will be used.

        2. If `name` is `None`, then the `name` of the `sa.Index` object will be used.

        3. If `indices` is not an `sa.Index` object, and if `name` is `None`, then the `default_name` will be used as the `name`.
        """
        from ..core.indices.index import Index

        if isinstance(data, dict):
            indices = data.get("indices")
            if isinstance(indices, Index):
                data["from_sa_index"] = True
                if data.get("variable_names") is None:
                    data["variable_names"] = indices.variable_names
                if data.get("name") is None:
                    data["name"] = indices.name
            else:
                data["from_sa_index"] = False
                if data.get("name") is None:
                    data["name"] = data.get("default_name")

        return data

    @field_validator("variable_names", mode="before")
    @classmethod
    def validate_variable_names(cls, v: list[Hashable] | None) -> list[Hashable] | None:
        """Validate the `variable_names` parameter.

        Rules: If `variable_names` is not `None`, then it must be a list of hashable items with no duplicates.
        """
        if v is not None:
            if not isinstance(v, list):
                raise TypeError("variable_names must be a list of Hashable or `None`.")
            if not all(isinstance(x, Hashable) for x in v):
                raise TypeError("All elements in variable_names must be Hashable.")
            if len(set(v)) != len(v):
                raise ValueError("variable_names must not contain duplicate values.")
        return v

    @model_validator(mode="after")
    def generate_variable_names(self) -> IndexValidator:
        """Generate variable names if they are not provided.

        Rules:

        1. For a `pd.MultiIndex`:

            a. If the `pd.MultiIndex` has level names and `variable_names` is `None`, then the level names of the `pd.MultiIndex` will be used as the `variable_names`.

            b. If `variable_names` is provided, then the level names of the `pd.MultiIndex` will be set to the `variable_names`.

            c. If the `pd.MultiIndex` has no level names and `variable_names` is `None`, then `variable_names` will be generated using `variable_names_prefix` and the level names of the `pd.MultiIndex` will be set accordingly.

        2. For a `pd.Index` which is not a `pd.MultiIndex`:

            a. If the `pd.Index` has a name and `variable_names` is `None`, then the name of the `pd.Index` will be used as the `variable_names`.

            b. If `variable_names` is provided, then the name of the `pd.Index` will be set to the `variable_names[0]`.

            c. If the `pd.Index` has no name and `variable_names` is `None`, then `variable_names` will be generated using `variable_names_prefix` and the name of the `pd.Index` will be set accordingly.
        """
        if self.indices is not None:
            if isinstance(self.indices, pd.MultiIndex):
                if set(self.indices.names) != {None} and self.variable_names is None:
                    self.variable_names = list(self.indices.names)

                elif self.variable_names is not None:
                    if len(self.variable_names) != self.indices.nlevels:
                        raise ValueError(
                            "The number of variable names must match the number of levels in the underlying pd.MultiIndex."
                        )
                    self.indices.names = self.variable_names

                else:
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

                if self.indices.name is not None and self.variable_names is None:
                    self.variable_names = [self.indices.name]

                elif self.variable_names is not None:
                    self.indices.name = self.variable_names[0]

                else:
                    if self.variable_names_prefix is None:
                        raise ValueError(
                            "If variable_names is None, then variable_names_prefix must be passed."
                        )

                    self.variable_names = [self.variable_names_prefix]
                    self.indices.name = self.variable_names_prefix

        return self
