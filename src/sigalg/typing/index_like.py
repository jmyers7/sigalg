from __future__ import annotations

from collections.abc import Hashable
from typing import Annotated, Any

import pandas as pd
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from ..core.indices.index import Index


class _IndexLikeValidator:
    """Validator for `IndexLike` objects.

    Rules:

    1. Given a `pd.Index` (including `pd.MultiIndex`), will check for duplicate values and preserve the index as is.

    2. Given a list of tuples, will check that all tuples have the same length and that there are no duplicate values, and will coerce into a `pd.MultiIndex` if the tuples have length > 1, or a `pd.Index` if the tuples have length 1.

    3. Given a list of hashable items, will check for duplicate values and coerce into a `pd.Index`.

    4. Given an `sa.Index` object, will extract and return the underlying `pd.Index`.

    Examples
    --------
    >>> import pandas as pd
    >>> import sigalg as sa
    >>> from sigalg.typing.index_like import _IndexLikeValidator

    Rule 1: Preserve a `pd.Index`.

    >>> indices = pd.Index([1, 2, 3])
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2, 3], dtype='int64')

    Rule 1: Preserve a `pd.MultiIndex`.

    >>> indices = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 2),
                (3, 4)],
               )

    Rule 2: Coerce a list of ordered pairs into a `pd.MultiIndex`.

    >>> indices = [(1, 2), (3, 4)]
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 2),
                (3, 4)],
                )

    Rule 3: Coerce a list of integers into a `pd.Index`.

    >>> indices = [1, 2]
    >>> validated_index = _IndexLikeValidator.validate(indices)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2], dtype='int64')

    Rule 4: Extract the underlying `pd.Index` from an `sa.Index`.

    >>> I = sa.Index([1, 2, 3])
    >>> validated_index = _IndexLikeValidator.validate(I)
    >>> print(validated_index)  # doctest: +NORMALIZE_WHITESPACE
    Index([1, 2, 3], dtype='int64', name='index')
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
        elif isinstance(v, Index):
            return v.data.copy()
        else:
            raise ValueError("Expected list[Hashable], list[tuple], or pd.Index")


IndexLike = Annotated[pd.Index | Index, _IndexLikeValidator]
