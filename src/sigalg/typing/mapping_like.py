from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Annotated, Any

import numpy as np
import pandas as pd
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from .index_like import _IndexLikeValidator


class _MappingLikeValidator:
    """Validator for mapping-like objects.

    Rules:

    1. Given a `dict`:
        a. Will coerce into a `pd.Series` if the values of the dict are 1-dimensional, i.e., not tuples.
        b. Will coerce into a `pd.DataFrame` if the values are tuples of length > 1.

    2. Given an `np.ndarray`:
        a. Will coerce into a `pd.Series` if the array is 1-dimensional.
        b. Will coerce into a `pd.DataFrame` if the array is 2-dimensional.
        c. Will coerce into a `pd.Series` with a multi-index whose number of levels is equal to the dimension of the array, if the array is n-dimensional with n > 2.

    3. Preserves `pd.Series` and `pd.DataFrame` as is.

    4. Preserves `Callable` as is.

    Examples
    --------
    Coerce a dict into a `pd.Series`.

    >>> import pandas as pd
    >>> from sigalg.typing.mapping_like import _MappingLikeValidator
    >>> mapping = {"a": 1, "b": 2}
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
    a    1
    b    2
    dtype: int64

    Coerce a dict with length-1 tuples into a `pd.Series`.

    >>> mapping = {"a": (1,), "b": (2,)}
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
    a    1
    b    2
    dtype: int64

    Coerce a dict with length-2 tuples into a `pd.DataFrame`.

    >>> mapping = {"a": (1, 2), "b": (3, 4)}
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
       0  1
    a  1  2
    b  3  4

    Coerce a dict with length-2 tuples as keys into a `pd.DataFrame` with a `pd.MultiIndex` for an index.

    >>> mapping = {("a", "x"): (1, 2), ("b", "y"): (3, 4)}
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
         0  1
    a x  1  2
    b y  3  4

    Preserve `pd.Series`.

    >>> mapping = pd.Series([1, 2])
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
    0    1
    1    2
    dtype: int64

    Preserve a `pd.Series` with a custom index.

    >>> mapping = pd.Series([1, 2], index=["a", "b"])
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
    a    1
    b    2
    dtype: int64

    Preserve a `pd.Series` with a custom `pd.MultiIndex` for index.

    >>> mapping = pd.Series([1, 2], index=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")]))
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
    a  x    1
    b  y    2
    dtype: int64

    Preserve a `pd.DataFrame`.

    >>> mapping = pd.DataFrame([[1, 2], [3, 4]])
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
       0  1
    0  1  2
    1  3  4

    Preserve a `pd.DataFrame` with a custom index and columns.

    >>> mapping = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["num_1", "num_2"])
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
       num_1  num_2
    a      1      2
    b      3      4

    Preserve a `pd.DataFrame` with a custom `pd.MultiIndex` for index and custom columns.

    >>> mapping = pd.DataFrame([[1, 2], [3, 4]], index=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")]), columns=["num_1", "num_2"])
    >>> validated_mapping = _MappingLikeValidator.validate(mapping)
    >>> print(validated_mapping)  # doctest: +NORMALIZE_WHITESPACE
         num_1  num_2
    a x      1      2
    b y      3      4
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v: Any) -> pd.Series | pd.DataFrame:

        if isinstance(v, dict):
            if any(isinstance(value, tuple) for value in v.values()):
                if not all(isinstance(value, tuple) for value in v.values()):
                    raise ValueError(
                        "If the mapping contains a tuple value, all values must be tuples."
                    )

                lengths = {len(value) for value in v.values()}
                if len(lengths) != 1:
                    raise ValueError(
                        "All tuples in the mapping must have the same length."
                    )

                if lengths == {1}:
                    return pd.Series(
                        [v[key][0] for key in v.keys()],
                        index=cls._generate_index(v),
                    )
                else:
                    return pd.DataFrame(
                        v.values(),
                        index=cls._generate_index(v),
                    )

            else:
                return pd.Series(v.values(), index=cls._generate_index(v))

        elif isinstance(v, pd.Series | pd.DataFrame):
            return v.copy()

        elif isinstance(v, Callable):
            return v

        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                return pd.Series(v)
            elif v.ndim == 2:
                return pd.DataFrame(v)
            else:
                index = pd.MultiIndex.from_product([range(dim) for dim in v.shape])
                return pd.Series(v.ravel(), index=index)

        else:
            raise ValueError(
                "Expected dict[Hashable, Any], np.ndarray, pd.Series, or pd.DataFrame."
            )

    @staticmethod
    def _generate_index(v: dict[Hashable, Any]) -> pd.Index:
        return _IndexLikeValidator.validate(v=list(v.keys()))


MappingLike = Annotated[pd.Series | pd.DataFrame | Callable, _MappingLikeValidator]
