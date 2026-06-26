from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema

from ..core.base.domain import Domain
from ..core.base.index import Index
from .index_validator import IndexLike, _IndexLikeValidator


class _MappingLikeValidator:
    """Validator for mapping-like objects.

    Will coerce a dict into a `pd.Series` or `pd.DataFrame` depending on the values. Will coerce an `np.ndarray` into a `pd.Series` or `pd.DataFrame` depending on the dimensions of the array. Preserves `pd.Series` and `pd.DataFrame` as is.

    Raises
    ------
    ValueError
        If the keys of the provided dict are not all hashable (if applicable), if the values of the dict contain mixed tuple and non-tuple types (if applicable),
        if the values of the dict contain tuples of inconsistent lengths (if applicable), or if the provided object is not a dict with hashable keys, a `pd.Series` or `pd.DataFrame`.

    Examples
    --------
    Coerce a dict into a `pd.Series`.

    >>> import pandas as pd
    >>> from sigalg.validation.mapping_validator import _MappingLikeValidator
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

        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                return pd.Series(v)
            else:
                return pd.DataFrame(v)

        else:
            raise ValueError(
                "Expected dict[Hashable, Any], np.ndarray, pd.Series, or pd.DataFrame."
            )

    @staticmethod
    def _generate_index(v: dict[Hashable, Any]) -> pd.Index:
        return _IndexLikeValidator.validate(v=list(v.keys()))


MappingLike = Annotated[pd.Series | pd.DataFrame, _MappingLikeValidator]


class MappingValidator(BaseModel):
    """Validate SigAlg objects that have an underlying `MappingLike` data structure.

    Designed to validate `MappingLike` objects against a `Domain` and `Index` instance, if they are provided. If these are not provided, sensible defaults are generated. See the Examples section below for usage.

    Parameters
    ----------
    mapping : MappingLike
        The mapping to be validated.
    domain : Domain | None, default=None
        The (optional) domain against which to validate the mapping.
    index : Index | None, default=None
        The (optional) index against which to validate the mapping.
    name : Hashable
        The name of the mapping.
    kind : Literal["any", "probabilities"], default="any"
        Whether the outputs of the mapping are probabilities or not.

    Examples
    --------
    Coerce an "out-of-order" dict into a `pd.Series` with the correct order against a provided `SampleSpace`.

    >>> import pandas as pd
    >>> from sigalg.core import Index, SampleSpace
    >>> from sigalg.validation.mapping_validator import MappingValidator
    >>> Omega = SampleSpace(["a", "b", "c"], variable_names=["omega"])
    >>> mapping = {"b": 2, "a": 1, "c": 3}
    >>> v = MappingValidator(mapping=mapping, domain=Omega, name="X")
    >>> print(v.mapping)  # doctest: +NORMALIZE_WHITESPACE
    omega
    a    1
    b    2
    c    3
    dtype: int64

    Coerce an "out-of-order" dict into a `pd.DataFrame` with the correct order against a provided `SampleSpace` with column labels from a provided `Index`.

    >>> Omega = SampleSpace(["a", "b", "c"], variable_names=["omega"])
    >>> mapping = {"b": (3, 4), "a": (1, 2), "c": (5, 6)}
    >>> I = Index(["odd", "even"])
    >>> v = MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
    >>> print(v.mapping)  # doctest: +NORMALIZE_WHITESPACE
    index  odd  even
    omega
    a        1     2
    b        3     4
    c        5     6

    Generate a default (2-dimensional) `SampleSpace` and `Index` from a plain `dict`.

    >>> mapping = {("a", 1): (1, 2), ("b", 2): (3, 4), ("c", 3): (5, 6)}
    >>> v = MappingValidator(mapping=mapping, name="X")
    >>> print(v.mapping)  # doctest: +NORMALIZE_WHITESPACE
    index              0  1
    point_0 point_1
          a       1    1  2
          b       2    3  4
          c       3    5  6
    >>> print(v.domain)  # doctest: +NORMALIZE_WHITESPACE
    Domain 'D':
    point_0  point_1
          a        1
          b        2
          c        3
    >>> print(v.index)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I':
     index
         0
         1

    Correct an "out-of-order" `pd.DataFrame` against a provided `SampleSpace` and `Index`.

    >>> mapping = pd.DataFrame(
    ...     [
    ...         (4, 3),
    ...         (2, 1),
    ...         (6, 5),
    ...     ],
    ...     index=["b", "a", "c"],
    ...     columns=["even", "odd"],
    ... )
    >>> v = MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
    >>> print(v.mapping)  # doctest: +NORMALIZE_WHITESPACE
    index  odd  even
    omega
    a        1     2
    b        3     4
    c        5     6

    Validate a mapping of "probabilities" that do not sum to 1.

    >>> mapping = {"a": 0.6, "b": 0.2, "c": 0.1}
    >>> v = MappingValidator(
    ...     mapping=mapping, domain=Omega, name="X", kind="probabilities"
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for MappingValidator
      Value error, The probabilities must sum to 1. ...

    Validate a mapping of "probabilities" that includes numbers outside the range [0, 1].

    >>> mapping = {"a": 0.6, "b": 0.2, "c": -0.1}
    >>> v = MappingValidator(
    ...     mapping=mapping, domain=Omega, name="X", kind="probabilities"
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for MappingValidator
      Value error, All probability values must be between 0 and 1. ...
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mapping: MappingLike | Callable | None = None
    domain: Domain | None = None
    output_name: Hashable | None = None
    index: Index | IndexLike | None = None
    name: Hashable | None = None
    kind: Literal["any", "probabilities"] = "any"

    @model_validator(mode="after")
    def _validate_mapping_name_and_output_name(self) -> MappingValidator:
        if isinstance(self.mapping, pd.Series):
            if (self.mapping.name is not None and self.output_name is not None) and (
                self.mapping.name != self.output_name
            ):
                raise ValueError(
                    "The name of the pd.Series must match the output name."
                )
            if self.mapping.name is None and self.output_name is not None:
                self.mapping.name = self.output_name
            if self.mapping.name is not None and self.output_name is None:
                self.output_name = self.mapping.name

        return self

    @model_validator(mode="after")
    def _validate_fun_and_set_argument_data(self) -> MappingValidator:
        if isinstance(self.mapping, Callable):
            sig = inspect.signature(self.mapping)
            if self.domain is not None and self.domain.variable_names != list(
                sig.parameters.keys()
            ):
                raise ValueError(
                    "The provided function's arguments do not match the domain's variable names in the same order."
                )

            if not all(
                (
                    param.kind
                    in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
                )
                for param in sig.parameters.values()
            ):
                raise ValueError(
                    "Multivariate functions must have all arguments as keyword-only arguments."
                )

            self._argument_names = list(sig.parameters.keys())

        return self

    @model_validator(mode="after")
    def _validate_domain_against_mapping(self) -> MappingValidator:
        if self.mapping is not None and not isinstance(self.mapping, Callable):
            if self.domain is not None:
                if len(self.mapping.index) != len(self.domain) or set(
                    self.mapping.index
                ) != set(self.domain):
                    raise ValueError(
                        "The mapping must contain an entry for every sample point in sample_space."
                    )

                if isinstance(self.domain.data, pd.MultiIndex) and isinstance(
                    self.mapping.index, pd.MultiIndex
                ):
                    if set(self.mapping.index.names) != {None}:
                        if self.domain.variable_names != self.mapping.index.names:
                            raise ValueError(
                                "If the mapping index is a MultiIndex, its level names if not None must match the variable names of the sample space."
                            )
                    else:
                        self.mapping.index.names = self.domain.variable_names

                    self.mapping = self.mapping.reindex(self.domain.data)

                if not isinstance(self.domain.data, pd.MultiIndex) and not isinstance(
                    self.mapping.index, pd.MultiIndex
                ):
                    if self.mapping.index.name is not None:
                        if self.domain.variable_names != [self.mapping.index.name]:
                            raise ValueError(
                                "If the mapping index is not a MultiIndex, its name if not None must match the variable name of the sample space."
                            )
                    else:
                        self.mapping.index.name = self.domain.variable_names[0]

                    self.mapping = self.mapping.reindex(self.domain.data)

        return self

    @model_validator(mode="after")
    def _generate_domain(self) -> MappingValidator:
        if self.mapping is not None and not isinstance(self.mapping, Callable):
            if self.domain is None:
                if isinstance(self.mapping.index, pd.MultiIndex):
                    if set(self.mapping.index.names) == {None}:
                        self.mapping.index.names = [
                            f"point_{i}" for i in range(self.mapping.index.nlevels)
                        ]
                    self.domain = Domain(indices=self.mapping.index)
                else:
                    if self.mapping.index.name is None:
                        self.mapping.index.name = "point"
                    self.domain = Domain(indices=self.mapping.index)

            self._argument_names = self.domain.variable_names

        return self

    @model_validator(mode="after")
    def _validate_index(self) -> MappingValidator:
        if self.mapping is not None:
            if isinstance(self.mapping, pd.DataFrame):
                if self.index is not None:
                    if isinstance(self.index.data, pd.MultiIndex):
                        raise ValueError(
                            "The mapping columns cannot be validated against a pd.MultiIndex."
                        )
                    if self.mapping.shape[1] != len(self.index):
                        raise ValueError(
                            "The length of the provided index does not match the dimension of the outputs of the mapping."
                        )

                    default_cols = pd.Index(range(self.mapping.shape[1]))
                    if not self.mapping.columns.equals(default_cols):
                        if set(self.mapping.columns) != set(self.index):
                            raise ValueError(
                                "The columns of the mapping must match the provided Index."
                            )

                        if self.mapping.columns.name is not None:
                            if (
                                self.mapping.columns.name
                                != self.index.variable_names[0]
                            ):
                                raise ValueError(
                                    "If the mapping columns have a name, it must match the name of the provided Index."
                                )

                        self.mapping = self.mapping.reindex(columns=self.index.data)

                    else:
                        self.mapping.columns = self.index.data

            else:
                self.index = None

        return self

    @model_validator(mode="after")
    def _generate_index(self) -> MappingValidator:
        if self.mapping is not None:
            if self.index is None and isinstance(self.mapping, pd.DataFrame):
                if isinstance(self.mapping.columns, pd.MultiIndex):
                    raise ValueError("The mapping columns cannot be a pd.MultiIndex.")
                self.index = Index(indices=self.mapping.columns)
                self.mapping.columns = self.index.data

            if isinstance(self.mapping, pd.Series):
                if self.mapping.name is None:
                    self.mapping.name = self.output_name

        return self

    @model_validator(mode="after")
    def _validate_probabilities(self) -> MappingValidator:
        if self.data is not None:
            if self.kind == "probabilities":
                if isinstance(self.data, pd.Series):
                    if not self.data.apply(lambda x: 0 <= x <= 1).all():
                        raise ValueError(
                            "All probability values must be between 0 and 1."
                        )
                    if np.abs(self.data.sum() - 1.0) >= 1e-8:
                        raise ValueError("The probabilities must sum to 1.")
                else:
                    raise ValueError(
                        "data must be a pd.Series when kind is 'probabilities'."
                    )

        return self

    @property
    def argument_names(self) -> list[Hashable]:  # noqa: D102
        return getattr(self, "_argument_names", None)

    @property
    def num_arguments(self) -> int:  # noqa: D102
        if hasattr(self, "_argument_names"):
            return len(self._argument_names)
        else:
            return None

    @property
    def signature(self) -> inspect.Signature:  # noqa: D102
        if self.fun is not None:
            return inspect.signature(self.fun)
        else:
            return None

    @property
    def data(self) -> pd.Series | pd.DataFrame | None:  # noqa: D102
        if isinstance(self.mapping, Callable) and self.domain is not None:
            if isinstance(self.domain.data, pd.MultiIndex):
                _data = self.domain.data.map(
                    lambda argument: self.mapping(
                        **dict(zip(self.domain.data.names, argument))
                    )
                ).to_series()

            else:
                _data = self.domain.data.map(
                    lambda argument: self.mapping(**{self._argument_names[0]: argument})
                ).to_series()

            _data.index = self.domain.data
            _data.name = self.output_name

        elif (
            self.mapping is not None
            and not isinstance(self.mapping, Callable)
            and self.domain is not None
        ):
            _data = self.mapping

        else:
            _data = None

        return _data

    @property
    def fun(self) -> Callable | None:  # noqa: D102
        if isinstance(self.mapping, pd.Series):

            def make_function(s: pd.Series):
                names = s.index.names
                arguments = [
                    inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                    for name in names
                ]
                sig = inspect.Signature(arguments)

                def function(*args, **kwargs):
                    bound = sig.bind(*args, **kwargs)
                    key = tuple(bound.arguments[name] for name in names)
                    return s[key[0] if len(key) == 1 else key]

                function.__signature__ = sig

                return function

            _fun = make_function(self.mapping)
            self.output_name = self.mapping.name

        elif isinstance(self.mapping, Callable):
            _fun = self.mapping

        else:
            _fun = None

        return _fun
