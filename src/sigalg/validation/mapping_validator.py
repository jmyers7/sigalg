from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    model_validator,
)

from ..core.indices.index import Index
from ..core.indices.time import Time
from ..core.spaces.domain import Domain
from ..core.spaces.sample_space import SampleSpace
from ..typing.index_like import IndexLike
from ..typing.mapping_like import MappingLike


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
    Domain 'X':
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
    >>> print(v.data)  # doctest: +NORMALIZE_WHITESPACE
    index  odd  even
    omega
    a        1     2
    b        3     4
    c        5     6

    Validate a mapping of "probabilities" that do not sum to 1.

    >>> mapping = {"a": 0.6, "b": 0.2, "c": 0.1}
    >>> v = MappingValidator(
    ...     mapping=mapping, domain=Omega, name="X", kind="probability"
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for MappingValidator
      Value error, Probability values must sum to 1. ...

    Validate a mapping of "probabilities" that includes numbers outside the range [0, 1].

    >>> mapping = {"a": 0.6, "b": 0.2, "c": -0.1}
    >>> v = MappingValidator(
    ...     mapping=mapping, domain=Omega, name="X", kind="probability"
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for MappingValidator
      Value error, All measure values in the mapping must be non-negative. ...
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: Domain | None = None
    kind: Literal["any", "measure", "probability"] = "any"
    mapping: MappingLike | None = None
    index: Index | IndexLike | None = None
    index_kind: Literal["any", "time"] = "any"
    domain_kind: Literal["any", "sample_space"] = "any"
    multi_dim_inputs: bool = False
    multi_dim_outputs: bool = False
    output_name: Hashable | None = None
    name: Hashable | None = None

    _data: pd.Series | pd.DataFrame | None = PrivateAttr(default=None)
    _fun: Callable | None = PrivateAttr(default=None)
    _argument_names: list[Hashable] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_names(self) -> MappingValidator:
        """Validate the `mapping` name and `output_name` parameter.

        Rules: The only type of `mapping` that carries a name is a `pd.Series`.

        1. If the series carries a name and `output_name` is not `None`, and if they do not agree, raise an exception.

        2. If the series does not carry a name, but `output_name` is not `None`, copy `output_name` as the name of the series.

        3. If the series carries a name, but `output_name` is `None`, copy the name of the series to `output_name`.
        """
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
    def _validate_fun(self) -> MappingValidator:
        """Validate the `mapping` if it is a callable.

        Rules: The only `mapping` that carries argument names is a callable.

        1. If `domain` is not `None`, check that the argument names of the callable match the variable names of the domain. If they do not match, raise an exception.

        2. Check that all arguments of the callable are keyword-only arguments. If they are not, raise an exception.
        """
        if isinstance(self.mapping, Callable):
            sig = inspect.signature(self.mapping)
            if self.domain is not None and set(self.domain.variable_names) != set(
                sig.parameters.keys()
            ):
                raise ValueError(
                    "The provided function's arguments do not match the domain's variable names."
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

        return self

    @model_validator(mode="after")
    def _validate_domain(self) -> MappingValidator:
        """Validate the `domain` against the `mapping` if both are provided.

        Rules: The only `mapping` that carries domain information is a `pd.Series` or `pd.DataFrame`.

        1. If the `mapping` index does not equal `domain` as a set, raise an exception.

        2. Suppose both the `mapping` index and `domain` are `pd.MultiIndex`.

            a. If the `mapping` index has level names that are not `None`, check that they match the variable names of the domain. If they do not match, raise an exception.

            b. If the `mapping` index has level names that are all `None`, copy the variable names of the domain to the level names of the `mapping` index. Then, reindex the `mapping` against the `domain` so that the order of the `mapping` matches the order of the `domain`.

        3. Suppose that both the `mapping` index and `domain` are not `pd.MultiIndex`.

            a. If the `mapping` index has a name that is not `None`, check that it matches the variable name of the domain. If they do not match, raise an exception.

            b. If the `mapping` index has a name that is `None`, copy the variable name of the domain to the name of the `mapping` index. Then, reindex the `mapping` against the `domain` so that the order of the `mapping` matches the order of the `domain`.

        4. If one of the `mapping` index and `domain` is a `pd.MultiIndex` and the other is not, raise an exception.
        """
        if (
            self.mapping is not None
            and not isinstance(self.mapping, Callable)
            and self.domain is not None
        ):
            if len(self.mapping.index) != len(self.domain) or set(
                self.mapping.index
            ) != set(self.domain):
                raise ValueError(
                    "The mapping must contain an entry for every point in the domain."
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

            elif not isinstance(self.domain.data, pd.MultiIndex) and not isinstance(
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

            else:
                raise ValueError(
                    "The mapping index and the domain must either both be MultiIndex or both not be MultiIndex."
                )

        return self

    @model_validator(mode="after")
    def _generate_domain(self) -> MappingValidator:
        """Generate a default `domain` if it is not provided.

        Rules: The only `mapping` that carries domain information is a `pd.Series` or `pd.DataFrame`.

        1. If the `domain` is `None`:

            a. If the `mapping` index is a `pd.MultiIndex` whose names are all `None`, generate default level names for the `mapping` index based on the `domain_kind` parameter.

            b. If the `mapping` index is not a `pd.MultiIndex` and its name is `None`, generate a default name for the `mapping` index based on the `domain_kind` parameter.

            c. Generate a default `domain` based on the `mapping` index and the `domain_kind` parameter.

        2. Set the `_argument_names` attribute to the variable names of the `domain`.
        """
        if self.mapping is not None and not isinstance(self.mapping, Callable):
            if self.domain is None:
                if isinstance(self.mapping.index, pd.MultiIndex):
                    if set(self.mapping.index.names) == {None}:
                        self.mapping.index.names = [
                            f"point_{i}" if self.domain_kind == "any" else f"sample_{i}"
                            for i in range(self.mapping.index.nlevels)
                        ]
                else:
                    if self.mapping.index.name is None:
                        self.mapping.index.name = (
                            "point" if self.domain_kind == "any" else "sample"
                        )

                self.domain = (
                    Domain(indices=self.mapping.index)
                    if self.domain_kind == "any"
                    else SampleSpace(indices=self.mapping.index)
                )

        return self

    @model_validator(mode="after")
    def _generate_argument_names(self) -> MappingValidator:
        """Generate argument names.

        Rules:

        1. If the `mapping` is not `None` and is not a callable, set `_argument_names` to the variable names of the `domain`.

        2. If the `mapping` is a callable, set `_argument_names` to the argument names of the callable.

        3. If the `mapping` is `None`, set `_argument_names` to `None`.
        """
        if self.mapping is not None:
            if not isinstance(self.mapping, Callable):
                self._argument_names = self.domain.variable_names
            elif isinstance(self.mapping, Callable):
                self._argument_names = list(
                    inspect.signature(self.mapping).parameters.keys()
                )
            else:
                self._argument_names = None
        else:
            self._argument_names = None

        return self

    @model_validator(mode="after")
    def _generate_fun(self) -> MappingValidator:
        """Generate the function.

        Rules:

        1. If the `mapping` is a `pd.Series`, generate a function that takes keyword-only arguments corresponding to the variable names of the `domain` and returns the corresponding value from the `mapping`.

        2. If the `mapping` is a callable, set `_fun` to the `mapping`.

        3. If the `mapping` is `None`, set `_fun` to `None`.
        """
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

            self._fun = make_function(self.mapping)

        elif isinstance(self.mapping, Callable):
            self._fun = self.mapping

        else:
            self._fun = None

        return self

    @model_validator(mode="after")
    def _generate_data(self) -> MappingValidator:
        """Generate the data.

        Rules:

        1. If the `mapping` is a callable and the `domain` is not `None`, generate a `pd.Series` or `pd.DataFrame` by applying the callable to each point in the `domain`.

        2. If the `mapping` is not `None` and is not a callable, set the data to the `mapping`.

        3. If the `mapping` is `None`, set the data to `None`.
        """
        if isinstance(self.mapping, Callable) and self.domain is not None:
            if isinstance(self.domain.data, pd.MultiIndex):
                self._data = self.domain.data.map(
                    lambda argument: self.mapping(
                        **dict(zip(self.domain.data.names, argument))
                    )
                ).to_series()

            else:
                self._data = self.domain.data.map(
                    lambda argument: self.mapping(**{self.domain.data.name: argument})
                ).to_series()

            self._data.index = self.domain.data
            self._data.name = self.output_name

            if self.multi_dim_outputs:
                if isinstance(self._data.iloc[0], tuple):
                    tuple_length = len(self._data.iloc[0])
                    if all(
                        isinstance(value, tuple) and len(value) == tuple_length
                        for value in self._data
                    ):
                        self._data = pd.DataFrame(
                            self._data.tolist(),
                            index=self._data.index,
                        )

        elif self.mapping is not None and not isinstance(self.mapping, Callable):
            self._data = self.mapping

        else:
            self._data = None

        return self

    @model_validator(mode="after")
    def _validate_index(self) -> MappingValidator:
        """Validate the index against the data.

        Rules: The only data that carries index information is a `pd.DataFrame`. Suppose `index` is provided.

        1.. Check that the underlying data of `index` is not a `pd.MultiIndex`. If it is, raise an exception.

        2. Check that the length of the `index` matches the number of columns in the `data`. If it does not, raise an exception.

        3. If the columns of the `data` are not the default integer range:

            a. Check that the set of columns matches the set of the `index`. If they do not match, raise an exception.

            b. Check that if the column index has a name, it matches the name of the `index`. If they do not match, raise an exception.

            c. Reindex (i.e., re-order) the `data` against the `index`.
        """
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                if self.index is not None:
                    if isinstance(self.index.data, pd.MultiIndex):
                        raise ValueError(
                            "The mapping columns cannot be validated against a pd.MultiIndex."
                        )
                    if self.data.shape[1] != len(self.index):
                        raise ValueError(
                            "The length of the provided index does not match the dimension of the outputs of the mapping."
                        )

                    default_cols = pd.Index(range(self.data.shape[1]))
                    if not self.data.columns.equals(default_cols):
                        if set(self.data.columns) != set(self.index):
                            raise ValueError(
                                "The columns of the mapping must match the provided Index."
                            )

                        if self.data.columns.name is not None:
                            if self.data.columns.name != self.index.variable_names[0]:
                                raise ValueError(
                                    "If the mapping columns have a name, it must match the name of the provided Index."
                                )

                        self._data = self.data.reindex(columns=self.index.data)

                    else:
                        self.data.columns = self.index.data

        return self

    @model_validator(mode="after")
    def _generate_index(self) -> MappingValidator:
        """Generate the index if it is not provided.

        Rules:

        1. If the `data` is a `pd.DataFrame` and `index` is `None`, generate an `Index` or `Time` instance from the columns of the `data` based on the `index_kind` parameter. Then, set the columns of the `data` to the underlying data of the generated index.

        2. If the `data` is a `pd.Series`, set the `index` to `None`.

        3. If the `data` is `None`, set the `index` to `None`.
        """
        if self.data is not None:
            if self.index is None and isinstance(self.data, pd.DataFrame):
                if isinstance(self.data.columns, pd.MultiIndex):
                    raise ValueError("The mapping columns cannot be a pd.MultiIndex.")
                self.index = (
                    Index(indices=self.data.columns)
                    if self.index_kind == "any"
                    else Time(indices=self.data.columns)
                )
                self.data.columns = self.index.data

            if isinstance(self.data, pd.Series):
                if self.data.name is None:
                    self.data.name = self.output_name
                self.index = None

        return self

    @model_validator(mode="after")
    def _validate_kind(self) -> MappingValidator:
        """Validate the `kind` parameter against the `data`.

        Rules:

        1. If the `data` is not `None` and the `kind` is "measure" or "probability", check that all values in the `data` are non-negative. If any value is negative, raise an exception.

        2. If the `kind` is "probability", check that the sum of the values in the `data` is equal to 1 (within a tolerance of 1e-8). If the sum is not equal to 1, raise an exception.

        3. If the `data` is not a `pd.Series` when the `kind` is "probability" or "measure", raise an exception.
        """
        if self.data is not None:
            if self.kind == "measure" or self.kind == "probability":
                if isinstance(self.data, pd.Series):
                    if (self.data < 0).any():
                        raise ValueError(
                            "All measure values in the mapping must be non-negative."
                        )
                    if (
                        self.kind == "probability"
                        and np.abs(self.data.sum() - 1.0) >= 1e-8
                    ):
                        raise ValueError("Probability values must sum to 1.")
                else:
                    raise ValueError(
                        "data must be a pd.Series when kind is 'probability' or 'measure'."
                    )
        return self

    @property
    def num_arguments(self) -> int:  # noqa: D102
        return len(self.argument_names) if self.argument_names is not None else 0

    @property
    def signature(self) -> inspect.Signature:  # noqa: D102
        return inspect.signature(self.fun) if self.fun is not None else None

    @property
    def data(self) -> pd.Series | pd.DataFrame | None:  # noqa: D102
        return self._data

    @property
    def fun(self) -> Callable | None:  # noqa: D102
        return self._fun

    @property
    def argument_names(self) -> list[Hashable] | None:  # noqa: D102
        return self._argument_names
