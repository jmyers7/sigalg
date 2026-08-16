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

from ..core.indices.index import Index  # noqa: TC001
from ..core.spaces.domain import Domain  # noqa: TC001
from ..typing.mapping_like import MappingLike  # noqa: TC001

PandasLike = pd.Series | pd.DataFrame


class MappingValidator(BaseModel):
    """Validate input data for SigAlg objects that are constructed from `MappingLike` objects.

    Parameters
    ----------
    domain : Domain | None, default=None
        The domain against which to validate the mapping.
    mapping : MappingLike | None, default=None
        The mapping to be validated.
    kind : Literal["any", "measure", "probability", "param_measure", "param_probability"], default="any"
        The kind of mapping being validated.
    domain_kind : Literal["Domain", "SampleSpace"], default="Domain"
        The kind of the domain object.
    index : Index | None, default=None
        The index against which to validate the mapping.
    index_kind : Literal["Index", "Time"], default="Index"
        The kind of the index object.
    multi_dim_outputs : bool, default=False
        Whether the outputs of the mapping are multi-dimensional.
    output_name : Hashable | None, default=None
        The name of the output dimension of the mapping. Only used if the outputs are 1-dimensional.
    name : Hashable | None, default=None
        The name of the mapping.

    Examples
    --------
    >>> import pandas as pd
    >>> import sigalg as sa

    Coerce an "out-of-order" dict into a `pd.Series` with the correct order against a provided `SampleSpace`.

    >>> Omega = sa.SampleSpace(["a", "b", "c"], variable_names=["omega"])
    >>> mapping = {"b": 2, "a": 1, "c": 3}
    >>> v = sa.validation.MappingValidator(mapping=mapping, domain=Omega, name="X")
    >>> print(v.data)  # doctest: +NORMALIZE_WHITESPACE
    omega
    a    1
    b    2
    c    3
    dtype: int64

    Coerce an "out-of-order" dict into a `pd.DataFrame` with the correct order against a provided `SampleSpace` with column labels from a provided `Index`.

    >>> Omega = sa.SampleSpace(["a", "b", "c"], variable_names=["omega"])
    >>> mapping = {"b": (3, 4), "a": (1, 2), "c": (5, 6)}
    >>> I = sa.Index(["odd", "even"])
    >>> v = sa.validation.MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
    >>> print(v.data)  # doctest: +NORMALIZE_WHITESPACE
    i      odd  even
    omega
    a        1     2
    b        3     4
    c        5     6

    Generate a default (2-dimensional) `SampleSpace` and `Index` from a plain `dict`.

    >>> mapping = {("a", 1): (1, 2), ("b", 2): (3, 4), ("c", 3): (5, 6)}
    >>> v = MappingValidator(mapping=mapping, name="X")
    >>> print(v.data)  # doctest: +NORMALIZE_WHITESPACE
    i        0  1
    x_0 x_1
    a   1    1  2
    b   2    3  4
    c   3    5  6

    Correct an "out-of-order" `pd.DataFrame` against a provided `SampleSpace` and `Index`.

    >>> mapping = pd.DataFrame(
    ...     [
    ...         (4, 3),
    ...         (2, 1),
    ...         (6, 5),
    ...     ],
    ...     index=["b", "a", "c"],
    ...     columns=pd.Index(["even", "odd"], name="i"),
    ... )
    >>> v = MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
    >>> print(v.data)  # doctest: +NORMALIZE_WHITESPACE
    i      odd  even
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
    mapping: MappingLike | None = None
    kind: Literal[
        "any", "measure", "probability", "param_measure", "param_probability"
    ] = "any"
    domain_kind: Literal["Domain", "SampleSpace"] = "Domain"
    index: Index | None = None
    index_kind: Literal["Index", "Time"] = "Index"
    multi_dim_outputs: bool = False
    output_name: Hashable | None = None
    parameter_names: list[Hashable] | None = None
    name: Hashable | None = None

    _data: PandasLike | None = PrivateAttr(default=None)
    _function: Callable | None = PrivateAttr(default=None)
    _argument_names: list[Hashable] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_name(self) -> MappingValidator:
        """Set the name of the `mapping` using `output_name`, if the former is a `pd.Series` and the latter is provided.

        Rules: If `output_name` is not `None`, overwrite the name of the series with `output_name`.
        """
        if isinstance(self.mapping, pd.Series) and self.output_name is not None:
            self.mapping.name = self.output_name

        return self

    @model_validator(mode="after")
    def validate_function_and_domain(self) -> MappingValidator:
        """Validate the `mapping` against the `domain` if the former is a callable and the latter is provided.

        Rules:

        1. If `domain` is not `None`, check that the argument names of the callable match the variable names of the domain. If they do not match, raise an exception.

        2. Check that all arguments of the callable are either all keyword-only arguments, or all positional keywords. If they are not, raise an exception.
        """
        if isinstance(self.mapping, Callable):
            sig = inspect.signature(self.mapping)
            param_names = list(sig.parameters.keys())
            params = sig.parameters.values()
            domain_var_names = self.domain.variable_names if self.domain else None

            if all(
                param.kind
                in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
                for param in params
            ):
                if self.domain is not None and set(domain_var_names) != set(
                    param_names
                ):
                    raise ValueError(
                        "The provided function's arguments do not match the domain's variable names."
                    )

            elif (
                all(
                    (
                        param.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.VAR_POSITIONAL,
                        )
                    )
                    for param in params
                )
                or len(param_names) == 1
            ):
                if self.domain is not None:
                    if len(domain_var_names) != len(param_names):
                        raise ValueError(
                            "If all parameters of the mapping (callable) are positional-only, or if there is only one parameter, and if the domain is given, the number of parameters must equal the number of domain variables."
                        )

                    domain_params = [
                        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                        for name in domain_var_names
                    ]
                    keyword_sig = inspect.Signature(domain_params)
                    original_mapping = self.mapping

                    def keyword_only_mapping(**kwargs):
                        keyword_to_pos = sig.bind(*kwargs.values())
                        return original_mapping(*keyword_to_pos.arguments.values())

                    keyword_only_mapping.__signature__ = keyword_sig
                    self.mapping = keyword_only_mapping

            else:
                raise ValueError(
                    "The mapping (if a callable) must have either all keyword-only parameters or positional-only parameters. In the former case, the names of the parameters must match the variable names of the domain (if given). In the latter case, the number of parameters must equal the number of domain variables (if the domain is given), and the mapping will be converted to a keyword-only mapping using the domain variable names."
                )

        return self

    @model_validator(mode="after")
    def validate_pandas_and_domain(self) -> MappingValidator:
        """Validate the `mapping` against the `domain` (if provided) if the former is a `pd.Series` or `pd.DataFrame` whose index has nonempty level names.

        Rules:

        1. The `mapping` index and `domain.data` must either both be a `pd.MultiIndex`, or not. If this is not true, raise an exception.

        2. If the `mapping` index has level names (or a name) that are/is not `None`, check they/it matches the variable names of the domain as a set. If they do not match, raise an exception. If the `mapping` index is a `pd.MultiIndex`, reorder the level names to match the domain variable names.

        3. Check that the mapping contains an entry for every point in the domain. If not, raise an exception.

        4. Reorder the mapping index to match the domain.
        """
        if isinstance(self.mapping, PandasLike) and self.domain is not None:
            if isinstance(self.mapping.index, pd.MultiIndex) != isinstance(
                self.domain.data, pd.MultiIndex
            ):
                raise TypeError(
                    "Both the index of the mapping and the underlying data of the domain must be multi-indices, or they must both be not"
                )
            if self.mapping.index.nlevels != self.domain.data.nlevels:
                raise ValueError(
                    "The number of levels of the mapping index must match the number of variables of the domain."
                )

            if set(self.mapping.index.names) != {None}:
                if set(self.mapping.index.names) != set(self.domain.variable_names):
                    raise ValueError(
                        "The names of the levels of the index of mapping do not match the variable names of the provided domain."
                    )

                if isinstance(self.mapping.index, pd.MultiIndex):
                    self.mapping = self.mapping.reorder_levels(
                        self.domain.variable_names
                    )

            if len(self.mapping.index) != len(self.domain) or set(
                self.mapping.index
            ) != set(self.domain):
                raise ValueError(
                    "The mapping must contain an entry for every point in the domain."
                )

            self.mapping = self.mapping.reindex(self.domain.data)

        return self

    @model_validator(mode="after")
    def generate_domain(self) -> MappingValidator:
        """Generate domain variable names if `mapping` is a `pd.Series` or `pd.DataFrame` whose index has empty level names and `domain` is not provided."""
        from .index_validator import IndexValidator

        if (
            isinstance(self.mapping, PandasLike)
            and set(self.mapping.index.names) == {None}
            and self.domain is None
        ):
            if self.domain_kind == "Domain":
                variable_names_prefix = "x"
            elif self.domain_kind == "SampleSpace":
                variable_names_prefix = "s"

            self.mapping.index = IndexValidator(
                indices=self.mapping.index,
                variable_names_prefix=variable_names_prefix,
            ).data

        return self

    @model_validator(mode="after")
    def generate_data(self) -> MappingValidator:
        """Generate `data`.

        Rules:

        1. If the `mapping` is a callable and the `domain` is not `None`, generate a `pd.Series` or `pd.DataFrame` by applying the callable to each point in the `domain`. Set this to `data`.

        2. If the `mapping` is a `pd.Series` or `pd.DataFrame`, set the `data` to the `mapping`.

        3. If the `mapping` is a callable and `domain` is `None`, set `data` to `mapping`.

        4. If the `mapping` is `None`, set the data to `None`.
        """
        if isinstance(self.mapping, Callable) and self.domain is not None:
            self._data = self.domain.data.to_frame().apply(
                lambda x: self.mapping(**x), axis=1, result_type="expand"
            )
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

        elif isinstance(self.mapping, Callable) or isinstance(self.mapping, PandasLike):
            self._data = self.mapping

        else:
            self._data = None

        return self

    @model_validator(mode="after")
    def validate_index(self) -> MappingValidator:
        """Validate `data` against `index` (if given), in the case that `data` is a `pd.DataFrame` whose columns are not the default.

        Rules:

        1. Check that the underlying data of `index` is not a `pd.MultiIndex`. If it is, raise an exception.

        2. Check that the length of the `index` matches the number of columns in the `data`. If it does not, raise an exception.

        3. Check that the set of columns matches the set of the `index`. If they do not match, raise an exception.

        4. Check that the name of the column index matches the name of the `index`. If they do not match, raise an exception.

        5. Reindex (i.e., re-order) the `data` against the `index`.
        """
        if isinstance(self.data, pd.DataFrame) and self.index is not None:
            default_cols = pd.Index(range(self.data.shape[1]))
            if not (
                self.data.columns.equals(default_cols)
                and self.data.columns.name is None
            ):
                if isinstance(self.index.data, pd.MultiIndex):
                    raise ValueError(
                        "The mapping columns cannot be validated against a pd.MultiIndex."
                    )
                if self.data.shape[1] != len(self.index):
                    raise ValueError(
                        "The length of the provided index does not match the dimension of the outputs of the mapping."
                    )
                if set(self.data.columns) != set(self.index):
                    raise ValueError(
                        "The indices in the provided index do not match the column names of the provided mapping."
                    )

                if self.data.columns.name != self.index.variable_names[0]:
                    raise ValueError(
                        "The name of the column index of the mapping must match the name of the provided Index."
                    )

                self._data = self.data.reindex(columns=self.index.data)

        return self

    @model_validator(mode="after")
    def generate_index(self) -> MappingValidator:
        """Generate the `index` variable name if `index` is `None` and if `data` is a `pd.DataFrame` with whose column index has no name."""
        if (
            isinstance(self.data, pd.DataFrame)
            and self.index is None
            and self.data.columns.name is None
        ):
            if self.index_kind == "Index":
                self.data.columns.name = "i"
            elif self.index_kind == "Time":
                self.data.columns.name = "t"

        return self

    @model_validator(mode="after")
    def overwrite_default_index(self) -> MappingValidator:
        """If `data` is a `pd.DataFrame` with default columns and `index` is provided, overwrite the columns."""
        if isinstance(self.data, pd.DataFrame) and self.index is not None:
            default_cols = pd.Index(range(self.data.shape[1]))
            if (
                self.data.columns.equals(default_cols)
                and self.data.columns.name is None
            ):
                # if isinstance(self.index.data, pd.MultiIndex):
                #     raise ValueError("Cannot set the column index to a pd.MultiIndex.")
                if self.data.shape[1] != len(self.index):
                    raise ValueError(
                        "The length of the provided index does not match the dimension of the outputs of the mapping."
                    )
                self.data.columns = self.index.data

        return self

    @model_validator(mode="after")
    def validate_kind(self) -> MappingValidator:
        """Validate the `kind` parameter against the `data`.

        Rules:

        1. If the `data` is not `None` and the `kind` is "measure" or "probability", check that all values in the `data` are non-negative. If any value is negative, raise an exception.

        2. If the `kind` is "probability", check that the sum of the values in the `data` is equal to 1 (within a tolerance of 1e-8). If the sum is not equal to 1, raise an exception.

        3. If the `data` is not a `pd.Series` when the `kind` is "probability" or "measure", raise an exception.
        """
        if self.data is not None:
            self.validate_mapping_kind(
                data=self.data, kind=self.kind, parameter_names=self.parameter_names
            )

        return self

    @staticmethod
    def validate_mapping_kind(  # noqa: D102
        data: pd.Series,
        kind: Literal[
            "any",
            "measure",
            "probability",
            "param_measure",
            "param_probability",
        ] = "any",
        parameter_names: list[Hashable] | None = None,
    ) -> None:
        if kind == "measure" or kind == "probability":
            if isinstance(data, pd.Series):
                if (data < 0).any():
                    raise ValueError(
                        "All measure values in the mapping must be non-negative."
                    )

                if kind == "probability" and np.abs(data.sum() - 1.0) >= 1e-8:
                    raise ValueError("Probability values must sum to 1.")

            else:
                raise ValueError(
                    "data must be a pd.Series when kind is 'probability' or 'measure'."
                )

        elif kind == "param_measure" or kind == "param_probability":
            if isinstance(data, pd.Series):
                if (data < 0).any():
                    raise ValueError(
                        "All measure values in the mapping must be non-negative."
                    )

                if kind == "param_probability" and not all(
                    np.abs(data.groupby(parameter_names).sum() - 1) < 1e-8
                ):
                    raise ValueError(
                        "For each parameter level, the values of the probability measure must sum to 1."
                    )

            else:
                raise ValueError(
                    "data must be a pd.Series when kind is 'probability' or 'measure'."
                )

    @property
    def data(self) -> PandasLike | None:  # noqa: D102
        return self._data
