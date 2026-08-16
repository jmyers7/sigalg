from __future__ import annotations  # noqa: I001

from collections.abc import Hashable  # noqa: TC003
from typing import TYPE_CHECKING, Literal  # noqa: F401

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


from ..core.spaces.domain import Domain  # noqa: TC001


class ParametrizedDomainConstructor(BaseModel):
    """Construct parametrized domains for SigAlg objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    component_domain: Domain
    parameter_domain: Domain | None = None
    complete_domain: Domain | None = None
    parameter_names: list[Hashable] | None = None
    parameter_domain_name: Hashable | None = None
    complete_domain_name: Hashable | None = None

    @model_validator(mode="after")
    def normalize_parameter_names(self) -> ParametrizedDomainConstructor:  # noqa: D102
        if self.parameter_names is None and self.parameter_domain is None:
            raise TypeError(
                "One or the other of parameter_domain or parameter_names must be given."
            )

        if self.parameter_domain is not None:
            if self.parameter_names is not None:
                if len(self.parameter_names) != self.parameter_domain.dimension:
                    raise ValueError(
                        "If both parameter_names and parameter_domain are given, the number of parameter names must match the dimension of parameter_domain."
                    )

                if self.parameter_domain.dimension > 1:
                    data = self.parameter_domain.data.rename(self.parameter_names)
                else:
                    data = self.parameter_domain.data.rename(self.parameter_names[0])

                self.parameter_domain = type(self.parameter_domain)._from_validated(
                    data=data, name=self.parameter_domain.name
                )

            else:
                self.parameter_names = self.parameter_domain.variable_names

        return self

    @model_validator(mode="after")
    def normalize_parameter_domain_name(self) -> ParametrizedDomainConstructor:  # noqa: D102
        if self.parameter_domain is not None:
            if self.parameter_domain_name is not None:
                self.parameter_domain.name = self.parameter_domain_name
            else:
                self.parameter_domain_name = self.parameter_domain.name

        elif self.parameter_domain_name is None:
            self.parameter_domain_name = "Theta"

        return self

    @model_validator(mode="after")
    def generate_and_validate_complete_domain(self) -> ParametrizedDomainConstructor:  # noqa: D102
        if self.complete_domain is None:
            if self.parameter_domain is not None:
                self.complete_domain = Domain.cartesian_product(
                    factors=[self.parameter_domain, self.component_domain],
                    name=self.complete_domain_name,
                )
                self.complete_domain_name = (
                    self.complete_domain_name
                    if self.complete_domain_name
                    else self.complete_domain.name
                )

        elif self.parameter_domain is not None:
            raise TypeError("Cannot pass both parameter_domain and complete_domain.")

        else:
            if self.complete_domain_name is not None:
                self.complete_domain.name = self.complete_domain_name
            else:
                self.complete_domain_name = self.component_domain.name

            if (
                self.parameter_names
                != self.complete_domain.variable_names[: len(self.parameter_names)]
            ):
                raise ValueError(
                    "If parameter_names and complete_domain are both given, the parameter names must appear first in the list of variables of the latter."
                )

        return self

    def __repr__(self):  # noqa: D105
        return "\n".join(
            f"{name} = {repr(value)}" for name, value in vars(self).items()
        )
