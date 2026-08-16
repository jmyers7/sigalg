from __future__ import annotations

from collections.abc import Hashable  # noqa: TC003
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from ..typing.index_like import IndexLike  # noqa: TC001


class DomainIndexValidator(BaseModel):
    """Validate Domain/Index pairs for SigAlg objects that are constructed from them.

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The domain object to be validated.
    domain_kind : Literal["Domain", "SampleSpace"], default="Domain"
        The kind of the domain object.
    domain_name : Hashable | None, default=None
        The name of the domain object.
    index : IndexLike | None, default=None
        The index object to be validated.
    index_kind : Literal["Index", "Time"], default="Index"
        The kind of the index object.
    index_name : Hashable | None, default=None
        The name of the index object.

    Examples
    --------
    >>> import sigalg as sa

    `Domain` and `Index` objects are preserved as is, no matter the values of the other parameters.

    >>> domain = sa.Domain.from_sequence(size=2, name="X")
    >>> index = sa.Index.from_sequence(size=2, name="I")
    >>> v = sa.validation.DomainIndexValidator(
    ...     domain=domain,
    ...     domain_kind="SampleSpace",
    ...     domain_name="Omega",
    ...     index=index,
    ...     index_kind="Time",
    ...     index_name="T",
    ... )
    >>> v
    domain = Domain(num_points=2, name=X)
    domain_kind = 'Domain'
    domain_name = 'X'
    index = Index(num_indices=2, name=I)
    index_kind = 'Index'
    index_name = 'I'
    >>> v.domain is domain
    True
    >>> v.index is index
    True

    `IndexLike` objects may be passed into the validator and they will be coerced into objects determined by the other parameters pass into the validator.

    >>> domain = [0, 1]
    >>> index = [0, 1]
    >>> v = sa.validation.DomainIndexValidator(
    ...     domain=domain,
    ...     domain_kind="SampleSpace",
    ...     domain_name="S",
    ...     index=index,
    ...     index_kind="Time",
    ...     index_name="Q",
    ... )
    >>> v
    domain = SampleSpace(num_samples=2, name=S)
    domain_kind = 'SampleSpace'
    domain_name = 'S'
    index = Time(start=0, stop=1, is_discrete=True, name=Q)
    index_kind = 'Time'
    index_name = 'Q'

    The `name` parameters are optional, and will default to the names determined by the `kind` parameters.

    >>> domain = [0, 1]
    >>> index = [0, 1]
    >>> v = sa.validation.DomainIndexValidator(
    ...     domain=domain,
    ...     domain_kind="SampleSpace",
    ...     index=index,
    ...     index_kind="Time",
    ... )
    >>> v
    domain = SampleSpace(num_samples=2, name=Omega)
    domain_kind = 'SampleSpace'
    domain_name = 'Omega'
    index = Time(start=0, stop=1, is_discrete=True, name=T)
    index_kind = 'Time'
    index_name = 'T'

    The `kind` parameters are also optional, and will default to `Domain` and `Index`.

    >>> domain = [0, 1]
    >>> index = [0, 1]
    >>> v = sa.validation.DomainIndexValidator(
    ...     domain=domain,
    ...     index=index,
    ... )
    >>> v
    domain = Domain(num_points=2, name=X)
    domain_kind = 'Domain'
    domain_name = 'X'
    index = Index(num_indices=2, name=I)
    index_kind = 'Index'
    index_name = 'I'

    However, even if the `kind` parameters are not passed, the `name` parameters still work.

    >>> domain = [0, 1]
    >>> index = [0, 1]
    >>> v = sa.validation.DomainIndexValidator(
    ...     domain=domain,
    ...     domain_name="Y",
    ...     index=index,
    ...     index_name="J",
    ... )
    >>> v
    domain = Domain(num_points=2, name=Y)
    domain_kind = 'Domain'
    domain_name = 'Y'
    index = Index(num_indices=2, name=J)
    index_kind = 'Index'
    index_name = 'J'
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: IndexLike | None = None
    domain_kind: Literal["Domain", "SampleSpace"] = "Domain"
    domain_name: Hashable | None = None
    index: IndexLike | None = None
    index_kind: Literal["Index", "Time"] = "Index"
    index_name: Hashable | None = None

    @model_validator(mode="after")
    def coerce_domain(self) -> DomainIndexValidator:  # noqa: D102
        from ..core.spaces.domain import Domain
        from ..core.spaces.sample_space import SampleSpace

        if self.domain is not None:
            if not isinstance(self.domain, Domain):
                domain_class = Domain if self.domain_kind == "Domain" else SampleSpace
                self.domain_name = (
                    self.domain_name if self.domain_name else domain_class._default_name
                )
                self.domain = domain_class(self.domain, name=self.domain_name)

            self.domain_name = self.domain.name
            self.domain_kind = type(self.domain).__name__

        return self

    @model_validator(mode="after")
    def coerce_index(self) -> DomainIndexValidator:  # noqa: D102
        from ..core.indices.index import Index
        from ..core.indices.time import Time

        if self.index is not None:
            if not isinstance(self.index, Index):
                index_class = Index if self.index_kind == "Index" else Time
                self.index_name = (
                    self.index_name if self.index_name else index_class._default_name
                )
                self.index = index_class(self.index, name=self.index_name)

            self.index_name = self.index.name
            self.index_kind = type(self.index).__name__

        return self

    def __repr__(self):  # noqa: D105
        return "\n".join(
            f"{name} = {repr(value)}" for name, value in vars(self).items()
        )
