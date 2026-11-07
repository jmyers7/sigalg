"""A marker class for a European option."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .claim import Claim

if TYPE_CHECKING:
    from numbers import Real


class EuropeanOption(Claim):
    """A marker class for a European option."""

    _repr_name = "EuropeanOption"
    _str_name = "European option"

    # --------------------- properties --------------------- #

    @property
    def strike(self) -> Real | None:
        """Get the strike price of the option."""
        return self._strike

    @property
    def option_type(self) -> Literal["call", "put"] | None:
        """Get the option type."""
        return self._option_type
