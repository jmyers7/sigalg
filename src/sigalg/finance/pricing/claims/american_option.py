"""A marker class for an American option."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .claim import Claim

if TYPE_CHECKING:
    from numbers import Real


class AmericanOption(Claim):
    """A marker class for an American option."""

    _repr_name = "AmericanOption"
    _str_name = "American option"

    # --------------------- properties --------------------- #

    @property
    def strike(self) -> Real | None:
        """Get the strike price of the option."""
        return self._strike

    @property
    def option_type(self) -> Literal["call", "put"] | None:
        """Get the option type."""
        return self._option_type
