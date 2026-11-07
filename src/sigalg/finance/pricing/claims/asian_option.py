"""Later."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .claim import Claim

if TYPE_CHECKING:
    from numbers import Real


class AsianOption(Claim):
    """Pass."""

    _repr_name = "AsianOption"
    _str_name = "Asian option"

    # --------------------- properties --------------------- #

    @property
    def strike(self) -> Real | None:
        """Get the strike price of the option."""
        return self._strike

    @property
    def option_type(self) -> Literal["call", "put"] | None:
        """Get the option type."""
        return self._option_type
