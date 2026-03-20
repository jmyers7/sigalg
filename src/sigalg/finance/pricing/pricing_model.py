"""Later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import TYPE_CHECKING

from ...processes.base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ..claims.claim import Claim


class PricingModel(ABC, StochasticProcess):
    """Abstract base class for various types of pricing models."""

    @abstractmethod
    def replicating_portfolio(
        self, claim: Claim
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        r"""Compute the replicating portfolio for a given contingent claim relative to this pricing model.

        Parameters
        ----------
        claim : Claim
            A contingent claim for which to compute the replicating portfolio.

        Returns
        -------
           bank_value, underlying_units, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
               A tuple containing the bank account process, the underlying units process, the total portfolio value process, and the risk-neutral price of the claim.
        """
        pass
