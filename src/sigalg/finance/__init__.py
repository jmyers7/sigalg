"""Module containing components for financial mathematics, including derivative pricing models."""

from .pricing import BinomialPricingModel, discount, european_option

__all__ = [
    "BinomialPricingModel",
    "european_option",
    "discount",
]
