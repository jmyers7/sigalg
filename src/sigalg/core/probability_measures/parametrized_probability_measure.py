from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base.sample_space import SampleSpace
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .probability_measure import ProbabilityMeasure


class ParametrizedProbabilityMeasure:
    """Pass."""

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sig_alg: SigmaAlgebra | None = None,
        name: Hashable | None = "P",
    ) -> None:
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra or None.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable object or None.")

        self._sig_alg = sig_alg
        self._name = name

        # caches
        self._sample_space: SampleSpace | None = None
        self._parametrization: Callable[..., dict[Hashable, Real]] | None = None
        self._parameter_names: list[Hashable] | None = None

    def from_callable(
        self,
        parametrization: Callable[..., dict[Hashable, Real]],
        parameter_names: list[Hashable] | None = None,
    ) -> ParametrizedProbabilityMeasure:
        """Pass."""
        if not isinstance(parametrization, Callable):
            raise TypeError("parametrization must be a callable object.")

        self._parametrization = parametrization
        self._parameter_names = (
            parameter_names
            if parameter_names is not None
            else list(inspect.signature(parametrization).parameters.keys())
        )
        return self

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Pass."""
        return self._sig_alg

    @property
    def sample_space(self) -> SampleSpace | None:
        """Pass."""
        return self.sig_alg.sample_space if self.sig_alg is not None else None

    @property
    def parametrization(self) -> Callable[..., dict[Hashable, Real]] | None:
        """Pass."""
        return self._parametrization

    @property
    def parameter_names(self) -> list[Hashable] | None:
        """Pass."""
        return self._parameter_names

    @property
    def name(self) -> Hashable | None:
        """Pass."""
        return self._name

    # --------------------- data access methods --------------------- #

    def __call__(self, atom_id: Hashable, **parameters) -> Real:
        """Pass."""
        if atom_id not in self.sig_alg.atom_ids:
            raise ValueError("atom_id must be a valid atom_id in the sigma algebra.")

        matched_parameters = {
            parameter_name: parameter
            for parameter_name, parameter in parameters.items()
            if parameter_name in self.parameter_names
        }
        unmatched_parameters = {
            parameter_name: parameter
            for parameter_name, parameter in parameters.items()
            if parameter_name not in self.parameter_names
        }

        if unmatched_parameters:
            raise ValueError(
                "There are unknown parameters passed into the __call__ method."
            )

        return self.parametrization(**matched_parameters)[atom_id]

    def at(self, **parameters) -> ProbabilityMeasure:
        """Pass."""
        from .probability_measure import ProbabilityMeasure

        matched_parameters = {
            parameter_name: parameter
            for parameter_name, parameter in parameters.items()
            if parameter_name in self.parameter_names
        }
        unmatched_parameters = {
            parameter_name: parameter
            for parameter_name, parameter in parameters.items()
            if parameter_name not in self.parameter_names
        }

        if unmatched_parameters:
            raise ValueError("There are unknown parameters passed into the at method.")

        outputs = self.parametrization(**matched_parameters)
        name = f"{self.name}({', '.join(f'{k}={v}' for k, v in matched_parameters.items())})"

        return ProbabilityMeasure(sig_alg=self.sig_alg, name=name).from_dict(outputs)

    # --------------------- arithmetic operations --------------------- #

    def __add__(
        self, other: ParametrizedProbabilityMeasure
    ) -> ParametrizedProbabilityMeasure:
        """Pass."""
        if not isinstance(other, ParametrizedProbabilityMeasure):
            raise TypeError(
                "other must be an instance of ParametrizedProbabilityMeasure."
            )
        if self.sig_alg != other.sig_alg:
            raise ValueError(
                "Both parametrized probability measures must be defined on the same sigma-algebra."
            )

        def new_parametrization(**parameters) -> dict[Hashable, Real]:
            self_parameters = {
                parameter_name: parameters[parameter_name]
                for parameter_name in self.parameter_names
            }
            other_parameters = {
                parameter_name: parameters[parameter_name]
                for parameter_name in other.parameter_names
            }
            return {
                atom_id: self(atom_id, **self_parameters)
                + other(atom_id, **other_parameters)
                for atom_id in self.sig_alg.atom_ids
            }

        new_name = f"({self.name} + {other.name})"
        return ParametrizedProbabilityMeasure(
            sig_alg=self.sig_alg, name=new_name
        ).from_callable(
            new_parametrization,
            parameter_names=list(
                set(self.parameter_names) | set(other.parameter_names)
            ),
        )
