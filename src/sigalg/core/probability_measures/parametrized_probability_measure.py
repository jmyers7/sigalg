"""A class representing a parametrized probability measure."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd
from scipy.stats import rv_discrete

from ..base.multivariate_function import MultivariateFunction

if TYPE_CHECKING:
    from ..base.domain import Domain
    from ..base.sample_space import SampleSpace
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .probability_measure import ProbabilityMeasure


class ParametrizedProbabilityMeasure(MultivariateFunction):
    r"""A class representing a parametrized probability measure.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra associated with the parametrized probability measure.
    parameter_domain : Domain | None, default=None
        The domain of the parameters for the parametrized probability measure.
    domain : Domain | None, default=None
        The domain of the parametrized probability measure. If not provided, it will be inferred from the sigma-algebra and parameter domain.
    name : Hashable, default="P"
        The name of the parametrized probability measure.

    Raises
    ------
    TypeError
        If `sig_alg` is not a `SigmaAlgebra` instance (if given), or if `parameter_domain` or `domain` is not a `Domain` instance (if given), or if `name` is not hashable.
    ValueError
        If `sig_alg`, `parameter_domain`, and `domain` are not provided in any other combination than (`sig_alg`, `parameter_domain`, `None`), (`sig_alg`, `None`, `domain`), (`None`, `parameter_domain`, `None`) or (`None`, `None`, `None`).

    Examples
    --------
    >>> from math import comb
    >>> from sigalg.core import (
    ...     Domain,
    ...     ParametrizedProbabilityMeasure,
    ...     SampleSpace,
    ...     SigmaAlgebra,
    ... )
    >>> Omega = SampleSpace().from_sequence(size=3, data_name=["omega"])
    >>> F = SigmaAlgebra.power_set(Omega)
    >>> Theta = Domain(name="Theta").from_list([0.0, 0.25, 0.75, 1.0], data_name=["theta"])
    >>> def P_func(*, theta, omega):
    ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
    >>> P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=Theta).from_callable(
    ...     P_func
    ... )
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized probability measure 'P':
                 probability
    theta omega
    0.00  0           1.0000
          1           0.0000
          2           0.0000
    0.25  0           0.5625
          1           0.3750
          2           0.0625
    0.75  0           0.0625
          1           0.3750
          2           0.5625
    1.00  0           0.0000
          1           0.0000
          2           1.0000

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be an event space and $\Theta$ a nonempty set. A *parametrized probability measure* is a function

    $$
    P : \mathcal{F} \times \Theta \to \mathbb{R}
    $$

    such that, for each fixed $\theta \in \Theta$, the partial function

    $$
    P(-, \theta): \mathcal{F} \to \mathbb{R}, \quad A \mapsto P(A,\theta),
    $$

    is a probability measure on the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain* and elements $\theta\in \Theta$ are called *parameters*.
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sig_alg: SigmaAlgebra | None = None,
        parameter_domain: Domain | None = None,
        domain: Domain | None = None,
        name: Hashable = "P",
    ) -> None:
        from ..base.domain import Domain
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("If given, sig_alg must be a SigmaAlgebra instance.")
        if parameter_domain is not None and not isinstance(parameter_domain, Domain):
            raise TypeError("If given, parameter_domain must be a Domain instance.")
        if domain is not None and not isinstance(domain, Domain):
            raise TypeError("If given, domain must be a Domain instance.")
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable object.")

        parameters_given = (
            sig_alg is not None,
            parameter_domain is not None,
            domain is not None,
        )
        if parameters_given == (1, 1, 1):
            raise ValueError(
                "If sig_alg and parameter_domain are given, domain must be None."
            )
        elif parameters_given == (1, 1, 0):
            tuples = list(product(parameter_domain.data, sig_alg.atom_space.data))
            tuples = [self._flatten(t) for t in tuples]
            data = pd.MultiIndex.from_tuples(
                tuples,
                names=parameter_domain.data.names + sig_alg.atom_space.data.names,
            )
            domain = Domain().from_pandas(data)
        elif parameters_given == (1, 0, 1):
            pass
        elif parameters_given == (1, 0, 0):
            raise ValueError(
                "If sig_alg is given, parameter_domain or domain must also be given (but not both)."
            )
        elif parameters_given == (0, 1, 1):
            raise ValueError(
                "If parameter_domain is given, sig_alg must be given and domain must be None."
            )
        elif parameters_given == (0, 1, 0):
            pass
        elif parameters_given == (0, 0, 1):
            raise ValueError("If domain is given, sig_alg must also be given.")
        elif parameters_given == (0, 0, 0):
            pass

        self._parameter_domain = parameter_domain
        self._sig_alg = sig_alg

        super().__init__(domain=domain, name=name)

    @staticmethod
    def _flatten(t):
        if isinstance(t[0], tuple) and isinstance(t[1], tuple):
            return t[0] + t[1]
        if isinstance(t[0], tuple) and not isinstance(t[1], tuple):
            return t[0] + (t[1],)
        if not isinstance(t[0], tuple) and isinstance(t[1], tuple):
            return (t[0],) + t[1]
        if not isinstance(t[0], tuple) and not isinstance(t[1], tuple):
            return t

    def from_callable(
        self,
        function: Callable,
        output_name: Hashable = "probability",
    ) -> ParametrizedProbabilityMeasure:
        """Initialize the parametrized probability measure from a callable.

        The sigma-algebra must be set before calling this method. The callable should take keyword-only arguments corresponding to the parametrized probability measure's parameters and atom identifiers of the sigma-algebra, and return a real number representing the probability of the atom under the specified parameters.

        Parameters
        ----------
        function : Callable
            A callable that takes keyword-only arguments corresponding to the parametrized probability measure's parameters and atom identifiers of the sigma-algebra, and returns a real number representing the probability of the atom under the specified parameters.
        output_name : Hashable, default="probability"
            The name of the output variable for the parametrized probability measure.

        Notes
        -----
        The method does not check that the provided callable actually yields valid probabilities. It is the user's responsibility to ensure that the callable is valid for the intended use case.

        Examples
        --------
        >>> from math import comb
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedProbabilityMeasure,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=3, data_name=["omega"])
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> Theta = Domain(name="Theta").from_list([0.0, 0.25, 0.75, 1.0], data_name=["theta"])
        >>> def P_func(*, theta, omega):
        ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
        >>> P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=Theta).from_callable(
        ...     P_func
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
                 probability
        theta omega
        0.00  0       1.0000
              1       0.0000
              2       0.0000
        0.25  0       0.5625
              1       0.3750
              2       0.0625
        0.75  0       0.0625
              1       0.3750
              2       0.5625
        1.00  0       0.0000
              1       0.0000
              2       1.0000
        """
        if self.sig_alg is None:
            raise ValueError(
                "Cannot initialize from a callable without a sigma-algebra."
            )
        atom_id_params = self.sig_alg.atom_space.data_name
        callable_params = inspect.signature(function).parameters
        if not set(atom_id_params).issubset(callable_params):
            raise ValueError(
                "The provided callable must accept keyword-only arguments corresponding to the atom identifiers of the sigma-algebra."
            )

        return super().from_callable(function=function, output_name=output_name)

    def from_scipy(
        self,
        dist: rv_discrete,
        support: tuple[Hashable, list],
        output_name: Hashable = "probability",
    ) -> ParametrizedProbabilityMeasure:
        """Initialize the parametrized probability measure from a SciPy probability distribution.

        To use this method, the `parameter_domain` must be set during initialization with appropriate parameter values for the SciPy distribution (see the SciPy docs). The `support` argument should be a tuple containing the name of the support variable and a list of its possible values.

        This method will automatically create a sigma-algebra as the power set of the support and will construct the domain of the parametrized probability measure accordingly.

        Parameters
        ----------
        dist : rv_discrete
            A discrete SciPy probability distribution.
        support : tuple[Hashable, list]
            A tuple containing the name of the support variable and a list of its possible values.
        output_name : Hashable, default="probability"
            The name of the output variable for the parametrized probability measure.

        Raises
        ------
        ValueError
            If `parameter_domain` is not set, or if `sig_alg` is already set (as it will be automatically created), or if `support` is not a 2-tuple.
        TypeError
            If `dist` is not a discrete SciPy distribution (rv_discrete).

        Examples
        --------
        >>> from scipy.stats import binom, hypergeom
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedProbabilityMeasure,
        ... )
        >>> Theta_P = Domain(name="Theta_P").from_list([(2, 0.25), (3, 0.75)], data_name=["n", "p"])
        >>> P = ParametrizedProbabilityMeasure(parameter_domain=Theta_P).from_scipy(
        ...     dist=binom, support=("k", [0, 1, 2, 3])
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
                    probability
        n p    k
        2 0.25 0       0.562500
               1       0.375000
               2       0.062500
               3       0.000000
        3 0.75 0       0.015625
               1       0.140625
               2       0.421875
               3       0.421875
        >>> Theta_Q = Domain(name="Theta_Q").from_list(
        ...     [(5, 3, 3), (10, 5, 5)], data_name=["M", "n", "N"]
        ... )
        >>> Q = ParametrizedProbabilityMeasure(parameter_domain=Theta_Q, name="Q").from_scipy(
        ...     dist=hypergeom, support=("k", [0, 1, 2, 3, 4, 5])
        ... )
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q':
                  probability
        M  n N k
        5  3 3 0     0.000000
               1     0.300000
               2     0.600000
               3     0.100000
               4     0.000000
               5     0.000000
        10 5 5 0     0.003968
               1     0.099206
               2     0.396825
               3     0.396825
               4     0.099206
               5     0.003968
        """
        from ..base.domain import Domain
        from ..base.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self.parameter_domain is None:
            raise ValueError("parameter_domain must be set before calling from_scipy")
        if self.sig_alg is not None:
            raise ValueError(
                "sig_alg must be None before calling from_scipy, as it will be automatically created."
            )
        if not isinstance(dist, rv_discrete):
            raise TypeError("dist must be a discrete scipy distribution (rv_discrete)")
        if not isinstance(support, tuple) or len(support) != 2:
            raise ValueError("support must be a 2-tuple (name, values)")
        if not isinstance(support[0], Hashable):
            raise TypeError("support[0] must be hashable")
        if not isinstance(support[1], list):
            raise TypeError("support[1] must be a list")

        sample_space = SampleSpace().from_list(support[1], data_name=[support[0]])
        self._sig_alg = SigmaAlgebra.power_set(sample_space)
        parameters = self.parameter_domain.data_name

        tuples = list(
            product(self.parameter_domain.data, self._sig_alg.atom_space.data)
        )
        tuples = [self._flatten(t) for t in tuples]
        data = pd.MultiIndex.from_tuples(
            tuples,
            names=self.parameter_domain.data.names
            + self._sig_alg.atom_space.data.names,
        )
        self._domain = Domain().from_pandas(data)

        parameter_names = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in parameters
        ] + [inspect.Parameter(support[0], inspect.Parameter.KEYWORD_ONLY)]
        sig = inspect.Signature(parameter_names)

        def func(**kwargs):
            bound = sig.bind(**kwargs)
            return dist.pmf(**bound.arguments)

        func.__signature__ = sig

        return self.from_callable(function=func, output_name=output_name)

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra of the multivariate function.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra associated with the multivariate function, or None if not set.
        """
        return self._sig_alg

    @property
    def sample_space(self) -> SampleSpace | None:
        """Get the sample space of the multivariate function.

        Returns
        -------
        sample_space : SampleSpace | None
            The sample space associated with the multivariate function, or None if the sigma-algebra is not set.
        """
        return self.sig_alg.sample_space if self.sig_alg is not None else None

    @property
    def parameter_domain(self) -> Domain | None:
        """Get the parameter domain of the parametrized probability measure.

        Returns
        -------
        parameter_domain : Domain | None
            The parameter domain associated with the parametrized probability measure, or None if not set.
        """
        return self._parameter_domain

    # --------------------- data access methods --------------------- #

    def __call__(
        self, *args, **kwargs
    ) -> (
        Real
        | MultivariateFunction
        | ParametrizedProbabilityMeasure
        | ProbabilityMeasure
    ):
        """Call the parametrized probability measure with the given parameters.

        The __call__ method can be used in several ways:

        1. If called with a single positional argument, or a single keyword argument `event`, the argument is assumed to either be an instance of `Event` in the sigma-algebra or a list of sample points corresponding to an event in the sigma-algebra. The method will return an instance of `MultivariateFunction` representing the parametrized probability measure evaluated at the given event.
        2. If called with a single positional argument and keyword arguments, the positional argument is assumed to be an instance of `Event` in the sigma-algebra or a list of sample points corresponding to an event in the sigma-algebra. The method will return one of two objects: If all parameters are provided as keyword arguments, it will return a `Real` value representing the probability of the event under the specified parameters. If not all parameters are provided, it will return an instance of `MultivariateFunction` representing the parametrized probability measure evaluated at the given event and the specified parameters.
        3. If called with no positional arguments and an `event` keyword argument, the method will return one of the objects described in points 1 and 2, depending on whether all, some, or no parameters are provided as keyword arguments.
        4. If called with no positional arguments and keyword arguments corresponding to parameters in the parameter domain, the method will return a `ParametrizedProbabilityMeasure` instance if not all parameters are provided, or a `ProbabilityMeasure` instance if all parameters are provided.

        Parameters
        ----------
        *args : tuple
            Positional arguments. If a single positional argument is provided, it is assumed to be an instance of `Event` in the sigma-algebra or a list of sample points corresponding to an event in the sigma-algebra.
        **kwargs : dict
            Keyword arguments. If an `event` keyword argument is provided, it is assumed to be an instance of `Event` in the sigma-algebra or a list of sample points corresponding to an event in the sigma-algebra. Other keyword arguments are assumed to correspond to parameters in the parameter domain.

        Raises
        ------
        ValueError
            If an invalid combination of positional and keyword arguments is provided, or if the event (if passed) is not in the domain of the parametrized probability measure.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedProbabilityMeasure,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: ("a", "a"),
        ...         1: ("a", "b"),
        ...         2: ("b", "c"),
        ...         3: ("b", "c"),
        ...     }
        ... )
        >>> parameter_domain = Domain().from_list(
        ...     [(0, 0), (0, 1), (1, 1)], data_name=["alpha", "beta"]
        ... )
        >>> def P_func(*, alpha, beta, F_0, F_1):
        ...     if (alpha, beta, F_0, F_1) == (0, 0, "a", "a"):
        ...         return 0.1
        ...     elif (alpha, beta, F_0, F_1) == (0, 0, "a", "b"):
        ...         return 0.2
        ...     elif (alpha, beta, F_0, F_1) == (0, 0, "b", "c"):
        ...         return 0.7
        ...     elif (alpha, beta, F_0, F_1) == (0, 1, "a", "a"):
        ...         return 0.3
        ...     elif (alpha, beta, F_0, F_1) == (0, 1, "a", "b"):
        ...         return 0.3
        ...     elif (alpha, beta, F_0, F_1) == (0, 1, "b", "c"):
        ...         return 0.4
        ...     elif (alpha, beta, F_0, F_1) == (1, 1, "a", "a"):
        ...         return 0.5
        ...     elif (alpha, beta, F_0, F_1) == (1, 1, "a", "b"):
        ...         return 0.3
        ...     elif (alpha, beta, F_0, F_1) == (1, 1, "b", "c"):
        ...         return 0.2
        >>> P = ParametrizedProbabilityMeasure(
        ...     sig_alg=F, parameter_domain=parameter_domain
        ... ).from_callable(P_func)
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
                            probability
        alpha beta F_0 F_1
        0     0    a   a            0.1
                       b            0.2
                   b   c            0.7
              1    a   a            0.3
                       b            0.3
                   b   c            0.4
        1     1    a   a            0.5
                       b            0.3
                   b   c            0.2
        >>> A = F.get_event([0, 1])
        >>> # Call with an `Event` instance as a positional argument
        >>> print(P(A))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)':
                    probability
        alpha beta
        0     0             0.3
              1             0.6
        1     1             0.8
        >>> # Call with an `Event` instance as a keyword argument
        >>> print(P(event=A))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)':
                    probability
        alpha beta
        0     0             0.3
              1             0.6
        1     1             0.8
        >>> # Call with a list of sample points as a positional argument
        >>> print(P([0, 1]))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)':
                    probability
        alpha beta
        0     0             0.3
              1             0.6
        1     1             0.8
        >>> # Call with a list of sample points as a keyword argument
        >>> print(P(event=[0, 1]))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)':
                    probability
        alpha beta
        0     0             0.3
              1             0.6
        1     1             0.8
        >>> # Call with an `Event` instance as a positional argument and some parameters as keyword arguments
        >>> print(P(A, beta=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)(beta=1)':
               probability
        alpha
        0              0.6
        1              0.8
        >>> # Call with a list of sample points as a positional argument and some parameters as keyword arguments
        >>> print(P([0, 1], beta=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)(beta=1)':
               probability
        alpha
        0              0.6
        1              0.8
        >>> # Call with an `Event` instance as a keyword argument and some parameters as keyword arguments
        >>> print(P(event=A, beta=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)(beta=1)':
               probability
        alpha
        0              0.6
        1              0.8
        >>> # Call with a list of sample points as a keyword argument and some parameters as keyword arguments
        >>> print(P(event=[0, 1], beta=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'P(A)(beta=1)':
               probability
        alpha
        0              0.6
        1              0.8
        >>> # Call with an `Event` instance as a positional argument and all parameters as keyword arguments
        >>> print(round(P(A, alpha=0, beta=0), 1))
        0.3
        >>> # Call with a list of sample points as a positional argument and all parameters as keyword arguments
        >>> print(round(P([0, 1], alpha=0, beta=0), 1))
        0.3
        >>> # Call with an `Event` instance as a keyword argument and all parameters as keyword arguments
        >>> print(round(P(event=A, alpha=0, beta=0), 1))
        0.3
        >>> # Call with a list of sample points as a keyword argument and all parameters as keyword arguments
        >>> print(round(P(event=[0, 1], alpha=0, beta=0), 1))
        0.3
        >>> # Call with some parameters as keyword arguments but no event
        >>> print(P(beta=1))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P(beta=1)':
                       probability
        alpha F_0 F_1
        0     a   a            0.3
                  b            0.3
              b   c            0.4
        1     a   a            0.5
                  b            0.3
              b   c            0.2
        >>> # Call with all parameters as keyword arguments but no event
        >>> print(P(alpha=0, beta=0))  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P(alpha=0, beta=0)':
                 probability
        F_0 F_1
        a   a            0.1
            b            0.2
        b   c            0.7
        >>>
        >>> # Obtain a probability measure with iterative calls
        >>> print(P(beta=1)(alpha=0))  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P(beta=1)(alpha=0)':
                 probability
        F_0 F_1
        a   a            0.3
            b            0.3
        b   c            0.4
        """
        from ..base.event import Event
        from .probability_measure import ProbabilityMeasure

        if self.sig_alg is None or self.domain is None:
            return super().__call__(*args, **kwargs)

        else:
            if len(args) == 0 and len(kwargs) == 1 and "event" in kwargs:
                event = kwargs.pop("event")
                args = (event,)

            if len(args) == 1:
                event = args[0]

                if isinstance(event, list):
                    event = self.sig_alg.get_event(event)
                elif not isinstance(event, Event):
                    raise TypeError(
                        "The provided event (as a positional or keyword argument) must be an instance of Event or a list of sample points."
                    )

                if not event.sig_alg <= self.sig_alg:
                    raise ValueError(
                        "Event is not in the domain of the parametrized probability measure."
                    )

                atom_coordinates = self.sig_alg.atom_space.data_name
                sig_alg_data_df = pd.DataFrame(
                    self.sig_alg.data.to_list(), columns=atom_coordinates
                )
                combined_data = pd.concat(
                    [
                        event.indicator.data,
                        sig_alg_data_df,
                    ],
                    axis=1,
                )
                atom_indicator = (
                    combined_data.drop_duplicates()
                    .set_index(atom_coordinates)
                    .squeeze()
                )
                data = (
                    self.data.unstack(atom_coordinates)
                    .dot(atom_indicator)
                    .rename("probability")
                )

                function = MultivariateFunction(
                    domain=self.parameter_domain,
                    name=f"{self.name}({event.name})",
                ).from_pandas(data)

                if not kwargs:
                    return function
                else:
                    return function(**kwargs)

            elif len(args) == 0 and "event" in kwargs:
                event = kwargs.pop("event")
                return self.__call__(event, **kwargs)

            elif len(args) == 0:
                atom_names = set(self.sig_alg.atom_space.data_name)
                param_names = {
                    name
                    for name in self.domain.data_name
                    if name not in self.sig_alg.atom_space.data_name
                }
                provided_names = set(kwargs.keys())

                if provided_names & atom_names:
                    return super().__call__(**kwargs)

                if not provided_names <= param_names:
                    unknown_names = provided_names - param_names
                    raise ValueError(
                        f"Unknown parameter names: {unknown_names}. "
                        f"Expected parameters from {param_names}"
                    )

                partial_function = super().__call__(**kwargs)

                if provided_names == param_names:
                    return ProbabilityMeasure(
                        sig_alg=self.sig_alg, name=partial_function.name
                    ).from_pandas(partial_function.data)
                else:
                    return ParametrizedProbabilityMeasure(
                        sig_alg=self.sig_alg,
                        domain=partial_function.domain,
                        name=partial_function.name,
                    ).from_callable(partial_function.function)

            else:
                raise ValueError(
                    "Invalid combination of positional and keyword arguments."
                )

    # --------------------- representation --------------------- #

    def __repr__(self):
        """Pass."""
        if self.function is None:
            return f"Parametrized probability measure '{self.name}': empty"
        else:
            if self.data is not None:
                return f"Parametrized probability measure '{self.name}':\n{self.data.to_frame()}"
            else:
                parameter_list = ", ".join(self.argument_names)
                return (
                    f"Parametrized probability measure '{self.name}({parameter_list})'"
                )
