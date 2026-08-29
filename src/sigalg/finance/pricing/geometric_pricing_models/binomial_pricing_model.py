"""A class modeling a binomial pricing model."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import bernoulli, binom

from .geometric_pricing_model import GeometricPricingModel

if TYPE_CHECKING:
    from ....core.indices.index import Index
    from ....core.measures.probability_measure import ProbabilityMeasure
    from ....processes.base.stochastic_process import StochasticProcess
    from ....processes.stopping_times.stopping_time import StoppingTime
    from ..claims.claim import Claim


class BinomialPricingModel(GeometricPricingModel):
    r"""A class modeling a binomial pricing model.

    The base constructor is not meant to be called directly by users. Instead, the user should call the `generate` class method. See the Examples section below for usage.

    See the Notes section below for the mathematical details.

    Examples
    --------
    Given parameters of a binomial model, enumerate all length-3 price trajectories of in `dense` enumeration mode.

    >>> from sigalg.core import Time
    >>> from sigalg.finance import BinomialPricingModel
    >>> T = Time.discrete(length=3)
    >>> S = BinomialPricingModel.generate(
    ...     mode="enum",
    ...     enum_mode="dense",
    ...     initial_price=100,
    ...     up_factor=1.1,
    ...     up_prob=0.7,
    ...     risk_free_rate=0.01,
    ...     index=T,
    ... )
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Binomial price process 'S':
    time      0           1           2           3
    sample
    0       100  110.000000  121.000000  133.100000
    1       100  110.000000  121.000000  110.000000
    2       100  110.000000  100.000000  110.000000
    3       100  110.000000  100.000000   90.909091
    4       100   90.909091  100.000000  110.000000
    5       100   90.909091  100.000000   90.909091
    6       100   90.909091   82.644628   90.909091
    7       100   90.909091   82.644628   75.131480

    Enumerate all length-3 price trajectories in `sparse` enumeration mode.

    >>> S.enum_mode = "sparse"
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Binomial price process 'S':
    time        0           1           2           3
    sample
    0       100.0  110.000000  121.000000  133.100000
    1       100.0   90.909091  100.000000  110.000000
    2       100.0   90.909091   82.644628   90.909091
    3       100.0   90.909091   82.644628   75.131480

    Simulate ten length-3 trajectories in `sim` mode.

    >>> S.n_trajectories = 10
    >>> S.random_state = 42
    >>> S.mode = "sim"
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Binomial price process 'S':
    time      0           1           2           3
    sample
    0       100   90.909091  100.000000   90.909091
    1       100  110.000000  121.000000  110.000000
    2       100   90.909091   82.644628   90.909091
    3       100  110.000000  121.000000  110.000000
    4       100  110.000000  100.000000  110.000000
    5       100  110.000000  121.000000  133.100000
    6       100   90.909091  100.000000   90.909091
    7       100  110.000000  100.000000   90.909091
    8       100   90.909091  100.000000  110.000000
    9       100  110.000000  121.000000  133.100000

    Notes
    -----
    This class produces a binomial model for the price proccess $S_t$ of a risky asset. Beginning from its initial price $S_0$, and given a time horizon $T$, this model supposes that the price process evolves according to the following dynamics:

    $$
    S_{t+1} = S_t Z_{t+1},
    $$

    for each $t=0,1,\ldots,T-1$, where each $Z_t$ is a random variable that takes the value $u>0$ with some probability $p$ and the value $d>0$ with probability $1-p$. We assume that $d < u$. The probability $p$ is called the *real-world probability*, the factors $u$ and $d$ are called the *up-factor* and *down-factor*, respectively, and the process $Z_t$ is called the *driving process* of the model.
    """

    _repr_name = "Binomial price process"
    # _properties = GeometricPricingModel._properties + [
    #     "_initial_price",
    #     "_risk_free_rate",
    #     "_risk_free_gross_return",
    #     "_up_prob",
    #     "_down_prob",
    #     "_up_factor",
    #     "_down_factor",
    #     "_enum_mode",
    #     "_sparse_price_array",
    # ]

    # --------------------- constructors --------------------- #

    @classmethod
    def generate(
        cls,
        mode: Literal["enum", "sim"],
        initial_price: Real,
        risk_free_rate: Real,
        up_prob: Real,
        up_factor: Real,
        down_factor: Real | None = None,
        enum_mode: Literal["dense", "sparse"] = "dense",
        n_trajectories: int | None = None,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "S",
        random_state: int | np.random.Generator | None = None,
    ) -> BinomialPricingModel:
        r"""Generate trajectories of the binomial pricing model by either exhaustive enumeration (in sparse or dense mode) or Monte Carlo simulation.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        mode : Literal["enum", "sim"]
            Whether to generate trajectories by exhuastive enumeration or Monte Carlo simulation.
        initial_price : Real
            The initial price of the risky asset.
        risk_free_rate : Real
            The risk-free rate of the non-risky asset, which must be positive.
        up_prob : Real
            The probability of an upward move in the price of the risky asset.
        up_factor : Real
            The up-factor of the model. Must have `down_factor < up_factor`.
        down_factor : Real | None, default=None
            The down-factor of the model. Must have `down_factor < up_factor`. If `None`, then it will be set to the reciprocal of the up-factor, provided the latter is greater than 1.
        enum_mode : Literal["dense", "sparse"], default="dense"
            The mode of enumeration. If the generation mode is set to `sim`, this parameter is ignore.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. If the generation mode is set to `enum`, this parameter is ignored.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        name : Hashable, default="S"
            The name of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded. If the generation mode is set to `enum`, this parameter is ignored.

        Returns
        -------
        self : BinomialPricingModel
            The current instance with generated trajectories.

        Notes
        -----
        Suppose that $S_t$ is the price process of the underlying asset, and that $u$ and $d$ are the up- and down-factors of the model, respectively. Then $S_t$ is a random walk on the set of prices

        $$
        \{S_0 u^m d^n : m,n \geq 0\}.
        $$

        At a fixed time horizon $T$, the final price $S_T$ takes one of the $T+1$ values in the set

        $$
        \{S_0 u^{n} d^{T-n} : 0\leq n \leq T\}.
        $$

        For a fixed $n$, there are exactly $\binom{T}{n}$ many random walks (i.e., price trajectories) that terminate at the final price $S_T = S_0 u^{n} d^{T-n}$, and thus a total of $2^T = \sum_{n=0}^T \binom{T}{n}$ many random walks that end at *some* final price.

        This method enumerates these price trajectories in one of two modes: either exhuastive enumeration (`enum` mode) or Monte Carlo simulation (`sim` mode). In `enum` mode, the method may also be run in `dense` enumeration mode or `sparse` mode. In `dense` mode, the method enumerates all $2^T$ price trajectories of the model. In `sparse` mode, the method enumerates only $T+1$ price trajectories of the model which are of the special forms:

        $$
        \begin{gather*}
        S_0 \to S_0 u \to S_0u^2 \to S_0u^3 \to \ldots \to S_0u^{T-1} \to S_0 u^T \\
        S_0 \to S_0 d \to S_0du \to S_0du^2 \to \ldots \to S_0du^{T-2} \to S_0 du^{T-1} \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^2u \to \ldots \to S_0d^2u^{T-3} \to S_0 d^2u^{T-2} \\
        \cdots \quad \cdots \quad \cdots \quad \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^3 \to \ldots \to S_0d^{T-1} \to S_0 d^T
        \end{gather*}
        $$

        The `dense` mode of enumeration should only be used for small values of $T$, as the number of price trajectories grows exponentially in $T$.
        """
        if not isinstance(initial_price, Real):
            raise TypeError("initial_price must be a real number.")
        if initial_price <= 0:
            raise ValueError("initial_price must be positive.")
        if not isinstance(risk_free_rate, Real):
            raise TypeError("risk_free_rate must be a real number.")
        if risk_free_rate <= 0:
            raise ValueError("risk_free_rate must be positive.")
        if not isinstance(up_factor, Real):
            raise TypeError("up_factor must be a real number.")
        if up_factor <= 0:
            raise ValueError("up_factor must be positive.")
        if not isinstance(up_prob, Real):
            raise TypeError("up_prob must be a real number.")
        if not (0 <= up_prob <= 1):
            raise ValueError("up_prob must be in the interval [0, 1].")
        if down_factor is not None:
            if not isinstance(down_factor, Real):
                raise TypeError("down_factor must be a real number.")
            if down_factor <= 0:
                raise ValueError("down_factor must be positive.")
        if not isinstance(enum_mode, str):
            raise TypeError("enum_mode must be a string.")
        if enum_mode not in {"sparse", "dense"}:
            raise ValueError("enum_mode must be either 'sparse' or 'dense'.")

        index, random_state = cls._validate_and_return_generation_params(
            index=index,
            length=length,
            mode=mode,
            random_state=random_state,
        )
        process = cls(index=index, name=name)
        process._mode = mode
        process._n_trajectories = n_trajectories
        process._random_state = random_state

        process._initial_price = initial_price
        process._risk_free_rate = risk_free_rate
        process._up_prob = up_prob
        process._down_prob = 1 - up_prob
        process._up_factor = up_factor
        process._enum_mode = enum_mode

        if down_factor is None:
            if up_factor <= 1:
                raise ValueError(
                    "If down_factor is None, then up_factor must be greater than 1."
                )
            else:
                down_factor = 1 / up_factor
        if not (down_factor < up_factor):
            raise ValueError("We must have down_factor < up_factor.")

        process._down_factor = down_factor

        if mode == "enum":
            return process._enumeration_logic()
        else:
            return process._simulation_logic()

    # --------------------- generation methods --------------------- #

    def _enumeration_subclass_hook(self) -> pd.DataFrame:
        if self.enum_mode == "sparse":
            return pd.DataFrame(self.sparse_price_array)

        elif self.enum_mode == "dense":
            S = self.initial_price * self.driving_process.cumprod()
            S.insert_rv(state=self.initial_price, time=0, in_place=True)
            return S.data

        else:
            raise ValueError("enum_mode must be either 'sparse' or 'dense'.")

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
        return self._generate_prob_measure(prob=self.up_prob, name="P")

    def _generate_prob_measure(self, prob: Real, name: Hashable) -> ProbabilityMeasure:
        from ....core.measures.probability_measure import ProbabilityMeasure
        from ....core.sigma_algebras.sigma_algebra import SigmaAlgebra

        T = self.time[-1]

        if self.enum_mode == "sparse":
            probs = dict(
                zip(
                    self.sample_space,
                    binom(n=T, p=1 - prob).pmf(range(T + 1)),
                )
            )

            return ProbabilityMeasure(
                sig_alg=SigmaAlgebra.power_set(self.sample_space),
                mapping=probs,
                name=name,
            )

        elif self.enum_mode == "dense":
            values = list(product([0, 1], repeat=T))

            distribution = bernoulli(1 - prob)
            element_wise_probabilities = distribution.pmf(values)
            probs = pd.Series(
                data=np.prod(element_wise_probabilities, axis=1),
                index=self.sample_space.data,
            )
            probs /= probs.sum()

            return ProbabilityMeasure(
                sig_alg=SigmaAlgebra.power_set(self.sample_space),
                mapping=probs,
                name=name,
            )

        else:
            raise ValueError("enum_mode must be either 'sparse' or 'dense'.")

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    @property
    def driving_process(self) -> StochasticProcess:
        r"""Return the driving process of the binomial pricing model.

        The driving process is an IID process $Z_t$ representing the up and down movements of the underlying asset in the binomial model. It takes the value $u$ with probability $p$ and the value $d$ with probability $1-p$, where $u$ is the up-factor, $d$ is the down-factor, and $p$ is the real-world probability of an up move. The driving process is defined for times $t=1,2,\ldots,T$, where $T$ is the final time of the model.

        This property should only be used for small values of $T$, as the number of price trajectories is equal to $2^T$.

        Returns
        -------
        driving_process : StochasticProcess
            The driving process of the binomial pricing model.
        """
        from ....processes.types.iid_process import IIDProcess

        if self._driving_process is None:
            T = self.time[1:]
            p = self.up_prob
            u = self.up_factor
            d = self.down_factor
            support = {0: u, 1: d}

            if self.mode == "enum":
                self._driving_process = IIDProcess.generate(
                    mode="enum",
                    distribution=bernoulli(1 - p),
                    support=support,
                    index=T,
                    name="driving_process",
                )

            elif self.mode == "sim":
                self._driving_process = IIDProcess.generate(
                    mode="sim",
                    distribution=bernoulli(1 - p),
                    support=support,
                    n_trajectories=self.n_trajectories,
                    index=T,
                    random_state=self.random_state,
                    name="driving_process",
                )

            else:
                print(self.mode)
                raise ValueError("mode must be either 'enum' or 'sim'.")

        return self._driving_process

    @property
    def sparse_price_array(self) -> np.ndarray | None:
        """Pass."""
        if self._sparse_price_array is None and (
            self.up_factor is not None
            and self.down_factor is not None
            and self.time is not None
            and self.initial_price is not None
        ):
            u = self.up_factor
            d = self.down_factor
            T = self.time[-1]

            total_powers = np.tile(np.array(range(1, T + 1)), reps=[T + 1, 1])
            self._u_powers = np.maximum(
                0, total_powers - np.array(range(T + 1)).reshape(-1, 1)
            )
            self._d_powers = total_powers - self._u_powers

            price_factors = (np.ones(shape=(T + 1, T)) * u**self._u_powers) * (
                np.ones(shape=(T + 1, T)) * d**self._d_powers
            )

            self._sparse_price_array = self.initial_price * np.insert(
                arr=price_factors,
                obj=[0],
                values=np.ones(shape=(T + 1, 1)),
                axis=1,
            )

        return self._sparse_price_array

    # --------------------- probability methods --------------------- #

    def risk_neutral_probs(self) -> tuple[Real, Real]:
        r"""Get the risk-neutral probabilities of the model.

        See the Notes section below for the mathematical details.

        Raises
        ------
        ValueError
            If `generate` has not been called first to generate price trajectories, or if the no-arbitrage condition is violated.

        Returns
        -------
        risk_neutral_probs : tuple[Real, Real]
            The risk neutral probabilities as a tuple `(q_u, q_d)`, where `q_u` is the risk-neutral probability of an up move, and `q_d` is the risk-neutral probability of a down move.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel
        >>> S_0 = 100
        >>> u = 1.1
        >>> d = 0.9
        >>> p = 0.7
        >>> r = 0.01
        >>> S = BinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     up_prob=p,
        ...     risk_free_rate=r,
        ...     length=3,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        time      0      1      2      3
        sample
        0       100  110.0  121.0  133.1
        1       100  110.0  121.0  108.9
        2       100  110.0   99.0  108.9
        3       100  110.0   99.0   89.1
        4       100   90.0   99.0  108.9
        5       100   90.0   99.0   89.1
        6       100   90.0   81.0   89.1
        7       100   90.0   81.0   72.9
        >>> risk_neutral_probs = S.risk_neutral_probs()
        >>> print(risk_neutral_probs)
        (0.5499999999999997, 0.4500000000000003)

        Notes
        -----
        Let $S_t$ be a binomial price model with up-factor $u$, down-factor $d$, and risk-free gross return $R$. The *risk-neutral probabilities* are real numbers $q_u$ and $q_d$ such that

        $$
        R = q_uu + q_dd, \quad q_u + q_d = 1, \quad q_u,q_d \geq 0.
        $$

        Provided that the *no-arbitrage condition* holds, namely that

        $$
        d < R < u,
        $$

        there are unique real numbers $q_u$ and $q_d$ that satisfy these three contraints. The second and third constraints together guarantee that $q_u$ and $q_d$ may be used as probabilities of an up move and down move in the model, respectively. If $Q$ denotes the induced probability measure, called the *equivalent martingale measure*, then the first constraint guarantees that the discounted price process is a martingale:

        $$
        S_t = \frac{1}{R}E_Q\left( S_{t+1} \mid S_t\right),
        $$

        for all $t\geq 0$.
        """
        R, u, d = self.risk_free_gross_return, self.up_factor, self.down_factor

        if None in [R, u, d]:
            raise ValueError(
                "One of the parameters needed to generate the risk-neutral probabilities is None. Be sure to call 'generate' first."
            )
        if R <= d or R >= u:
            raise ValueError(
                "no-arbitrage condition violated: down_factor < risk_free_gross_return < up_factor"
            )

        q_u = (R - d) / (u - d)
        q_d = 1 - q_u

        return q_u, q_d

    @property
    def EMMs(self) -> ProbabilityMeasure:
        r"""Return the equivalent martingale measure of the model.

        See the Notes section below for the mathematical details.

        Raises
        ------
            If one of the parameters used to generate the EMM is none. This likely means that the `generate` method was not called first.

        Returns
        -------
        EMM : ParametrizedProbabilityMeasure
            The equivalent martingale measure of the model.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel
        >>> S_0 = 4
        >>> u = 1.2
        >>> d = 0.9
        >>> p = 0.6
        >>> r = 0.01
        >>> S = BinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     up_prob=p,
        ...     risk_free_rate=r,
        ...     length=2,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        time    0    1     2
        sample
        0       4  4.8  5.76
        1       4  4.8  4.32
        2       4  3.6  4.32
        3       4  3.6  3.24
        >>> Q = S.EMMs
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        sample
        0          0.134444
        1          0.232222
        2          0.232222
        3          0.401111
        >>> is_martingale_wrt_real_world_measure = S.discount(r).is_martingale()
        >>> print(is_martingale_wrt_real_world_measure)
        False
        >>> is_martingale_wrt_EMM = S.discount(r).is_martingale(prob_measure=Q)
        >>> print(is_martingale_wrt_EMM)
        True

        Notes
        -----
        Let $S_t$ be a binomial price model with up-factor $u$, down-factor $d$, and risk-free gross return $R$. Given the no-arbitrage condition

        $$
        d \leq R \leq u,
        $$

        there is a unique pair of risk-neutral probabilities $q_u$ and $q_d$, defining a new probability measure $Q$ on the price process, where $q_u$ is the probability of an up move and $q_d$ is the probability of a down move. See the Notes section of the docstring for the `risk_neutral_probs` method for further details. The probability measure $Q$ is called the *equivalent martingale measure*, or *EMM*. The name comes from the fact that the discounted price process is a martingale with respect to the EMM:

        $$
        S_t = \frac{1}{R} E_{Q}(S_{t+1} \mid S_t),
        $$

        for each $t\geq 0$.
        """
        if None in [
            self.risk_free_gross_return,
            self.up_factor,
            self.down_factor,
            self.time,
        ]:
            raise ValueError(
                "One of the parameters needed to generate the EMMs is None. Be sure to call 'generate' first."
            )

        if self._emms is None:
            if self.enum_mode == "sparse":
                self._emms = self._generate_prob_measure(
                    prob=self.risk_neutral_probs()[0], name="Q"
                )

            elif self.enum_mode == "dense":
                self._emms = self._generate_prob_measure(
                    prob=self.risk_neutral_probs()[0], name="Q"
                )

            else:
                raise ValueError("enum_mode must be either 'sparse' or 'dense'.")

        return self._emms

    # --------------------- properties --------------------- #

    @property
    def initial_price(self) -> Real | None:
        """Pass."""
        return self._initial_price

    @initial_price.setter
    def initial_price(self, value: Real) -> None:
        """Pass."""
        self._initial_price = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def risk_free_rate(self) -> Real | None:
        """Pass."""
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: Real) -> None:
        """Pass."""
        self._risk_free_rate = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def risk_free_gross_return(self) -> Real | None:
        """Pass."""
        if self._risk_free_gross_return is None and self.risk_free_rate is not None:
            self._risk_free_gross_return = self.risk_free_rate + 1

        return self._risk_free_gross_return

    @property
    def up_prob(self) -> Real | None:
        """Pass."""
        return self._up_prob

    @up_prob.setter
    def up_prob(self, value: Real) -> None:
        """Pass."""
        self._up_prob = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def down_prob(self) -> Real | None:
        """Pass."""
        return self._down_prob

    @down_prob.setter
    def down_prob(self, value: Real) -> None:
        """Pass."""
        self._down_prob = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def up_factor(self) -> Real | None:
        """Pass."""
        return self._up_factor

    @up_factor.setter
    def up_factor(self, value: Real) -> None:
        """Pass."""
        self._up_factor = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def down_factor(self) -> Real | None:
        """Pass."""
        return self._down_factor

    @down_factor.setter
    def down_factor(self, value: Real) -> None:
        """Pass."""
        self._down_factor = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def enum_mode(self) -> Literal["dense", "sparse"] | None:
        """Pass."""
        return self._enum_mode

    @enum_mode.setter
    def enum_mode(self, value: Literal["dense", "sparse"]) -> None:
        """Pass."""
        self._enum_mode = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    # --------------------- finance methods --------------------- #

    def price(self, claim: Claim, emm: ProbabilityMeasure | None = None) -> Real:
        """Price a claim under the model."""
        pass

    def replicating_portfolio(
        self, claim: Claim
    ) -> tuple[
        StochasticProcess, StochasticProcess, StochasticProcess, Real, StoppingTime
    ]:
        """Compute the replicating portfolio for a given contingent claim.

        Parameters
        ----------
        claim : Claim
            The contingent claim for which to compute the replicating portfolio.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.finance import AsianOption, BinomialPricingModel, EuropeanOption
        >>> S_0 = 100
        >>> u = 1.1
        >>> p = 0.7
        >>> r = 0.01
        >>> T = Time.discrete(length=3)
        >>> S = BinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     up_prob=p,
        ...     risk_free_rate=r,
        ...     index=T,
        ... )
        >>> print(S) # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        time      0           1           2           3
        sample
        0       100  110.000000  121.000000  133.100000
        1       100  110.000000  121.000000  110.000000
        2       100  110.000000  100.000000  110.000000
        3       100  110.000000  100.000000   90.909091
        4       100   90.909091  100.000000  110.000000
        5       100   90.909091  100.000000   90.909091
        6       100   90.909091   82.644628   90.909091
        7       100   90.909091   82.644628   75.131480
        >>> K = 100
        >>> euro_call = EuropeanOption.from_model(model=S, strike=K, option_type="call")
        >>> B, Delta, V, price, tau = S.replicating_portfolio(claim=euro_call)
        >>> print(B)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'B':
        time            0          1          2
        sample
        0      -50.150931 -73.822294 -99.009901
        1      -50.150931 -73.822294 -99.009901
        2      -50.150931 -73.822294 -47.147572
        3      -50.150931 -73.822294 -47.147572
        4      -50.150931 -24.674118 -47.147572
        5      -50.150931 -24.674118 -47.147572
        6      -50.150931 -24.674118   0.000000
        7      -50.150931 -24.674118   0.000000
        >>> print(Delta) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Delta':
        time           0         1        2
        sample
        0       0.587304  0.797939  1.00000
        1       0.587304  0.797939  1.00000
        2       0.587304  0.797939  0.52381
        3       0.587304  0.797939  0.52381
        4       0.587304  0.301542  0.52381
        5       0.587304  0.301542  0.52381
        6       0.587304  0.301542 -0.00000
        7       0.587304  0.301542 -0.00000
        >>> print(V) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'V':
        time           0          1          2     3
        sample
        0       8.579463  13.950993  21.990099  33.1
        1       8.579463  13.950993  21.990099  10.0
        2       8.579463  13.950993   5.233380  10.0
        3       8.579463  13.950993   5.233380   0.0
        4       8.579463   2.738827   5.233380  10.0
        5       8.579463   2.738827   5.233380   0.0
        6       8.579463   2.738827   0.000000   0.0
        7       8.579463   2.738827   0.000000   0.0
        >>> print(price)
        8.57946313365138
        """
        T = self.time[-1]
        B_arr, Delta_arr, V_arr, tau_arr = self._initialize_replicating_arrays()
        V_arr[:, -1], tau_arr[:, -1] = claim._backward_induction_base_case()

        for t in reversed(range(T)):
            V_forward, S_forward, S_curr = self._extract_tree_nodes(t, V_arr)

            B_curr, Delta_curr, V_curr, tau_curr = claim._backward_induction(
                enum_mode=self.enum_mode,
                V_forward=V_forward,
                S_forward=S_forward,
                S_curr=S_curr,
                risk_free_rate=self.risk_free_rate,
                risk_neutral_prob=self.risk_neutral_probs()[0],
            )

            B_arr[:, t], Delta_arr[:, t], V_arr[:, t], tau_arr[:, t] = (
                self._broadcast_node_values(
                    t,
                    B_curr,
                    Delta_curr,
                    V_curr,
                    tau_curr,
                )
            )

        tau_arr = np.where(
            tau_arr.max(axis=1) == 0,
            np.inf,
            np.argmax(tau_arr, axis=1),
        )

        B, Delta, V, price, tau = self._convert_replicating_arrays_to_processes(
            B_arr, Delta_arr, V_arr, tau_arr
        )

        return B, Delta, V, price, tau

    def _initialize_replicating_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]

        if self.enum_mode == "dense":
            B_arr = np.zeros(shape=(2**T, T))
            Delta_arr = np.zeros(shape=(2**T, T))
            V_arr = np.zeros(shape=(2**T, T + 1))
            tau_arr = np.zeros(shape=(2**T, T + 1))

        elif self.enum_mode == "sparse":
            B_arr = np.zeros(shape=(T + 1, T))
            Delta_arr = np.zeros(shape=(T + 1, T))
            V_arr = np.zeros(shape=(T + 1, T + 1))
            tau_arr = np.zeros(shape=(T + 1, T + 1))

        return B_arr, Delta_arr, V_arr, tau_arr

    def _extract_tree_nodes(
        self, t: int, V_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]
        S_arr = self.data.values

        if self.enum_mode == "dense":
            V_forward = V_arr[:: (2 ** (T - t - 1)), t + 1]  # shape (2^(t+1),)
            S_forward = S_arr[:: (2 ** (T - t - 1)), t + 1]  # shape (2^(t+1),)
            S_curr = S_arr[:: (2 ** (T - t)), t]  # shape (2^t,)

        elif self.enum_mode == "sparse":
            V_forward = V_arr[: t + 2, t + 1]  # shape (t+2,)
            S_forward = self.data.values[: t + 2, t + 1]  # shape(t+2,)
            S_curr = self.data.values[: t + 1, t]  # shape (t+1,)

        return V_forward, S_forward, S_curr

    def _broadcast_node_values(
        self,
        t: int,
        B_curr: np.ndarray,
        Delta_curr: np.ndarray,
        V_curr: np.ndarray,
        tau_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]

        if self.enum_mode == "dense":
            # all have shape (2^T,)
            B = np.repeat(B_curr, repeats=2 ** (T - t))
            Delta = np.repeat(Delta_curr, repeats=2 ** (T - t))
            V = np.repeat(V_curr, repeats=2 ** (T - t))
            tau = np.repeat(tau_curr, repeats=2 ** (T - t))

        elif self.enum_mode == "sparse":
            # all have shape (T+1,)
            B = np.concatenate((B_curr, np.repeat(B_curr[-1], T - t)))
            Delta = np.concatenate((Delta_curr, np.repeat(Delta_curr[-1], T - t)))
            V = np.concatenate((V_curr, np.repeat(V_curr[-1], T - t)))
            tau = np.concatenate((tau_curr, np.repeat(tau_curr[-1], T - t)))

        return B, Delta, V, tau

    def _convert_replicating_arrays_to_processes(
        self,
        B_arr: np.ndarray,
        Delta_arr: np.ndarray,
        V_arr: np.ndarray,
        tau_arr: np.ndarray,
    ) -> tuple[
        StochasticProcess, StochasticProcess, StochasticProcess, Real, StoppingTime
    ]:
        from ....processes.base.stochastic_process import StochasticProcess
        from ....processes.stopping_times.stopping_time import StoppingTime

        B = StochasticProcess(
            *self.prob_space,
            index=self.time[:-1],
            name="B",
            mapping=B_arr,
        )

        Delta = StochasticProcess(
            *self.prob_space,
            index=self.time[:-1],
            name="Delta",
            mapping=Delta_arr,
        )

        V = StochasticProcess(
            *self.prob_space,
            index=self.time,
            name="V",
            mapping=V_arr,
        )

        price = V[0].data.to_numpy()[0]

        tau = StoppingTime.from_filtration(process=self, mapping=tau_arr, name="tau")

        return B, Delta, V, price, tau
