"""A class modeling a trinomial pricing model."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .geometric_pricing_model import GeometricPricingModel

if TYPE_CHECKING:
    from ....core.indices.index import Index
    from ....core.measures.parametrized_probability_measure import (
        ParametrizedProbabilityMeasure,
    )
    from ....core.measures.probability_measure import ProbabilityMeasure
    from ....processes.base.stochastic_process import StochasticProcess
    from ..claims.claim import Claim


class TrinomialPricingModel(GeometricPricingModel):
    r"""A class modeling a trinomial pricing model.

    The base constructor is not meant to be called directly by users. Instead, the user should call the `generate` class method. See the Examples section below for usage.

    See the Notes section below for the mathematical details.

    Examples
    --------
    Given parameters of a trinomial model, generate all length-2 price trajectories in `enum` mode.

    >>> from sigalg.finance import TrinomialPricingModel
    >>> S_0 = 4
    >>> u = 1.2
    >>> m = 1.1
    >>> d = 0.9
    >>> p_u = 0.6
    >>> p_d = 0.1
    >>> r = 0.01
    >>> S = TrinomialPricingModel.generate(
    ...     mode="enum",
    ...     initial_price=S_0,
    ...     up_factor=u,
    ...     middle_factor=m,
    ...     down_factor=d,
    ...     up_prob=p_u,
    ...     down_prob=p_d,
    ...     risk_free_rate=r,
    ...     length=2,
    ... )
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Trinomial price process 'S':
    time    0    1     2
    sample
    0       4  4.8  5.76
    1       4  4.8  5.28
    2       4  4.8  4.32
    3       4  4.4  5.28
    4       4  4.4  4.84
    5       4  4.4  3.96
    6       4  3.6  4.32
    7       4  3.6  3.96
    8       4  3.6  3.24

    Simulate ten length-2 trajectories in `sim` mode.

    >>> S.n_trajectories = 10
    >>> S.random_state = 42
    >>> S.mode = "sim"
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Trinomial price process 'S':
    time    0    1     2
    sample
    0       4  4.4  4.84
    1       4  4.8  4.32
    2       4  4.4  5.28
    3       4  4.8  5.28
    4       4  4.4  5.28
    5       4  4.8  5.76
    6       4  4.4  4.84
    7       4  3.6  3.96
    8       4  4.8  5.76
    9       4  4.8  5.28

    Notes
    -----
    This class produces a trinomial model for the price proccess $S_t$ of a risky asset. Beginning from its initial price $S_0$, and given a time horizon $T$, this model supposes that the price process evolves according to the following dynamics:

    $$
    S_{t+1} = S_t Z_{t+1},
    $$

    for each $t=0,1,\ldots,T-1$, where each $Z_t$ is a random variable that takes the value $u>0$ with some probability $p_u$, the value $m>0$ with some probability $p_m$, and the value $d>0$ with some probability $p_d$. We assume that $d < m < u$. The probabilities $p_u$, $p_m$ and $p_d$ are called the *real-world probabilities*, the factors $u$, $m$ and $d$ are called the *up-factor*, *middle-factor* and *down-factor*, respectively, and the process $Z_t$ is called the *driving process* of the model.
    """

    _repr_name = "Trinomial price process"
    # _properties = GeometricPricingModel._properties + [
    #     "_initial_price",
    #     "_risk_free_rate",
    #     "_risk_free_gross_return",
    #     "_up_prob",
    #     "_middle_prob",
    #     "_down_prob",
    #     "_up_factor",
    #     "_middle_factor",
    #     "_down_factor",
    # ]

    # --------------------- constructors --------------------- #

    @classmethod
    def generate(
        cls,
        mode: Literal["enum", "sim"],
        initial_price: Real,
        risk_free_rate: Real,
        up_prob: Real,
        down_prob: Real,
        up_factor: Real,
        middle_factor: Real,
        down_factor: Real,
        n_trajectories: int | None = None,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "S",
        random_state: int | np.random.Generator | None = None,
    ) -> TrinomialPricingModel:
        """Generate trajectories of the trinomial pricing model by either exhaustive enumeration or Monte Carlo simulation.

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
        down_prob : Real
            The probability of a downward move in the price of the risky asset.
        up_factor : Real
            The up-factor of the model. Must have `down_factor < middle_factor < up_factor`.
        middle_factor : Real
            The middle-factor of the model. Must have `down_factor < middle_factor < up_factor`.
        down_factor : Real
            The down-factor of the model. Must have `down_factor < middle_factor < up_factor`.
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
        self : TrinomialPricingModel
            The current instance with generated trajectories.
        """
        if not isinstance(initial_price, Real):
            raise TypeError("initial_price must be a real number.")
        if initial_price <= 0:
            raise ValueError("initial_price must be positive.")
        if not isinstance(risk_free_rate, Real):
            raise TypeError("risk_free_rate must be a real number.")
        if risk_free_rate <= 0:
            raise ValueError("risk_free_rate must be positive.")
        if not isinstance(up_prob, Real):
            raise TypeError("up_prob must be a real number.")
        if not (0 <= up_prob <= 1):
            raise ValueError("up_prob must be in the interval [0, 1].")
        if not isinstance(down_prob, Real):
            raise TypeError("down_prob must be a real number.")
        if not (0 <= down_prob <= 1):
            raise ValueError("down_prob must be in the interval [0, 1].")
        if not isinstance(up_factor, Real):
            raise TypeError("up_factor must be a real number.")
        if up_factor <= 0:
            raise ValueError("up_factor must be positive.")
        if not isinstance(middle_factor, Real):
            raise TypeError("middle_factor must be a real number.")
        if middle_factor <= 0:
            raise ValueError("middle_factor must be positive.")
        if not isinstance(down_factor, Real):
            raise TypeError("down_factor must be a real number.")
        if down_factor <= 0:
            raise ValueError("down_factor must be positive.")
        if up_prob + down_prob > 1:
            raise ValueError("The sum of up_prob and down_prob must be at most 1.")
        if not (down_factor < middle_factor < up_factor):
            raise ValueError("We must have down_factor < middle_factor < up_factor.")

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
        process._middle_prob = 1 - up_prob - down_prob
        process._down_prob = down_prob
        process._up_factor = up_factor
        process._middle_factor = middle_factor
        process._down_factor = down_factor

        if mode == "enum":
            return process._enumeration_logic()
        else:
            return process._simulation_logic()

    # --------------------- generation methods --------------------- #

    def _enumeration_subclass_hook(self) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
        return self.driving_process.prob_measure

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    @property
    def driving_process(self) -> StochasticProcess:
        """Pass."""
        from scipy.stats import multinomial

        from ....processes.types.iid_process import IIDProcess

        if self._driving_process is None:
            T = self.time[1:]
            p_u = self.up_prob
            p_m = self.middle_prob
            p_d = self.down_prob
            u = self.up_factor
            m = self.middle_factor
            d = self.down_factor
            support = {0: u, 1: m, 2: d}

            if self.mode == "enum":
                self._driving_process = IIDProcess.generate(
                    mode="enum",
                    distribution=multinomial(1, [p_u, p_m, p_d]),
                    support=support,
                    index=T,
                    name="driving_process",
                )

            elif self.mode == "sim":
                self._driving_process = IIDProcess.generate(
                    mode="sim",
                    distribution=multinomial(1, [p_u, p_m, p_d]),
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

    # --------------------- probability methods --------------------- #

    def risk_neutral_probs(self, theta: Real) -> tuple[Real, Real, Real]:
        r"""Get the risk-neutral probabilities of the model.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        theta : Real
            A real number parameterizing the risk-neutral probabilities. See the Notes section for further explanation.

        Raises
        ------
        TypeError
            If `theta` is not a real number.
        ValueError
            If `generate` has not been called first to generate price trajectories, or if the no-arbitrage condition is violated, or `theta` is not in the open inverval (0, 1).

        Returns
        -------
        risk_neutral_probs : tuple[Real, Real, Real]
            The risk neutral probabilities as a tuple `(q_u, q_m, q_d)`, where `q_u` is the risk-neutral probability of an up move, `q_m` is the risk-neutral probability of a middle move, and `q_d` is the risk-neutral probability of a down move.

        Examples
        --------
        >>> from sigalg.finance import TrinomialPricingModel
        >>> S_0 = 4
        >>> u = 1.2
        >>> m = 1.1
        >>> d = 0.9
        >>> p_u = 0.6
        >>> p_d = 0.1
        >>> r = 0.01
        >>> S = TrinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     middle_factor=m,
        ...     down_factor=d,
        ...     up_prob=p_u,
        ...     down_prob=p_d,
        ...     risk_free_rate=r,
        ...     length=2,
        ... )
        >>> print(S) # doctest: +NORMALIZE_WHITESPACE
        Trinomial price process 'S':
        time    0    1     2
        sample
        0       4  4.8  5.76
        1       4  4.8  5.28
        2       4  4.8  4.32
        3       4  4.4  5.28
        4       4  4.4  4.84
        5       4  4.4  3.96
        6       4  3.6  4.32
        7       4  3.6  3.96
        8       4  3.6  3.24
        >>> risk_neutral_probs = S.risk_neutral_probs(theta=0.42)
        >>> print(risk_neutral_probs)
        (0.15400000000000003, 0.31899999999999973, 0.5270000000000001)

        Notes
        -----
        Let $S_t$ be a trinomial price model with up-factor $u$, middle-factor $m$, down-factor $d$, and risk-free gross return $R$. The *risk-neutral probabilities* are real numbers $q_u$, $q_m$ and $q_d$ such that

        $$
        R = q_uu + q_mm + q_dd, \quad q_u + q_m + q_d = 1, \quad q_u,q_m,q_d \geq 0.
        $$

        The first pair of equations are linear in the three unknowns $q_u$, $q_m$ and $q_d$, and hence there is a $1$-dimensional space of solutions. Indeed, solving the system yields

        $$
        q_u = \frac{(m-d)q_d + (R-m)}{u-m}
        $$

        and

        $$
        q_m = \frac{(d-u)q_d + (u-R)}{u-m},
        $$

        where $q_d$ is a free parameter.

        One may show that if $q_u$, $q_m$ and $q_d$ are solutions to the first two linear equations and $q_u\geq 0$, then necessarily

        $$
        \frac{m-R}{m-d} \leq q_d.
        $$

        Then, if $q_m,q_d\geq 0$ as well, we have that $q_d \leq 1$, and so $d \leq R$. Similarly, we get from $q_m \geq 0$ that

        $$
        q_d \leq \frac{u-R}{u-d},
        $$

        and so $R\leq u$ provided that $q_d\geq 0$. We have therefore shown that *if* $q_u$, $q_m$ and $q_d$ satisfy all three constraints listed above (the two linear equations and the inequalities), then we necessarily have the *no-arbitrage condition*

        $$
        d \leq R \leq u,
        $$

        along with the inequalities

        $$
        \max\left(\frac{m-R}{m-d},0\right) \leq q_d \leq \frac{u-R}{u-d}.
        $$

        Conversely, if these inequalities hold, as well as the no-arbitrage condition, then we have

        $$
        0 \leq q_u \leq 1.
        $$

        If we define $q_u$ and $q_m$ by the equations

        $$
        q_u = \frac{(m-d)q_d + (R-m)}{u-m} \quad \text{and} \quad q_m = \frac{(d-u)q_d + (u-R)}{u-m}
        $$

        given above, then the three numbers $q_u$, $q_m$ and $q_d$ satisfy the three original constraints defining risk-neutral probabilities. We have thus uncovered both necessary and sufficient conditions for solutions to these constraints.

        In particular, in the presence of the no-arbitrage condition, all solutions may be parametrized by $\theta\in [0,1]$, by setting

        $$
        q_d = (b-a)(\theta-1) + b, \quad a = \max\left(\frac{m-R}{m-d},0\right), \quad b = \frac{u-R}{u-d},
        $$

        and then letting $q_u$ and $q_m$ be given by the expressions in terms of $q_d$ described above.

        The second pair of constraints show that the numbers $q_u$, $q_m$ and $q_d$ may be used for the probabilities of an up move, a middle move, and a down move in the trinomial model. If we use these probabilities instead of the real-world probabilities, and let $Q$ be the induced probability measure (called an *equivalent martingale measure*), then the first constraint shows that the discounted price process is a martingale:

        $$
        S_t = \frac{1}{R}E_Q\left( S_{t+1} \mid S_t\right),
        $$

        for each $t\geq 0$.
        """
        if not isinstance(theta, Real):
            raise TypeError("The parameter theta must be a real number.")
        if theta <= 0 or theta >= 1:
            raise ValueError("The parameter theta must be in the open interval (0,1).")

        R = self.risk_free_gross_return
        u = self.up_factor
        m = self.middle_factor
        d = self.down_factor

        if None in [R, u, m, d]:
            raise ValueError(
                "One of the parameters needed to generate the risk-neutral probabilities is None. Be sure to call 'generate' first."
            )
        if R <= d or R >= u:
            raise ValueError(
                "There is arbitrage in the model. The risk-free gross return R must be in the interval [down_factor, up_factor]."
            )

        a = max((m - R) / (m - d), 0)
        b = (u - R) / (u - d)
        q_d = (b - a) * (theta - 1) + b
        q_u = ((m - d) * q_d + (R - m)) / (u - m)
        q_m = ((d - u) * q_d + (u - R)) / (u - m)

        return q_u, q_m, q_d

    @property
    def EMMs(self) -> ParametrizedProbabilityMeasure:
        r"""Return the equivalent martingale measures of the model.

        See the Notes section below for the mathematical details.

        Raises
        ------
        ValueError
            If one of the parameters used to generate the EMMs is none. This likely means that the `generate` method was not called first.

        Returns
        -------
        EMMs : ParametrizedProbabilityMeasure
            The equivalent martingale measures of the model.

        Examples
        --------
        >>> from sigalg.finance import TrinomialPricingModel
        >>> S_0 = 4
        >>> u = 1.2
        >>> m = 1.1
        >>> d = 0.9
        >>> p_u = 0.6
        >>> p_d = 0.1
        >>> r = 0.01
        >>> S = TrinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     middle_factor=m,
        ...     down_factor=d,
        ...     up_prob=p_u,
        ...     down_prob=p_d,
        ...     risk_free_rate=r,
        ...     length=2,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Trinomial price process 'S':
        time    0    1     2
        sample
        0       4  4.8  5.76
        1       4  4.8  5.28
        2       4  4.8  4.32
        3       4  4.4  5.28
        4       4  4.4  4.84
        5       4  4.4  3.96
        6       4  3.6  4.32
        7       4  3.6  3.96
        8       4  3.6  3.24
        >>> Q = S.EMMs
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q(theta, sample)'
        >>> print(Q(theta=0.5))  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q(theta=0.5)':
                probability
        sample
        0          0.033611
        1          0.050417
        2          0.099306
        3          0.050417
        4          0.075625
        5          0.148958
        6          0.099306
        7          0.148958
        8          0.293403
        >>> print(Q(theta=0.5, sample=4))
        0.07562500000000004
        >>> is_martingale_wrt_real_world_measure = S.discount(r).is_martingale()
        >>> print(is_martingale_wrt_real_world_measure)
        False
        >>> is_martingale_wrt_EMM = S.discount(r).is_martingale(prob_measure=Q(theta=0.5))
        >>> print(is_martingale_wrt_EMM)
        True

        Notes
        -----
        Let $S_t$ be a trinomial price model with up-factor $u$, middle-factor $m$, down-factor $d$, and risk-free gross return $R$. Given the no-arbitrage condition

        $$
        d \leq R \leq u,
        $$

        there is a $1$-parameter family of risk-neutral probabilities $q_u(\theta)$, $q_m(\theta)$ and $q_d(\theta)$, defining a new probability measure $Q(\theta)$ on the price process, where $q_u(\theta)$ is the probability of an up move, $q_m(\theta)$ is the probability of a middle move, and $q_d(\theta)$ is the probability of a down move. See the Notes section of the docstring for the `risk_neutral_probs` method for further details. The probability measures $Q(\theta)$, parametrized by $\theta\in [0,1]$, are called *equivalent martingale measures*, or *EMMs*. The name comes from the fact that the discounted price process is a martingale with respect to an EMM:

        $$
        S_t = \frac{1}{R} E_{Q(\theta)}(S_{t+1} \mid S_t),
        $$

        for each $t\geq 0$.
        """
        from scipy.stats import multinomial

        from ....core.measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ....processes.types.iid_process import IIDProcess

        if None in [
            self.risk_free_gross_return,
            self.up_factor,
            self.middle_factor,
            self.down_factor,
            self.time,
        ]:
            raise ValueError(
                "One of the parameters needed to generate the EMMs is None. Be sure to call 'generate' first."
            )

        if self._emms is None:
            T = self.time[1:]

            def mapping(*, theta, sample):
                q_u, q_m, q_d = self.risk_neutral_probs(theta=theta)

                prob_measure = IIDProcess.generate(
                    mode="enum",
                    distribution=multinomial(1, [q_u, q_m, q_d]),
                    support=[0, 1, 2],
                    index=T,
                ).prob_measure

                return prob_measure(sample)

            self._emms = ParametrizedProbabilityMeasure(
                domain=self.sample_space, mapping=mapping, name="Q"
            )

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
    def middle_prob(self) -> Real | None:
        """Pass."""
        return self._middle_prob

    @middle_prob.setter
    def middle_prob(self, value: Real) -> None:
        """Pass."""
        self._middle_prob = value
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
    def middle_factor(self) -> Real | None:
        """Pass."""
        return self._middle_factor

    @middle_factor.setter
    def middle_factor(self, value: Real) -> None:
        """Pass."""
        self._middle_factor = value
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

    # --------------------- finance methods --------------------- #

    def price(self, claim: Claim, emm: ProbabilityMeasure | None = None) -> Real:
        """Price a claim under the model."""
        pass
