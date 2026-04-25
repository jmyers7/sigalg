"""A class representing a random variable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .random_vector import RandomVector

if TYPE_CHECKING:
    from ..base.event import Event
    from .random_vector import RandomVector


class RandomVariable(RandomVector):
    r"""A class representing a random variable.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : SampleSpace | None, default=None
        The sample space over which the random vector is defined. The `None` value indicates that the domain will be generated later through a method like `from_dict`, `from_pandas`, or `from_numpy`.
    name : Hashable | None, default="X"
        The name of the random vector.
    **kwargs
        Additional keyword arguments for subclass constructors.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, RandomVariable
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> outputs = dict(zip(Omega, [0.1, 0.3, 0.5]))
    >>> X = RandomVariable(domain=Omega, name="X").from_dict(outputs)
    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
    Random variable 'X':
                X
        sample
        0       0.1
        1       0.3
        2       0.5

    Notes
    -----
    Given a probability space $(\Omega,\mathcal{F},P)$, a *random variable* is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}$, where $\mathbb{R}$ is equipped with its Borel $\sigma$-algebra.

    An instance `X` of `RandomVariable` is SigAlg's representation of a random variable $X$. Such an instance may be constructed with a `domain` parameter representing $\Omega$, and a dictionary parameter `outputs` representing the mapping $\omega \to X(\omega)$. (Other construction methods exist besides this canonical one.)

    The probability measure $P$ may be represented by setting the `probability_measure` attribute of `X` to an instance of `ProbabilityMeasure` after construction. If not set explicitly, this measure defaults to the uniform measure on $\Omega$.

    The $\sigma$-algebra $\mathcal{F}$ is not carried by the instance `X`. In particular, SigAlg does not enforce the measurability requirement for random vectors on construction. However, `X` does carry a method `is_measurable` for checking measurability after construction relative to an instance of `SigmaAlgebra`.

    See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
    """

    # --------------------- constructors --------------------- #

    def from_randint(
        self,
        low: int,
        high: int,
        random_state: int | None = None,
    ) -> RandomVariable:
        """Generate a random variable with integer outputs uniformly sampled from the range [low, high).

        For this construction method, the `domain` must be provided at construction.

        Parameters
        ----------
        low : int
            The lower bound (inclusive) of the random integers.
        high : int
            The upper bound (exclusive) of the random integers.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.

        Returns
        -------
        self : RandomVariable
            A random variable with integer outputs uniformly sampled from the range [low, high).

        Examples
        --------
        >>> from sigalg.core import RandomVariable, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVariable(domain=Omega).from_randint(low=0, high=5, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        sample
        0       0
        1       3
        2       3
        """
        return super().from_randint(
            low=low, high=high, dim=1, random_state=random_state
        )

    def from_randnorm(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        random_state: int | None = None,
    ) -> RandomVariable:
        """Generate a random variable with outputs sampled from a normal distribution.

        Parameters
        ----------
        loc : float, default=0.0
            The mean (center) of the normal distribution.
        scale : float, default=1.0
            The standard deviation (spread or width) of the normal distribution.
        random_state : int | None, default=None
            An optional seed for the random number generator to ensure reproducibility. If `None`, the random number generator is not seeded.

        Returns
        -------
        self : RandomVariable
            A random variable with outputs sampled from a normal distribution.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVariable(domain=Omega).from_randnorm(random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                    X
        sample
        0       0.304717
        1      -1.039984
        2       0.750451
        """
        return super().from_randnorm(
            loc=loc, scale=scale, dim=1, random_state=random_state
        )

    # --------------------- factory methods --------------------- #

    @classmethod
    def indicator_of(cls, event: Event) -> RandomVariable:
        """Create the indicator random variable of a given event.

        Parameters
        ----------
        event : Event
            The event for which the indicator random variable is to be created.

        Raises
        ------
        TypeError
            If `event` is not an instance of `Event`.

        Returns
        -------
        indicator_rv : RandomVariable
            The indicator random variable of the given event.

        Examples
        --------
        >>> from sigalg.core import RandomVariable, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> print(Omega)
        Sample space 'Omega':
        [0, 1, 2]
        >>> A = Omega.get_event([0, 1])
        >>> I_A = RandomVariable.indicator_of(event=A)
        >>> print(I_A) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'I_A':
                I_A
        sample
        0         1
        1         1
        2         0
        """
        return super().indicator_of(event=event, dim=1)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the random variable.

        Returns
        -------
        repr_str : str
            The string representation of the random variable.
        """
        data = self.data.to_frame()
        data.columns = [self.name] if self.name is not None else ["value"]

        if self.name is None:
            return f"Random variable:\n{data}"
        else:
            return f"Random variable '{self.name}':\n{data}"
