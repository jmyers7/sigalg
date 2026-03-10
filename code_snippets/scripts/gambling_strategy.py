"""Model a gambling strategy on a binary-outcome game as a predictable process."""

from sigalg.core import RandomVariable, Time
from sigalg.processes import ProcessTransforms, RandomWalk, StochasticProcess

T = Time.discrete(start=0, stop=3)  # (1)!

Y = RandomWalk(p=0.4, time=T, name="Y").from_enumeration()  # (2)!


def f0(Y: StochasticProcess) -> RandomVariable:  # (3)!
    return RandomVariable(domain=Y.domain).from_constant(1)


def f1(Y: StochasticProcess) -> RandomVariable:  # (4)!
    return 2 * (Y[1] > Y[0])


def f2(Y: StochasticProcess) -> RandomVariable:  # (5)!
    return 2 * (Y[2] > Y[1]) + 1 * (Y[1] > Y[0])


X = ProcessTransforms.transform(  # (6)!
    process=Y, functions=[f0, f1, f2], time=T[:-1]
).with_name("X")

winnings = X.ito_integral(Y)  # (7)!
expected_winnings = winnings.last_rv.expectation().item()  # (8)!

print("Is the game unfair to the bettor?", Y.is_supermartingale())  # (9)!
print("\nWhich games are winners and which are losers?\n", Y.increments())  # (10)!
print("\nBettor's strategy:\n", X)  # (11)!
print(
    "\nIs the bettor's strategy adapted?", X.is_adapted(Y.natural_filtration)
)  # (12)!
print("\nBettor's winnings:\n", winnings)  # (13)!
print(
    "\nIs the bettor's strategy a losing strategy?",
    winnings.is_supermartingale(Y.natural_filtration),
)  # (14)!
print("\nExpected winnings:", f"{expected_winnings:0.2f}")  # (15)!
