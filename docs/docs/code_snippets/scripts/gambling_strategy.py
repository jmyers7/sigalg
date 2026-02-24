"""Model a gambling strategy on a binary-outcome game as a predictable process."""

from sigalg.core import RandomVariable, Time
from sigalg.processes import ProcessTransforms, RandomWalk, StochasticProcess

T = Time.discrete(start=0, stop=3)  # (1)!

Y = RandomWalk(  # (2)!
    p=0.4,
    time=T,
    name="Y",
).from_enumeration()


def f1(Y: StochasticProcess) -> RandomVariable:  # (3)!
    return RandomVariable(domain=Y.domain).from_constant(1)


def f2(Y: StochasticProcess) -> RandomVariable:  # (4)!
    Y0, Y1, *_ = Y
    strategy = {}
    for trajectory in Y.domain:
        strategy[trajectory] = 2 if Y1(trajectory) > Y0(trajectory) else 0
    return RandomVariable(domain=Y.domain).from_dict(strategy)


def f3(Y: StochasticProcess) -> RandomVariable:  # (5)!
    Y0, Y1, Y2, *_ = Y
    strategy = {}
    for trajectory in Y.domain:
        if Y2(trajectory) > Y1(trajectory) and Y1(trajectory) > Y0(trajectory):
            strategy[trajectory] = 3
        elif Y2(trajectory) > Y1(trajectory) and Y1(trajectory) < Y0(trajectory):
            strategy[trajectory] = 2
        elif Y2(trajectory) < Y1(trajectory) and Y1(trajectory) > Y0(trajectory):
            strategy[trajectory] = 1
        else:
            strategy[trajectory] = 0
    return RandomVariable(domain=Y.domain).from_dict(strategy)


X = ProcessTransforms.transform(  # (6)!
    process=Y,
    functions=[f1, f2, f3],
    time=T[1:],
).with_name("X")

winnings = X.ito_integral(Y)  # (7)!
expected_winnings = winnings.last_rv.expectation().item()  # (8)!

print("Is the game unfair to the bettor?", Y.is_supermartingale())  # (9)!
print("\nWhich games are winners and which are losers?\n", Y.increments())  # (10)!
print("\nBettor's strategy:\n", X)  # (11)!
print(
    "\nIs the bettor's strategy predictable?", X.is_predictable(Y.natural_filtration)
)  # (12)!
print("\nBettor's winnings:\n", winnings)  # (13)!
print(
    "\nIs the bettor's strategy a winning strategy?",
    winnings.is_supermartingale(),
)  # (14)!
print("\nExpected winnings:", f"{expected_winnings:0.2f}")  # (15)!
