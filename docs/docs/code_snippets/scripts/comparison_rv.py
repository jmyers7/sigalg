"""Demonstrate comparing random vectors and variables."""

from sigalg.core import RandomVariable, RandomVector, SampleSpace

Omega = SampleSpace().from_sequence(size=2)  # (1)!
X = RandomVector(domain=Omega).from_dict(
    {
        0: (1, 2),
        1: (2, 3),
    }
)
Y = RandomVector(domain=Omega, name="Y").from_dict(
    {
        0: (1, 3),
        1: (1, 4),
    }
)
Z = RandomVector(domain=Omega, name="Z").from_dict(
    {
        0: (2, 3),
        1: (4, 5),
    }
)
W = RandomVariable(domain=Omega, name="W").from_dict(
    {
        0: 2,
        1: -3,
    }
)

print(X, "\n")  # (2)!
print(Y, "\n")
print(Z, "\n")
print(W, "\n")
print(X < Y, "\n")
print(X <= Y, "\n")
print(X > Y, "\n")
print(X >= Y, "\n")
print(X <= 2, "\n")
print(1 > W, "\n")
print(
    f"Are *any* entries of 'X' less than the corresponding entries of 'Y'? {(X < Y).any()}\n"
)
print(
    f"Are *all* entries of 'X' less than the corresponding entries of 'Z'? {(X < Z).all()}"
)
