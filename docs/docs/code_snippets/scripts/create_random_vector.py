"""Create random vectors and random variables from different data sources."""

import numpy as np
import pandas as pd

from sigalg.core import RandomVariable, RandomVector, SampleSpace

Omega = SampleSpace().from_sequence(size=4)  # (1)!

dict_2d = dict(zip(Omega, [(1, 2), (3, 4), (3, 4), (7, 8)]))
dict_1d = dict(zip(Omega, [1, 2, 3, 4]))
df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=["a", "b", "c"], columns=["A", "B"])
s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
arr = np.array([[1, 2], [3, 4], [5, 6]])

X = RandomVector(domain=Omega).from_dict(dict_2d)  # (2)!
Y = RandomVariable(domain=Omega, name="Y").from_dict(dict_1d)  # (3)!
Z = RandomVector(name="Z").from_pandas(df)  # (4)!
W = RandomVariable(name="W").from_pandas(s)  # (5)!
U = RandomVector(name="U").from_numpy(arr)  # (6)!
A = RandomVector(domain=Omega, name="A").from_randint(  # (7)!
    low=0, high=10, dim=3, random_state=42
)
B = RandomVector(domain=Omega, name="B").from_randnorm(  # (8)!
    loc=1, scale=2, dim=2, random_state=42
)
E = Omega.get_event([0, 1], name="E")  # (9)!
I = RandomVector.indicator_of(event=E, dim=3)

print(X)
print(X.domain, "\n")
print(Y)
print(Y.domain, "\n")
print(Z)
print(Z.domain.with_name("Z_domain"), "\n")
print(W)
print(W.domain.with_name("W_domain"), "\n")
print(U)
print(U.domain.with_name("U_domain"), "\n")
print(A)
print(A.domain, "\n")
print(B)
print(B.domain, "\n")
print(I)
print(I.domain)
