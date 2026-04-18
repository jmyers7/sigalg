"""Create random vectors and random variables from different data sources."""

import numpy as np
import pandas as pd

from sigalg.core import RandomVariable, RandomVector, SampleSpace

Omega = SampleSpace().from_sequence(size=4)  # (1)!

dict_2d = dict(zip(Omega, [(1, 2), (3, 4), (3, 4), (7, 8)]))  # (2)!
X = RandomVector(domain=Omega).from_dict(dict_2d)
print(X)
print(X.domain, "\n")

dict_1d = dict(zip(Omega, [1, 2, 3, 4]))  # (3)!
Y = RandomVariable(domain=Omega, name="Y").from_dict(dict_1d)
print(Y)
print(Y.domain, "\n")

df = pd.DataFrame(  # (4)!
    [[1, 2], [3, 4], [5, 6]], index=["a", "b", "c"], columns=["A", "B"]
)
Z = RandomVector(name="Z").from_pandas(df)
print(Z)
print(Z.domain.with_name("Z_domain"), "\n")

s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])  # (5)!
W = RandomVariable(name="W").from_pandas(s)
print(W)
print(W.domain.with_name("W_domain"), "\n")

arr = np.array([[1, 2], [3, 4], [5, 6]])  # (6)!
U = RandomVector(name="U").from_numpy(arr)
print(U)
print(U.domain.with_name("U_domain"), "\n")

A = RandomVector(domain=Omega, name="A").from_randint(  # (7)!
    low=0, high=10, dim=3, random_state=42
)
print(A)
print(A.domain, "\n")

B = RandomVector(domain=Omega, name="B").from_randnorm(  # (8)!
    loc=1, scale=2, dim=2, random_state=42
)
print(B)
print(B.domain)
