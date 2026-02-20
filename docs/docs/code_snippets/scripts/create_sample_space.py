"""Instantiate a SampleSpace from different data sources."""

import pandas as pd

from sigalg.core import SampleSpace

Omega_from_list = SampleSpace(  # (1)!
    name="Omega_from_list",
).from_list(["H", "T"])

Omega_from_sequence = SampleSpace(  # (2)!
    name="Omega_from_sequence",
).from_sequence(size=6, initial_index=1)

Omega_with_prefixes = SampleSpace(  # (3)!
    name="Omega_with_prefixes",
).from_sequence(size=4, prefix="omega")

data = pd.Index(["red", "green", "blue"])
Omega_from_pandas = SampleSpace(  # (4)!
    name="Omega_from_pandas",
).from_pandas(data)

print(Omega_from_list)
print("\n", Omega_from_sequence)
print("\n", Omega_with_prefixes)
print("\n", Omega_from_pandas)
