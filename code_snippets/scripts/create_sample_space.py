"""Instantiating a SampleSpace from different data sources."""

import pandas as pd

from sigalg.core import SampleSpace

# Create a sample space from a list
Omega_from_list = SampleSpace(
    name="Omega_from_list",
).from_list(["H", "T"])

# Create a sample space from a sequence of numbers
Omega_from_sequence = SampleSpace(
    name="Omega_from_sequence",
).from_sequence(size=6, initial_index=1)

# Create a sample space from a sequence of numbers with prefixes
Omega_with_prefixes = SampleSpace(
    name="Omega_with_prefixes",
).from_sequence(size=4, prefix="omega")

# Create a sample space from a pd.Index
data = pd.Index(["red", "green", "blue"])
Omega_from_pandas = SampleSpace(
    name="Omega_from_pandas",
).from_pandas(data)

print(Omega_from_list)
print("\n", Omega_from_sequence)
print("\n", Omega_with_prefixes)
print("\n", Omega_from_pandas)
