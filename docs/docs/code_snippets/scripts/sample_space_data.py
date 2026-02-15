"""Accessing the underlying data in a sample space."""

from sigalg.core import SampleSpace

# Create a sample space Omega = [s_0, s_1, s_2, s_3, s_4]
Omega = SampleSpace().from_sequence(size=5, prefix="s")

# Access the underlying data of the sample space as a pd.Index
data = Omega.data

print(Omega)
print("\nThe `data` attribute of the sample space:\n", Omega.data)
