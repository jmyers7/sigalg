# from __future__ import annotations
# from collections import OrderedDict
# from typing import List
# from ..sigma_algebras.sigma_algebra import SigmaAlgebra
# from ..time.time import Time
# from numbers import Real


# class FilteredSigmaAlgebra(SigmaAlgebra):
#     """
#     A filtered sigma-algebra is a sigma-algebra equipped with a filtration,
#     i.e., an increasing family of sub-sigma-algebras.

#     Attributes
#     ----------
#     filtration : OrderedDict
#         An ordered dictionary mapping time indices to sigma-algebras.
#     """

#     def __init__(
#         self,
#         filtration: List[SigmaAlgebra],
#         time: Time,
#     ) -> None:

#         items = list(zip(time, filtration))
#         self.filtration = OrderedDict(items)
#         self.time = time

#         ambient = list(self.filtration.values())[-1]
#         super().__init__(ambient.sample_space, ambient.atom_ids)
#         self._validate_filtration()

#     def at(self, t: Real) -> SigmaAlgebra:
#         if t not in self.time:
#             raise KeyError(f"time {t} not in time index")
#         return self.filtration[t]

#     def __getitem__(self, t: Real) -> SigmaAlgebra:
#         return self.at(t)

#     def __len__(self) -> int:
#         return len(self.filtration)

#     def __iter__(self):
#         return iter(self.filtration.items())

#     def _validate_filtration(self) -> None:

#         levels = list(self.filtration.items())

#         if not levels:
#             raise ValueError("Filtration cannot be empty.")

#         for t, Ft in self:
#             if not isinstance(Ft, SigmaAlgebra):
#                 raise TypeError(f"F_{t} must be a SigmaAlgebra instance.")

#         sample_spaces = [Ft.sample_space for _, Ft in levels]
#         first_space = sample_spaces[0]
#         if not all(sp is first_space for sp in sample_spaces):
#             raise ValueError("All F_t must share the same SampleSpace object.")

#         for (s, Fs), (t, Ft) in zip(levels[:-1], levels[1:]):
#             if not Fs.is_subalgebra_of(Ft):
#                 raise ValueError(f"Filtration not increasing: F_{s} ⊄ F_{t}.")

#     def __repr__(self) -> str:
#         cls = self.__class__.__name__
#         return f"{cls}(n_levels={len(self)})"
