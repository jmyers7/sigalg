# from ..spaces import SampleSpace


# class ProbabilityMeasure:
#     def __init__(self, sample_space, function):
#         if not isinstance(sample_space, SampleSpace):
#             raise TypeError("sample_space must be an instance of SampleSpace.")
#         self._sample_space = sample_space
#         self._measure_function = function

#         total_prob = self._measure_function(sample_space.as_event())
#         if abs(total_prob - 1.0) > 1e-8:
#             raise ValueError(
#                 f"Measure function invalid: P(sample_space)={total_prob}, should be 1.0."
#             )

#     @property
#     def sample_space(self):
#         return self._sample_space

#     def __call__(self, event) -> float:
#         p = self._measure_function(event)
#         return p
