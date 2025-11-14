# from .....old.sample_points_index import SamplePointsIndex
# from ..time import Time
# from .sample_space import SampleSpace
# from .probability_space import ProbabilitySpace
# from ..probability_measures import ProbabilityMeasure
# import pandas as pd
# from itertools import product


# def create_sequence_space(
#     state_space: list,
#     time: Time,
#     sequence_names: SamplePointsIndex | None = None,
#     threshold: int = 1000,
# ) -> SampleSpace:
#     if not isinstance(state_space, list) or len(state_space) == 0:
#         raise ValueError("state_space must be a non-empty list")

#     sequence_length = len(time)
#     sample_space_cardinality = len(state_space) ** sequence_length

#     if sample_space_cardinality > threshold:
#         raise ValueError(
#             f"Sample space size {sample_space_cardinality} exceeds threshold of {threshold}. "
#         )

#     sequences = list(product(state_space, repeat=sequence_length))

#     if sequence_names is None:
#         sequence_names = pd.Index(
#             [f"omega{i+1}" for i in range(len(sequences))],
#             name="sequence",
#         )

#     if sequence_names is not None and len(sequence_names) != len(sequences):
#         raise ValueError(
#             f"sequence_names must have length {len(sequences)}, "
#             f"got {len(sequence_names)}"
#         )

#     sample_points = SamplePointsIndex(sequence_names, name="sequence")

#     return SampleSpace(data=sequences, sample_point_indices=sample_points, columns=time)


# def add_probability_measure(
#     sample_space: SampleSpace, prob_measure: ProbabilityMeasure
# ) -> ProbabilitySpace:
#     return ProbabilitySpace(sample_space=sample_space, prob_measure=prob_measure)
