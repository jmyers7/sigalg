# import pandas as pd
# from .sample_space import SampleSpace
# from ..probability_measures import ProbabilityMeasure
# from .event import Event


class ProbabilitySpace:
    pass
    # _metadata = SampleSpace._metadata + ["prob_measure", "probabilities"]

    # @property
    # def _constructor(self):
    #     return ProbabilitySpace

    # def __init__(
    #     self,
    #     sample_space=None,
    #     prob_measure: ProbabilityMeasure = None,
    #     sample_points: list = None,
    #     data=None,
    #     index=None,
    #     columns=None,
    #     dtype=None,
    #     copy=None,
    # ) -> None:
    #     super().__init__(
    #         sample_space=sample_space,
    #         sample_points=sample_points,
    #         data=data,
    #         index=index,
    #         columns=columns,
    #         dtype=dtype,
    #         copy=copy,
    #     )

    #     self.prob_measure = prob_measure
    #     if sample_space is not None and prob_measure is not None:
    #         self.probabilities = sample_space.apply(
    #             lambda row: prob_measure(Event(sample_space, [row.name])), axis=1
    #         )
    #     else:
    #         self.probabilities = pd.Series(dtype=float)
    #     self.probabilities.name = "probability"