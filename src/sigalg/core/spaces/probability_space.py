from ..probability_measures import ProbabilityMeasure
from .sample_space import SampleSpace


class ProbabilitySpace(SampleSpace):

    def __init__(self, indices, probabilities=None):
        super().__init__(indices)

        if probabilities is None:
            self._probability_measure = ProbabilityMeasure.uniform(self)
        else:
            self._probability_measure = ProbabilityMeasure(self, probabilities)

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        return self._probability_measure

    @property
    def sample_space(self):
        underlying_sample_space = SampleSpace(self.index)
        return underlying_sample_space

    def set_probability_measure(self, probability_measure: ProbabilityMeasure):
        if not isinstance(probability_measure, ProbabilityMeasure):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure instance."
            )
        if probability_measure.sample_space != self:
            raise ValueError(
                "probability_measure must be defined on this sample space."
            )
        self._probability_measure = probability_measure

    def P(self, key):
        return self._probability_measure(key)

    # def __getitem__(self, key):
    #     if isinstance(key, list):
    #         event_prob = self.probability_measure(Event(self, key))
    #         probs = (
    #             {
    #                 sample_point: self.P(sample_point) / event_prob
    #                 for sample_point in key
    #             }
    #             if event_prob > 0
    #             else 0.0
    #         )
    #         prob_measure = ProbabilityMeasure(self, probs) if event_prob > 0 else None
    #         return Event(
    #             sample_space=self,
    #             event_indices=key,
    #             probability=event_prob,
    #             probability_measure=prob_measure,
    #         )
    #     return self._index[key]

    def __repr__(self):
        return f"ProbabilitySpace({list(self._index)}, P={self._probability_measure.probabilities.to_dict()})"

    def __eq__(self, other):
        return (
            isinstance(other, ProbabilitySpace)
            and super().__eq__(other)
            and self._probability_measure == other._probability_measure
        )

    def __hash__(self):
        return hash(
            (super().__hash__(), tuple(self._probability_measure.probabilities.items()))
        )

    @staticmethod
    def uniform(indices):
        return ProbabilitySpace(indices, probabilities=None)
