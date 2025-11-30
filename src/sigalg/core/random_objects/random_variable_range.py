import pandas as pd

from ..featurized_spaces import FeaturizedProbabilitySpace, FeaturizedSampleSpace


class RandomVariableRangeWithProbability(FeaturizedProbabilitySpace):

    def __repr__(self):
        df = pd.concat(
            [self.feature_embedding.values, self.probability_measure.values.to_frame()],
            axis=1,
        )
        return f"Range with probabilites:\n{df}"


class RandomVariableRange(FeaturizedSampleSpace):

    def __repr__(self):
        return f"Range:\n{self.feature_embedding.values}"
