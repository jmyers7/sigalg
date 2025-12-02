import pandas as pd

from ..featurized_spaces.feature_embedding import FeatureEmbedding
from ..featurized_spaces.featurized_probability_space import FeaturizedProbabilitySpace


class RandomVariableRangeWithProbability(FeaturizedProbabilitySpace):

    def __repr__(self):
        df = pd.concat(
            [self.feature_embedding.values, self.probability_measure.values.to_frame()],
            axis=1,
        )
        return f"Range with probabilites:\n{df}"


class RandomVariableRange(FeatureEmbedding):

    def __repr__(self):
        return f"Range:\n{self.values}"
