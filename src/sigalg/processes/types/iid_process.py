from collections.abc import Hashable
from itertools import product

import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ..base.stochastic_process import StochasticProcess


class IIDProcess(StochasticProcess):

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_enumerated_trajectories(
        cls,
        rv: rv_frozen,
        support: list,
        time: Time,
        name: Hashable | None = "X",
    ):
        all_trajectories = list(product(support, repeat=len(time)))
        data = pd.DataFrame(data=all_trajectories)

        if data.shape[1] == 1:
            outputs = data.iloc[:, 0].to_dict()
            data = data.iloc[:, 0]
            if data.name is None:
                data.name = name
        else:
            outputs = data.apply(lambda row: tuple(row), axis=1).to_dict()

        domain = SampleSpace.from_pandas(data=data.index)
        domain.data_name = "trajectory"
        process = cls(outputs=outputs, domain=domain, vector_index=time, name=name)
        process.data = data
        process.vector_index = time
        return process
