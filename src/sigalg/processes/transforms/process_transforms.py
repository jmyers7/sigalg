from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base.stochastic_process import StochasticProcess


class ProcessTransforms:

    @staticmethod
    def pointwise_map(
        process: StochasticProcess, f: Callable[[any], any]
    ) -> StochasticProcess:
        # transformed = process.values.map(f)
        # return StochasticProcess(
        #     time=process.time,
        #     name=process.name + "_mapped",
        # )
        pass

    @staticmethod
    def time_shift(process: StochasticProcess, shift: int) -> StochasticProcess:
        # shifted_data = process._data.shift(periods=shift, axis=1)
        # return SampleSpaceFeatures(
        #     features=shifted_data,
        #     sample_space=process.sample_space,
        #     feature_index=list(process._data.columns),
        # )
        pass

    @staticmethod
    def cumulative_sum(process: StochasticProcess) -> StochasticProcess:
        # cumsum_data = process._data.cumsum(axis=1)
        # return SampleSpaceFeatures(
        #     features=cumsum_data,
        #     sample_space=process.sample_space,
        #     feature_index=list(process._data.columns),
        # )
        pass

    @staticmethod
    def running_maximum(process: StochasticProcess) -> StochasticProcess:
        # max_data = process._data.cummax(axis=1)
        # return SampleSpaceFeatures(
        #     features=max_data,
        #     sample_space=process.sample_space,
        #     feature_index=list(process._data.columns),
        # )
        pass

    @staticmethod
    def moving_average(process: StochasticProcess, window: int) -> StochasticProcess:
        # ma_data = process._data.rolling(window=window, axis=1).mean()
        # return SampleSpaceFeatures(
        #     features=ma_data,
        #     sample_space=process.sample_space,
        #     feature_index=list(process._data.columns),
        # )
        pass

    @staticmethod
    def compose(
        process1: StochasticProcess,
        process2: StochasticProcess,
        op: Callable[[float, float], float],
    ) -> StochasticProcess:
        # if not process1.sample_space == process2.sample_space:
        #     raise ValueError("Processes must have the same sample space")
        # result_data = op(process1._data, process2._data)
        # return SampleSpaceFeatures(
        #     features=result_data,
        #     sample_space=process1.sample_space,
        #     feature_index=list(process1._data.columns),
        # )
        pass

    @staticmethod
    def stopped_process(
        process: StochasticProcess, stopping_times: dict[Hashable, int]
    ) -> StochasticProcess:
        # stopped_data = process._data.copy()

        # for omega in process.sample_space.index:
        #     if omega in stopping_times:
        #         tau = stopping_times[omega]
        #         # Get column positions
        #         cols = list(process._data.columns)
        #         tau_idx = cols.index(tau) if tau in cols else len(cols) - 1
        #         # After tau, keep the value constant
        #         for j in range(tau_idx + 1, len(cols)):
        #             stopped_data.loc[omega, cols[j]] = stopped_data.loc[
        #                 omega, cols[tau_idx]
        #             ]

        # return SampleSpaceFeatures(
        #     features=stopped_data,
        #     sample_space=process.sample_space,
        #     feature_index=list(process._data.columns),
        # )
        pass


class ProcessTransformMethods:

    def map(self, f: Callable[[float], float]):
        return ProcessTransforms.pointwise_map(self, f)

    def shift(self, periods: int):
        return ProcessTransforms.time_shift(self, periods)

    def cumsum(self):
        return ProcessTransforms.cumulative_sum(self)

    def running_max(self):
        return ProcessTransforms.running_maximum(self)

    def moving_average(self, window: int):
        return ProcessTransforms.moving_average(self, window)
