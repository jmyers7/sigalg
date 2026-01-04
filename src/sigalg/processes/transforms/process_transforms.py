from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base.stochastic_process import StochasticProcess


class ProcessTransforms:
    """A collection of static methods for transforming stochastic processes."""

    # @staticmethod
    # def pointwise_map(
    #     process: StochasticProcess, f: Callable[[any], any]
    # ) -> StochasticProcess:
    # transformed = process.values.map(f)
    # return StochasticProcess(
    #     time=process.time,
    #     name=process.name + "_mapped",
    # )
    # pass

    # @staticmethod
    # def time_shift(process: StochasticProcess, shift: int) -> StochasticProcess:
    # shifted_data = process._data.shift(periods=shift, axis=1)
    # return SampleSpaceFeatures(
    #     features=shifted_data,
    #     sample_space=process.sample_space,
    #     feature_index=list(process._data.columns),
    # )
    # pass

    @staticmethod
    def cumsum(process: StochasticProcess) -> StochasticProcess:
        from ..base.stochastic_process import StochasticProcess

        if not process.time.is_discrete:
            raise ValueError(
                "Cumulative sum is only defined for discrete-time processes."
            )
        cumsum_data = process.data.cumsum(axis=1)
        return StochasticProcess(
            domain=process.domain,
            name=f"{process.name}_cumsum" if process.name is not None else None,
            vector_index=process.time,
        ).from_pandas(cumsum_data)

    @staticmethod
    def diff(process: StochasticProcess) -> StochasticProcess:
        from ..base.stochastic_process import StochasticProcess

        if not process.time.is_discrete:
            raise ValueError("Difference is only defined for discrete-time processes.")
        if process.dimension == 1:
            raise ValueError("Difference is not defined for one-dimensional processes.")
        diff_data = process.data.diff(axis=1)
        return StochasticProcess(
            domain=process.domain,
            name=f"{process.name}_diff" if process.name is not None else None,
            vector_index=process.time,
        ).from_pandas(diff_data)

    # @staticmethod
    # def running_maximum(process: StochasticProcess) -> StochasticProcess:
    # max_data = process._data.cummax(axis=1)
    # return SampleSpaceFeatures(
    #     features=max_data,
    #     sample_space=process.sample_space,
    #     feature_index=list(process._data.columns),
    # )
    # pass

    # @staticmethod
    # def moving_average(process: StochasticProcess, window: int) -> StochasticProcess:
    # ma_data = process._data.rolling(window=window, axis=1).mean()
    # return SampleSpaceFeatures(
    #     features=ma_data,
    #     sample_space=process.sample_space,
    #     feature_index=list(process._data.columns),
    # )
    # pass

    # @staticmethod
    # def compose(
    #     process1: StochasticProcess,
    #     process2: StochasticProcess,
    #     op: Callable[[float, float], float],
    # ) -> StochasticProcess:
    # if not process1.sample_space == process2.sample_space:
    #     raise ValueError("Processes must have the same sample space")
    # result_data = op(process1._data, process2._data)
    # return SampleSpaceFeatures(
    #     features=result_data,
    #     sample_space=process1.sample_space,
    #     feature_index=list(process1._data.columns),
    # )
    # pass

    # @staticmethod
    # def stopped_process(
    #     process: StochasticProcess, stopping_times: dict[Hashable, int]
    # ) -> StochasticProcess:
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
    # pass


class ProcessTransformMethods:

    #     def map(self, f: Callable[[float], float]):
    #         return ProcessTransforms.pointwise_map(self, f)

    #     def shift(self, periods: int):
    #         return ProcessTransforms.time_shift(self, periods)

    def cumsum(self):
        return ProcessTransforms.cumsum(self)

    def diff(self):
        return ProcessTransforms.diff(self)


#     def running_max(self):
#         return ProcessTransforms.running_maximum(self)

#     def moving_average(self, window: int):
#         return ProcessTransforms.moving_average(self, window)
