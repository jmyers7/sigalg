from collections.abc import Hashable, Mapping

from ...core.base.index import Index
from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.random_objects.random_variable import RandomVariable
from ...core.random_objects.random_vector import RandomVector


class StochasticProcess(RandomVector):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: Mapping[Hashable, Hashable],
        domain: SampleSpace,
        name: Hashable | None = "X",
        initial_vector_index: int = 0,
        vector_index: Time | None = None,
    ) -> None:
        super().__init__(
            outputs=outputs,
            domain=domain,
            name=name,
            initial_vector_index=initial_vector_index,
            vector_index=vector_index,
        )

        if vector_index is not None and not isinstance(vector_index, Time):
            raise TypeError("vector_index must be a Time object.")

    # --------------------- properties --------------------- #

    @property
    def vector_index(self) -> Time | None:
        """Get the time index of a stochastic process of length 2 or greater.

        Returns
        -------
        vector_index : Time
            The time index of the stochastic process.
        """
        if self.dimension == 1:
            return None
        elif self._vector_index is not None:
            return self._vector_index
        else:
            self._vector_index = Time.discrete(
                start=self.initial_vector_index,
                length=self.dimension,
                name="T",
                data_name="time",
            )
        return self._vector_index

    @vector_index.setter
    def vector_index(self, vector_index: Time | Index) -> None:

        if self.dimension == 1:
            raise ValueError(
                "Cannot set vector_index for a 1-dimensional RandomVector."
            )

        if not isinstance(vector_index, (Time, Index)):
            raise TypeError("vector_index must be a Time or Index.")
        if len(vector_index) != self.dimension:
            raise ValueError(
                "vector_index size must match the dimension of the RandomVector."
            )
        self._vector_index = vector_index
        self.data.columns = vector_index.data

    @property
    def time(self) -> Time | None:
        """Get the time index of a stochastic process of length 2 or greater.

        This attribute is an alias for `vector_index`.

        Returns
        -------
        time : Time
            The time index of the stochastic process.
        """
        return self.vector_index

    # --------------------- data access methods --------------------- #

    @property
    def rv_at(self):
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, stochastic_process):
            self.stochastic_process = stochastic_process

        def __getitem__(self, time_idx) -> RandomVariable:

            if self.stochastic_process.time.is_discrete:
                if time_idx not in self.stochastic_process.time:
                    raise ValueError(f"Time {time_idx} not in process time index")
                else:
                    name = (
                        f"{self.stochastic_process.name}_{time_idx}"
                        if self.stochastic_process.name is not None
                        else None
                    )
                    return self.stochastic_process.get_component_rv(time_idx).with_name(
                        name
                    )
            else:
                nearest_time = self.stochastic_process.time.find_nearest_time(time_idx)
                name = (
                    f"{self.stochastic_process.name}_{nearest_time}"
                    if self.stochastic_process.name is not None
                    else None
                )
                return self.stochastic_process.get_component_rv(nearest_time).with_name(
                    name
                )
