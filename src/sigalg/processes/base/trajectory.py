from ...core.featurized_spaces.sample_point_features import SamplePointFeatures


class Trajectory(SamplePointFeatures):

    def __init__(self, features):
        super().__init__(name=features.name, values=features)
        self._values.index.name = "time"
        self._values.name = self.name

    @property
    def value_at(self):
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key):
            if key not in self.parent.values.index:
                raise ValueError(f"Time {key} not in trajectory time index")
            return self.parent.values[key]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Trajectory(name={self.name}, length={len(self)})"

    def __str__(self) -> str:
        series_repr = repr(self._values)
        lines = series_repr.split("\n")
        data_lines = [
            line
            for line in lines
            if not line.startswith(("Name:", "Length:", "dtype:"))
        ]
        data_str = "\n".join(data_lines)
        return f"Trajectory '{self.name}'\n" f"Length: {len(self)}\n\n" f"{data_str}"
