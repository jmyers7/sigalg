from ..featurized_spaces import FeaturizedProbabilitySpace


class RandomVariableRange(FeaturizedProbabilitySpace):

    def __repr__(self):
        series_repr = repr(self._values)
        lines = series_repr.split("\n")
        data_lines = [
            line
            for line in lines
            if not line.startswith(("Name:", "Length:", "dtype:"))
        ]
        data_str = "\n".join(data_lines)
        return (
            f"Range of '{self.name}'\n"
            f"Number of features: {len(self)}\n\n"
            f"{data_str}"
        )
