"""Empty marker class for function domains."""

from collections.abc import Hashable

from ...validation.index_validator import IndexLike
from .index import Index


class Domain(Index):
    """Marker class for function domains."""

    def __init__(
        self,
        indices: IndexLike | None = None,
        name: Hashable = "D",
        variable_names: list[Hashable] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            indices=indices,
            name=name,
            variable_names=variable_names,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return a string representation of the domain.

        Returns
        -------
        repr_str : str
            String representation of the domain.
        """
        if self.data is None:
            return f"Domain '{self.name}': empty"
        else:
            return (
                f"Domain '{self.name}':\n{self.data.to_frame().to_string(index=False)}"
            )
