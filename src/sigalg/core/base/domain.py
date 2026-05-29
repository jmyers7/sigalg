"""Empty marker class for function domains."""

from collections.abc import Hashable

from .index import Index


class Domain(Index):
    """Empty marker class for function domains."""

    def __init__(self, name: Hashable = "D") -> None:
        super().__init__(name=name)

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
            return f"Domain '{self.name}':\n{self.data.to_frame().to_string(index=False)}"
