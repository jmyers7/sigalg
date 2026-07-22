"""Empty marker class for function domains."""

from ..indices.index import Index


class Domain(Index):
    """Empty marker class for function domains."""

    _default_name = "D"
    _repr_name = "Domain"
    _variable_names_prefix = "point"
