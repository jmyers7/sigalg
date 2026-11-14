# sigalg/style.py
from functools import wraps
from importlib import resources

import matplotlib.pyplot as plt

_DEFAULT = str(resources.files("sigalg") / "style" / "sigalg.mplstyle")
_current = _DEFAULT


def set_style(style=None):  # noqa: D103
    global _current
    _current = _DEFAULT if style is None else style


def get_style():  # noqa: D103
    return _current


def use_globally():  # noqa: D103
    plt.style.use(_current)


def styled(plot_func):  # noqa: D103
    @wraps(plot_func)
    def wrapper(*args, **kwargs):
        with plt.style.context(get_style()):
            return plot_func(*args, **kwargs)

    return wrapper
