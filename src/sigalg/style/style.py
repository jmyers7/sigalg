from importlib import resources
from typing import Literal

import matplotlib.pyplot as plt

_DEFAULT = str(resources.files("sigalg") / "style" / "sigalg.mplstyle")
_BLOG = str(resources.files("sigalg") / "style" / "sigalg-blog.mplstyle")
_current = _DEFAULT


def set_style(style: Literal["default", "blog"] = "default"):  # noqa: D103
    global _current
    _current = _DEFAULT if style == "default" else _BLOG
    plt.style.use(_current)


def get_style():  # noqa: D103
    return _current


def use_globally():  # noqa: D103
    plt.style.use(_current)
