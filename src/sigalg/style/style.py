from importlib import resources
from typing import Literal

import matplotlib.pyplot as plt

_DEFAULT = str(resources.files("sigalg") / "style" / "sigalg.mplstyle")
_BLACK = str(resources.files("sigalg") / "style" / "sigalg-black.mplstyle")
_BLOG = str(resources.files("sigalg") / "style" / "sigalg-blog.mplstyle")
_current = _DEFAULT


def set_style(style: Literal["default", "black", "blog"] = "default"):  # noqa: D103
    global _current
    if style == "black":
        _current = _BLACK
    elif style == "blog":
        _current = _BLOG
    else:
        _current = _DEFAULT

    plt.style.use(_current)


def get_style():  # noqa: D103
    return _current


def use_globally():  # noqa: D103
    plt.style.use(_current)
