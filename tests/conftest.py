from unittest.mock import patch

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def no_plt_show():
    with patch("matplotlib.pyplot.show"):
        yield
