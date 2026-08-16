import numpy as np
import pandas as pd


def random_tuples(
    size: int,
    sample_range: tuple[int, int] | None = None,
    dim: int = 1,
    random_state: int | np.random.Generator | None = None,
) -> pd.Index:
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    if sample_range is None:
        sample_range = (0, size)

    sample_range = list(range(sample_range[0], sample_range[1]))
    if len(sample_range) < size:
        raise ValueError("sample_range must have at least 'size' elements.")
    tuples = rng.choice(sample_range, size=size, replace=False)

    if dim > 1:
        extra_dims = rng.choice(sample_range, size=(size, dim - 1))
        tuples = np.hstack((tuples.reshape(-1, 1), extra_dims)).tolist()

    tuples = [tuple(t) if isinstance(t, list) else int(t) for t in tuples]

    if dim == 1:
        data = pd.Index(tuples)
    else:
        data = pd.MultiIndex.from_tuples(tuples)

    return data
