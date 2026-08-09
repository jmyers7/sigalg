import re
from collections import Counter

import pandas as pd


def _to_df(
    data: pd.Series | pd.DataFrame,
    suffix: str | None = None,
    subscript_index_flag: bool = False,
) -> pd.DataFrame:
    if suffix is None:
        suffix = ""
    if isinstance(data, pd.DataFrame):
        result = data.add_suffix(suffix)
    else:
        result = data.to_frame().add_suffix(suffix)

    if subscript_index_flag:
        result.index.names = [f"{name}_d" for name in result.index.names]
    return result


def _add_suffix(lst: list, suffix: str) -> list:
    return [f"{name}{suffix}" for name in lst]


def _flatten(t):
    return tuple(
        x for item in t for x in (item if isinstance(item, tuple) else (item,))
    )


def _subscript_var_names(lists, grouped: bool = False):
    names = [x for names in lists for x in names]
    if set(Counter(names).values()) == {1}:
        return names if not grouped else lists

    def base(s):
        m = re.fullmatch(r"(.+)_(\d+)", s)
        return (s, None) if not m else (m.group(1), int(m.group(2)))

    tuples = [base(s) for lst in lists for s in lst]
    bases = [t[0] for t in tuples]
    common_bases = {base for base, count in Counter(bases).items() if count >= 2}

    for base in common_bases:
        idx = 0
        for t in tuples:
            if t[0] == base:
                tuples[tuples.index(t)] = (t[0], idx)
                idx += 1

    result = [f"{t[0]}_{t[1]}" if t[1] is not None else t[0] for t in tuples]

    if grouped:
        grouped_result = []
        for lst in lists:
            grouped_result.append(result[: len(lst)])
            del result[: len(lst)]
        return grouped_result
    else:
        return result
