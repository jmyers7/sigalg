from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Hashable
    from numbers import Real

    import pandas as pd


# TODO: write docstring
def compute_conditional_prob_measure(
    self_data: pd.Series,
    restricted_self_data: pd.Series,
    atom_data: pd.Series | pd.DataFrame,
    given_data: pd.Series | pd.DataFrame,
    given_variable_names: list[Hashable],
    return_raw_data: bool = False,
    ascend: bool = False,
):
    """Pass."""
    import pandas as pd

    from .function_helpers import ascend_from_atom_space
    from .utils import to_df

    restricted_measure_name = restricted_self_data.name
    restricted_self_data = to_df(restricted_self_data).copy()
    restricted_self_data.index.names = given_variable_names

    atom_data = to_df(atom_data)
    atom_data.columns = given_variable_names

    cross_data = pd.merge(
        left=restricted_self_data.reset_index(),
        right=self_data.index.to_frame(),
        how="cross",
    )

    self_and_sub_data = pd.merge(
        left=self_data.reset_index(), right=atom_data.reset_index()
    )

    prob_data = pd.merge(
        left=self_and_sub_data, right=restricted_self_data.reset_index()
    )
    prob_data["probs"] = prob_data[self_data.name] / prob_data[restricted_measure_name]
    prob_data = prob_data[given_variable_names + self_data.index.names + ["probs"]]

    data = pd.merge(left=cross_data, right=prob_data, how="outer").set_index(
        given_variable_names + self_data.index.names
    )

    if return_raw_data:
        return data.rename(columns={restricted_measure_name: "restricted_probs"})

    mask = data["probs"].isna() & (data[restricted_measure_name] < 1e-10)
    data.loc[mask, "probs"] = 1 / len(self_data)
    data = data.fillna(0.0, inplace=True)["probs"].sort_index()

    if ascend:
        data = ascend_from_atom_space(
            self_data=data.reorder_levels(self_data.index.names + given_variable_names),
            sig_alg_data=given_data,
            parameter_names=self_data.index.names,
        )

        data = data.reorder_levels(
            given_data.index.names + self_data.index.names
        ).sort_index()

    return data


# TODO: write docstring
def compute_radon_nikodym(
    self_data: pd.Series,
    base_measure_data: pd.Series,
    sig_alg_data: pd.Series | pd.DataFrame,
    parameter_names: list[Hashable] | None = None,
    given_data: pd.Series | pd.DataFrame | None = None,
    given_variable_names: list[Hashable] | None = None,
    atom_data: pd.Series | pd.DataFrame | None = None,
    restricted_self_data: pd.Series | None = None,
    return_type: Literal["param", "non_param", None] = None,
) -> pd.Series:
    """Pass."""
    from .function_helpers import ascend_from_atom_space

    if given_data is None:
        data = (self_data / base_measure_data).fillna(0.0)
        data = self_data.divide(base_measure_data, axis=0).fillna(0.0)
        data.name = None
        return ascend_from_atom_space(
            self_data=data, sig_alg_data=sig_alg_data, parameter_names=parameter_names
        )

    else:
        conditional_data = compute_conditional_prob_measure(
            self_data=self_data,
            restricted_self_data=restricted_self_data,
            atom_data=atom_data,
            given_data=given_data,
            given_variable_names=given_variable_names,
            return_raw_data=True,
        )
        conditional_data["derivative"] = conditional_data["probs"].divide(
            base_measure_data, axis=0
        )

        mask = conditional_data["derivative"].isna() & (
            conditional_data["restricted_probs"] < 1e-10
        )
        conditional_data.loc[mask, "derivative"] = 0.0
        derivative_data = conditional_data.fillna(0.0, inplace=True)[
            "derivative"
        ].sort_index()

        data = ascend_from_atom_space(
            self_data=derivative_data,
            sig_alg_data=sig_alg_data,
            parameter_names=given_variable_names,
        )

        if return_type == "param":
            return data

        else:
            return ascend_from_atom_space(
                self_data=data,
                sig_alg_data=given_data,
                parameter_names=sig_alg_data.index.names,
            )


def compute_surprisal(
    self_data: pd.Series,
    base_measure_data: pd.Series,
    sig_alg_data: pd.Series | pd.DataFrame,
    parameter_names: list[Hashable] | None = None,
    given_data: pd.Series | pd.DataFrame | None = None,
    given_variable_names: list[Hashable] | None = None,
    atom_data: pd.Series | pd.DataFrame | None = None,
    restricted_self_data: pd.Series | None = None,
    base: Literal["e", "2", "10"] = "e",
) -> Real | pd.Series:
    """Pass."""
    import numpy as np

    data = compute_radon_nikodym(
        self_data=self_data,
        base_measure_data=base_measure_data,
        sig_alg_data=sig_alg_data,
        parameter_names=parameter_names,
        given_data=given_data,
        given_variable_names=given_variable_names,
        atom_data=atom_data,
        restricted_self_data=restricted_self_data,
        return_type="param",
    )

    if base == "e":
        log = np.log
    elif base == "2":
        log = np.log2
    else:
        log = np.log10

    with np.errstate(divide="ignore"):
        data = -log(data)
    data = data.mask(np.isinf(data), 0)

    return data
