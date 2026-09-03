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
    given_variable_names: list[Hashable],
    base_measure_data: pd.Series | None = None,
    return_raw_data: bool = False,
) -> pd.Series:
    """Pass."""
    import pandas as pd

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

    if base_measure_data is not None:
        default_dist = base_measure_data.where(
            base_measure_data == 0, 1 / (base_measure_data != 0).sum()
        )

        data.loc[mask, "probs"] = pd.merge(
            left=data.loc[mask, "probs"].reset_index().drop(columns="probs"),
            right=default_dist.rename("probs").reset_index(),
        ).set_index(list(data.index.names))

    else:
        data.loc[mask, "probs"] = 1 / len(self_data)

    data = data.fillna(0.0, inplace=True)["probs"].sort_index()

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
    ascend: bool = True,
    return_conditional_data: bool = False,
) -> pd.Series:
    """Pass."""
    from .function_helpers import ascend_from_atom_space

    if given_data is None:
        data = self_data.divide(base_measure_data, axis=0).fillna(0.0)
        data.name = None

        if ascend:
            return ascend_from_atom_space(
                self_data=data,
                sig_alg_data=sig_alg_data,
                parameter_names=parameter_names,
            )

        else:
            return data

    else:
        conditional_data = compute_conditional_prob_measure(
            self_data=self_data,
            restricted_self_data=restricted_self_data,
            atom_data=atom_data,
            given_variable_names=given_variable_names,
            base_measure_data=base_measure_data,
        )

        data = conditional_data.divide(base_measure_data, axis=0)
        data = data.fillna(0.0, inplace=True).sort_index()

        if ascend:
            if return_conditional_data:
                return ascend_from_atom_space(
                    self_data=data,
                    sig_alg_data=sig_alg_data,
                    parameter_names=given_variable_names,
                ), conditional_data
            else:
                return ascend_from_atom_space(
                    self_data=data,
                    sig_alg_data=sig_alg_data,
                    parameter_names=given_variable_names,
                )

        else:
            if return_conditional_data:
                return data, conditional_data
            else:
                return data


# TODO: write docstring
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
    ascend: bool = True,
    return_conditional_data: bool = False,
) -> Real | pd.Series:
    """Pass."""
    import numpy as np

    if return_conditional_data:
        data, conditional_data = compute_radon_nikodym(
            self_data=self_data,
            base_measure_data=base_measure_data,
            sig_alg_data=sig_alg_data,
            parameter_names=parameter_names,
            given_data=given_data,
            given_variable_names=given_variable_names,
            atom_data=atom_data,
            restricted_self_data=restricted_self_data,
            ascend=ascend,
            return_conditional_data=True,
        )
    else:
        data = compute_radon_nikodym(
            self_data=self_data,
            base_measure_data=base_measure_data,
            sig_alg_data=sig_alg_data,
            parameter_names=parameter_names,
            given_data=given_data,
            given_variable_names=given_variable_names,
            atom_data=atom_data,
            restricted_self_data=restricted_self_data,
            ascend=ascend,
            return_conditional_data=False,
        )

    if base == "e":
        log = np.log
    elif base == "2":
        log = np.log2
    else:
        log = np.log10

    with np.errstate(divide="ignore"):
        data = -log(data)

    if return_conditional_data:
        return data.mask(np.isinf(data), 0), conditional_data
    else:
        return data.mask(np.isinf(data), 0)


def compute_entropy(
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
    from .function_helpers import compute_integral

    if given_data is None:
        data = compute_surprisal(
            self_data=self_data,
            base_measure_data=base_measure_data,
            sig_alg_data=sig_alg_data,
            parameter_names=parameter_names,
            given_data=given_data,
            given_variable_names=given_variable_names,
            atom_data=atom_data,
            restricted_self_data=restricted_self_data,
            base=base,
            ascend=False,
        )

        return compute_integral(
            function_atom_data=data.unstack(level=parameter_names)
            if parameter_names is not None
            else data,
            measure_data=self_data,
            indicator_data=None,
            function_parameter_names=parameter_names,
            measure_parameter_names=parameter_names,
        )

    else:
        data, conditional_data = compute_surprisal(
            self_data=self_data,
            base_measure_data=base_measure_data,
            sig_alg_data=sig_alg_data,
            given_data=given_data,
            given_variable_names=given_variable_names,
            atom_data=atom_data,
            restricted_self_data=restricted_self_data,
            base=base,
            ascend=False,
            return_conditional_data=True,
        )

        inner_integral = compute_integral(
            function_atom_data=data.unstack(level=given_variable_names),
            measure_data=conditional_data,
            indicator_data=None,
            function_parameter_names=given_variable_names,
            measure_parameter_names=given_variable_names,
        )

        return compute_integral(
            function_atom_data=inner_integral,
            measure_data=restricted_self_data,
            indicator_data=None,
            function_parameter_names=None,
            measure_parameter_names=None,
        )
