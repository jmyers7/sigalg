from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..sigma_algebras.filtration import Filtration
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


def plot_information_flow(
    sigma_algebras: list[SigmaAlgebra] | None = None,
    filtration: Filtration | None = None,
    labels: list[str] | None = None,
    show_atom_labels: bool = True,
    show_atom_counts: bool = True,
    **style_kwargs,
) -> go.Figure:
    """Plot the information flow between a sequence of sigma algebras or a filtration.

    Parameters
    ----------
    sigma_algebras : list[SigmaAlgebra] | None, default=None
        A list of SigmaAlgebra objects to visualize. If None, the `filtration`
        parameter must be provided.
    filtration : Filtration | None, default=None
        A Filtration object whose sigma algebras will be visualized. If None,
        the `sigma_algebras` parameter must be provided.
    labels : list[str] | None, default=None
        Labels for each sigma algebra or filtration time point. If None,
        default labels will be used.
    show_atom_labels : bool, default=True
        Whether to display atom labels on the nodes. Default is True.
    show_atom_counts : bool, default=True
        Whether to display the count of samples in each atom on the nodes. Default is True.
    **style_kwargs
        Additional styling keyword arguments for the Sankey diagram.

    Returns
    -------
    plot : go.Figure
        A Plotly Figure object representing the information flow Sankey diagram.
    """
    if sigma_algebras is None and filtration is None:
        raise ValueError("Either sigma_algebras or filtration must be provided.")
    if sigma_algebras is None:
        sigma_algebras = filtration.sigma_algebras
    if filtration is not None:
        if labels is None:
            labels = [f"t={t}" for t in filtration.time.data]
    else:
        if labels is None:
            labels = [alg.name for alg in sigma_algebras]

    if len([alg.name for alg in sigma_algebras]) != len(
        {alg.name for alg in sigma_algebras}
    ):
        raise ValueError("All sigma algebras must have unique names.")

    for alg in sigma_algebras:
        alg.data.rename(alg.name, inplace=True)
    atoms_df = pd.concat([alg.data for alg in sigma_algebras], axis=1)

    node_labels, atom_to_node = _build_node_labels(
        atoms_df=atoms_df, show_atom_counts=show_atom_counts
    )

    sources, targets, values = _build_sankey_links(
        atoms_df=atoms_df, atom_to_node=atom_to_node
    )

    fig = _create_sankey_figure(
        node_labels=node_labels,
        sources=sources,
        targets=targets,
        values=values,
        show_atom_labels=show_atom_labels,
        **style_kwargs,
    )

    _add_column_headers(
        fig=fig,
        labels=labels,
        **style_kwargs,
    )

    return fig


def _build_node_labels(
    atoms_df: pd.DataFrame, show_atom_counts: bool
) -> tuple[list[str], dict]:

    node_labels = []
    atom_to_node = {}
    node_offset = 0

    for label in atoms_df.columns:
        atom_ids = atoms_df[label].unique()
        atom_to_node[label] = {}

        for atom_id in atom_ids:
            if isinstance(atom_id, tuple):
                atom_id_display = tuple(
                    x.item() if hasattr(x, "item") else x for x in atom_id
                )
            elif hasattr(atom_id, "item"):
                atom_id_display = atom_id.item()
            else:
                atom_id_display = atom_id

            if show_atom_counts:
                count = (atoms_df[label] == atom_id).sum()
                node_labels.append(f"Atom {atom_id_display}<br>(n={count})")
            else:
                node_labels.append(f"Atom {atom_id_display}")

            atom_to_node[label][atom_id] = node_offset
            node_offset += 1

    return node_labels, atom_to_node


def _build_sankey_links(
    atoms_df: pd.DataFrame, atom_to_node: dict
) -> tuple[list[int], list[int], list[int]]:

    sources = []
    targets = []
    values = []

    for src_alg_name, target_alg_name in zip(
        atoms_df.columns[:-1], atoms_df.columns[1:]
    ):
        flow_counts = (
            atoms_df.groupby([src_alg_name, target_alg_name])
            .size()
            .reset_index(name="count")
        )

        for _, row in flow_counts.iterrows():
            source_atom = row[src_alg_name]
            target_atom = row[target_alg_name]
            count = row["count"]

            sources.append(atom_to_node[src_alg_name][source_atom])
            targets.append(atom_to_node[target_alg_name][target_atom])
            values.append(count)

    return sources, targets, values


def _create_sankey_figure(
    node_labels: list[str],
    sources: list[int],
    targets: list[int],
    values: list[int],
    show_atom_labels: bool,
    node_color: str | None = None,
    link_color: str | None = None,
    height: int | None = None,
    width: int | None = None,
    font_family: str | None = None,
    font_size: int | None = None,
    node_font_size: int | None = None,
    font_color: str | None = None,
    title: str | None = None,
    background_color: str | None = None,
    margins: dict | None = None,
    **kwargs,
) -> go.Figure:

    node_params = {"line": {"color": "black", "width": 2}, "hoverinfo": "skip"}
    if show_atom_labels:
        node_params["label"] = node_labels
    else:
        node_params["label"] = [""] * len(node_labels)
    if node_color is not None:
        node_params["color"] = node_color

    link_params = {
        "source": sources,
        "target": targets,
        "value": values,
        "hoverinfo": "skip",
    }
    if link_color is not None:
        link_params["color"] = link_color

    fig = go.Figure(data=[go.Sankey(node=node_params, link=link_params)])

    layout_params = {}
    if margins is not None:
        layout_params["margin"] = margins
    else:
        layout_params["margin"] = {"t": 40, "b": 80, "l": 40, "r": 40}

    if height is not None:
        layout_params["height"] = height
    if width is not None:
        layout_params["width"] = width
    if background_color is not None:
        layout_params["paper_bgcolor"] = background_color
        layout_params["plot_bgcolor"] = background_color

    effective_node_font_size = (
        node_font_size if node_font_size is not None else font_size
    )
    if effective_node_font_size is None:
        effective_node_font_size = 14
    layout_params["font"] = {}
    layout_params["font"]["size"] = effective_node_font_size

    if any([font_family, effective_node_font_size, font_color]):
        layout_params["font"] = {}
        if font_family:
            layout_params["font"]["family"] = font_family
        if effective_node_font_size:
            layout_params["font"]["size"] = effective_node_font_size
        if font_color:
            layout_params["font"]["color"] = font_color

    if title is not None:
        layout_params["title"] = {
            "text": title,
            "font": {
                "family": font_family or "Arial",
                "size": font_size or 20,
                "color": font_color or "black",
            },
        }

    fig.update_layout(**layout_params)

    return fig


def _add_column_headers(
    fig: go.Figure,
    labels: list[str],
    label_y: float = -0.15,
    font_family: str | None = None,
    font_size: int | None = None,
    column_font_size: int | None = None,
    font_color: str | None = None,
    **kwargs,
) -> None:

    num_cols = len(labels)

    effective_column_font_size = (
        column_font_size if column_font_size is not None else font_size
    )
    if effective_column_font_size is None:
        effective_column_font_size = 16

    for i, label in enumerate(labels):
        x_pos = i / (num_cols - 1) if num_cols > 1 else 0.5

        fig.add_annotation(
            x=x_pos,
            y=label_y,
            xref="paper",
            yref="paper",
            text=str(label),
            showarrow=False,
            font={
                "size": effective_column_font_size,
                "family": font_family or "Arial",
                "color": font_color or "black",
            },
            xanchor="center",
        )
