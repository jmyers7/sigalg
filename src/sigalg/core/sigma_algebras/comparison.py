from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from .filtration import Filtration
    from .sigma_algebra import SigmaAlgebra


def is_subalgebra(sub_algebra: SigmaAlgebra, super_algebra: SigmaAlgebra) -> bool:
    sub_atoms = sub_algebra.atom_id_to_event.values()
    super_atoms = super_algebra.atom_id_to_event.values()
    for super_atom in super_atoms:
        if not any(super_atom <= sub_atom for sub_atom in sub_atoms):
            return False
    return True


def is_refinement(coarser_algebra: SigmaAlgebra, finer_algebra: SigmaAlgebra) -> bool:
    return is_subalgebra(sub_algebra=coarser_algebra, super_algebra=finer_algebra)


def plot_information_flow(
    sigma_algebras: list[SigmaAlgebra] | None = None,
    filtration: Filtration | None = None,
    labels: list[str] | None = None,
    show_atom_labels: bool = True,
    show_atom_counts: bool = True,
    node_label_font_size: int | None = None,
    column_header_font_size: int | None = None,
    **style_kwargs,
) -> go.Figure:

    if sigma_algebras is None and filtration is None:
        raise ValueError("Either sigma_algebras or filtration must be provided.")
    if sigma_algebras is None:
        sigma_algebras = filtration.sigma_algebras
    if filtration is not None:
        if labels is None:
            labels = [f"t={t}" for t in filtration.time.values]
    else:
        if labels is None:
            labels = [alg.name for alg in sigma_algebras]

    if len([alg.name for alg in sigma_algebras]) != len(
        {alg.name for alg in sigma_algebras}
    ):
        raise ValueError("All sigma algebras must have unique names.")

    atoms_df = pd.concat([alg.values for alg in sigma_algebras], axis=1)

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
        node_label_font_size=node_label_font_size,
        **style_kwargs,
    )

    _add_column_headers(
        fig=fig,
        labels=labels,
        column_header_font_size=column_header_font_size,
        **style_kwargs,
    )

    return fig


# --------------------- Internal Helpers --------------------- #


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
            if show_atom_counts:
                count = (atoms_df[label] == atom_id).sum()
                node_labels.append(f"Atom {atom_id}<br>(n={count})")
            else:
                node_labels.append(f"Atom {atom_id}")

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
    node_label_font_size: int | None = None,
    font_color: str | None = None,
    title: str | None = None,
    title_font_size: int | None = None,
    title_y: float | None = None,
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
        layout_params["margin"] = {"t": 80, "b": 40, "l": 40, "r": 40}

    if height is not None:
        layout_params["height"] = height
    if width is not None:
        layout_params["width"] = width
    if background_color is not None:
        layout_params["paper_bgcolor"] = background_color
        layout_params["plot_bgcolor"] = background_color

    effective_node_font_size = (
        node_label_font_size if node_label_font_size is not None else font_size
    )

    if any([font_family, effective_node_font_size, font_color]):
        layout_params["font"] = {}
        if font_family:
            layout_params["font"]["family"] = font_family
        if effective_node_font_size:
            layout_params["font"]["size"] = effective_node_font_size
        if font_color:
            layout_params["font"]["color"] = font_color

    fig.update_layout(**layout_params)

    if title is not None:
        fig.add_annotation(
            x=0.5,
            y=title_y if title_y is not None else 1.15,
            xref="paper",
            yref="paper",
            text=title,
            showarrow=False,
            font={
                "size": title_font_size or 16,
                "family": font_family or "Arial",
                "color": font_color or "black",
            },
            xanchor="center",
            yanchor="bottom",
        )

    return fig


def _add_column_headers(
    fig: go.Figure,
    labels: list[str],
    label_y: float = 1.1,
    font_family: str | None = None,
    font_size: int | None = None,
    column_header_font_size: int | None = None,
    font_color: str | None = None,
    **kwargs,
) -> None:

    num_cols = len(labels)

    header_size = (
        column_header_font_size if column_header_font_size is not None else font_size
    )
    if header_size is None:
        header_size = 12
    header_size += 2

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
                "size": header_size,
                "family": font_family or "Arial",
                "color": font_color or "black",
            },
            xanchor="center",
        )
