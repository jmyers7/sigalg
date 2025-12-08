import pandas as pd
import plotly.graph_objects as go

from .sigma_algebra import SigmaAlgebra


def is_sub_algebra(sub: SigmaAlgebra, super: SigmaAlgebra) -> bool:
    sub_atoms = sub.to_events().values()
    super_atoms = super.to_events().values()
    for super_atom in super_atoms:
        if not any(super_atom <= sub_atom for sub_atom in sub_atoms):
            return False
    return True


def compare_sigma_algebras(
    sigma_algebras,
    labels=None,
    title=None,
    color_palette=None,
    height: int = 600,
    node_thickness: int = 25,
    node_pad: int = 20,
    font_family: str = "monospace",
    font_size: int = 12,
    background_color: str = "#FFFFFF",
    text_color: str = "#000000",
    show_atom_counts: bool = True,
) -> go.Figure:
    """
    Create a Sankey diagram comparing multiple sigma algebras.

    Shows the refinement relationships between sigma algebras by visualizing
    how samples flow between atoms of different sigma algebras.

    Parameters
    ----------
    sigma_algebras : List[SigmaAlgebra]
        List of 2 or more sigma algebras to compare. They must all be defined
        on the same sample space.
    labels : Optional[List[str]]
        Labels for each sigma algebra. If None, uses sigma algebra names.
    title : Optional[str]
        Title for the diagram. If None, generates a default title.
    color_palette : Optional[List[str]]
        List of RGBA color strings for the flows. If None, uses default palette.
    height : int
        Height of the figure in pixels.
    node_thickness : int
        Thickness of the nodes.
    node_pad : int
        Vertical padding between nodes.
    font_family : str
        Font family for text.
    font_size : int
        Font size for labels.
    background_color : str
        Background color for the plot.
    text_color : str
        Color for text labels.
    show_atom_counts : bool
        Whether to show sample counts in node labels.

    Returns
    -------
    go.Figure
        Plotly Sankey diagram figure.

    Examples
    --------
    >>> import sigalg as sa
    >>> import numpy as np
    >>>
    >>> # Create sample space
    >>> sample_space = sa.SampleSpace(list(range(32)))
    >>>
    >>> # Create coarse sigma algebra
    >>> np.random.seed(42)
    >>> atom_ids_1 = np.random.choice([0, 1, 2], size=len(sample_space))
    >>> F1 = sa.SigmaAlgebra(
    ...     dict(zip(sample_space, atom_ids_1)),
    ...     sample_space=sample_space,
    ...     name="F1"
    ... )
    >>>
    >>> # Create finer sigma algebra
    >>> atom_ids_2 = [aid * 2 + hash(sid) % 2 for sid, aid in zip(sample_space, atom_ids_1)]
    >>> F2 = sa.SigmaAlgebra(
    ...     dict(zip(sample_space, atom_ids_2)),
    ...     sample_space=sample_space,
    ...     name="F2"
    ... )
    >>>
    >>> # Create Sankey diagram
    >>> fig = sa.compare_sigma_algebras_sankey([F1, F2])
    >>> fig.show()
    """
    # Validate input
    if len(sigma_algebras) < 2:
        raise ValueError("Need at least 2 sigma algebras to compare")

    # Check all sigma algebras have same sample space
    sample_space = sigma_algebras[0].sample_space
    for sa_obj in sigma_algebras[1:]:
        if sa_obj.sample_space != sample_space:
            raise ValueError("All sigma algebras must have the same sample space")

    # Set default labels
    if labels is None:
        labels = [
            sa_obj.name or f"σ-algebra {i+1}" for i, sa_obj in enumerate(sigma_algebras)
        ]

    if len(labels) != len(sigma_algebras):
        raise ValueError("Number of labels must match number of sigma algebras")

    # Set default color palette
    if color_palette is None:
        color_palette = [
            "rgba(255, 195, 0, 0.6)",  # Gold
            "rgba(51, 153, 255, 0.6)",  # Blue
            "rgba(255, 51, 153, 0.6)",  # Pink
            "rgba(153, 102, 255, 0.6)",  # Purple
            "rgba(255, 159, 64, 0.6)",  # Orange
            "rgba(46, 204, 113, 0.6)",  # Green
        ]

    # Build combined dataframe
    atom_cols = {}
    for i, sa_obj in enumerate(sigma_algebras):
        atom_cols[f"atom_{i}"] = sa_obj.values

    df_combined = pd.concat(atom_cols.values(), axis=1, keys=atom_cols.keys())

    # Build node labels and mappings
    all_node_labels = []
    atom_maps = []
    offset = 0

    for i, _ in enumerate(sigma_algebras):
        atoms = sorted(df_combined[f"atom_{i}"].unique())

        if show_atom_counts:
            counts = df_combined[f"atom_{i}"].value_counts()
            node_labels = [f"{labels[i]}\nAtom {a}\n(n={counts[a]})" for a in atoms]
        else:
            node_labels = [f"{labels[i]}\nAtom {a}" for a in atoms]

        all_node_labels.extend(node_labels)
        atom_maps.append({atom: offset + j for j, atom in enumerate(atoms)})
        offset += len(atoms)

    # Build links between consecutive sigma algebras
    sources = []
    targets = []
    values = []
    colors = []

    for i in range(len(sigma_algebras) - 1):
        flow_counts = (
            df_combined.groupby([f"atom_{i}", f"atom_{i+1}"])
            .size()
            .reset_index(name="count")
        )

        for _, row in flow_counts.iterrows():
            source_atom = row[f"atom_{i}"]
            target_atom = row[f"atom_{i+1}"]
            count = row["count"]

            sources.append(atom_maps[i][source_atom])
            targets.append(atom_maps[i + 1][target_atom])
            values.append(count)
            colors.append(color_palette[source_atom % len(color_palette)])

    # Create node colors (solid versions of flow colors)
    node_colors = []
    for i, _ in enumerate(sigma_algebras):
        atoms = sorted(df_combined[f"atom_{i}"].unique())
        for atom in atoms:
            # Convert rgba to solid color
            rgba = color_palette[atom % len(color_palette)]
            node_colors.append(rgba.replace("0.6", "0.8"))

    # Generate default title
    if title is None:
        if len(sigma_algebras) == 2:
            title = f"Comparison: {labels[0]} → {labels[1]}"
        else:
            title = "Refinement chain: " + " → ".join(labels)

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": node_pad,
                    "thickness": node_thickness,
                    "line": {"color": text_color, "width": 1},
                    "label": all_node_labels,
                    "color": node_colors,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": colors,
                },
            )
        ]
    )

    fig.update_layout(
        title={
            "text": title,
            "font": {"family": font_family, "size": font_size + 2, "color": text_color},
        },
        font={"family": font_family, "size": font_size, "color": text_color},
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        height=height,
    )

    return fig
