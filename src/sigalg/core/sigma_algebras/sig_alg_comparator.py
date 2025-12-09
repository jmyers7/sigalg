from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from .sigma_algebra import SigmaAlgebra


class _SankeyPlotMethods(ABC):

    # --------------------- properties --------------------- #

    @property
    @abstractmethod
    def df_combined(self) -> pd.DataFrame:
        pass

    # --------------------- methods --------------------- #

    def _flow_counts(
        self, source_alg_idx: Hashable, target_alg_idx: Hashable
    ) -> pd.DataFrame:
        return (
            self.df_combined.groupby([source_alg_idx, target_alg_idx])
            .size()
            .reset_index(name="count")
        )

    def _get_node_labels(self, show_atom_counts: bool) -> tuple[list, dict]:
        all_node_labels = []
        atom_maps = {}
        offset = 0
        for alg in self.sigma_algebras:
            alg_idx = self._alg_name_to_idx[alg.name]
            if show_atom_counts:
                node_labels = [
                    f"Atom {atom_id}<br>(n={cardinality})"
                    for atom_id, cardinality in alg.atom_id_to_cardinality.items()
                ]
            else:
                node_labels = [f"{alg_idx}\natom {atom_id}" for atom_id in alg.atom_ids]

            all_node_labels.extend(node_labels)
            atom_maps[alg_idx] = {
                atom: offset + j for j, atom in enumerate(alg.atom_ids)
            }
            offset += len(alg.atom_ids)
        return all_node_labels, atom_maps

    def _get_sankey_parameters(self, atom_maps: dict) -> tuple[list, list, list]:
        sources = []
        targets = []
        values = []
        for source_alg_idx, target_alg_idx in zip(self.index[:-1], self.index[1:]):
            counts = self._flow_counts(source_alg_idx, target_alg_idx)
            for _, row in counts.iterrows():
                source_atom_id = row[source_alg_idx]
                target_atom_id = row[target_alg_idx]
                count = row["count"]
                sources.append(atom_maps[source_alg_idx][source_atom_id])
                targets.append(atom_maps[target_alg_idx][target_atom_id])
                values.append(count)
        return sources, targets, values


class SigAlgComparator(_SankeyPlotMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self, sigma_algebras: list[SigmaAlgebra], index: pd.Index | None = None
    ):
        self._validate_parameters(sigma_algebras=sigma_algebras, index=index)
        self._sigma_algebras = sigma_algebras
        self._names = [alg.name for alg in sigma_algebras]
        self._index = index if index is not None else pd.Index(self._names)
        self._df_combined = pd.concat(
            [alg.values for alg in self.sigma_algebras], axis=1
        )
        self._df_combined.columns = self._index
        self._alg_name_to_idx = dict(zip(self._names, self._index))
        self._idx_to_pos = {idx: pos for pos, idx in enumerate(self._index)}

    # --------------------- properties --------------------- #

    @property
    def sigma_algebras(self) -> list[SigmaAlgebra]:
        return self._sigma_algebras.copy()

    @property
    def index(self) -> pd.Index:
        return self._index.copy()

    @property
    def names(self) -> list[str]:
        return self._names.copy()

    @property
    def df_combined(self) -> pd.DataFrame:
        return self._df_combined.copy()

    @property
    def alg_name_to_idx(self) -> dict[str, int]:
        return self._alg_name_to_idx.copy()

    # --------------------- comparison methods --------------------- #

    def is_refinement(
        self, coarser_algebra_idx: Hashable, finer_algebra_idx: Hashable
    ) -> bool:
        return is_refinement(
            coarser_algebra=self.sigma_algebras[self._idx_to_pos[coarser_algebra_idx]],
            finer_algebra=self.sigma_algebras[self._idx_to_pos[finer_algebra_idx]],
        )

    def is_subalgebra(
        self, sub_algebra_idx: Hashable, super_algebra_idx: Hashable
    ) -> bool:
        return is_subalgebra(
            sub_algebra=self.sigma_algebras[self._idx_to_pos[sub_algebra_idx]],
            super_algebra=self.sigma_algebras[self._idx_to_pos[super_algebra_idx]],
        )

    def plot_flow(
        self,
        show_atom_counts: bool = True,
        node_color: str | None = None,
        link_color: str | None = None,
        height: int | None = None,
        width: int | None = None,
        font_family: str | None = None,
        font_size: int | None = None,
        font_color: str | None = None,
        title: str | None = None,
        title_size: int | None = None,
        background_color: str | None = None,
        margins: dict = None,
    ) -> go.Figure:

        if margins is None:
            margins = {"l": 40, "r": 40, "t": 40, "b": 40}

        all_node_labels, atom_maps = self._get_node_labels(show_atom_counts)
        sources, targets, values = self._get_sankey_parameters(atom_maps)

        node_parameters = {
            "label": all_node_labels,
            "line": {"color": "black", "width": 2},
        }
        if node_color is not None:
            node_parameters["color"] = node_color
        link_parameters = {"source": sources, "target": targets, "value": values}
        if link_color is not None:
            link_parameters["color"] = link_color

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=node_parameters,
                    link=link_parameters,
                )
            ]
        )

        fig_parameters = {"margin": margins}
        if height is not None:
            fig_parameters["height"] = height
        if width is not None:
            fig_parameters["width"] = width
        if title is not None:
            fig_parameters["title"] = {
                "text": title,
                "font": {"size": title_size} if title_size is not None else {},
            }
        if background_color is not None:
            fig_parameters["paper_bgcolor"] = background_color
            fig_parameters["plot_bgcolor"] = background_color
        if font_family is not None or font_size is not None or font_color is not None:
            fig_parameters["font"] = {}
            if font_family is not None:
                fig_parameters["font"]["family"] = font_family
            if font_size is not None:
                fig_parameters["font"]["size"] = font_size
            if font_color is not None:
                fig_parameters["font"]["color"] = font_color

        fig.update_layout(**fig_parameters)

        num_algs = len(self.sigma_algebras)
        for i, alg_idx in enumerate(self.index):
            x_position = i / (num_algs - 1) if num_algs > 1 else 0.5
            fig.add_annotation(
                x=x_position,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"{alg_idx}",
                showarrow=False,
                font={
                    "size": (font_size or 12) + 2,
                    "family": font_family or "Arial",
                    "color": font_color or "black",
                },
                xanchor="center",
            )

        return fig

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"CompareSigmaAlgebras({', '.join(self.names)})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sigma_algebras: list[SigmaAlgebra], index: pd.Index | None
    ) -> None:
        from .sigma_algebra import SigmaAlgebra

        if len(sigma_algebras) < 2:
            raise ValueError("Need at least 2 sigma algebras to compare")
        for alg in sigma_algebras:
            if not isinstance(alg, SigmaAlgebra):
                raise ValueError(
                    "All sigma algebras need to be instances of SigmaAlgebra."
                )
        sample_space = sigma_algebras[0].sample_space
        for alg in sigma_algebras[1:]:
            if alg.sample_space != sample_space:
                raise ValueError("All sigma algebras must have the same sample space")
        if index is not None and not isinstance(index, pd.Index):
            raise TypeError("index must be a pandas Index object.")
        if index is not None and len(index) != len(sigma_algebras):
            raise ValueError(
                "If provided, the length of index must match the number of sigma algebras."
            )


def is_subalgebra(sub_algebra: SigmaAlgebra, super_algebra: SigmaAlgebra) -> bool:
    sub_atoms = sub_algebra.atom_id_to_event.values()
    super_atoms = super_algebra.atom_id_to_event.values()
    for super_atom in super_atoms:
        if not any(super_atom <= sub_atom for sub_atom in sub_atoms):
            return False
    return True


def is_refinement(coarser_algebra: SigmaAlgebra, finer_algebra: SigmaAlgebra) -> bool:
    return is_subalgebra(sub_algebra=coarser_algebra, super_algebra=finer_algebra)
