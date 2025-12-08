from abc import ABC, abstractmethod
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

    def _flow_counts(self, source_name: str, target_name: str) -> pd.DataFrame:
        return (
            self.df_combined.groupby([source_name, target_name])
            .size()
            .reset_index(name="count")
        )

    def _get_node_labels(self, show_atom_counts: bool) -> tuple[list, dict]:
        all_node_labels = []
        atom_maps = {}
        offset = 0
        for alg in self.sigma_algebras:
            if show_atom_counts:
                node_labels = [
                    f"{alg.name}\nAtom {atom_id}\n(n={cardinality})"
                    for atom_id, cardinality in alg.atom_id_to_cardinality.items()
                ]
            else:
                node_labels = [
                    f"{alg.name}\nAtom {atom_id}" for atom_id in alg.atom_ids
                ]

            all_node_labels.extend(node_labels)
            atom_maps[alg.name] = {
                atom: offset + j for j, atom in enumerate(alg.atom_ids)
            }
            offset += len(alg.atom_ids)
        return all_node_labels, atom_maps

    def _get_sankey_parameters(self, atom_maps: dict) -> tuple[list, list, list]:
        sources = []
        targets = []
        values = []
        for source_alg_name, target_alg_name in zip(self.names[:-1], self.names[1:]):
            counts = self._flow_counts(source_alg_name, target_alg_name)
            for _, row in counts.iterrows():
                source_atom_id = row[source_alg_name]
                target_atom_id = row[target_alg_name]
                count = row["count"]
                sources.append(atom_maps[source_alg_name][source_atom_id])
                targets.append(atom_maps[target_alg_name][target_atom_id])
                values.append(count)
        return sources, targets, values


class CompareSigmaAlgebras(_SankeyPlotMethods):

    # --------------------- constructor --------------------- #

    def __init__(self, sigma_algebras: list[SigmaAlgebra]):
        self._validate_parameters(sigma_algebras=sigma_algebras)
        self.sigma_algebras = sigma_algebras
        self.names = [alg.name for alg in sigma_algebras]
        self._df_combined = pd.concat(
            [alg.values for alg in self.sigma_algebras], axis=1
        )

    # --------------------- properties --------------------- #

    @property
    def df_combined(self) -> pd.DataFrame:
        return self._df_combined.copy()

    # --------------------- comparison methods --------------------- #

    def is_refinement(self, coarser_algebra_idx: int, finer_algebra_idx: int) -> bool:
        return is_refinement(
            coarser_algebra=self.sigma_algebras[coarser_algebra_idx],
            finer_algebra=self.sigma_algebras[finer_algebra_idx],
        )

    def is_subalgebra(self, sub_algebra_idx: int, super_algebra_idx: int) -> bool:
        return is_subalgebra(
            sub_algebra=self.sigma_algebras[sub_algebra_idx],
            super_algebra=self.sigma_algebras[super_algebra_idx],
        )

    def refinement_chain(self) -> list[int] | None:
        from itertools import permutations

        n = len(self.sigma_algebras)
        for perm in permutations(range(n)):
            valid = True
            for k in range(n - 1):
                if not self.is_refinement(perm[k + 1], perm[k]):
                    valid = False
                    break
            if valid:
                return list(perm)
        return None

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
    ) -> go.Figure:

        all_node_labels, atom_maps = self._get_node_labels(show_atom_counts)
        sources, targets, values = self._get_sankey_parameters(atom_maps)

        node_parameters = {"label": all_node_labels}
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

        fig_parameters = {}
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

        return fig

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"CompareSigmaAlgebras({', '.join(self.names)})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(sigma_algebras: list[SigmaAlgebra]) -> None:
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


def is_subalgebra(sub_algebra: SigmaAlgebra, super_algebra: SigmaAlgebra) -> bool:
    sub_atoms = sub_algebra.atom_id_to_event.values()
    super_atoms = super_algebra.atom_id_to_event.values()
    for super_atom in super_atoms:
        if not any(super_atom <= sub_atom for sub_atom in sub_atoms):
            return False
    return True


def is_refinement(coarser_algebra: SigmaAlgebra, finer_algebra: SigmaAlgebra) -> bool:
    return is_subalgebra(sub_algebra=coarser_algebra, super_algebra=finer_algebra)
