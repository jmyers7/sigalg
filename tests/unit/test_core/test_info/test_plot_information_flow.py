import plotly.graph_objects as go
import pytest

from sigalg.core import Filtration, SampleSpace, SigmaAlgebra, Time
from sigalg.core.info.plot_information_flow import plot_information_flow


class TestInputValidation:

    def test_both_parameters_none_raises_error(self):
        """Test that providing neither sigma_algebras nor filtration raises ValueError."""
        with pytest.raises(
            ValueError, match="Either sigma_algebras or filtration must be provided"
        ):
            plot_information_flow()

    def test_non_unique_names_raises_error(self):
        """Test that sigma algebras with duplicate names raise ValueError."""
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        F1 = SigmaAlgebra.trivial(sample_space, name="F")
        F2 = SigmaAlgebra.power_set(sample_space, name="F")

        with pytest.raises(
            ValueError, match="All sigma algebras must have unique names"
        ):
            plot_information_flow(sigma_algebras=[F1, F2])


class TestWithSigmaAlgebras:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def sigma_algebras(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space, name="F0")
        middle = SigmaAlgebra(
            sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1},
            sample_space=sample_space,
            name="F1",
        )
        power_set = SigmaAlgebra.power_set(sample_space, name="F2")
        return [trivial, middle, power_set]

    def test_returns_figure_object(self, sigma_algebras):
        """Test that plot_information_flow returns a Plotly Figure object."""
        fig = plot_information_flow(sigma_algebras=sigma_algebras)
        assert isinstance(fig, go.Figure)

    def test_with_default_labels(self, sigma_algebras):
        """Test that default labels use sigma algebra names."""
        fig = plot_information_flow(sigma_algebras=sigma_algebras)

        # Check that annotations were added for column headers
        assert len(fig.layout.annotations) == 3
        assert fig.layout.annotations[0].text == "F0"
        assert fig.layout.annotations[1].text == "F1"
        assert fig.layout.annotations[2].text == "F2"

    @pytest.mark.parametrize(
        "custom_labels",
        [
            pytest.param(["Time 0", "Time 1", "Time 2"], id="custom_time_labels"),
            pytest.param(["A", "B", "C"], id="single_char_labels"),
        ],
    )
    def test_with_custom_labels(self, sigma_algebras, custom_labels):
        """Test that custom labels are applied correctly."""
        fig = plot_information_flow(sigma_algebras=sigma_algebras, labels=custom_labels)

        assert len(fig.layout.annotations) == 3
        for i, label in enumerate(custom_labels):
            assert fig.layout.annotations[i].text == label

    @pytest.mark.parametrize(
        "show_atom_labels,show_atom_counts",
        [
            pytest.param(True, True, id="show_both"),
            pytest.param(True, False, id="show_labels_only"),
            pytest.param(False, True, id="show_counts_only"),
            pytest.param(False, False, id="show_neither"),
        ],
    )
    def test_atom_display_options(
        self, sigma_algebras, show_atom_labels, show_atom_counts
    ):
        """Test different combinations of atom label and count display options."""
        fig = plot_information_flow(
            sigma_algebras=sigma_algebras,
            show_atom_labels=show_atom_labels,
            show_atom_counts=show_atom_counts,
        )
        assert isinstance(fig, go.Figure)

    def test_with_two_sigma_algebras(self, sample_space):
        """Test with minimal case of two sigma algebras."""
        trivial = SigmaAlgebra.trivial(sample_space, name="Start")
        power_set = SigmaAlgebra.power_set(sample_space, name="End")

        fig = plot_information_flow(sigma_algebras=[trivial, power_set])
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) == 2

    def test_with_single_sigma_algebra(self, sample_space):
        """Test with edge case of single sigma algebra."""
        trivial = SigmaAlgebra.trivial(sample_space, name="Alone")

        fig = plot_information_flow(sigma_algebras=[trivial])
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) == 1


class TestWithFiltration:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space, name="F0")
        middle = SigmaAlgebra(
            sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1},
            sample_space=sample_space,
            name="F1",
        )
        power_set = SigmaAlgebra.power_set(sample_space, name="F2")
        time = Time.discrete(start=0, length=3)
        return Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="Ft"
        )

    def test_returns_figure_object(self, filtration):
        """Test that plot_information_flow with filtration returns a Plotly Figure object."""
        fig = plot_information_flow(filtration=filtration)
        assert isinstance(fig, go.Figure)

    def test_default_labels_from_time(self, filtration):
        """Test that default labels for filtration use time indices."""
        fig = plot_information_flow(filtration=filtration)

        assert len(fig.layout.annotations) == 3
        assert fig.layout.annotations[0].text == "t=0"
        assert fig.layout.annotations[1].text == "t=1"
        assert fig.layout.annotations[2].text == "t=2"

    def test_custom_labels_override_time(self, filtration):
        """Test that custom labels override default time labels for filtration."""
        custom_labels = ["Early", "Middle", "Late"]
        fig = plot_information_flow(filtration=filtration, labels=custom_labels)

        assert len(fig.layout.annotations) == 3
        for i, label in enumerate(custom_labels):
            assert fig.layout.annotations[i].text == label

    def test_with_continuous_time_filtration(self, sample_space):
        """Test plot_information_flow with continuous time filtration."""
        trivial = SigmaAlgebra.trivial(sample_space, name="F0")
        power_set = SigmaAlgebra.power_set(sample_space, name="F1")
        time = Time.continuous(start=0.0, stop=1.0, num_points=2)
        filtration = Filtration(
            sigma_algebras=[trivial, power_set], time=time, name="Ft"
        )

        fig = plot_information_flow(filtration=filtration)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) == 2
        assert fig.layout.annotations[0].text == "t=0.0"
        assert fig.layout.annotations[1].text == "t=1.0"


class TestStylingOptions:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def sigma_algebras(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space, name="A")
        power_set = SigmaAlgebra.power_set(sample_space, name="B")
        return [trivial, power_set]

    @pytest.mark.parametrize(
        "style_param,value",
        [
            pytest.param("node_color", "lightblue", id="node_color"),
            pytest.param("link_color", "rgba(0,0,0,0.2)", id="link_color"),
            pytest.param("height", 600, id="height"),
            pytest.param("width", 800, id="width"),
            pytest.param("font_family", "Courier New", id="font_family"),
            pytest.param("font_size", 18, id="font_size"),
            pytest.param("font_color", "darkblue", id="font_color"),
            pytest.param("title", "Information Flow", id="title"),
            pytest.param("background_color", "#f0f0f0", id="background_color"),
        ],
    )
    def test_individual_style_parameters(self, sigma_algebras, style_param, value):
        """Test individual styling parameters."""
        kwargs = {style_param: value}
        fig = plot_information_flow(sigma_algebras=sigma_algebras, **kwargs)
        assert isinstance(fig, go.Figure)

    def test_multiple_style_parameters(self, sigma_algebras):
        """Test multiple styling parameters together."""
        fig = plot_information_flow(
            sigma_algebras=sigma_algebras,
            height=700,
            width=900,
            font_family="Arial",
            font_size=16,
            node_color="lightgreen",
            link_color="rgba(100,100,100,0.3)",
            background_color="white",
            title="Test Plot",
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 700
        assert fig.layout.width == 900

    def test_margins_parameter(self, sigma_algebras):
        """Test custom margins parameter."""
        custom_margins = {"t": 50, "b": 100, "l": 50, "r": 50}
        fig = plot_information_flow(
            sigma_algebras=sigma_algebras, margins=custom_margins
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.margin.t == 50
        assert fig.layout.margin.b == 100
        assert fig.layout.margin.l == 50
        assert fig.layout.margin.r == 50

    def test_node_font_size_parameter(self, sigma_algebras):
        """Test node_font_size parameter."""
        fig = plot_information_flow(
            sigma_algebras=sigma_algebras,
            node_font_size=20,
        )
        assert isinstance(fig, go.Figure)

    def test_column_font_size_parameter(self, sigma_algebras):
        """Test column_font_size parameter."""
        fig = plot_information_flow(
            sigma_algebras=sigma_algebras,
            column_font_size=22,
        )
        assert isinstance(fig, go.Figure)

    def test_label_y_parameter(self, sigma_algebras):
        """Test label_y parameter for column header positioning."""
        fig = plot_information_flow(
            sigma_algebras=sigma_algebras,
            label_y=-0.2,
        )
        assert isinstance(fig, go.Figure)
        # Check that annotations have the custom y position
        for annotation in fig.layout.annotations:
            assert annotation.y == -0.2
