import pytest

import sigalg as sa


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def trivial_algebra(self, sample_space):
        return sa.SigmaAlgebra.trivial(sample_space, name="trivial")

    @pytest.fixture
    def power_set_algebra(self, sample_space):
        return sa.SigmaAlgebra.power_set(sample_space, name="power_set")

    @pytest.fixture
    def middle_algebra(self, sample_space):
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        return sa.SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    def test_construction_with_two_algebras(self, trivial_algebra, power_set_algebra):
        comparator = sa.SigmaAlgebraComparator([trivial_algebra, power_set_algebra])
        assert len(comparator.sigma_algebras) == 2
        assert comparator.sigma_algebras[0].sample_space == trivial_algebra.sample_space
        assert (
            comparator.sigma_algebras[1].sample_space == power_set_algebra.sample_space
        )

    def test_construction_with_three_algebras(
        self, trivial_algebra, middle_algebra, power_set_algebra
    ):
        comparator = sa.SigmaAlgebraComparator(
            [trivial_algebra, middle_algebra, power_set_algebra]
        )
        assert len(comparator.sigma_algebras) == 3

    def test_construction_sets_names(self, trivial_algebra, power_set_algebra):
        comparator = sa.SigmaAlgebraComparator([trivial_algebra, power_set_algebra])
        assert comparator.names == ["trivial", "power_set"]

    def test_construction_with_custom_names(self, sample_space):
        alg1 = sa.SigmaAlgebra.trivial(sample_space)
        alg1.name = "Trivial"
        alg2 = sa.SigmaAlgebra.power_set(sample_space)
        alg2.name = "PowerSet"
        comparator = sa.SigmaAlgebraComparator([alg1, alg2])
        assert comparator.names == ["Trivial", "PowerSet"]

    def test_construction_creates_combined_dataframe(
        self, trivial_algebra, power_set_algebra
    ):
        comparator = sa.SigmaAlgebraComparator([trivial_algebra, power_set_algebra])
        assert comparator._df_combined.shape[0] == 4
        assert comparator._df_combined.shape[1] == 2

    def test_construction_with_custom_index(self, trivial_algebra, power_set_algebra):
        import pandas as pd

        custom_index = pd.Index(["F_0", "F_1"])
        comparator = sa.SigmaAlgebraComparator(
            [trivial_algebra, power_set_algebra], index=custom_index
        )
        assert list(comparator.index) == ["F_0", "F_1"]
        assert list(comparator._df_combined.columns) == ["F_0", "F_1"]

    def test_construction_with_numeric_index(self, trivial_algebra, power_set_algebra):
        import pandas as pd

        custom_index = pd.Index([0, 1])
        comparator = sa.SigmaAlgebraComparator(
            [trivial_algebra, power_set_algebra], index=custom_index
        )
        assert list(comparator.index) == [0, 1]

    def test_construction_without_index_uses_names(
        self, trivial_algebra, power_set_algebra
    ):
        comparator = sa.SigmaAlgebraComparator([trivial_algebra, power_set_algebra])
        assert list(comparator.index) == ["trivial", "power_set"]


class TestValidation:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def other_sample_space(self):
        return sa.SampleSpace(["a", "b", "c"])

    def test_construction_with_less_than_two_algebras_raises_error(self, sample_space):
        algebra = sa.SigmaAlgebra.trivial(sample_space)
        with pytest.raises(ValueError, match="at least 2 sigma algebras"):
            sa.SigmaAlgebraComparator([algebra])

    def test_construction_with_empty_list_raises_error(self):
        with pytest.raises(ValueError, match="at least 2 sigma algebras"):
            sa.SigmaAlgebraComparator([])

    def test_construction_with_different_sample_spaces_raises_error(
        self, sample_space, other_sample_space
    ):
        alg1 = sa.SigmaAlgebra.trivial(sample_space)
        alg2 = sa.SigmaAlgebra.trivial(other_sample_space)
        with pytest.raises(ValueError, match="same sample space"):
            sa.SigmaAlgebraComparator([alg1, alg2])

    def test_construction_with_non_sigma_algebra_raises_error(self, sample_space):
        alg = sa.SigmaAlgebra.trivial(sample_space)
        with pytest.raises(ValueError, match="instances of SigmaAlgebra"):
            sa.SigmaAlgebraComparator([alg, "not an algebra"])

    def test_construction_with_wrong_index_length_raises_error(self, sample_space):
        import pandas as pd

        alg1 = sa.SigmaAlgebra.trivial(sample_space)
        alg2 = sa.SigmaAlgebra.power_set(sample_space)
        wrong_index = pd.Index(["F_0"])  # Only 1 element for 2 algebras
        with pytest.raises(ValueError, match="length of index must match"):
            sa.SigmaAlgebraComparator([alg1, alg2], index=wrong_index)

    def test_construction_with_non_index_raises_error(self, sample_space):
        alg1 = sa.SigmaAlgebra.trivial(sample_space)
        alg2 = sa.SigmaAlgebra.power_set(sample_space)
        with pytest.raises(TypeError, match="must be a pandas Index"):
            sa.SigmaAlgebraComparator([alg1, alg2], index=["F_0", "F_1"])


class TestProperties:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def comparator(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        return sa.SigmaAlgebraComparator([trivial, power_set])

    def test_df_combined_returns_dataframe(self, comparator):
        df = comparator.df_combined
        assert df.shape[0] == 4
        assert df.shape[1] == 2

    def test_df_combined_returns_copy(self, comparator):
        df1 = comparator.df_combined
        df1.iloc[0, 0] = 999
        df2 = comparator.df_combined
        assert df2.iloc[0, 0] != 999

    def test_index_property_returns_correct_values(self, comparator):
        import pandas as pd

        assert isinstance(comparator.index, pd.Index)
        assert len(comparator.index) == 2

    def test_index_property_returns_copy(self, sample_space):
        import pandas as pd

        alg1 = sa.SigmaAlgebra.trivial(sample_space)
        alg2 = sa.SigmaAlgebra.power_set(sample_space)
        custom_index = pd.Index(["F_0", "F_1"])
        comparator = sa.SigmaAlgebraComparator([alg1, alg2], index=custom_index)
        idx1 = comparator.index
        # Verify it's a copy by checking we can't modify the original
        assert idx1 is not comparator._index

    def test_alg_name_to_idx_with_custom_index(self, sample_space):
        import pandas as pd

        alg1 = sa.SigmaAlgebra.trivial(sample_space, name="first")
        alg2 = sa.SigmaAlgebra.power_set(sample_space, name="second")
        custom_index = pd.Index(["F_0", "F_1"])
        comparator = sa.SigmaAlgebraComparator([alg1, alg2], index=custom_index)
        assert comparator.alg_name_to_idx == {"first": "F_0", "second": "F_1"}


class TestIsRefinement:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def comparator(self, sample_space):
        import pandas as pd

        trivial = sa.SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        custom_index = pd.Index(["F_0", "F_1", "F_2"])
        return sa.SigmaAlgebraComparator([trivial, middle, power_set], index=custom_index)

    def test_trivial_refines_middle(self, comparator):
        assert comparator.is_refinement("F_0", "F_1")

    def test_middle_refines_power_set(self, comparator):
        assert comparator.is_refinement("F_1", "F_2")

    def test_trivial_refines_power_set(self, comparator):
        assert comparator.is_refinement("F_0", "F_2")

    def test_middle_does_not_refine_trivial(self, comparator):
        assert not comparator.is_refinement("F_1", "F_0")

    def test_power_set_does_not_refine_middle(self, comparator):
        assert not comparator.is_refinement("F_2", "F_1")

    def test_algebra_refines_itself(self, comparator):
        assert comparator.is_refinement("F_0", "F_0")
        assert comparator.is_refinement("F_1", "F_1")
        assert comparator.is_refinement("F_2", "F_2")


class TestIsSubalgebra:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def comparator(self, sample_space):
        import pandas as pd

        trivial = sa.SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        custom_index = pd.Index([0, 1, 2])
        return sa.SigmaAlgebraComparator([trivial, middle, power_set], index=custom_index)

    def test_trivial_is_subalgebra_of_middle(self, comparator):
        assert comparator.is_subalgebra(0, 1)

    def test_middle_is_subalgebra_of_power_set(self, comparator):
        assert comparator.is_subalgebra(1, 2)

    def test_trivial_is_subalgebra_of_power_set(self, comparator):
        assert comparator.is_subalgebra(0, 2)

    def test_middle_is_not_subalgebra_of_trivial(self, comparator):
        assert not comparator.is_subalgebra(1, 0)

    def test_power_set_is_not_subalgebra_of_middle(self, comparator):
        assert not comparator.is_subalgebra(2, 1)

    def test_algebra_is_subalgebra_of_itself(self, comparator):
        assert comparator.is_subalgebra(0, 0)
        assert comparator.is_subalgebra(1, 1)
        assert comparator.is_subalgebra(2, 2)


class TestPlotFlow:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def comparator(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space, name="trivial")
        power_set = sa.SigmaAlgebra.power_set(sample_space, name="power_set")
        return sa.SigmaAlgebraComparator([trivial, power_set])

    def test_plot_flow_returns_figure(self, comparator):
        import plotly.graph_objects as go

        fig = comparator.plot_flow()
        assert isinstance(fig, go.Figure)

    def test_plot_flow_with_show_atom_counts_false(self, comparator):
        import plotly.graph_objects as go

        fig = comparator.plot_flow(show_atom_counts=False)
        assert isinstance(fig, go.Figure)

    def test_plot_flow_with_custom_title(self, comparator):
        fig = comparator.plot_flow(title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_plot_flow_with_custom_height(self, comparator):
        fig = comparator.plot_flow(height=800)
        assert fig.layout.height == 800

    def test_plot_flow_with_custom_colors(self, comparator):
        fig = comparator.plot_flow(
            node_color="blue", link_color="red", background_color="white"
        )
        assert fig.layout.paper_bgcolor == "white"
        assert fig.layout.plot_bgcolor == "white"


class TestHelperFunctionIsSubalgebra:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    def test_trivial_is_subalgebra_of_power_set(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        assert sa.is_subalgebra(trivial, power_set)

    def test_power_set_is_not_subalgebra_of_trivial(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        assert not sa.is_subalgebra(power_set, trivial)

    def test_algebra_is_subalgebra_of_itself(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        assert sa.is_subalgebra(trivial, trivial)

    def test_coarser_is_subalgebra_of_finer(self, sample_space):
        atom_ids_coarse = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_coarse, sample_space=sample_space
        )
        atom_ids_fine = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_fine, sample_space=sample_space
        )
        assert sa.is_subalgebra(coarse, fine)

    def test_finer_is_not_subalgebra_of_coarser(self, sample_space):
        atom_ids_coarse = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_coarse, sample_space=sample_space
        )
        atom_ids_fine = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_fine, sample_space=sample_space
        )
        assert not sa.is_subalgebra(fine, coarse)

    def test_incomparable_algebras_neither_is_subalgebra(self, sample_space):
        atom_ids1 = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        alg1 = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids1, sample_space=sample_space
        )
        atom_ids2 = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        alg2 = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids2, sample_space=sample_space
        )
        assert not sa.is_subalgebra(alg1, alg2)
        assert not sa.is_subalgebra(alg2, alg1)


class TestHelperFunctionIsRefinement:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    def test_finer_refines_coarser(self, sample_space):
        atom_ids_coarse = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_coarse, sample_space=sample_space
        )
        atom_ids_fine = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_fine, sample_space=sample_space
        )
        assert sa.is_refinement(coarse, fine)

    def test_coarser_does_not_refine_finer(self, sample_space):
        atom_ids_coarse = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_coarse, sample_space=sample_space
        )
        atom_ids_fine = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids_fine, sample_space=sample_space
        )
        assert not sa.is_refinement(fine, coarse)

    def test_power_set_refines_trivial(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        assert sa.is_refinement(trivial, power_set)

    def test_trivial_does_not_refine_power_set(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        assert not sa.is_refinement(power_set, trivial)

    def test_algebra_refines_itself(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        assert sa.is_refinement(trivial, trivial)
