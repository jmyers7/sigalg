import pandas as pd
import pytest

from sigalg.core import Index, SampleSpace
from sigalg.validation.sample_space_mapping_in import SampleSpaceMappingIn


class TestWithDictMapping:
    def test_sample_space_out_of_order(self):
        Omega = SampleSpace().from_sequence(size=3)
        dict_mapping = {2: "c", 0: "a", 1: "b"}
        v = SampleSpaceMappingIn(mapping=dict_mapping, sample_space=Omega)
        expected_mapping = {0: "a", 1: "b", 2: "c"}

        assert v.mapping == expected_mapping

    def test_sample_space_do_not_match(self):
        Omega = SampleSpace().from_sequence(size=3)
        dict_mapping = {0: "a", 1: "b", 3: "d"}

        with pytest.raises(
            ValueError,
            match="mapping must contain an entry for every sample index in sample_space.",
        ):
            _ = SampleSpaceMappingIn(mapping=dict_mapping, sample_space=Omega)

    def test_generated_1d_sample_space(self):
        dict_mapping = {0: "a", 1: "b", 2: "c"}
        v = SampleSpaceMappingIn(mapping=dict_mapping)
        expected_sample_space = SampleSpace().from_sequence(size=3)

        assert v.sample_space == expected_sample_space
        assert v.sample_space.name == "Omega"

    def test_generated_2d_sample_space(self):
        dict_mapping = {(0, "x"): "a", (1, "y"): "b", (2, "z"): "c"}
        v = SampleSpaceMappingIn(mapping=dict_mapping)
        expected_sample_space = SampleSpace().from_list([(0, "x"), (1, "y"), (2, "z")])

        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space == expected_sample_space
        assert v.sample_space.name == "Omega"


class TestWithSeriesMapping:
    def test_1d_sample_space_out_of_order(self):
        Omega = SampleSpace().from_sequence(size=3)
        series_mapping = pd.Series(["c", "a", "b"], index=[2, 0, 1])
        v = SampleSpaceMappingIn(mapping=series_mapping, sample_space=Omega)
        expected_mapping = pd.Series(["a", "b", "c"], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.sample_space is Omega

    def test_2d_sample_space_out_of_order(self):
        Omega = SampleSpace().from_list(
            [
                ("blue", "circle"),
                ("red", "square"),
                ("green", "triangle"),
            ],
            variable_names=["color", "shape"],
        )
        series_mapping = pd.Series(
            ["c", "a", "b"],
            index=pd.MultiIndex.from_tuples(
                [("green", "triangle"), ("blue", "circle"), ("red", "square")]
            ),
        )
        v = SampleSpaceMappingIn(mapping=series_mapping, sample_space=Omega)
        expected_mapping = pd.Series(
            ["a", "b", "c"],
            index=Omega.data,
        )

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.sample_space is Omega

    def test_1d_sample_space_do_not_match(self):
        Omega = SampleSpace().from_sequence(size=3)
        series_mapping = pd.Series(["a", "b", "d"], index=[0, 1, 3])

        with pytest.raises(
            ValueError,
            match="mapping must contain an entry for every sample index in sample_space.",
        ):
            _ = SampleSpaceMappingIn(mapping=series_mapping, sample_space=Omega)

    def test_2d_sample_space_do_not_match(self):
        Omega = SampleSpace().from_list(
            [
                ("blue", "circle"),
                ("red", "square"),
                ("green", "triangle"),
            ],
            variable_names=["color", "shape"],
        )
        series_mapping = pd.Series(
            ["a", "b", "d"],
            index=pd.MultiIndex.from_tuples(
                [("blue", "circle"), ("red", "square"), ("yellow", "hexagon")]
            ),
        )

        with pytest.raises(
            ValueError,
            match="mapping must contain an entry for every sample index in sample_space.",
        ):
            _ = SampleSpaceMappingIn(mapping=series_mapping, sample_space=Omega)

    def test_generated_1d_sample_space_with_no_name(self):
        series_mapping = pd.Series(["a", "b", "c"], index=[0, 1, 2])
        v = SampleSpaceMappingIn(mapping=series_mapping)
        expected_sample_space = SampleSpace().from_sequence(size=3)

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "Omega"

    def test_generated_2d_sample_space_with_no_name(self):
        series_mapping = pd.Series(
            ["a", "b", "c"],
            index=pd.MultiIndex.from_tuples([(0, "x"), (1, "y"), (2, "z")]),
        )
        v = SampleSpaceMappingIn(mapping=series_mapping)
        expected_sample_space = SampleSpace().from_list([(0, "x"), (1, "y"), (2, "z")])

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "Omega"

    def test_generated_1d_sample_space_with_name(self):
        series_mapping = pd.Series(["a", "b", "c"], index=pd.Index([0, 1, 2], name="S"))
        v = SampleSpaceMappingIn(mapping=series_mapping)
        expected_sample_space = SampleSpace(name="S").from_sequence(size=3)

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "S"

    def test_generated_2d_sample_space_with_names(self):
        series_mapping = pd.Series(
            ["a", "b", "c"],
            index=pd.MultiIndex.from_tuples(
                [(0, "x"), (1, "y"), (2, "z")], names=["number", "letter"]
            ),
        )
        v = SampleSpaceMappingIn(mapping=series_mapping)
        expected_sample_space = SampleSpace(name="Omega").from_list(
            [(0, "x"), (1, "y"), (2, "z")], variable_names=["number", "letter"]
        )

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "Omega"


class TestWithDataFrameMapping:
    def test_1d_sample_space_out_of_order(self):
        Omega = SampleSpace().from_sequence(size=3)
        I = Index().from_list(["letter", "number"])
        df_mapping = pd.DataFrame(
            [("c", 3), ("a", 1), ("b", 2)],
            index=pd.Index([2, 0, 1], name="Omega1"),
            columns=I.data,
        )
        v = SampleSpaceMappingIn(mapping=df_mapping, sample_space=Omega)
        expected_mapping = pd.DataFrame(
            [("a", 1), ("b", 2), ("c", 3)],
            index=Omega.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        assert v.sample_space is Omega

    def test_2d_sample_space_out_of_order(self):
        Omega = SampleSpace().from_list(
            [
                ("blue", "circle"),
                ("red", "square"),
                ("green", "triangle"),
            ],
            variable_names=["color", "shape"],
        )
        I = Index().from_list(["letter", "number"])
        df_mapping = pd.DataFrame(
            [
                ("c", 3),
                ("a", 1),
                ("b", 2),
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    ("green", "triangle"),
                    ("blue", "circle"),
                    ("red", "square"),
                ],
                names=Omega.variable_names,
            ),
            columns=I.data,
        )
        v = SampleSpaceMappingIn(mapping=df_mapping, sample_space=Omega)
        expected_mapping = pd.DataFrame(
            [
                ("a", 1),
                ("b", 2),
                ("c", 3),
            ],
            index=Omega.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        assert v.sample_space is Omega

    def test_1d_sample_space_do_not_match(self):
        Omega = SampleSpace().from_sequence(size=3)
        df_mapping = pd.DataFrame(
            [
                ("a", 1),
                ("b", 2),
                ("d", 4),
            ],
            index=[0, 1, 3],
        )

        with pytest.raises(
            ValueError,
            match="mapping must contain an entry for every sample index in sample_space.",
        ):
            _ = SampleSpaceMappingIn(mapping=df_mapping, sample_space=Omega)

    def test_2d_sample_space_do_not_match(self):
        Omega = SampleSpace().from_list(
            [
                ("blue", "circle"),
                ("red", "square"),
                ("green", "triangle"),
            ],
            variable_names=["color", "shape"],
        )
        df_mapping = pd.DataFrame(
            [
                ("a", 1),
                ("b", 2),
                ("d", 4),
            ],
            index=[
                ("blue", "circle"),
                ("red", "square"),
                ("yellow", "hexagon"),
            ],
        )

        with pytest.raises(
            ValueError,
            match="mapping must contain an entry for every sample index in sample_space.",
        ):
            _ = SampleSpaceMappingIn(mapping=df_mapping, sample_space=Omega)

    def test_index_out_of_order(self):
        df_mapping = pd.DataFrame(
            [(1, "a"), (2, "b"), (3, "c")],
            index=[0, 1, 2],
            columns=["number", "letter"],
        )
        I = Index().from_list(["letter", "number"])
        v = SampleSpaceMappingIn(mapping=df_mapping, index=I)
        expected_mapping = pd.DataFrame(
            [("a", 1), ("b", 2), ("c", 3)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        assert v.index is I

    def test_index_do_not_match(self):
        df_mapping = pd.DataFrame(
            [(1, "a"), (2, "b"), (3, "c")],
            index=[0, 1, 2],
            columns=["number", "letter"],
        )
        I = Index().from_list(["letter", "value"])

        with pytest.raises(
            ValueError,
            match="If the mapping is a data frame, its columns must match the provided index.",
        ):
            _ = SampleSpaceMappingIn(mapping=df_mapping, index=I)

    def test_generated_1d_sample_space_with_no_name(self):
        df_mapping = pd.DataFrame(
            [("a", 1), ("b", 2), ("c", 3)],
            index=[0, 1, 2],
            columns=["letter", "number"],
        )
        v = SampleSpaceMappingIn(mapping=df_mapping)
        expected_sample_space = SampleSpace().from_sequence(size=3)

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "Omega"

    def test_generated_2d_sample_space_with_no_name(self):
        df_mapping = pd.DataFrame(
            [
                ("a", 1),
                ("b", 2),
                ("c", 3),
            ],
            index=pd.MultiIndex.from_tuples([(0, "x"), (1, "y"), (2, "z")]),
            columns=["letter", "number"],
        )
        v = SampleSpaceMappingIn(mapping=df_mapping)
        expected_sample_space = SampleSpace().from_list([(0, "x"), (1, "y"), (2, "z")])

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "Omega"

    def test_generated_1d_sample_space_with_name(self):
        df_mapping = pd.DataFrame(
            [("a", 1), ("b", 2), ("c", 3)],
            index=pd.Index([0, 1, 2], name="S"),
            columns=["letter", "number"],
        )
        v = SampleSpaceMappingIn(mapping=df_mapping)
        expected_sample_space = SampleSpace(name="S").from_sequence(size=3)

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "S"

    def test_generated_2d_sample_space_with_names(self):
        df_mapping = pd.DataFrame(
            [
                ("a", 1),
                ("b", 2),
                ("c", 3),
            ],
            index=pd.MultiIndex.from_tuples(
                [(0, "x"), (1, "y"), (2, "z")], names=["number", "letter"]
            ),
            columns=["value", "rank"],
        )
        v = SampleSpaceMappingIn(mapping=df_mapping)
        expected_sample_space = SampleSpace(name="Omega").from_list(
            [(0, "x"), (1, "y"), (2, "z")], variable_names=["number", "letter"]
        )

        pd.testing.assert_index_equal(v.mapping.index, expected_sample_space.data)
        pd.testing.assert_index_equal(v.sample_space.data, expected_sample_space.data)
        assert v.sample_space.name == "Omega"

    def test_generated_index_with_no_name(self):
        df_mapping = pd.DataFrame(
            [("a", 1), ("b", 2), ("c", 3)],
            index=[0, 1, 2],
            columns=["letter", "number"],
        )
        v = SampleSpaceMappingIn(mapping=df_mapping)
        expected_index = Index().from_list(["letter", "number"])

        pd.testing.assert_index_equal(v.mapping.columns, expected_index.data)
        pd.testing.assert_index_equal(v.index.data, expected_index.data)
        assert v.index.name == "I"
