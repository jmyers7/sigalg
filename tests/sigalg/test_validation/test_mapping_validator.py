import pandas as pd
import pytest

from sigalg.core import Domain, Index, SampleSpace
from sigalg.validation.mapping_validator import MappingValidator


@pytest.fixture
def Omega():
    return SampleSpace(["a", "b", "c"], variable_names=["omega"])


@pytest.fixture
def I():  # noqa: E743
    return Index(["odd", "even"], variable_names=["parity"])


@pytest.fixture
def Omega2D():
    return SampleSpace([("a", 1), ("b", 2), ("c", 3)], variable_names=["letter", "num"])


class TestFromDictWithDomainAndNoIndex:
    def test_with_1D_keys_and_1D_values(self, Omega):
        mapping = {"b": 2, "a": 1, "c": 3}
        v = MappingValidator(mapping=mapping, domain=Omega)
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_length_1_tuple_values(self, Omega):
        mapping = {"b": (2,), "a": (1,), "c": (3,)}
        v = MappingValidator(mapping=mapping, domain=Omega, output_name="x")
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data, name="x")

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name == "x"
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_2D_values(self, Omega):
        mapping = {"b": (3, 4), "a": (1, 2), "c": (5, 6)}
        v = MappingValidator(mapping=mapping, domain=Omega, name="X")
        expected_index = Index(indices=[0, 1], name="I")
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=Omega.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, expected_index.data)
        assert v.domain is Omega
        assert v.index == expected_index
        assert v.output_name is None
        assert v.name == "X"
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_1D_values(self, Omega):
        mapping = {("b",): 2, ("a",): 1, ("c",): 3}
        v = MappingValidator(mapping=mapping, domain=Omega)
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_length_1_tuple_values(self, Omega):
        mapping = {("b",): (2,), ("a",): (1,), ("c",): (3,)}
        v = MappingValidator(mapping=mapping, domain=Omega, name="X")
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name is None
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_2D_values(self, Omega):
        mapping = {("b",): (3, 4), ("a",): (1, 2), ("c",): (5, 6)}
        v = MappingValidator(mapping=mapping, domain=Omega, name="X")
        expected_index = Index(indices=[0, 1], name="I")
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=Omega.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, expected_index.data)
        assert v.domain is Omega
        assert v.index == expected_index
        assert v.name == "X"
        assert v.kind == "any"

    def test_with_2D_keys_and_1D_values(self, Omega2D):
        mapping = {("b", 2): 2, ("a", 1): 1, ("c", 3): 3}
        v = MappingValidator(
            mapping=mapping, domain=Omega2D, name="X", output_name="num"
        )
        expected_mapping = pd.Series([1, 2, 3], index=Omega2D.data, name="num")

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega2D
        assert v.output_name == "num"
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_length_1_tuple_values(self, Omega2D):
        mapping = {("b", 2): (2,), ("a", 1): (1,), ("c", 3): (3,)}
        v = MappingValidator(mapping=mapping, domain=Omega2D, name="X")
        expected_mapping = pd.Series([1, 2, 3], index=Omega2D.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega2D
        assert v.output_name is None
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_2D_values(self, Omega2D):
        mapping = {("b", 2): (2, 4), ("a", 1): (1, 2), ("c", 3): (3, 6)}
        v = MappingValidator(mapping=mapping, domain=Omega2D, name="X")
        expected_index = Index(indices=[0, 1], name="I")
        expected_mapping = pd.DataFrame(
            [[1, 2], [2, 4], [3, 6]], index=Omega2D.data, columns=expected_index.data
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, expected_index.data)
        assert v.domain is Omega2D
        assert v.name == "X"
        assert v.index == expected_index
        assert v.kind == "any"

    def test_incompatible_keys_and_sample_space_raises(self, Omega):
        mapping = {"d": 4, "e": 5, "f": 6}

        with pytest.raises(
            ValueError,
            match="The mapping must contain an entry for every point in the domain",
        ):
            MappingValidator(mapping=mapping, domain=Omega, name="X")

    def test_values_mix_of_tuples_and_non_tuples_raises(self, Omega):
        mapping = {"a": (1,), "b": 2, "c": 3}

        with pytest.raises(
            ValueError,
            match="If the mapping contains a tuple value, all values must be tuples.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, name="X")

    def test_inconsistent_tuple_lengths_raises(self, Omega):
        mapping = {"a": (1, 2), "b": (3, 4, 5), "c": (6, 7)}

        with pytest.raises(
            ValueError,
            match="All tuples in the mapping must have the same length.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, name="X")


class TestFromDictWithDomainAndIndex:
    def test_with_1D_keys_and_1D_values(self, Omega, I):
        mapping = {"b": 2, "a": 1, "c": 3}
        v = MappingValidator(mapping=mapping, domain=Omega, index=I)
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_length_1_tuple_values(self, Omega, I):
        mapping = {"b": (2,), "a": (1,), "c": (3,)}
        v = MappingValidator(mapping=mapping, domain=Omega, index=I)
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_2D_values(self, Omega, I):
        mapping = {"b": (3, 4), "a": (1, 2), "c": (5, 6)}
        v = MappingValidator(mapping=mapping, domain=Omega, index=I)
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=Omega.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, I.data)
        assert v.domain is Omega
        assert v.index is I
        assert v.output_name is None
        assert v.name is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_1D_values(self, Omega, I):
        mapping = {("b",): 2, ("a",): 1, ("c",): 3}
        v = MappingValidator(mapping=mapping, domain=Omega, index=I)
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_length_1_tuple_values(self, Omega, I):
        mapping = {("b",): (2,), ("a",): (1,), ("c",): (3,)}
        v = MappingValidator(mapping=mapping, domain=Omega, index=I)
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_2D_values(self, Omega, I):
        mapping = {("b",): (3, 4), ("a",): (1, 2), ("c",): (5, 6)}
        v = MappingValidator(mapping=mapping, domain=Omega, index=I)
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=Omega.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, I.data)
        assert v.domain is Omega
        assert v.index == I
        assert v.output_name is None
        assert v.name is None
        assert v.kind == "any"

    def test_with_2D_keys_and_1D_values(self, Omega2D, I):
        mapping = {("b", 2): 2, ("a", 1): 1, ("c", 3): 3}
        v = MappingValidator(mapping=mapping, domain=Omega2D, index=I)
        expected_mapping = pd.Series([1, 2, 3], index=Omega2D.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega2D
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_length_1_tuple_values(self, Omega2D, I):
        mapping = {("b", 2): (2,), ("a", 1): (1,), ("c", 3): (3,)}
        v = MappingValidator(mapping=mapping, domain=Omega2D, index=I)
        expected_mapping = pd.Series([1, 2, 3], index=Omega2D.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega2D
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_2D_values(self, Omega2D, I):
        mapping = {("b", 2): (2, 4), ("a", 1): (1, 2), ("c", 3): (3, 6)}
        v = MappingValidator(mapping=mapping, domain=Omega2D, index=I)
        expected_mapping = pd.DataFrame(
            [[1, 2], [2, 4], [3, 6]], index=Omega2D.data, columns=I.data
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, I.data)
        assert v.domain is Omega2D
        assert v.output_name is None
        assert v.name is None
        assert v.index == I
        assert v.kind == "any"

    def test_incompatible_keys_and_sample_space_raises(self, Omega, I):
        mapping = {"d": 4, "e": 5, "f": 6}

        with pytest.raises(
            ValueError,
            match="The mapping must contain an entry for every point in the domain",
        ):
            MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")

    def test_values_mix_of_tuples_and_non_tuples_raises(self, Omega, I):
        mapping = {"a": (1,), "b": 2, "c": 3}

        with pytest.raises(
            ValueError,
            match="If the mapping contains a tuple value, all values must be tuples.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")

    def test_inconsistent_tuple_lengths_raises(self, Omega, I):
        mapping = {"a": (1, 2), "b": (3, 4, 5), "c": (6, 7)}

        with pytest.raises(
            ValueError,
            match="All tuples in the mapping must have the same length.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")

    def test_validate_columns_against_2D_index_raises(self, Omega2D):
        mapping = {("b", 2): (2, 4), ("a", 1): (1, 2), ("c", 3): (3, 6)}
        I = Index([(1, 2), (3, 4)])

        with pytest.raises(
            ValueError,
            match="The mapping columns cannot be validated against a pd.MultiIndex.",
        ):
            MappingValidator(mapping=mapping, domain=Omega2D, index=I, name="X")

    def test_incompatible_index_length_raises(self, Omega2D):
        mapping = {("b", 2): (2, 4), ("a", 1): (1, 2), ("c", 3): (3, 6)}
        I = Index(["first", "second", "third"])

        with pytest.raises(
            ValueError,
            match="The length of the provided index does not match the dimension of the outputs of the mapping.",
        ):
            MappingValidator(mapping=mapping, domain=Omega2D, index=I, name="X")


class TestFromDictWithNoDomainAndNoIndex:
    def test_with_1D_keys_and_1D_values(self):
        mapping = {"a": 1, "b": 2, "c": 3}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_length_1_tuple_values(self):
        mapping = {"a": (1,), "b": (2,), "c": (3,)}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_2D_values(self):
        mapping = {"a": (1, 2), "b": (3, 4), "c": (5, 6)}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain(["a", "b", "c"])
        expected_index = Index([0, 1])
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=expected_domain.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index == expected_index
        assert v.kind == "any"

    def test_with_2D_keys_and_1D_values(self):
        mapping = {("a", 1): 1, ("b", 2): 2, ("c", 3): 3}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain([("a", 1), ("b", 2), ("c", 3)])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_length_1_tuple_values(self):
        mapping = {("a", 1): (1,), ("b", 2): (2,), ("c", 3): (3,)}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain([("a", 1), ("b", 2), ("c", 3)])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_2D_values(self):
        mapping = {("a", 1): (1, 2), ("b", 2): (3, 4), ("c", 3): (5, 6)}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain([("a", 1), ("b", 2), ("c", 3)])
        expected_index = Index([0, 1])
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=expected_domain.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index == expected_index
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_1D_values(self):
        mapping = {("a",): 1, ("b",): 2, ("c",): 3}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_length_1_tuple_values(self):
        mapping = {("a",): (1,), ("b",): (2,), ("c",): (3,)}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_2D_values(self):
        mapping = {("a",): (1, 2), ("b",): (3, 4), ("c",): (5, 6)}
        v = MappingValidator(mapping=mapping)
        expected_domain = Domain(["a", "b", "c"])
        expected_index = Index([0, 1])
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=expected_domain.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index == expected_index
        assert v.kind == "any"

    def test_values_mix_of_tuples_and_non_tuples_raises(self):
        mapping = {"a": (1,), "b": 2, "c": 3}

        with pytest.raises(
            ValueError,
            match="If the mapping contains a tuple value, all values must be tuples.",
        ):
            MappingValidator(mapping=mapping, name="X")

    def test_inconsistent_tuple_lengths_raises(self):
        mapping = {"a": (1, 2), "b": (3, 4, 5), "c": (6, 7)}

        with pytest.raises(
            ValueError,
            match="All tuples in the mapping must have the same length.",
        ):
            MappingValidator(mapping=mapping, name="X")


class TestFromDictWithNoDomainAndIndex:
    def test_with_1D_keys_and_1D_values(self, I):
        mapping = {"a": 1, "b": 2, "c": 3}
        v = MappingValidator(mapping=mapping, index=I)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_length_1_tuple_values(self, I):
        mapping = {"a": (1,), "b": (2,), "c": (3,)}
        v = MappingValidator(mapping=mapping, index=I)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_1D_keys_and_2D_values(self, I):
        mapping = {"a": (1, 2), "b": (3, 4), "c": (5, 6)}
        v = MappingValidator(mapping=mapping, index=I)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=expected_domain.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, I.data)
        assert v.domain == expected_domain
        assert v.index is I
        assert v.output_name is None
        assert v.name is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_1D_values(self, I):
        mapping = {("a",): 1, ("b",): 2, ("c",): 3}
        v = MappingValidator(mapping=mapping, index=I)
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_length_1_tuple_values(self, I):
        mapping = {("a",): (1,), ("b",): (2,), ("c",): (3,)}
        expected_domain = Domain(["a", "b", "c"])
        v = MappingValidator(mapping=mapping, index=I)
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_length_1_tuple_keys_and_2D_values(self, I):
        mapping = {("a",): (1, 2), ("b",): (3, 4), ("c",): (5, 6)}
        expected_domain = Domain(["a", "b", "c"])
        v = MappingValidator(mapping=mapping, index=I)
        expected_mapping = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=expected_domain.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, I.data)
        assert v.domain == expected_domain
        assert v.index == I
        assert v.output_name is None
        assert v.name is None
        assert v.kind == "any"

    def test_with_2D_keys_and_1D_values(self, I):
        mapping = {("a", 1): 1, ("b", 2): 2, ("c", 3): 3}
        v = MappingValidator(mapping=mapping, index=I)
        expected_domain = Domain([("a", 1), ("b", 2), ("c", 3)])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_length_1_tuple_values(self, I):
        mapping = {("a", 1): (1,), ("b", 2): (2,), ("c", 3): (3,)}
        v = MappingValidator(mapping=mapping, index=I)
        expected_domain = Domain([("a", 1), ("b", 2), ("c", 3)])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index is None
        assert v.kind == "any"

    def test_with_2D_keys_and_2D_values(self, I):
        mapping = {("a", 1): (1, 2), ("b", 2): (2, 4), ("c", 3): (3, 6)}
        v = MappingValidator(mapping=mapping, index=I)
        expected_domain = Domain([("a", 1), ("b", 2), ("c", 3)])
        expected_mapping = pd.DataFrame(
            [[1, 2], [2, 4], [3, 6]], index=expected_domain.data, columns=I.data
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, I.data)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name is None
        assert v.index == I
        assert v.kind == "any"

    def test_values_mix_of_tuples_and_non_tuples_raises(self, I):
        mapping = {"a": (1,), "b": 2, "c": 3}

        with pytest.raises(
            ValueError,
            match="If the mapping contains a tuple value, all values must be tuples.",
        ):
            MappingValidator(mapping=mapping, index=I, name="X")

    def test_inconsistent_tuple_lengths_raises(self, I):
        mapping = {"a": (1, 2), "b": (3, 4, 5), "c": (6, 7)}

        with pytest.raises(
            ValueError,
            match="All tuples in the mapping must have the same length.",
        ):
            MappingValidator(mapping=mapping, index=I, name="X")

    def test_validate_columns_against_2D_index_raises(self):
        mapping = {("b", 2): (2, 4), ("a", 1): (1, 2), ("c", 3): (3, 6)}
        I = Index([(1, 2), (3, 4)])

        with pytest.raises(
            ValueError,
            match="The mapping columns cannot be validated against a pd.MultiIndex.",
        ):
            MappingValidator(mapping=mapping, index=I, name="X")

    def test_incompatible_index_length_raises(self):
        mapping = {("b", 2): (2, 4), ("a", 1): (1, 2), ("c", 3): (3, 6)}
        I = Index(["first", "second", "third"])

        with pytest.raises(
            ValueError,
            match="The length of the provided index does not match the dimension of the outputs of the mapping.",
        ):
            MappingValidator(mapping=mapping, index=I, name="X")


class TestFromSeriesWithSampleSpace:
    def test_series_with_index_with_no_name_into_series(self, Omega, I):
        mapping = pd.Series([2, 1, 3], index=["b", "a", "c"])
        v = MappingValidator(
            mapping=mapping, domain=Omega, index=I, name="X", output_name="num"
        )
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data, name="num")

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name == "num"
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_series_with_index_with_name_into_series(self, Omega, I):
        mapping = pd.Series(
            [2, 1, 3],
            index=pd.Index(["b", "a", "c"], name="omega"),
            name="num",
        )
        v = MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
        expected_mapping = pd.Series([1, 2, 3], index=Omega.data, name="num")

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega
        assert v.output_name == "num"
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_series_with_multi_index_with_no_names_into_series(self, Omega2D, I):
        mapping = pd.Series(
            [2, 1, 3],
            index=pd.MultiIndex.from_tuples([("b", 2), ("a", 1), ("c", 3)]),
        )
        v = MappingValidator(mapping=mapping, domain=Omega2D, index=I, name="X")
        expected_mapping = pd.Series([1, 2, 3], index=Omega2D.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega2D
        assert v.output_name is None
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_series_with_multi_index_with_names_into_series(self, Omega2D):
        mapping = pd.Series(
            [2, 1, 3],
            index=pd.MultiIndex.from_tuples(
                [("b", 2), ("a", 1), ("c", 3)], names=["letter", "num"]
            ),
        )
        v = MappingValidator(
            mapping=mapping, domain=Omega2D, name="X", output_name="num"
        )
        expected_mapping = pd.Series([1, 2, 3], index=Omega2D.data, name="num")

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain is Omega2D
        assert v.output_name == "num"
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_series_with_mismatched_index_name_raises(self, Omega):
        mapping = pd.Series(
            [2, 1, 3], index=pd.Index(["b", "a", "c"], name="wrong_name")
        )

        with pytest.raises(
            ValueError,
            match="If the mapping index is not a MultiIndex, its name if not None must match the variable name of the sample space.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, name="X")

    def test_series_with_mismatched_index_level_names_raise(self, Omega2D):
        mapping = pd.Series(
            [2, 1, 3],
            index=pd.MultiIndex.from_tuples(
                [("b", 2), ("a", 1), ("c", 3)], names=["wrong_letter", "wrong_num"]
            ),
        )

        with pytest.raises(
            ValueError,
            match="If the mapping index is a MultiIndex, its level names if not None must match the variable names of the sample space.",
        ):
            MappingValidator(mapping=mapping, domain=Omega2D, name="X")


class TestFromSeriesWithNoDomain:
    def test_series_with_index_with_no_name_into_series(self):
        mapping = pd.Series([1, 2, 3], index=["a", "b", "c"])
        v = MappingValidator(mapping=mapping, name="X")
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_series_with_index_with_name_into_series(self):
        mapping = pd.Series(
            [1, 2, 3], index=pd.Index(["a", "b", "c"], name="omega"), name="num"
        )
        v = MappingValidator(mapping=mapping, name="X")
        expected_domain = Domain(["a", "b", "c"], variable_names=["omega"])
        expected_mapping = pd.Series(
            [1, 2, 3],
            index=expected_domain.data,
            name="num",
        )

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name == "num"
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_series_with_multi_index_with_no_names_into_series(self):
        mapping = pd.Series(
            [1, 2, 3],
            index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)]),
        )
        v = MappingValidator(mapping=mapping, name="X")
        expected_domain = Domain([("a", 1), ("b", 2), ("c", 3)])
        expected_mapping = pd.Series([1, 2, 3], index=expected_domain.data)

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name is None
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"

    def test_series_with_multi_index_with_names_into_series(self):
        mapping = pd.Series(
            [1, 2, 3],
            index=pd.MultiIndex.from_tuples(
                [("a", 1), ("b", 2), ("c", 3)], names=["letter", "num"]
            ),
        )
        v = MappingValidator(mapping=mapping, name="X", output_name="num")
        expected_domain = Domain(
            [("a", 1), ("b", 2), ("c", 3)], variable_names=["letter", "num"]
        )
        expected_mapping = pd.Series(
            [1, 2, 3],
            index=expected_domain.data,
            name="num",
        )

        pd.testing.assert_series_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.output_name == "num"
        assert v.name == "X"
        assert v.index is None
        assert v.kind == "any"


class TestFromDataFrameWithSampleSpaceAndIndex:
    def test_dataframe_with_no_names_for_index_and_columns(self, Omega, I):
        mapping = pd.DataFrame(
            [[4, 3], [2, 1], [6, 5]],
            index=["b", "a", "c"],
            columns=["even", "odd"],
        )
        v = MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
        expected_data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.data, expected_data)
        assert v.domain is Omega
        assert v.index is I
        assert v.name == "X"
        assert v.kind == "any"

    def test_dataframe_with_no_names_for_multi_index_and_columns(self, Omega2D, I):
        mapping = pd.DataFrame(
            [[4, 3], [2, 1], [6, 5]],
            index=pd.MultiIndex.from_tuples([("b", 2), ("a", 1), ("c", 3)]),
            columns=["even", "odd"],
        )
        v = MappingValidator(mapping=mapping, domain=Omega2D, index=I, name="X")
        expected_data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega2D.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.data, expected_data)
        assert v.domain is Omega2D
        assert v.index is I
        assert v.name == "X"
        assert v.kind == "any"

    def test_dataframe_with_names_for_index_and_columns(self, Omega, I):
        mapping = pd.DataFrame(
            [[4, 3], [2, 1], [6, 5]],
            index=pd.Index(["b", "a", "c"], name="omega"),
            columns=pd.Index(["even", "odd"], name="parity"),
        )
        v = MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
        expected_data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.data, expected_data)
        assert v.domain is Omega
        assert v.index is I
        assert v.name == "X"
        assert v.kind == "any"

    def test_dataframe_with_names_for_multi_index_and_columns(self, Omega2D, I):
        mapping = pd.DataFrame(
            [[4, 3], [2, 1], [6, 5]],
            index=pd.MultiIndex.from_tuples(
                [("b", 2), ("a", 1), ("c", 3)],
                names=["letter", "num"],
            ),
            columns=pd.Index(["even", "odd"], name="parity"),
        )
        v = MappingValidator(mapping=mapping, domain=Omega2D, index=I, name="X")
        expected_data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega2D.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.data, expected_data)
        assert v.domain is Omega2D
        assert v.index is I
        assert v.name == "X"
        assert v.kind == "any"

    def test_default_columns_filled(self, Omega, I):
        mapping = pd.DataFrame(
            [[3, 4], [1, 2], [5, 6]],
            index=["b", "a", "c"],
        )
        v = MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")
        expected_data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.data, expected_data)
        assert v.domain is Omega
        assert v.index is I
        assert v.name == "X"
        assert v.kind == "any"

    def test_incompatible_columns_with_index_raises(self, Omega, I):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=pd.Index(["foo", "bar"], name="parity"),
        )
        with pytest.raises(
            ValueError,
            match="The columns of the mapping must match the provided Index.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")

    def test_dataframe_with_mismatched_index_name_raises(self, Omega, I):
        mapping = pd.DataFrame(
            [[3, 4], [1, 2], [5, 6]],
            index=pd.Index(["b", "a", "c"], name="wrong_name"),
            columns=I.data,
        )

        with pytest.raises(
            ValueError,
            match="If the mapping index is not a MultiIndex, its name if not None must match the variable name of the sample space.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, name="X")

    def test_dataframe_with_mismatched_index_level_names_raise(self, Omega2D, I):
        mapping = pd.DataFrame(
            [[3, 4], [1, 2], [5, 6]],
            index=pd.MultiIndex.from_tuples(
                [("b", 2), ("a", 1), ("c", 3)], names=["wrong_letter", "wrong_num"]
            ),
            columns=I.data,
        )

        with pytest.raises(
            ValueError,
            match="If the mapping index is a MultiIndex, its level names if not None must match the variable names of the sample space.",
        ):
            MappingValidator(mapping=mapping, domain=Omega2D, name="X")

    def test_dataframe_with_mismatched_column_name_raises(self, Omega, I):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=pd.Index(["odd", "even"], name="wrong name"),
        )
        with pytest.raises(
            ValueError,
            match="If the mapping columns have a name, it must match the name of the provided Index.",
        ):
            MappingValidator(mapping=mapping, domain=Omega, index=I, name="X")


class TestFromDataFrameWithSampleSpaceAndNoIndex:
    def test_dataframe_with_no_names_for_index_and_columns(self, Omega):
        mapping = pd.DataFrame(
            [[3, 4], [1, 2], [5, 6]],
            index=["b", "a", "c"],
            columns=["odd", "even"],
        )
        v = MappingValidator(mapping=mapping, domain=Omega, name="X")
        expected_index = Index(["odd", "even"], name="I")
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, expected_index.data)
        assert v.domain is Omega
        assert v.index == expected_index
        assert v.index.name == "I"
        assert v.index.variable_names == ["index"]
        assert v.name == "X"
        assert v.kind == "any"

    def test_dataframe_with_names_for_index_and_columns(self, Omega):
        mapping = pd.DataFrame(
            [[3, 4], [1, 2], [5, 6]],
            index=pd.Index(["b", "a", "c"], name="omega"),
            columns=pd.Index(["odd", "even"], name="parity"),
        )
        v = MappingValidator(mapping=mapping, domain=Omega, name="X")
        expected_index = Index(["odd", "even"], name="I", variable_names=["parity"])
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, expected_index.data)
        assert v.domain is Omega
        assert v.index == expected_index
        assert v.index.name == "I"
        assert v.index.variable_names == ["parity"]
        assert v.name == "X"
        assert v.kind == "any"

    def test_dataframe_with_default_column_names(self, Omega):
        mapping = pd.DataFrame(
            [[3, 4], [1, 2], [5, 6]], index=pd.Index(["b", "a", "c"], name="omega")
        )
        v = MappingValidator(mapping=mapping, domain=Omega, name="X")
        expected_index = Index([0, 1], name="I")
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=Omega.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.index.data, expected_index.data)
        assert v.domain is Omega
        assert v.index == expected_index
        assert v.index.name == "I"
        assert v.index.variable_names == ["index"]
        assert v.name == "X"
        assert v.kind == "any"


class TestFromDataFrameWithNoDomainAndIndex:
    def test_dataframe_with_no_names_for_index_and_columns(self, I):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["a", "b", "c"],
            columns=["odd", "even"],
        )
        v = MappingValidator(mapping=mapping, index=I, name="f")
        expected_domain = Domain(["a", "b", "c"])
        expected_data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=expected_domain.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.data, expected_data)
        pd.testing.assert_index_equal(v.domain.data, expected_domain.data)
        assert v.domain == expected_domain
        assert v.domain.name == "X"
        assert v.domain.variable_names == ["point"]
        assert v.index is I
        assert v.index.name == "I"
        assert v.index.variable_names == ["parity"]
        assert v.name == "f"
        assert v.kind == "any"

    def test_dataframe_with_names_for_index_and_columns(self, I):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["a", "b", "c"], name="omega"),
            columns=pd.Index(["odd", "even"], name="parity"),
        )
        v = MappingValidator(mapping=mapping, index=I, name="f")
        expected_domain = Domain(["a", "b", "c"], variable_names=["omega"])
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=expected_domain.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.domain.data, expected_domain.data)
        assert v.domain == expected_domain
        assert v.domain.name == "X"
        assert v.domain.variable_names == ["omega"]
        assert v.index is I
        assert v.index.name == "I"
        assert v.index.variable_names == ["parity"]
        assert v.name == "f"
        assert v.kind == "any"

    def test_default_columns_filled(self, I):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["a", "b", "c"],
        )
        v = MappingValidator(mapping=mapping, index=I, name="f")
        expected_domain = Domain(["a", "b", "c"])
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=expected_domain.data,
            columns=I.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        assert v.domain == expected_domain
        assert v.index is I
        assert v.name == "f"
        assert v.kind == "any"


class TestFromDataFrameWithNoDomainAndNoIndex:
    def test_dataframe_with_no_names_for_index_and_columns(self):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["a", "b", "c"],
            columns=["odd", "even"],
        )
        v = MappingValidator(mapping=mapping, name="f")
        expected_domain = Domain(["a", "b", "c"])
        expected_index = Index(["odd", "even"])
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=expected_domain.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.domain.data, expected_domain.data)
        assert v.domain == expected_domain
        assert v.domain.name == "X"
        assert v.domain.variable_names == ["point"]
        assert v.index == expected_index
        assert v.index.name == "I"
        assert v.index.variable_names == ["index"]
        assert v.name == "f"
        assert v.kind == "any"

    def test_dataframe_with_names_for_index_and_columns(self):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["a", "b", "c"], name="letter"),
            columns=pd.Index(["odd", "even"], name="parity"),
        )
        v = MappingValidator(mapping=mapping, name="f")
        expected_domain = Domain(["a", "b", "c"], variable_names=["letter"])
        expected_index = Index(["odd", "even"], variable_names=["parity"])
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=expected_domain.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.domain.data, expected_domain.data)
        assert v.domain == expected_domain
        assert v.domain.name == "X"
        assert v.domain.variable_names == ["letter"]
        assert v.index == expected_index
        assert v.index.name == "I"
        assert v.index.variable_names == ["parity"]
        assert v.name == "f"
        assert v.kind == "any"

    def test_dataframe_with_default_index_and_columns(self):
        mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
        )
        v = MappingValidator(mapping=mapping, name="f")
        expected_domain = Domain([0, 1, 2])
        expected_index = Index([0, 1])
        expected_mapping = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=expected_domain.data,
            columns=expected_index.data,
        )

        pd.testing.assert_frame_equal(v.mapping, expected_mapping)
        pd.testing.assert_index_equal(v.domain.data, expected_domain.data)
        assert v.domain == expected_domain
        assert v.domain.name == "X"
        assert v.domain.variable_names == ["point"]
        assert v.index == expected_index
        assert v.index.name == "I"
        assert v.index.variable_names == ["index"]
        assert v.name == "f"
        assert v.kind == "any"
