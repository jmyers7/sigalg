from numbers import Real

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sigalg.core import (
    FeatureVector,
    Index,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.8,
                1: 0.2,
            }
        )

    @pytest.fixture
    def index(self):
        return Index().from_sequence(size=2)

    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters."""
        X = RandomVector()
        prob_space = ProbabilitySpace()

        assert X.point_outputs is None
        assert X.atom_outputs is None
        assert X.data is None
        assert X.atom_data is None
        assert X.components is None
        assert X.index is None
        assert X.generated_sig_alg is None
        assert X.prob_space == prob_space
        assert X.domain is None
        assert X.sig_alg is None
        assert X.prob_measure is None
        assert X.range is None

    def test_constructor_with_custom_parameters(self, Omega, F, P, index):
        """Test the constructor with custom parameters."""
        Y = RandomVector(sample_space=Omega, sig_alg=F, prob_measure=P, index=index, name="Y")
        prob_space = ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)

        assert Y.point_outputs is None
        assert Y.atom_outputs is None
        assert Y.data is None
        assert Y.atom_data is None
        assert Y.components is None
        assert Y.index == index
        assert Y.generated_sig_alg is None
        assert Y.prob_space == prob_space
        assert Y.domain == Omega
        assert Y.sig_alg == F
        assert Y.prob_measure == P
        assert Y.range is None


class TestFromDict:
    @pytest.fixture
    def F(self):
        Omega = SampleSpace().from_sequence(size=3)
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
            }
        )

    @pytest.fixture
    def dict_2d_point(self):
        return {0: (1, 2), 1: (3, 4), 2: (5, 6)}

    @pytest.fixture
    def dict_1d_point(self):
        return {0: 10, 1: 20, 2: 30}

    @pytest.fixture
    def dict_2d_atom(self):
        return {0: (1, 2), 1: (3, 4)}

    @pytest.fixture
    def dict_1d_atom(self):
        return {0: 10, 1: 20}

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_no_provided_domain_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with no provided domain and index at construction."""
        rv = RandomVector(name="Z").from_dict(
            outputs=dict_2d_point,
            type="point",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_domain = SampleSpace().from_list([0, 1, 2])
        expected_index = Index(name="index", data_name="Z").from_list(["Z_0", "Z_1"])
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["Z_0", "Z_1"], name="Z"),
        )

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, expected_data)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_provided_aligned_domain_no_provided_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with a provided aligned domain, but no provided index."""
        Omega = SampleSpace().from_sequence(size=3)
        rv = RandomVector(sample_space=Omega, name="Z").from_dict(
            outputs=dict_2d_point,
            type="point",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = Index(name="index", data_name="Z").from_list(["Z_0", "Z_1"])
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["Z_0", "Z_1"], name="Z"),
        )

        assert rv.domain == Omega
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, expected_data)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_provided_misaligned_domain_no_provided_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with a provided misaligned domain, but no provided index."""
        Omega = SampleSpace().from_list([0, 1])

        if not overwrite_domain:
            with pytest.raises(
                ValueError,
                match="mapping must contain an entry for every sample index in sample_space",
            ):
                rv = RandomVector(sample_space=Omega, name="Z").from_dict(
                    outputs=dict_2d_point,
                    type="point",
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, name="Z").from_dict(
                outputs=dict_2d_point,
                type="point",
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list([0, 1, 2])
            expected_index = Index(name="index", data_name="Z").from_list(
                ["Z_0", "Z_1"]
            )
            expected_data = pd.DataFrame(
                [(1, 2), (3, 4), (5, 6)],
                index=pd.Index([0, 1, 2], name="Omega"),
                columns=pd.Index(["Z_0", "Z_1"], name="Z"),
            )

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Z"
            pd.testing.assert_frame_equal(rv.data, expected_data)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_no_provided_domain_provided_correct_length_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with no provided domain, but a provided correct-length index."""
        index = Index(name="index", data_name="feature").from_list(["A", "B"])
        rv = RandomVector(index=index, name="Z").from_dict(
            outputs=dict_2d_point,
            type="point",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_domain = SampleSpace().from_list([0, 1, 2])
        expected_index = (
            Index(name="index", data_name="Z").from_list(["Z_0", "Z_1"])
            if overwrite_index
            else index
        )

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Z"
        assert rv.data.shape == (3, 2)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_no_provided_domain_provided_wrong_length_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with no provided domain, but a provided wrong-length index."""
        index = Index(name="index", data_name="feature").from_list(["A", "B", "C"])
        if not overwrite_index:
            with pytest.raises(
                ValueError,
                match="Length of index must match the dimension of the RandomVector.",
            ):
                rv = RandomVector(index=index, name="Z").from_dict(
                    outputs=dict_2d_point,
                    type="point",
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(index=index, name="Z").from_dict(
                outputs=dict_2d_point,
                type="point",
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list([0, 1, 2])
            expected_index = Index(name="index", data_name="Z").from_list(
                ["Z_0", "Z_1"]
            )

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Z"

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_provided_aligned_domain_provided_correct_length_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with both a provided aligned domain and correct-length index."""
        Omega = SampleSpace().from_sequence(size=3)
        index = Index(name="index", data_name="feature").from_list(["A", "B"])
        rv = RandomVector(sample_space=Omega, index=index, name="Z").from_dict(
            outputs=dict_2d_point,
            type="point",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = (
            Index(name="index", data_name="feature").from_list(["Z_0", "Z_1"])
            if overwrite_index
            else index
        )

        assert rv.domain == Omega
        assert rv.index == expected_index
        assert rv.name == "Z"
        assert rv.data.shape == (3, 2)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_provided_aligned_domain_provided_wrong_length_index(
        self,
        overwrite_domain,
        overwrite_index,
        dict_2d_point,
    ):
        """Test from_dict with provided aligned domain, but a provided wrong-length index."""
        Omega = SampleSpace().from_sequence(size=3)
        index = Index(name="index", data_name="feature").from_list(["A", "B", "C"])
        if not overwrite_index:
            with pytest.raises(
                ValueError,
                match="Length of index must match the dimension of the RandomVector.",
            ):
                rv = RandomVector(sample_space=Omega, index=index, name="Z").from_dict(
                    outputs=dict_2d_point,
                    type="point",
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, index=index, name="Z").from_dict(
                outputs=dict_2d_point,
                type="point",
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_index = Index(name="index", data_name="Z").from_list(
                ["Z_0", "Z_1"]
            )

            assert rv.domain == Omega
            assert rv.index == expected_index
            assert rv.name == "Z"

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_provided_misaligned_domain_provided_correct_length_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with a provided misaligned domain, and provided correct-length index."""
        Omega = SampleSpace().from_sequence(size=2)
        index = Index(name="index", data_name="feature").from_list(["A", "B"])

        if not overwrite_domain:
            with pytest.raises(
                ValueError,
                match="mapping must contain an entry for every sample index in sample_space",
            ):
                rv = RandomVector(sample_space=Omega, index=index, name="Z").from_dict(
                    outputs=dict_2d_point,
                    type="point",
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, index=index, name="Z").from_dict(
                outputs=dict_2d_point,
                type="point",
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list([0, 1, 2])
            expected_index = (
                Index(name="index", data_name="Z").from_list(["Z_0", "Z_1"])
                if overwrite_index
                else index
            )

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Z"

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_point_with_provided_misaligned_domain_provided_wrong_length_index(
        self, overwrite_domain, overwrite_index, dict_2d_point
    ):
        """Test from_dict with both a provided misaligned domain and wrong-length index."""
        Omega = SampleSpace().from_sequence(size=2)
        index = Index(name="index", data_name="feature").from_list(["A", "B", "C"])

        if (overwrite_domain, overwrite_index) == (True, True):
            rv = RandomVector(sample_space=Omega, index=index, name="Z").from_dict(
                outputs=dict_2d_point,
                type="point",
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list([0, 1, 2])
            expected_index = Index(name="index", data_name="Z").from_list(
                ["Z_0", "Z_1"]
            )

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Z"
        else:
            with pytest.raises(ValueError):
                rv = RandomVector(sample_space=Omega, index=index, name="Z").from_dict(
                    outputs=dict_2d_point,
                    type="point",
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_1d_point_with_no_provided_domain(
        self, overwrite_domain, overwrite_index, dict_1d_point
    ):
        """Test from_dict with no provided domain at construction for 1D output."""
        rv = RandomVector(name="Y").from_dict(
            outputs=dict_1d_point,
            type="point",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_domain = SampleSpace().from_list([0, 1, 2])
        expected_index = None
        expected_data = pd.Series(
            [10, 20, 30],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="Y",
        )

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, expected_data)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_1d_point_with_provided_aligned_domain(
        self, overwrite_domain, overwrite_index, dict_1d_point
    ):
        """Test from_dict with provided aligned domain at construction for 1D output."""
        Omega = SampleSpace().from_sequence(size=3)
        rv = RandomVector(sample_space=Omega, name="Y").from_dict(
            outputs=dict_1d_point,
            type="point",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = None
        expected_data = pd.Series(
            [10, 20, 30],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="Y",
        )

        assert rv.domain == Omega
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, expected_data)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_1d_point_with_provided_misaligned_domain(
        self, overwrite_domain, overwrite_index, dict_1d_point
    ):
        """Test from_dict with a provided misaligned domain for 1D output."""
        Omega = SampleSpace().from_list([0, 1])

        if not overwrite_domain:
            with pytest.raises(
                ValueError,
                match="mapping must contain an entry for every sample index in sample_space",
            ):
                rv = RandomVector(sample_space=Omega, name="Y").from_dict(
                    outputs=dict_1d_point,
                    type="point",
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, name="Y").from_dict(
                outputs=dict_1d_point,
                type="point",
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list([0, 1, 2])
            expected_index = None
            expected_data = pd.Series(
                [10, 20, 30],
                index=pd.Index([0, 1, 2], name="Omega"),
                name="Y",
            )

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Y"
            pd.testing.assert_series_equal(rv.data, expected_data)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_atom_with_no_provided_index(
        self, overwrite_domain, overwrite_index, F, dict_2d_atom
    ):
        """Test from_dict with type='atom' and no provided index."""
        Omega = SampleSpace().from_sequence(size=3)
        rv = RandomVector(sample_space=Omega, sig_alg=F, name="Z").from_dict(
            outputs=dict_2d_atom,
            type="atom",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = Index(name="index", data_name="Z").from_list(["Z_0", "Z_1"])
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (3, 4)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["Z_0", "Z_1"], name="Z"),
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="F"),
            columns=pd.Index(["Z_0", "Z_1"], name="Z"),
        )

        assert rv.domain == Omega
        assert rv.sig_alg == F
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, expected_data)
        pd.testing.assert_frame_equal(rv.atom_data, expected_atom_data)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_atom_with_provided_correct_length_index(
        self, overwrite_domain, overwrite_index, F, dict_2d_atom
    ):
        """Test from_dict with type='atom' and provided correct-length index."""
        Omega = SampleSpace().from_sequence(size=3)
        index = Index(name="index", data_name="feature").from_list(["A", "B"])
        rv = RandomVector(sample_space=Omega, sig_alg=F, index=index, name="Z").from_dict(
            outputs=dict_2d_atom,
            type="atom",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = (
            Index(name="index", data_name="feature").from_list(["Z_0", "Z_1"])
            if overwrite_index
            else index
        )

        assert rv.domain == Omega
        assert rv.sig_alg == F
        assert rv.index == expected_index
        assert rv.name == "Z"
        assert rv.data.shape == (3, 2)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_atom_with_provided_wrong_length_index(
        self, overwrite_domain, overwrite_index, F, dict_2d_atom
    ):
        """Test from_dict with type='atom' and provided wrong-length index."""
        Omega = SampleSpace().from_sequence(size=3)
        index = Index(name="index", data_name="feature").from_list(["A", "B", "C"])
        if not overwrite_index:
            with pytest.raises(
                ValueError,
                match="Length of index must match the dimension of the RandomVector.",
            ):
                rv = RandomVector(
                    sample_space=Omega, sig_alg=F, index=index, name="Z"
                ).from_dict(
                    outputs=dict_2d_atom,
                    type="atom",
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, sig_alg=F, index=index, name="Z").from_dict(
                outputs=dict_2d_atom,
                type="atom",
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_index = Index(name="index", data_name="feature").from_list(
                ["Z_0", "Z_1"]
            )

            assert rv.domain == Omega
            assert rv.sig_alg == F
            assert rv.index == expected_index
            assert rv.name == "Z"

    def test_2d_atom_raises_without_sigma_algebra(self, dict_2d_atom):
        """Test from_dict with type='atom' raises error when sigma algebra not provided."""
        with pytest.raises(
            ValueError,
            match="The sig_alg parameter must be set during construction for the from_dict method with type='atom'.",
        ):
            RandomVector(name="Z").from_dict(
                outputs=dict_2d_atom,
                type="atom",
            )

    def test_2d_atom_with_misaligned_atom_ids(self, F, dict_2d_point):
        """Test from_dict with type='atom' raises error when atom IDs don't match."""
        Omega = SampleSpace().from_sequence(size=3)
        with pytest.raises(
            ValueError,
            match="mapping must contain an entry for every sample index in sample_space",
        ):
            RandomVector(sample_space=Omega, sig_alg=F, name="Z").from_dict(
                outputs=dict_2d_point,
                type="atom",
            )

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_1d_atom_with_no_issues(
        self, overwrite_domain, overwrite_index, F, dict_1d_atom
    ):
        """Test from_dict with type='atom' and 1D outputs."""
        Omega = SampleSpace().from_sequence(size=3)
        rv = RandomVector(sample_space=Omega, sig_alg=F, name="Y").from_dict(
            outputs=dict_1d_atom,
            type="atom",
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = None
        expected_data = pd.Series(
            [10, 20, 20],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="Y",
        )
        expected_atom_data = pd.Series(
            [10, 20],
            index=pd.Index([0, 1], name="F"),
            name="Y",
        )

        assert rv.domain == Omega
        assert rv.sig_alg == F
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, expected_data)
        pd.testing.assert_series_equal(rv.atom_data, expected_atom_data)

    def test_1d_atom_raises_without_sigma_algebra(self, dict_1d_atom):
        """Test from_dict with type='atom' and 1D outputs raises error when sigma algebra not provided."""
        with pytest.raises(
            ValueError,
            match="The sig_alg parameter must be set during construction for the from_dict method with type='atom'.",
        ):
            RandomVector(name="Y").from_dict(
                outputs=dict_1d_atom,
                type="atom",
            )

    def test_1d_atom_with_misaligned_atom_ids(self, F, dict_1d_point):
        """Test from_dict with type='atom' and 1D outputs raises error when atom IDs don't match."""
        Omega = SampleSpace().from_sequence(size=3)
        with pytest.raises(
            ValueError,
            match="mapping must contain an entry for every sample index in sample_space",
        ):
            RandomVector(sample_space=Omega, sig_alg=F, name="Y").from_dict(
                outputs=dict_1d_point,
                type="atom",
            )

    def test_invalid_type_parameter_raises(self, dict_2d_point):
        """Test from_dict raises error for invalid type parameter."""
        with pytest.raises(
            ValueError,
            match="type must be either 'point' or 'atom'.",
        ):
            RandomVector(name="Z").from_dict(
                outputs=dict_2d_point,
                type="invalid",
            )

    def test_measurability_check_point_type(self, F):
        """Test from_dict raises error for non-measurable outputs with type='point'."""
        Omega = SampleSpace().from_sequence(size=3)
        non_measurable_outputs = {
            0: (1, 2),
            1: (3, 4),
            2: (5, 6),
        }
        with pytest.raises(
            ValueError,
            match="Random vector Z is not measurable",
        ):
            RandomVector(sample_space=Omega, sig_alg=F, name="Z").from_dict(
                outputs=non_measurable_outputs,
                type="point",
            )

    def test_empty_dict_raises(self):
        """Test from_dict raises error for empty dictionary."""
        with pytest.raises(StopIteration):
            RandomVector(name="Z").from_dict(
                outputs={},
                type="point",
            )


class TestFromPandas:
    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            data=[(1, 2), (3, 4), (5, 6)],
            index=pd.Index(["a", "b", "c"], name="letters"),
            columns=pd.Index(["black", "blue"], name="colors"),
        )

    @pytest.fixture
    def series(self):
        return pd.Series(
            data=[1, 2, 3],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="Y",
        )

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_no_provided_domain_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with no provided domain and index at construction."""
        rv = RandomVector(name="Z").from_pandas(
            data=df,
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_domain = SampleSpace().from_list(
            ["a", "b", "c"], variable_names=["letters"]
        )
        expected_index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue"])

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_provided_aligned_domain_no_provided_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with a provided aligned domain, but no provided index."""
        Omega = SampleSpace().from_list(["a", "b", "c"], variable_names=["letters"])
        rv = RandomVector(sample_space=Omega, name="Z").from_pandas(
            data=df,
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue"])

        assert rv.domain == Omega
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_provided_misaligned_domain_no_provided_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with a provided misaligned domain, but no provided index."""
        Omega = SampleSpace().from_list(["a", "b"], variable_names=["letters"])

        if not overwrite_domain:
            with pytest.raises(
                ValidationError,
                match="mapping must contain an entry for every sample index",
            ):
                rv = RandomVector(sample_space=Omega, name="Z").from_pandas(
                    data=df,
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, name="Z").from_pandas(
                data=df,
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list(
                ["a", "b", "c"], variable_names=["letters"]
            )
            expected_index = Index(
                name="index",
                data_name="colors",
            ).from_list(["black", "blue"])

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Z"
            pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_no_provided_domain_provided_aligned_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with a no provided domain, but a provided aligned index."""
        index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue"])
        rv = RandomVector(index=index, name="Z").from_pandas(
            data=df,
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_domain = SampleSpace().from_list(
            ["a", "b", "c"], variable_names=["letters"]
        )

        assert rv.domain == expected_domain
        assert rv.index == index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_no_provided_domain_provided_misaligned_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with no provided domain, but a provided misaligned index."""
        index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue", "red"])

        if not overwrite_index:
            with pytest.raises(
                ValueError,
                match="The existing index must match the column index of the data.",
            ):
                rv = RandomVector(index=index, name="Z").from_pandas(
                    data=df,
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(index=index, name="Z").from_pandas(
                data=df,
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list(
                ["a", "b", "c"], variable_names=["letters"]
            )
            expected_index = Index(
                name="index",
                data_name="colors",
            ).from_list(["black", "blue"])

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Z"
            pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_provided_aligned_domain_provided_aligned_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with both a provided aligned domain and index."""
        Omega = SampleSpace().from_list(["a", "b", "c"], variable_names=["letters"])
        index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue"])
        rv = RandomVector(sample_space=Omega, index=index, name="Z").from_pandas(
            data=df,
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )

        assert rv.domain == Omega
        assert rv.index == index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_provided_misaligned_domain_provided_aligned_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with a provided misaligned domain, and provided aligned index."""
        Omega = SampleSpace().from_list(["a", "b"], variable_names=["letters"])
        index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue"])

        if not overwrite_domain:
            with pytest.raises(
                ValidationError,
                match="mapping must contain an entry for every sample index",
            ):
                rv = RandomVector(sample_space=Omega, index=index, name="Z").from_pandas(
                    data=df,
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, index=index, name="Z").from_pandas(
                data=df,
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list(
                ["a", "b", "c"], variable_names=["letters"]
            )

            assert rv.domain == expected_domain
            assert rv.index == index
            assert rv.name == "Z"
            pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_provided_aligned_domain_provided_misaligned_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with provided aligned domain, but a provided misaligned index."""
        Omega = SampleSpace().from_list(["a", "b", "c"], variable_names=["letters"])
        index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue", "red"])

        if not overwrite_index:
            with pytest.raises(
                ValueError,
                match="The existing index must match the column index of the data.",
            ):
                rv = RandomVector(sample_space=Omega, index=index, name="Z").from_pandas(
                    data=df,
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, index=index, name="Z").from_pandas(
                data=df,
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_index = Index(
                name="index",
                data_name="colors",
            ).from_list(["black", "blue"])

            assert rv.domain == Omega
            assert rv.index == expected_index
            assert rv.name == "Z"
            pd.testing.assert_frame_equal(rv.data, df)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_2d_df_with_provided_misaligned_domain_provided_misaligned_index(
        self, overwrite_domain, overwrite_index, df
    ):
        """Test from_pandas with both provided misaligned domain, and provided misaligned index."""
        Omega = SampleSpace().from_list(["a", "b"], variable_names=["letters"])
        index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue", "red"])

        if (overwrite_domain, overwrite_index) == (True, True):
            rv = RandomVector(sample_space=Omega, index=index, name="Z").from_pandas(
                data=df,
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list(
                ["a", "b", "c"], variable_names=["letters"]
            )
            expected_index = Index(
                name="index",
                data_name="colors",
            ).from_list(["black", "blue"])

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Z"
            pd.testing.assert_frame_equal(rv.data, df)
        else:
            with pytest.raises(ValueError):
                rv = RandomVector(sample_space=Omega, index=index, name="Z").from_pandas(
                    data=df,
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_1d_series_with_no_provided_domain(
        self, overwrite_domain, overwrite_index, series
    ):
        """Test from_pandas with no provided domain at construction."""
        rv = RandomVector(name="Y").from_pandas(
            data=series,
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_domain = SampleSpace().from_list(
            ["a", "b", "c"], variable_names=["letters"]
        )
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, series)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_1d_series_with_provided_aligned_domain(
        self, overwrite_domain, overwrite_index, series
    ):
        """Test from_pandas with provided aligned domain at construction."""
        Omega = SampleSpace().from_list(["a", "b", "c"], variable_names=["letters"])
        rv = RandomVector(sample_space=Omega, name="Y").from_pandas(
            data=series,
            overwrite_domain=overwrite_domain,
            overwrite_index=overwrite_index,
        )
        expected_index = None

        assert rv.domain == Omega
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, series)

    @pytest.mark.parametrize(
        "overwrite_domain, overwrite_index",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_1d_series_with_provided_misaligned_domain(
        self, overwrite_domain, overwrite_index, series
    ):
        """Test from_pandas with a provided misaligned domain"""
        Omega = SampleSpace().from_list(["a", "b"], variable_names=["letters"])

        if not overwrite_domain:
            with pytest.raises(
                ValidationError,
                match="must contain an entry for every sample index in sample_space",
            ):
                rv = RandomVector(sample_space=Omega, name="Y").from_pandas(
                    data=series,
                    overwrite_domain=overwrite_domain,
                    overwrite_index=overwrite_index,
                )
        else:
            rv = RandomVector(sample_space=Omega, name="Y").from_pandas(
                data=series,
                overwrite_domain=overwrite_domain,
                overwrite_index=overwrite_index,
            )
            expected_domain = SampleSpace().from_list(
                ["a", "b", "c"], variable_names=["letters"]
            )
            expected_index = None

            assert rv.domain == expected_domain
            assert rv.index == expected_index
            assert rv.name == "Y"
            pd.testing.assert_series_equal(rv.data, series)


class TestFromNumPy:
    def test_from_numpy(self):
        """Test RandomVector.from_numpy method."""
        arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
        arr_flat = np.array([10, 20, 30])
        arr_col = np.array([[10], [20], [30]])
        rv_2d = RandomVector(name="X").from_numpy(array=arr_2d)
        rv_flat = RandomVector(name="Y").from_numpy(array=arr_flat)
        rv_col = RandomVector(name="Z").from_numpy(array=arr_col)

        expected_domain = rv_2d.domain

        expected_index_2d = Index(
            name="index",
            data_name="feature",
        ).from_list(list(range(2)))
        expected_index_flat = None
        expected_index_col = None

        assert rv_2d.domain == expected_domain
        assert rv_flat.domain == expected_domain
        assert rv_col.domain == expected_domain

        assert rv_2d.index == expected_index_2d
        assert rv_flat.index == expected_index_flat
        assert rv_col.index == expected_index_col

        assert rv_2d.name == "X"
        assert rv_flat.name == "Y"
        assert rv_col.name == "Z"

        assert rv_2d.data.shape == (3, 2)
        assert rv_flat.data.shape == (3,)
        assert rv_col.data.shape == (3,)

    def test_from_numpy_sets_default_probability_measure(self):
        """Test that from_numpy sets a default uniform probability measure."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        rv = RandomVector(name="W").from_numpy(array=arr)

        expected_domain = SampleSpace().from_sequence(size=3)
        expected_prob_measure = ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(expected_domain)
        )

        assert rv.prob_measure == expected_prob_measure

    def test_from_numpy_sets_default_sigma_algebra(self):
        """Test that from_numpy sets a default power set sigma algebra."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        rv = RandomVector(name="V").from_numpy(array=arr)

        expected_domain = SampleSpace().from_sequence(size=3)
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space=expected_domain)

        assert rv.sig_alg == expected_sigma_algebra


class TestFromConstant:
    def test_from_constant_2d(self):
        """Test the from_constant method with a 2-dimensional output."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega).from_constant(constant=(1, 2))
        expected_index = Index(name="X").from_sequence(size=2, prefix="X")
        expected_data = pd.DataFrame(
            [(1, 2)] * 3, index=Omega.data, columns=expected_index.data
        )

        pd.testing.assert_frame_equal(X.data, expected_data)

    def test_from_constant_1d(self):
        """Test the from_constant method with a 1-dimensional output."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega).from_constant(constant=2)
        expected_data = pd.Series(
            [
                2,
            ]
            * 3,
            index=Omega.data,
            name="X",
        )

        pd.testing.assert_series_equal(X.data, expected_data)


# --------------------- test properties --------------------- #


class TestPointOutputs:
    pass


class TestAtomOutputs:
    pass


class TestData:
    pass


class TestAtomData:
    pass


class TestComponents:
    pass


class TestIndex:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def random_vector_2d(self, Omega):
        outputs = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        return RandomVector(sample_space=Omega, name="X").from_dict(outputs)

    @pytest.fixture
    def random_vector_1d(self, Omega):
        outputs = {0: 10, 1: 20, 2: 30}
        return RandomVector(sample_space=Omega, name="Y").from_dict(outputs)

    def test_index_property_of_2d_random_vector(self, random_vector_2d):
        """Test index property of RandomVector."""
        expected_index = Index(name="X").from_list(["X_0", "X_1"], variable_names=["X"])

        assert random_vector_2d.index == expected_index
        assert random_vector_2d.index.name == "X"

    def test_index_property_of_1d_random_vector(self, random_vector_1d):
        """Test index property of 1D RandomVector."""
        expected_index = None
        assert random_vector_1d.index == expected_index


class TestGeneratedSigAlg:
    def test_generated_sigma_algebra_property(self):
        """Test generated_sigma_algebra property of RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        outputs_2d = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        outputs_1d = {0: 10, 1: 20, 2: 30}
        X = RandomVector(sample_space=Omega, name="X").from_dict(outputs_2d)
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(outputs_1d)
        expected_sigma_algebra_2d = SigmaAlgebra(
            sample_space=Omega,
            name="sigma(X)",
        ).from_dict(sample_id_to_atom_id=outputs_2d)
        expected_sigma_algebra_1d = SigmaAlgebra(
            sample_space=Omega,
            name="sigma(Y)",
        ).from_dict(sample_id_to_atom_id=outputs_1d)

        assert X.generated_sig_alg == expected_sigma_algebra_2d
        assert Y.generated_sig_alg == expected_sigma_algebra_1d


class TestProbSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.8,
                1: 0.2,
            }
        )

    @pytest.fixture
    def point_outputs(self):
        return {
            0: (1, 2),
            1: (1, 2),
            2: (3, 4),
        }

    def test_prob_space_with_defaults(self, Omega, point_outputs):
        """Test that default probability space has power-set sigma-algebra and uniform probability measure."""
        X = RandomVector(sample_space=Omega).from_dict(point_outputs)
        prob_space = ProbabilitySpace(sample_space=Omega)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == SigmaAlgebra.power_set(sample_space=Omega)
        assert X.prob_space.prob_measure == ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(sample_space=Omega)
        )

    def test_prob_space_with_custom_prob_measure(self, Omega, P, point_outputs):
        """Test constructor with custom probability measure sets sigma-algebra to the sigma-algebra of the probability measure."""
        X = RandomVector(sample_space=Omega, prob_measure=P).from_dict(point_outputs)
        prob_space = ProbabilitySpace(Omega, prob_measure=P)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == P.sig_alg
        assert X.prob_space.prob_measure == P

    def test_prob_space_with_custom_sigma_algebra(self, Omega, F, point_outputs):
        """Test constructor with custom sigma-algebra sets the probability measure to uniform over the sigma-algebra."""
        X = RandomVector(sample_space=Omega, sig_alg=F).from_dict(point_outputs)
        prob_space = ProbabilitySpace(Omega, F)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.prob_measure == ProbabilityMeasure.uniform(sig_alg=F)

    def test_prob_space_with_all_components(self, Omega, F, P, point_outputs):
        """Test constructor with all components."""
        prob_space = ProbabilitySpace(Omega, F, P)
        X = RandomVector(*prob_space).from_dict(point_outputs)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.prob_measure == P


class TestDomain:
    pass


class TestSigAlg:
    pass


class TestProbMeasure:
    pass


class TestRange:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 2,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.2,
                1: 0.1,
                2: 0.7,
            }
        )

    @pytest.fixture
    def point_outputs_2d(self):
        return {
            0: (1, 2),
            1: (1, 2),
            2: (3, 4),
            3: (3, 4),
        }

    @pytest.fixture
    def point_outputs_1d(self):
        return {
            0: 4,
            1: 4,
            2: 5,
            3: 6,
        }

    def test_range_2d_random_vector_with_str_name(self, Omega, F, P, point_outputs_2d):
        """Test range property of 2D RandomVector with string name."""
        X = RandomVector(Omega, F, P).from_dict(point_outputs_2d)
        expected_sig_alg = SigmaAlgebra().from_dict(
            {
                (1, 2): (1, 2),
                (3, 4): (3, 4),
            }
        )
        expected_pushforward = ProbabilityMeasure(
            sig_alg=expected_sig_alg,
            name="P_X",
        ).from_dict(
            {
                (1, 2): 0.2,
                (3, 4): 0.8,
            }
        )
        expected_range = ProbabilitySpace(
            sig_alg=expected_sig_alg, prob_measure=expected_pushforward
        )

        assert X.range == expected_range

    def test_range_1d_random_vector_with_str_name(self, Omega, F, P, point_outputs_1d):
        """Test range property of 1D RandomVector with string name."""
        X = RandomVector(Omega, F, P, name="X").from_dict(point_outputs_1d)
        expected_sig_alg = SigmaAlgebra().from_dict(
            {
                4: 4,
                5: 5,
                6: 6,
            }
        )
        expected_pushforward = ProbabilityMeasure(
            sig_alg=expected_sig_alg,
            name="P_X",
        ).from_dict(
            {
                4: 0.2,
                5: 0.1,
                6: 0.7,
            }
        )
        expected_range = ProbabilitySpace(
            sig_alg=expected_sig_alg, prob_measure=expected_pushforward
        )

        assert X.range == expected_range

    def test_range_2d_random_vector_with_int_name(self, Omega, F, P, point_outputs_2d):
        """Test range property of 2D RandomVector with int name."""
        X = RandomVector(Omega, F, P, name=42).from_dict(point_outputs_2d)
        expected_sig_alg = SigmaAlgebra().from_dict(
            {
                (1, 2): (1, 2),
                (3, 4): (3, 4),
            }
        )
        expected_pushforward = ProbabilityMeasure(
            sig_alg=expected_sig_alg, name="P_42"
        ).from_dict(
            {
                (1, 2): 0.2,
                (3, 4): 0.8,
            }
        )
        expected_range = ProbabilitySpace(
            sig_alg=expected_sig_alg, prob_measure=expected_pushforward
        )

        assert X.range == expected_range

    def test_range_1d_random_vector_with_int_name(self, Omega, F, P, point_outputs_1d):
        """Test range property of 1D RandomVector with int name."""
        X = RandomVector(Omega, F, P, name=42).from_dict(point_outputs_1d)
        expected_sig_alg = SigmaAlgebra().from_dict(
            {
                4: 4,
                5: 5,
                6: 6,
            }
        )
        expected_pushforward = ProbabilityMeasure(
            sig_alg=expected_sig_alg, name="P_42"
        ).from_dict(
            {
                4: 0.2,
                5: 0.1,
                6: 0.7,
            }
        )
        expected_range = ProbabilitySpace(
            sig_alg=expected_sig_alg, prob_measure=expected_pushforward
        )

        assert X.range == expected_range


# --------------------- test probability methods --------------------- #


class TestIsMeasurable:
    pass


# --------------------- test data access methods --------------------- #


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.3,
                1: 0.2,
                2: 0.5,
            }
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(*prob_space).from_dict(
            {
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            }
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(*prob_space, name="Y").from_dict(
            {
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            }
        )

    def test_call_method_on_sample_points(self, Omega, X, Y):
        """Test calling on sample points."""
        for sample_point in Omega:
            assert isinstance(X(sample_point), FeatureVector)
            assert isinstance(Y(sample_point), Real)
            pd.testing.assert_series_equal(
                X(sample_point).data, X.data.loc[sample_point]
            )
            assert Y(sample_point) == Y.data.loc[sample_point]

    def test_call_method_on_atoms(self, F, X, Y):
        """Test calling on atoms."""
        for atom_id, atom in F.atom_id_to_event.items():
            assert isinstance(X(atom), FeatureVector)
            assert isinstance(Y(atom), Real)
            pd.testing.assert_series_equal(X(atom).data, X.atom_data.loc[atom_id])
            assert Y(atom) == Y.atom_data.loc[atom_id]

    def test_call_with_no_data_raises(self, prob_space):
        """Test calling a RandomVector with no data."""
        X = RandomVector(*prob_space, name="X")

        with pytest.raises(
            ValueError, match="Cannot evaluate a random vector without outputs"
        ):
            X(0)

    def test_call_on_non_measurable_event_raises(self, Omega, X):
        """Test calling on a non-measurable event."""
        power_set = SigmaAlgebra.power_set(Omega)
        A = power_set.get_event([1, 2])

        with pytest.raises(
            ValueError,
            match="The provided event is not in the sigma-algebra of the random vector",
        ):
            X(A)

    def test_call_on_measurable_non_atom_raises(self, F, X):
        """Test calling on a measurable non-atom event."""
        A = F.get_event([0, 1, 2, 3])

        with pytest.raises(
            ValueError,
            match="The provided event is not an atom in the sigma-algebra of the random vector",
        ):
            X(A)

    def test_call_on_non_sample_point_raises(self, X):
        """Test calling on non-sample point."""

        with pytest.raises(
            ValueError,
            match="The provided sample point is not in the domain of the random vector",
        ):
            X("not_a_sample_point")


class TestGetComponentRV:
    pass


class TestGetSubVector:
    pass


class TestIterFeatures:
    def test_iter_features_of_2d_random_vector(self):
        """Test iter_features method of 2D RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        outputs = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        X = RandomVector(sample_space=Omega, name="X").from_dict(outputs)

        expected_features = {
            0: FeatureVector().from_pandas(
                data=pd.Series(
                    [1, 2],
                    index=pd.Index(["X_0", "X_1"], name="X"),
                    name=0,
                )
            ),
            1: FeatureVector().from_pandas(
                data=pd.Series(
                    [3, 4],
                    index=pd.Index(["X_0", "X_1"], name="X"),
                    name=1,
                )
            ),
            2: FeatureVector().from_pandas(
                data=pd.Series(
                    [5, 6],
                    index=pd.Index(["X_0", "X_1"], name="X"),
                    name=2,
                )
            ),
        }

        for sample_idx, feature_vector in X.iter_features():
            pd.testing.assert_series_equal(
                feature_vector.data, expected_features[sample_idx].data
            )

    def test_iter_features_of_1d_random_vector(self):
        """Test iter_features method of 1D RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        outputs = {0: 10, 1: 20, 2: 30}
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(outputs)

        expected_features = {
            0: 10,
            1: 20,
            2: 30,
        }

        for sample_idx, feature in Y.iter_features():
            assert feature == expected_features[sample_idx]


class TestApplyToFeatures:
    pass


# --------------------- equality --------------------- #


class TestEquality:
    pass


# --------------------- arithmetic --------------------- #


class TestArithmetic:
    def test_add_two_random_vectors(self):
        """Test adding two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Y = RandomVector(
            sample_space=Omega,
            name="Y",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X + Y
        expected_data = pd.DataFrame(
            [(11, 22), (33, 44), (55, 66)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X+Y)_0", "(X+Y)_1"], name="(X+Y)"),
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_vector_and_scalar(self):
        """Test adding a scalar to a RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = X + 10
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X+10)_0", "(X+10)_1"], name="(X+10)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_vector(self):
        """Test adding a RandomVector to a scalar (reverse add)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = 10 + X
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(10+X)_0", "(10+X)_1"], name="(10+X)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10+X)"

    def test_sub_two_random_vectors(self):
        """Test subtracting two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Y = RandomVector(
            sample_space=Omega,
            name="Y",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = X - Y
        expected_values = pd.DataFrame(
            [(9, 18), (27, 36), (45, 54)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X-Y)_0", "(X-Y)_1"], name="(X-Y)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_vector_and_scalar(self):
        """Test subtracting a scalar from a RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X - 5
        expected_data = pd.DataFrame(
            [(5, 15), (25, 35), (45, 55)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X-5)_0", "(X-5)_1"], name="(X-5)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_vector(self):
        """Test subtracting a RandomVector from a scalar (reverse sub)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = 10 - X
        expected_data = pd.DataFrame(
            [(9, 8), (7, 6), (5, 4)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(10-X)_0", "(10-X)_1"], name="(10-X)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10-X)"

    def test_mul_two_random_vectors(self):
        """Test multiplying two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Y = RandomVector(
            sample_space=Omega,
            name="Y",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X * Y
        expected_data = pd.DataFrame(
            [(20, 60), (120, 200), (300, 420)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X*Y)_0", "(X*Y)_1"], name="(X*Y)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*Y)"

    def test_mul_random_vector_and_scalar(self):
        """Test multiplying a RandomVector by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = X * 10
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X*10)_0", "(X*10)_1"], name="(X*10)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_vector(self):
        """Test multiplying a scalar by a RandomVector (reverse mul)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = 10 * X
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(10*X)_0", "(10*X)_1"], name="(10*X)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10*X)"

    def test_truediv_two_random_vectors(self):
        """Test dividing two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (100, 200), 1: (300, 400), 2: (500, 600)})
        Y = RandomVector(
            sample_space=Omega,
            name="Y",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X / Y
        expected_data = pd.DataFrame(
            [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X/Y)_0", "(X/Y)_1"], name="(X/Y)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/Y)"

    def test_truediv_random_vector_and_scalar(self):
        """Test dividing a RandomVector by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X / 10
        expected_data = pd.DataFrame(
            [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X/10)_0", "(X/10)_1"], name="(X/10)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_vector(self):
        """Test dividing a scalar by a RandomVector (reverse div)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (2, 4), 1: (5, 10), 2: (20, 25)})
        Z = 100 / X
        expected_data = pd.DataFrame(
            [(50.0, 25.0), (20.0, 10.0), (5.0, 4.0)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(100/X)_0", "(100/X)_1"], name="(100/X)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(100/X)"

    def test_pow_two_random_vectors(self):
        """Test exponentiating two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Y = RandomVector(
            sample_space=Omega,
            name="Y",
        ).from_dict({0: (2, 2), 1: (2, 2), 2: (2, 2)})
        Z = X**Y
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X**Y)_0", "(X**Y)_1"], name="(X**Y)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self):
        """Test exponentiating a RandomVector by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Z = X**2
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X**2)_0", "(X**2)_1"], name="(X**2)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self):
        """Test exponentiating a scalar by a RandomVector (reverse pow)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (0, 1)})
        Z = 2**X
        expected_data = pd.DataFrame(
            [(4, 8), (16, 32), (1, 2)],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(2**X)_0", "(2**X)_1"], name="(2**X)"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(2**X)"

    def test_add_with_different_probability_spaces_raises_error(self):
        """Test that adding RandomVectors with different probability spaces raises ValueError."""
        Omega1 = SampleSpace().from_list(["a", "b", "c"])
        Omega2 = SampleSpace().from_list(["x", "y", "z"])
        X = RandomVector(
            sample_space=Omega1,
            name="X",
        ).from_dict({"a": (1, 2), "b": (3, 4), "c": (5, 6)})
        Y = RandomVector(
            sample_space=Omega2,
            name="Y",
        ).from_dict({"x": (1, 2), "y": (3, 4), "z": (5, 6)})

        with pytest.raises(ValueError, match="incompatible probability spaces"):
            Z = X + Y  # noqa: F841

    def test_add_with_non_random_vector_raises_error(self):
        """Test that adding a non-RandomVector and non-scalar raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            sample_space=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})

        with pytest.raises(TypeError):
            Z = X + "invalid"  # noqa: F841


class TestArithmeticWithRandomVariable:
    def test_add_two_random_variables(self):
        """Test adding two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            outputs={0: 1, 1: 3, 2: 5},
        )
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict(
            outputs={0: 10, 1: 30, 2: 50},
        )
        Z = X + Y
        expected_values = pd.Series(
            [11, 33, 55],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X+Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_variable_and_scalar(self):
        """Test adding a scalar to a RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = X + 10
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X+10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_variable(self):
        """Test adding a RandomVariable to a scalar (reverse add)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = 10 + X
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(10+X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10+X)"

    def test_sub_two_random_variables(self):
        """Test subtracting two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = X - Y
        expected_values = pd.Series(
            [9, 27, 45],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X-Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_variable_and_scalar(self):
        """Test subtracting a scalar from a RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X - 5
        expected_values = pd.Series(
            [5, 25, 45],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X-5)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_variable(self):
        """Test subtracting a RandomVariable from a scalar (reverse sub)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = 10 - X
        expected_values = pd.Series(
            [9, 7, 5],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(10-X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10-X)"

    def test_mul_two_random_variables(self):
        """Test multiplying two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 6},
        )
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X * Y
        expected_values = pd.Series(
            [20, 120, 300],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X*Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*Y)"

    def test_mul_random_variable_and_scalar(self):
        """Test multiplying a RandomVariable by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = X * 10
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X*10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_variable(self):
        """Test multiplying a scalar by a RandomVariable (reverse mul)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = 10 * X
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(10*X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10*X)"

    def test_truediv_two_random_variables(self):
        """Test dividing two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 100, 1: 300, 2: 500},
        )
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X / Y
        expected_values = pd.Series(
            [10.0, 10.0, 10.0],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X/Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/Y)"

    def test_truediv_random_variable_and_scalar(self):
        """Test dividing a RandomVariable by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X / 10
        expected_values = pd.Series(
            [1.0, 3.0, 5.0],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X/10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_variable(self):
        """Test dividing a scalar by a RandomVariable (reverse div)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 2, 1: 5, 2: 20},
        )
        Z = 100 / X
        expected_values = pd.Series(
            [50.0, 20.0, 5.0],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(100/X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(100/X)"

    def test_pow_two_random_variables(self):
        """Test exponentiating two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 6},
        )
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict(
            {0: 2, 1: 2, 2: 2},
        )
        Z = X**Y
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X**Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**Y)"

    def test_pow_random_variable_and_scalar(self):
        """Test exponentiating a RandomVariable by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 6},
        )
        Z = X**2
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X**2)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_variable(self):
        """Test exponentiating a scalar by a RandomVariable (reverse pow)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 0},
        )
        Z = 2**X
        expected_values = pd.Series(
            [4, 16, 1],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(2**X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(2**X)"

    def test_add_with_different_probability_spaces_raises_error(self):
        """Test that adding RandomVariables with different probability spaces raises ValueError."""
        Omega1 = SampleSpace().from_list(["a", "b", "c"])
        Omega2 = SampleSpace().from_list(["x", "y", "z"])
        X = RandomVariable(sample_space=Omega1, name="X").from_dict(
            {"a": 1, "b": 3, "c": 5},
        )
        Y = RandomVariable(sample_space=Omega2, name="Y").from_dict(
            {"x": 1, "y": 3, "z": 5},
        )

        with pytest.raises(ValueError, match="incompatible probability spaces"):
            Z = X + Y  # noqa: F841

    def test_add_with_non_random_variable_raises_error(self):
        """Test that adding a non-RandomVariable and non-scalar raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )

        with pytest.raises(TypeError):
            Z = X + "invalid"  # noqa: F841


# --------------------- comparison --------------------- #


class TestComparisonOperators:
    def test_lt_two_random_vectors(self):
        """Test less than comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {0: (1, 2), 1: (2, 3), 2: (3, 4)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
        )
        result = X < Y
        expected_data = pd.DataFrame(
            [[False, True], [False, True], [False, False]],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X < Y)_0", "(X < Y)_1"], name="(X < Y)"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X < Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_le_two_random_vectors(self):
        """Test less than or equal comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {0: (1, 2), 1: (2, 3), 2: (3, 4)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {0: (1, 3), 1: (2, 4), 2: (3, 4)}
        )
        result = X <= Y
        expected_data = pd.DataFrame(
            [[True, True], [True, True], [True, True]],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X <= Y)_0", "(X <= Y)_1"], name="(X <= Y)"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X <= Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_gt_two_random_vectors(self):
        """Test greater than comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {0: (5, 6), 1: (3, 4), 2: (1, 2)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {0: (3, 5), 1: (3, 3), 2: (2, 3)}
        )
        result = X > Y
        expected_data = pd.DataFrame(
            [[True, True], [False, True], [False, False]],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X > Y)_0", "(X > Y)_1"], name="(X > Y)"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X > Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_ge_two_random_vectors(self):
        """Test greater than or equal comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {0: (5, 6), 1: (3, 4), 2: (1, 2)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {0: (5, 5), 1: (3, 4), 2: (2, 3)}
        )
        result = X >= Y
        expected_data = pd.DataFrame(
            [[True, True], [True, True], [False, False]],
            index=pd.Index([0, 1, 2], name="Omega"),
            columns=pd.Index(["(X >= Y)_0", "(X >= Y)_1"], name="(X >= Y)"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X >= Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_lt_random_variables(self):
        """Test less than comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict({0: 2, 1: 2, 2: 1})
        result = X < Y
        expected_data = pd.Series(
            [True, False, False],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X < Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X < Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_le_random_variables(self):
        """Test less than or equal comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict({0: 2, 1: 2, 2: 1})
        result = X <= Y
        expected_data = pd.Series(
            [True, True, False],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X <= Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X <= Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_gt_random_variables(self):
        """Test greater than comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict({0: 5, 1: 3, 2: 1})
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict({0: 2, 1: 3, 2: 2})
        result = X > Y
        expected_data = pd.Series(
            [True, False, False],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X > Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X > Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_ge_random_variables(self):
        """Test greater than or equal comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, name="X").from_dict({0: 5, 1: 3, 2: 1})
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict({0: 2, 1: 3, 2: 2})
        result = X >= Y
        expected_data = pd.Series(
            [True, True, False],
            index=pd.Index([0, 1, 2], name="Omega"),
            name="(X >= Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X >= Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_lt_random_vector_and_scalar(self):
        """Test less than comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(sample_space=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X < 5, 5 > X]
        expected_data = pd.DataFrame(
            [[True, True], [True, False]],
            index=Omega.data,
            columns=pd.Index(["(X < 5)_0", "(X < 5)_1"], name="(X < 5)"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X < 5)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_le_random_vector_and_scalar(self):
        """Test less than or equal comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(sample_space=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X <= 3, 3 >= X]
        expected_data = pd.DataFrame(
            [[True, True], [True, False]],
            index=Omega.data,
            columns=pd.Index(["(X <= 3)_0", "(X <= 3)_1"], name="(X <= 3)"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X <= 3)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_gt_random_vector_and_scalar(self):
        """Test greater than comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(sample_space=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X > 2, 2 < X]
        expected_data = pd.DataFrame(
            [[False, False], [True, True]],
            index=Omega.data,
            columns=pd.Index(["(X > 2)_0", "(X > 2)_1"], name="(X > 2)"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X > 2)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_ge_random_vector_and_scalar(self):
        """Test greater than or equal comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(sample_space=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X >= 2, 2 <= X]
        expected_data = pd.DataFrame(
            [[False, True], [True, True]],
            index=Omega.data,
            columns=pd.Index(["(X >= 2)_0", "(X >= 2)_1"], name="(X >= 2)"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X >= 2)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_lt_random_variable_and_scalar(self):
        """Test less than comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X < 3, 3 > X]
        expected_data = pd.Series([True, False], index=Omega.data, name="(X < 3)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X < 3)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_le_random_variable_and_scalar(self):
        """Test less than or equal comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X <= 3, 3 >= X]
        expected_data = pd.Series([True, True], index=Omega.data, name="(X <= 3)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X <= 3)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_gt_random_variable_and_scalar(self):
        """Test greater than comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X > 1, 1 < X]
        expected_data = pd.Series([False, True], index=Omega.data, name="(X > 1)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X > 1)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_ge_random_variable_and_scalar(self):
        """Test greater than or equal comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X >= 1, 1 <= X]
        expected_data = pd.Series([True, True], index=Omega.data, name="(X >= 1)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X >= 1)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_lt_with_different_domains_raises(self):
        """Test that comparing RandomVectors with different domains raises ValueError."""
        Omega1 = SampleSpace().from_list(["a", "b", "c"])
        Omega2 = SampleSpace().from_list(["x", "y", "z"])
        X = RandomVector(sample_space=Omega1, name="X").from_dict(
            {"a": (1, 2), "b": (3, 4), "c": (5, 6)}
        )
        Y = RandomVector(sample_space=Omega2, name="Y").from_dict(
            {"x": (1, 2), "y": (3, 4), "z": (5, 6)}
        )

        with pytest.raises(ValueError, match="must have the same domain"):
            _ = X < Y

    def test_lt_with_different_dimensions_raises(self):
        """Test that comparing RandomVectors with different dimensions raises ValueError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {0: (1, 2, 3), 1: (3, 4, 5), 2: (5, 6, 7)}
        )

        with pytest.raises(ValueError, match="must have the same dimension"):
            _ = X < Y

    def test_lt_with_non_random_vector_raises(self):
        """Test that comparing RandomVector with non-RandomVector raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        )

        with pytest.raises(TypeError, match="must be a RandomVector"):
            _ = X < "not a random vector"


class TestBooleanMethods:
    def test_all_returns_true_when_all_true(self):
        """Test that all() returns True when all values are True."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {"omega_0": (0, 1), "omega_1": (1, 2), "omega_2": (2, 3)}
        )
        result = X > Y

        assert result.all() is True

    def test_all_returns_false_when_some_false(self):
        """Test that all() returns False when some values are False."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        assert result.all() is False

    def test_any_returns_true_when_some_true(self):
        """Test that any() returns True when at least one value is True."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        assert result.any() is True

    def test_any_returns_false_when_all_false(self):
        """Test that any() returns False when all values are False."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {"omega_0": (5, 6), "omega_1": (7, 8), "omega_2": (9, 10)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        )
        result = X < Y

        assert result.any() is False

    def test_all_with_random_variable(self):
        """Test all() method with RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 2, "omega_2": 3}
        )
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict(
            {"omega_0": 0, "omega_1": 1, "omega_2": 2}
        )
        result = X > Y

        assert result.all() is True

    def test_any_with_random_variable(self):
        """Test any() method with RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVariable(sample_space=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 2, "omega_2": 3}
        )
        Y = RandomVariable(sample_space=Omega, name="Y").from_dict(
            {"omega_0": 2, "omega_1": 2, "omega_2": 1}
        )
        result = X < Y

        assert result.any() is True

    def test_bool_raises_value_error(self):
        """Test that __bool__() raises ValueError to prevent ambiguous boolean conversion."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        with pytest.raises(
            ValueError, match="truth value of a RandomVector is ambiguous"
        ):
            bool(result)

    def test_bool_in_if_statement_raises(self):
        """Test that using RandomVector in if statement raises ValueError."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(sample_space=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(sample_space=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        with pytest.raises(
            ValueError, match="truth value of a RandomVector is ambiguous"
        ):
            if result:
                pass
