import pandas as pd
import pytest

from sigalg.core import Event, ProbabilityMeasure, SampleSpace, SigmaAlgebra
from sigalg.core.random_objects.random_vector import RandomVector


class TestConstructor:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    def test_constructor_default_name(self, Omega, F):
        """Test the constructor of ProbabilityMeasure with default name."""
        probabilities = {0: 0.5, 1: 0.5}
        P = ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

        assert P.sample_space == Omega
        assert P.probabilities == probabilities
        assert P.name == "P"

    def test_constructor_custom_name(self):
        """Test the constructor of ProbabilityMeasure with custom name."""
        Omega = SampleSpace().from_list(["a", "b", "c"])
        F = SigmaAlgebra.power_set(Omega)
        probabilities = {"a": 0.2, "b": 0.3, "c": 0.5}
        Q = ProbabilityMeasure(sig_alg=F, name="Q").from_dict(
            probabilities=probabilities
        )

        assert Q.sample_space == Omega
        assert Q.probabilities == probabilities
        assert Q.name == "Q"

    def test_invalid_input_probabilities_not_summing_to_1(self, Omega, F):
        """Test that probabilities not summing to 1 raises ValueError."""
        probabilities = {0: 0.6, 1: 0.5}

        with pytest.raises(ValueError):
            ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_invalid_input_negative_and_greater_than_one_probability(self, Omega, F):
        """Test that negative and greater than one probabilities raise ValueError."""
        probabilities = {0: -0.1, 1: 1.1}

        with pytest.raises(ValueError):
            ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_invalid_input_non_numeric_probability(self, Omega, F):
        """Test that non-numeric probabilities raise TypeError."""
        probabilities = {0: "not a number", 1: 1.0}

        with pytest.raises(TypeError):
            ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_constructor_partition_sigma_algebra(self):
        """Test the constructor with a partition sigma-algebra."""
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        probabilities = {0: 0.3, 1: 0.3, 2: 0.2, 3: 0.2}
        P = ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

        assert P.sample_space == Omega
        assert P.sig_alg == F
        assert P.probabilities == probabilities
        assert not F.is_power_set


class TestFromPandas:
    def test_from_pandas_custom_name(self):
        """Test the from_pandas instance method of ProbabilityMeasure with custom name."""
        series_data = {0: 0.4, 1: 0.6}
        data = pd.Series(series_data, name="dummy_name")
        Q = ProbabilityMeasure(name="Q").from_pandas(data=data)
        data.name = "probability"

        pd.testing.assert_series_equal(Q.data, data)
        assert Q.name == "Q"

    def test_from_pandas_default_name(self):
        """Test the from_pandas instance method of ProbabilityMeasure with default name."""
        series_data = {0: 0.7, 1: 0.3}
        data = pd.Series(series_data, name="dummy_name")
        P = ProbabilityMeasure().from_pandas(data=data)
        data.name = "probability"

        pd.testing.assert_series_equal(P.data, data)
        assert P.name == "P"


class TestEquality:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    def test_non_equality_different_sample_spaces(self):
        """Test the __eq__ method for inequality with different sample spaces."""
        Omega1 = SampleSpace().from_sequence(size=2)
        Omega2 = SampleSpace().from_list(["a", "b"])
        F1 = SigmaAlgebra.power_set(Omega1)
        F2 = SigmaAlgebra.power_set(Omega2)
        P1 = ProbabilityMeasure(sig_alg=F1).from_dict(probabilities={0: 0.5, 1: 0.5})
        P2 = ProbabilityMeasure(sig_alg=F2).from_dict(
            probabilities={"a": 0.5, "b": 0.5}
        )

        assert P1 != P2

    def test_non_equality_different_probabilities(self, Omega, F):
        """Test the __eq__ method for inequality with different probabilities."""
        P1 = ProbabilityMeasure(sig_alg=F).from_dict(probabilities={0: 0.6, 1: 0.4})
        P2 = ProbabilityMeasure(sig_alg=F).from_dict(probabilities={0: 0.5, 1: 0.5})

        assert P1 != P2

    def test_equality_same_probabilities_and_sample_space(self, Omega, F):
        """Test the __eq__ method for equality with same probabilities and sample space."""
        P1 = ProbabilityMeasure(sig_alg=F).from_dict(probabilities={0: 0.5, 1: 0.5})
        P2 = ProbabilityMeasure(sig_alg=F).from_dict(probabilities={0: 0.5, 1: 0.5})

        assert P1 == P2

    def test_equality_same_components_different_names(self):
        """Test the __eq__ method for equality with same components but different names."""
        Omega_S = SampleSpace(name="S").from_list(["a", "b"])
        Omega_T = SampleSpace(name="T").from_list(["a", "b"])
        F_S = SigmaAlgebra.power_set(Omega_S)
        F_T = SigmaAlgebra.power_set(Omega_T)
        Q = ProbabilityMeasure(sig_alg=F_S, name="Q").from_dict(
            probabilities={"a": 0.2, "b": 0.8}
        )
        R = ProbabilityMeasure(sig_alg=F_T, name="R").from_dict(
            probabilities={"a": 0.2, "b": 0.8}
        )

        assert Q == R


class TestFromFeatures:
    def test_from_features(self):
        """Test adding a ProbabilityMeasure to the domain of a RandomVector."""
        Omega = SampleSpace().from_sequence(size=4)
        outputs = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
        X = RandomVector(domain=Omega, name="X").from_dict(outputs=outputs)

        def pmf(feature_vector):
            v0, v1 = feature_vector
            return 0.75**v0 * 0.25 ** (1 - v0) * 0.6**v1 * 0.4 ** (1 - v1)

        P = ProbabilityMeasure.from_features(rv=X, pmf=pmf)
        P_expected = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(Omega)
        ).from_dict(
            probabilities={0: 0.25 * 0.4, 1: 0.25 * 0.6, 2: 0.75 * 0.4, 3: 0.75 * 0.6}
        )

        assert P.sample_space == Omega
        assert P == P_expected


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        probabilities = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        return ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_call_list_of_indices(self, P):
        """Test the __call__ method of ProbabilityMeasure with list of indices."""
        indices = [0, 2]
        result = P(indices)
        expected = 0.4

        assert abs(result - expected) < 1e-9

    def test_call_single_hashable_index(self, P):
        """Test the __call__ method of ProbabilityMeasure with single hashable index."""
        result = P(1)
        expected = 0.2

        assert abs(result - expected) < 1e-9

    def test_call_event_instance(self, F, P):
        """Test the __call__ method of ProbabilityMeasure with event instance."""
        A = Event(sig_alg=F).from_list([1, 3])
        result = P(A)
        expected = 0.6

        assert abs(result - expected) < 1e-9

    def test_call_partition_sigma_algebra(self):
        """Test the __call__ method with partition sigma-algebra."""
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        probabilities = {0: 0.15, 1: 0.15, 2: 0.35, 3: 0.35}
        P = ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)
        A = F.get_event([0, 1])
        result = P(A)
        expected = 0.3

        assert abs(result - expected) < 1e-9


class TestUniform:
    def test_uniform(self):
        """Test the uniform probability measure constructor."""
        Omega = SampleSpace().from_list(["a", "b", "c", "d"])
        F = SigmaAlgebra.power_set(Omega)
        U = ProbabilityMeasure.uniform(sig_alg=F, name="U")
        expected_probabilities = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}

        assert U.probabilities == expected_probabilities
        assert U.name == "U"


class TestConditionalProbability:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        probabilities = {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.1}
        return ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_conditional_probability_subset_of_conditioning_event(self, F, P):
        """Test conditional_probability method when event A is subset of B."""
        A = Event(sig_alg=F).from_list([0, 1])
        B = Event(sig_alg=F).from_list([0, 1, 2])
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9

    def test_conditional_probability_non_trivial_overlap(self, F, P):
        """Test conditional_probability method with non-trivial overlap."""
        A = Event(sig_alg=F).from_list([0, 1])
        B = Event(sig_alg=F).from_list([1, 2])
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9

    def test_conditional_probability_no_overlap(self, F, P):
        """Test conditional_probability method with no overlap."""
        A = Event(sig_alg=F).from_list([2, 3])
        B = Event(sig_alg=F).from_list([0, 1])
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9

    def test_conditioning_on_impossible_event(self):
        """Test that conditional_probability raises ValueError when P(B) = 0."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra.power_set(Omega)
        probabilities = {0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0}
        P = ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)
        A = Event(sig_alg=F).from_list([0, 1])
        B = Event(sig_alg=F).from_list([2, 3])

        with pytest.raises(ValueError):
            P.conditional_probability(A, B)


class TestNonMeasurableSubsets:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        return SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    @pytest.fixture
    def P(self, F):
        probabilities = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        return ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_call_non_measurable_list(self, P):
        """Test that calling with non-measurable list of indices raises ValueError."""
        with pytest.raises(ValueError, match="do not form a measurable event"):
            P([0, 2])

    def test_call_partial_atom(self, P):
        """Test that calling with partial atom raises ValueError."""
        with pytest.raises(ValueError, match="do not form a measurable event"):
            P([1, 3])

    def test_call_single_point_from_multi_point_atom(self, P):
        """Test that calling with single point from multi-point atom raises ValueError."""
        with pytest.raises(ValueError, match="do not form a measurable event"):
            P([0])

    def test_call_event_from_different_sigma_algebra(self, Omega, P):
        """Test that calling with Event from different sigma-algebra raises ValueError."""
        F_other = SigmaAlgebra.power_set(Omega)
        A = Event(sig_alg=F_other).from_list([0, 1])

        with pytest.raises(ValueError, match="not in the domain"):
            P(A)

    def test_call_event_from_different_sample_space(self, F, P):
        """Test that calling with Event from different sample space raises ValueError."""
        Omega_other = SampleSpace().from_sequence(size=3)
        F_other = SigmaAlgebra.power_set(Omega_other)
        A = Event(sig_alg=F_other).from_list([0, 1])

        with pytest.raises(ValueError, match="not in the domain"):
            P(A)

    def test_call_valid_measurable_event(self, P):
        """Test that calling with valid measurable event works correctly."""
        result = P([0, 1])
        expected = 0.3

        assert abs(result - expected) < 1e-9

    def test_call_single_index_power_set(self):
        """Test that calling with single index works on power-set sigma-algebra."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra.power_set(Omega)
        probabilities = {0: 0.2, 1: 0.5, 2: 0.3}
        P = ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)
        result = P(1)
        expected = 0.5

        assert abs(result - expected) < 1e-9


class TestAreIndependent:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        probabilities = {0: 0.25**2, 1: 0.25 * 0.75, 2: 0.75 * 0.25, 3: 0.75**2}
        return ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_are_independent_events_independent(self, F, P):
        """Test the are_independent method with independent events."""
        A = Event(sig_alg=F).from_list([0, 1])
        B = Event(sig_alg=F).from_list([0, 2])
        result = P.are_independent(event1=A, event2=B)

        assert result

    def test_are_independent_events_dependent(self, F, P):
        """Test the are_independent method with dependent events."""
        A = Event(sig_alg=F).from_list([0, 1])
        B = Event(sig_alg=F).from_list([2, 3])
        result = P.are_independent(event1=A, event2=B)

        assert not result

    def test_are_independent_sigma_algebras_independent(self, Omega, P):
        """Test the are_independent method for independent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 0, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 1, 2: 0, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega, name="F1").from_dict(
            sample_id_to_atom_id=atom_ids_1
        )
        F2 = SigmaAlgebra(sample_space=Omega, name="F2").from_dict(
            sample_id_to_atom_id=atom_ids_2
        )
        result = P.are_independent(algebra1=F1, algebra2=F2)

        assert result

    def test_are_independent_sigma_algebras_dependent(self, Omega, P):
        """Test the are_independent method for dependent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 1, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 0, 2: 1, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega, name="F1").from_dict(
            sample_id_to_atom_id=atom_ids_1
        )
        F2 = SigmaAlgebra(sample_space=Omega, name="F2").from_dict(
            sample_id_to_atom_id=atom_ids_2
        )
        result = P.are_independent(algebra1=F1, algebra2=F2)

        assert not result

    def test_are_independent_raises_for_both_events_and_algebras(self):
        """Test that are_independent raises ValueError when both events and algebras are provided."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        probabilities = {0: 0.5, 1: 0.5}
        P = ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)
        A = Event(sig_alg=F).from_list([0])
        B = Event(sig_alg=F).from_list([1])
        F1 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 1}
        )
        F2 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 1}
        )

        with pytest.raises(ValueError, match="Must provide exactly one"):
            P.are_independent(event1=A, event2=B, algebra1=F1, algebra2=F2)

    def test_are_independent_raises_for_neither_events_nor_algebras(self):
        """Test that are_independent raises ValueError when neither events nor algebras are provided."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        probabilities = {0: 0.5, 1: 0.5}
        P = ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

        with pytest.raises(ValueError, match="Must provide exactly one"):
            P.are_independent()
