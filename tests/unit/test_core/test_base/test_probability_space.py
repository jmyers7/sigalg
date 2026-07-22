import pytest

from sigalg.core import (
    ProbabilityMeasure,
    MeasureSpace,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestConstructor:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.8,
                1: 0.2,
            },
        )

    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters."""
        prob_space = MeasureSpace()

        assert prob_space.sample_space is None
        assert prob_space.sig_alg is None
        assert prob_space.prob_measure is None

    def test_constructor_all_parameters(self, Omega, F, P):
        """Test constructing MeasureSpace with all parameters."""
        prob_space = MeasureSpace(Omega, F, P)

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg is F
        assert prob_space.prob_measure is P

    def test_constructor_only_sample_space(self, Omega):
        """Test constructing MeasureSpace with only sample_space."""
        prob_space = MeasureSpace(Omega)

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(Omega)
        )
        assert prob_space.sig_alg.sample_space is Omega
        assert prob_space.prob_measure.sample_space is Omega
        assert prob_space.prob_measure.sig_alg is prob_space.sig_alg

    def test_constructor_only_sig_alg(self, F):
        """Test constructing MeasureSpace with only sig_alg."""
        prob_space = MeasureSpace(sig_alg=F)

        assert prob_space.sample_space is F.sample_space
        assert prob_space.sig_alg is F
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(sig_alg=F)
        assert prob_space.prob_measure.sample_space is F.sample_space
        assert prob_space.prob_measure.sig_alg is F

    def test_constructor_only_prob_measure(self, P):
        """Test constructing MeasureSpace with only prob_measure."""
        prob_space = MeasureSpace(prob_measure=P)

        assert prob_space.sample_space is P.sample_space
        assert prob_space.sig_alg is P.sig_alg
        assert prob_space.prob_measure is P

    def test_constructor_sample_space_and_sig_alg(self, Omega, F):
        """Test constructing MeasureSpace with sample_space and sig_alg."""
        prob_space = MeasureSpace(Omega, sig_alg=F)

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg is F
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(sig_alg=F)
        assert prob_space.prob_measure.sample_space is Omega
        assert prob_space.prob_measure.sig_alg is F

    def test_constructor_sample_space_and_prob_measure(self, Omega, P):
        """Test constructing MeasureSpace with sample_space and prob_measure."""
        prob_space = MeasureSpace(Omega, prob_measure=P)

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg is P.sig_alg
        assert prob_space.prob_measure is P

    def test_constructor_sig_alg_and_prob_measure(self, F, P):
        """Test constructing MeasureSpace with sig_alg and prob_measure."""
        prob_space = MeasureSpace(sig_alg=F, prob_measure=P)

        assert prob_space.sample_space is P.sample_space
        assert prob_space.sig_alg is F
        assert prob_space.prob_measure is P

    def test_invalid_sample_space_type_raises(self):
        """Test that invalid sample_space type raises TypeError."""
        with pytest.raises(TypeError):
            MeasureSpace(sample_space="not a sample space")

    def test_invalid_sig_alg_type_raises(self):
        """Test that invalid sig_alg type raises TypeError."""
        with pytest.raises(TypeError):
            MeasureSpace(sig_alg="not a sigma algebra")

    def test_invalid_prob_measure_type_raises(self):
        """Test that invalid prob_measure type raises TypeError."""
        with pytest.raises(TypeError):
            MeasureSpace(prob_measure="not a probability measure")

    def test_mismatched_sample_space_and_sig_alg_raises(self, Omega):
        """Test that mismatched sample_space and sig_alg raises ValueError."""
        Omega_other = SampleSpace().from_sequence(size=2)
        F_other = SigmaAlgebra(sample_space=Omega_other, mapping={0: 0, 1: 1})

        with pytest.raises(ValueError):
            MeasureSpace(Omega, sig_alg=F_other)

    def test_mismatched_sample_space_and_prob_measure_raises(self, Omega):
        """Test that mismatched sample_space and prob_measure raises ValueError."""
        Omega_other = SampleSpace.from_sequence(size=2)
        P_other = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(Omega_other), mapping={0: 0.5, 1: 0.5}
        )

        with pytest.raises(ValueError):
            MeasureSpace(Omega, prob_measure=P_other)

    def test_mismatched_sig_alg_and_prob_measure_raises(self, Omega):
        """Test that mismatched sig_alg and prob_measure raises ValueError."""
        F1 = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )
        F2 = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 1,
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F2,
            mapping={
                0: 0.3,
                1: 0.7,
            },
        )

        with pytest.raises(ValueError):
            MeasureSpace(sig_alg=F1, prob_measure=P)


class TestFromMeasurableSet:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.15,
                1: 0.6,
                2: 0.25,
            },
        )

    def test_from_event_basic(self, F, P):
        """Test creating conditional probability space from basic event."""
        A = F.get_event([1, 2, 3], name="A")
        prob_space = MeasureSpace.from_event(event=A, prob_measure=P)

        assert prob_space.sample_space.name == "A"
        assert prob_space.sample_space == A.to_sample_space()
        assert prob_space.sig_alg.name == "F_A"
        assert prob_space.prob_measure.name == "P_A"

    def test_from_event_probabilities_sum_to_one(self, F, P):
        """Test that conditional probabilities sum to 1."""
        A = F.get_event([1, 2, 3], name="A")
        prob_space = MeasureSpace.from_event(event=A, prob_measure=P)
        total_prob = sum(prob_space.prob_measure.data)

        assert abs(total_prob - 1.0) < 1e-10

    def test_from_event_conditional_probabilities_correct(self, F, P):
        """Test that conditional probabilities are correctly calculated."""
        A = F.get_event([1, 2, 3], name="A")
        prob_space = MeasureSpace.from_event(event=A, prob_measure=P)

        assert abs(prob_space.prob_measure([1, 2]) - 0.6 / 0.85) < 1e-10
        assert abs(prob_space.prob_measure(sample_point=3) - 0.25 / 0.85) < 1e-10

    def test_from_event_sigma_algebra_structure_preserved(self, F, P):
        """Test that sigma-algebra structure is preserved in conditional space."""
        A = F.get_event([1, 2, 3], name="A")
        prob_space = MeasureSpace.from_event(event=A, prob_measure=P)
        expected_sig_alg = SigmaAlgebra(
            sample_space=A.to_sample_space(),
            mapping={
                1: 1,
                2: 1,
                3: 2,
            },
        )

        assert prob_space.sig_alg == expected_sig_alg

    def test_from_event_full_sample_space(self, Omega, F, P):
        """Test creating conditional space from full sample space."""
        full = F.get_event([0, 1, 2, 3], name="Omega")
        prob_space = MeasureSpace.from_event(event=full, prob_measure=P)

        assert prob_space.sample_space == Omega

    def test_from_event_invalid_event_type_raises(self, P):
        """Test that from_event with non-MeasurableSet raises TypeError."""
        with pytest.raises(TypeError, match="event must be an MeasurableSet instance"):
            MeasureSpace.from_event(event="not an event", prob_measure=P)

    def test_from_event_invalid_prob_measure_type_raises(self, F):
        """Test that from_event with non-ProbabilityMeasure raises TypeError."""
        A = F.get_event([1, 2])

        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure instance"
        ):
            MeasureSpace.from_event(event=A, prob_measure="not a prob measure")

    def test_from_event_event_not_in_domain_raises(self, Omega, P):
        """Test that from_event with event not in domain raises ValueError."""
        F_other = SigmaAlgebra.power_set(Omega)
        A = F_other.get_event([0, 1])

        with pytest.raises(
            ValueError, match="event must be in the domain.*of the given"
        ):
            MeasureSpace.from_event(event=A, prob_measure=P)

    def test_from_event_zero_probability_raises(self, F):
        """Test that from_event with zero probability event raises ValueError."""
        P_zero = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 1.0,
                1: 0.0,
                2: 0.0,
            },
        )
        A = F.get_event([1, 2])

        with pytest.raises(
            ValueError, match="Cannot create a probability space from.*0 probability"
        ):
            MeasureSpace.from_event(event=A, prob_measure=P_zero)


# --------------------- test properties --------------------- #


class TestSampleSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def point_probs(self):
        return {
            0: 0.2,
            1: 0.3,
            2: 0.2,
            3: 0.3,
        }

    def test_sample_space_getter_on_prob_space(self, Omega, F, P):
        """Test sample_space property getter."""
        prob_space = MeasureSpace(sample_space=Omega, sig_alg=F, prob_measure=P)

        assert prob_space.sample_space is Omega

    def test_sample_space_setter_on_empty_prob_space(self, Omega):
        """Test sample_space property setter on empty MeasureSpace."""
        prob_space = MeasureSpace()
        prob_space.sample_space = Omega

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(Omega)
        )
        assert prob_space.sig_alg.sample_space is Omega
        assert prob_space.prob_measure.sample_space is Omega
        assert prob_space.prob_measure.sig_alg is prob_space.sig_alg

    def test_sample_space_setter_on_nonempty_prob_space(self, Omega, F, P):
        """Test sample_space property setter on nonempty MeasureSpace."""
        prob_space = MeasureSpace(sample_space=Omega, sig_alg=F, prob_measure=P)
        Omega_new = SampleSpace(["a", "b", "c", "d"], name="Omega_new")
        prob_space.sample_space = Omega_new

        assert prob_space.sample_space is Omega_new
        assert prob_space.sig_alg.sample_space is Omega_new
        assert prob_space.prob_measure.sample_space is Omega_new

    def test_sample_space_setter_type_error(self):
        """Test sample_space setter with invalid type raises TypeError."""
        prob_space = MeasureSpace()

        with pytest.raises(TypeError, match="sample_space must be a SampleSpace"):
            prob_space.sample_space = "not a sample space"


class TestSigAlg:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            },
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    def test_sig_alg_getter_on_prob_space(self, Omega, F, P):
        """Test sig_alg property getter."""
        prob_space = MeasureSpace(sample_space=Omega, sig_alg=F, prob_measure=P)

        assert prob_space.sig_alg is F

    def test_sig_alg_setter_on_empty_prob_space(self, F):
        """Test sig_alg property setter on empty MeasureSpace."""
        prob_space = MeasureSpace()
        prob_space.sig_alg = F

        assert prob_space.sig_alg is F
        assert prob_space.sample_space is F.sample_space
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(sig_alg=F)
        assert prob_space.prob_measure.sig_alg is F

    def test_sig_alg_setter_on_nonempty_prob_space(self, Omega, F, G, P):
        """Test sig_alg property setter on nonempty MeasureSpace."""
        prob_space = MeasureSpace(sample_space=Omega, sig_alg=F, prob_measure=P)
        prob_space.sig_alg = G

        assert prob_space.sig_alg is G
        assert prob_space.sample_space is Omega
        assert prob_space.prob_measure.sig_alg is G

    def test_sig_alg_setter_type_error(self):
        """Test sig_alg setter with invalid type raises TypeError."""
        prob_space = MeasureSpace()

        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra"):
            prob_space.sig_alg = "not a sigma algebra"


class TestProbMeasure:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            },
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def Q(self, G):
        return ProbabilityMeasure(
            sig_alg=G,
            name="Q",
            mapping={
                0: 0.5,
                1: 0.5,
            },
        )

    def test_prob_measure_getter_on_prob_space(self, Omega, F, P):
        """Test prob_measure property getter."""
        prob_space = MeasureSpace(sample_space=Omega, sig_alg=F, prob_measure=P)

        assert prob_space.prob_measure is P

    def test_prob_measure_setter_on_empty_prob_space(self, P):
        """Test prob_measure property setter on empty MeasureSpace."""
        prob_space = MeasureSpace()
        prob_space.prob_measure = P

        assert prob_space.prob_measure is P
        assert prob_space.sig_alg is P.sig_alg
        assert prob_space.sample_space is P.sample_space

    def test_prob_measure_setter_on_nonempty_prob_space(self, Omega, F, P, Q):
        """Test prob_measure property setter on nonempty MeasureSpace."""
        prob_space = MeasureSpace(sample_space=Omega, sig_alg=F, prob_measure=P)
        prob_space.prob_measure = Q

        assert prob_space.prob_measure is Q
        assert prob_space.sig_alg is Q.sig_alg
        assert prob_space.sample_space is Omega

    def test_prob_measure_setter_type_error(self):
        """Test prob_measure setter with invalid type raises TypeError."""
        prob_space = MeasureSpace()

        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure"
        ):
            prob_space.prob_measure = "not a probability measure"


# --------------------- test equality --------------------- #


class TestEquality:
    def test_non_equality_different_probability_measures(self):
        """Test inequality when probability measures are different."""
        Omega = SampleSpace.from_sequence(size=2)
        prob_space1 = MeasureSpace(
            Omega,
            prob_measure=ProbabilityMeasure(
                sig_alg=SigmaAlgebra.power_set(Omega), mapping={0: 0.5, 1: 0.5}
            ),
        )
        prob_space2 = MeasureSpace(
            Omega,
            prob_measure=ProbabilityMeasure(
                sig_alg=SigmaAlgebra.power_set(Omega), mapping={0: 0.7, 1: 0.3}
            ),
        )

        assert prob_space1 != prob_space2

    def test_non_equality_different_sigma_algebras(self):
        """Test inequality when sigma algebras are different."""
        Omega = SampleSpace.from_sequence(size=3)
        prob_space1 = MeasureSpace(
            Omega,
            sig_alg=SigmaAlgebra(sample_space=Omega, mapping={0: 0, 1: 0, 2: 1}),
        )
        prob_space2 = MeasureSpace(
            Omega,
            sig_alg=SigmaAlgebra(sample_space=Omega, mapping={0: 0, 1: 1, 2: 1}),
        )

        assert prob_space1 != prob_space2

    def test_non_equality_different_sample_spaces(self):
        """Test inequality when sample spaces are different."""
        Omega1 = SampleSpace.from_sequence(size=2)
        Omega2 = SampleSpace(["a", "b"])
        prob_space1 = MeasureSpace(Omega1)
        prob_space2 = MeasureSpace(Omega2)

        assert prob_space1 != prob_space2

    def test_non_equality_wrong_type_string(self):
        """Test inequality when comparing to string."""
        Omega = SampleSpace.from_sequence(size=2)
        prob_space = MeasureSpace(Omega)
        other = "not a probability space"

        assert prob_space != other

    def test_non_equality_wrong_type_int(self):
        """Test inequality when comparing to integer."""
        Omega = SampleSpace.from_sequence(size=2)
        prob_space = MeasureSpace(Omega)
        other = 123

        assert prob_space != other

    def test_equality_same_components(self):
        """Test equality when all components are the same."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra(sample_space=Omega, mapping={0: 0, 1: 1})
        P = ProbabilityMeasure(sig_alg=F, mapping={0: 0.5, 1: 0.5})
        prob_space1 = MeasureSpace(Omega, F, P)
        prob_space2 = MeasureSpace(Omega, F, P)

        assert prob_space1 == prob_space2
