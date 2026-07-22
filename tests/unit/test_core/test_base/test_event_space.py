import pytest

from sigalg.core import (
    MeasurableSpace,
    ProbabilityMeasure,
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

    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters."""
        measurable_space = MeasurableSpace()

        assert measurable_space.sample_space is None
        assert measurable_space.sig_alg is None

    def test_constructor_all_parameters(self, Omega, F):
        """Test the constructor with all parameters."""
        measurable_space = MeasurableSpace(sample_space=Omega, sig_alg=F)

        assert measurable_space.sample_space is Omega
        assert measurable_space.sig_alg is F

    def test_constructor_only_sample_space(self, Omega):
        """Test the constructor with only the sample space."""
        measurable_space = MeasurableSpace(sample_space=Omega)

        assert measurable_space.sample_space is Omega
        assert measurable_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert measurable_space.sig_alg.sample_space is Omega

    def test_constructor_only_sig_alg(self, F):
        """Test the constructor with only the sigma-algebra."""
        measurable_space = MeasurableSpace(sig_alg=F)

        assert measurable_space.sample_space is F.sample_space
        assert measurable_space.sig_alg is F


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
                1: 0,
                2: 1,
                3: 1,
            },
        )

    def test_sample_space_getter(self, Omega, F):
        """Test sample_space property getter."""
        measurable_space = MeasurableSpace(sample_space=Omega, sig_alg=F)

        assert measurable_space.sample_space == Omega

    def test_sample_space_setter_on_empty_measurable_space(self, Omega):
        """Test sample_space property setter on empty MeasurableSpace."""
        measurable_space = MeasurableSpace()
        measurable_space.sample_space = Omega

        assert measurable_space.sample_space == Omega
        assert measurable_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert measurable_space.sig_alg.sample_space is Omega

    def test_sample_space_setter_on_nonempty_measurable_space(self, Omega, F):
        """Test sample_space property setter on nonempty MeasurableSpace."""
        measurable_space = MeasurableSpace(sample_space=Omega, sig_alg=F)
        Omega_new = SampleSpace(["a", "b", "c", "d"], name="Omega_new")
        measurable_space.sample_space = Omega_new

        assert measurable_space.sample_space is Omega_new
        assert measurable_space.sig_alg.sample_space is Omega_new


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

    def test_sig_alg_getter(self, Omega, F):
        """Test sig_alg property getter."""
        measurable_space = MeasurableSpace(sample_space=Omega, sig_alg=F)

        assert measurable_space.sig_alg is F

    def test_sig_alg_setter_on_empty_measurable_space(self, F):
        """Test sig_alg property setter on empty MeasurableSpace."""
        measurable_space = MeasurableSpace()
        measurable_space.sig_alg = F

        assert measurable_space.sig_alg is F
        assert measurable_space.sample_space is F.sample_space

    def test_sig_alg_setter_on_nonempty_measurable_space(self, Omega, F, G):
        """Test sig_alg property setter on nonempty MeasurableSpace."""
        measurable_space = MeasurableSpace(sample_space=Omega, sig_alg=F)
        measurable_space.sig_alg = G

        assert measurable_space.sig_alg is G
        assert measurable_space.sample_space is Omega

    def test_sig_alg_setter_type_error(self):
        """Test sig_alg setter with invalid type raises TypeError."""
        measurable_space = MeasurableSpace()

        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra"):
            measurable_space.sig_alg = "not a sigma algebra"

    def test_sig_alg_setter_value_error_different_sample_space(self, Omega, F):
        """Test sig_alg setter with different sample space raises ValueError."""
        measurable_space = MeasurableSpace(sample_space=Omega, sig_alg=F)
        Omega_other = SampleSpace.from_sequence(size=3)
        G = SigmaAlgebra(sample_space=Omega_other, name="G", mapping={0: 0, 1: 1, 2: 1})

        with pytest.raises(
            ValueError, match="New sig_alg must have the same sample space"
        ):
            measurable_space.sig_alg = G


# --------------------- test conversion methods --------------------- #


class TestMakeProbabilitySpace:
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

    def test_make_probability_space(self, Omega, F):
        """Test the make_probability_space method."""
        measurable_space = MeasurableSpace(sample_space=Omega, sig_alg=F)
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.8,
            },
        )
        prob_space = measurable_space.make_probability_space(prob_measure=P)

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg is F
        assert prob_space.prob_measure is P


# --------------------- test equality --------------------- #


class TestEquality:
    def test_non_equality_different_sample_spaces(self):
        """Test inequality when sample spaces are different."""
        Omega1 = SampleSpace.from_sequence(size=2)
        Omega2 = SampleSpace.from_sequence(size=3)
        F1 = SigmaAlgebra.power_set(Omega1)
        F2 = SigmaAlgebra.power_set(Omega2)
        measurable_space1 = MeasurableSpace(sample_space=Omega1, sig_alg=F1)
        measurable_space2 = MeasurableSpace(sample_space=Omega2, sig_alg=F2)

        assert measurable_space1 != measurable_space2

    def test_non_equality_different_sigma_algebras(self):
        """Test inequality when sigma algebras are different."""
        Omega = SampleSpace.from_sequence(size=3)
        F1 = SigmaAlgebra.power_set(Omega)
        F2 = SigmaAlgebra(sample_space=Omega, mapping={0: 0, 1: 0, 2: 1})
        measurable_space1 = MeasurableSpace(sample_space=Omega, sig_alg=F1)
        measurable_space2 = MeasurableSpace(sample_space=Omega, sig_alg=F2)

        assert measurable_space1 != measurable_space2

    def test_non_equality_wrong_type(self):
        """Test inequality when comparing to wrong type."""
        Omega = SampleSpace.from_sequence(size=2)
        measurable_space = MeasurableSpace(sample_space=Omega)
        other = "not an event space"

        assert measurable_space != other

    def test_equality_same_parameters(self):
        """Test equality when parameters are the same."""
        Omega = SampleSpace.from_sequence(size=3)
        F = SigmaAlgebra.power_set(Omega)
        measurable_space1 = MeasurableSpace(sample_space=Omega, sig_alg=F)
        measurable_space2 = MeasurableSpace(sample_space=Omega, sig_alg=F)

        assert measurable_space1 == measurable_space2
