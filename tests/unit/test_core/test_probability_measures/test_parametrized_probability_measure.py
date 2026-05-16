from collections.abc import Hashable
from numbers import Real

import pytest

from sigalg.core import (
    ParametrizedProbabilityMeasure,
    ProbabilityMeasure,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_with_no_parameters(self):
        """Test the constructor with no parameters."""
        P = ParametrizedProbabilityMeasure()

        assert P.name == "P"
        assert P.sig_alg is None
        assert P.parametrization is None

    def test_constructor_with_all_parameters(self):
        """Test the constructor with all parameters."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        Q = ParametrizedProbabilityMeasure(sig_alg=F, name="Q")

        assert Q.name == "Q"
        assert Q.sig_alg is F
        assert Q.parametrization is None

    def test_with_invalid_sig_alg_raises(self):
        """Test that the constructor raises an error when an invalid sig_alg is passed in."""
        with pytest.raises(
            TypeError, match="sig_alg must be an instance of SigmaAlgebra"
        ):
            ParametrizedProbabilityMeasure(sig_alg="not a sigma algebra")

    def test_with_invalid_name_raises(self):
        """Test that the constructor raises an error when an invalid name is passed in."""
        with pytest.raises(TypeError, match="name must be a hashable object or None"):
            ParametrizedProbabilityMeasure(name=["not a hashable object"])


class TestFromCallable:
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
                3: 1,
            }
        )

    def test_from_callable_with_single_variable_parametrization(self, F):
        """Test the from_callable method with a single-variable parametrization."""

        def parametrization(theta) -> dict[Hashable, Real]:  # noqa: D103
            if theta == 0:
                return {0: 0.2, 1: 0.8}
            elif theta == 1:
                return {0: 0.5, 1: 0.5}

        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(parametrization)

        assert P.parametrization is parametrization
        assert P.parameter_names == ["theta"]

    def test_from_callable_with_multi_variable_parametrization(self, F):
        """Test the from_callable method with a multi-variable parametrization."""

        def parametrization(alpha, beta) -> dict[Hashable, Real]:  # noqa: D103
            if (alpha, beta) == (0, 0):
                return {0: 0.8, 1: 0.2}
            elif (alpha, beta) == (1, 0):
                return {0: 0.5, 1: 0.5}
            elif (alpha, beta) == (0, 1):
                return {0: 0.2, 1: 0.8}
            elif (alpha, beta) == (1, 1):
                return {0: 0.1, 1: 0.9}

        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(parametrization)

        assert P.parametrization is parametrization
        assert P.parameter_names == ["alpha", "beta"]

    def test_with_invalid_parametrization_raises(self, F):
        """Test that the from_callable method raises an error when an invalid parametrization is passed in."""
        P = ParametrizedProbabilityMeasure(sig_alg=F)

        with pytest.raises(
            TypeError, match="parametrization must be a callable object"
        ):
            P.from_callable(parametrization="not a callable object")


# --------------------- test data access methods --------------------- #


class TestCall:
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
                3: 1,
            }
        )

    @pytest.fixture
    def single_variable_parametrization(self):
        """Fixture for a single-variable parametrization."""

        def parametrization(theta) -> dict[Hashable, Real]:  # noqa: D103
            if theta == 0:
                return {0: 0.2, 1: 0.8}
            elif theta == 1:
                return {0: 0.5, 1: 0.5}

        return parametrization

    @pytest.fixture
    def multi_variable_parametrization(self):
        """Fixture for a multi-variable parametrization."""

        def parametrization(alpha, beta) -> dict[Hashable, Real]:  # noqa: D103
            if (alpha, beta) == (0, 0):
                return {0: 0.8, 1: 0.2}
            elif (alpha, beta) == (1, 0):
                return {0: 0.5, 1: 0.5}
            elif (alpha, beta) == (0, 1):
                return {0: 0.2, 1: 0.8}
            elif (alpha, beta) == (1, 1):
                return {0: 0.1, 1: 0.9}

        return parametrization

    def test_call_with_single_variable_parametrization(
        self, F, single_variable_parametrization
    ):
        """Test the __call__ method with a single-variable parametrization."""
        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(
            single_variable_parametrization
        )

        assert P(0, theta=0) == 0.2
        assert P(1, theta=0) == 0.8
        assert P(0, theta=1) == 0.5
        assert P(1, theta=1) == 0.5

    def test_call_with_multi_variable_parametrization(
        self, F, multi_variable_parametrization
    ):
        """Test the __call__ method with a multi-variable parametrization."""
        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(
            multi_variable_parametrization
        )

        assert P(0, alpha=0, beta=0) == 0.8
        assert P(1, alpha=0, beta=0) == 0.2
        assert P(0, alpha=1, beta=0) == 0.5
        assert P(1, alpha=1, beta=0) == 0.5
        assert P(0, alpha=0, beta=1) == 0.2
        assert P(1, alpha=0, beta=1) == 0.8
        assert P(0, alpha=1, beta=1) == 0.1
        assert P(1, alpha=1, beta=1) == 0.9

    def test_call_with_unknown_parameters_raises(
        self, F, single_variable_parametrization
    ):
        """Test that the __call__ method raises an error when unknown parameters are passed in."""
        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(
            single_variable_parametrization
        )

        with pytest.raises(ValueError, match="unknown parameters"):
            P(0, theta=0, unknown_parameter=1)

    def test_with_unknown_sample_point_raises(self, F, single_variable_parametrization):
        """Test that the __call__ method raises an error when an unknown sample point is passed in."""
        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(
            single_variable_parametrization
        )

        with pytest.raises(
            ValueError, match="atom_id must be a valid atom_id in the sigma algebra"
        ):
            P(2, theta=0)


class TestAtMethod:
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
                3: 1,
            }
        )

    @pytest.fixture
    def single_variable_parametrization(self):
        """Fixture for a single-variable parametrization."""

        def parametrization(theta) -> dict[Hashable, Real]:  # noqa: D103
            if theta == 0:
                return {0: 0.2, 1: 0.8}
            elif theta == 1:
                return {0: 0.5, 1: 0.5}

        return parametrization

    @pytest.fixture
    def multi_variable_parametrization(self):
        """Fixture for a multi-variable parametrization."""

        def parametrization(alpha, beta) -> dict[Hashable, Real]:  # noqa: D103
            if (alpha, beta) == (0, 0):
                return {0: 0.8, 1: 0.2}
            elif (alpha, beta) == (1, 0):
                return {0: 0.5, 1: 0.5}
            elif (alpha, beta) == (0, 1):
                return {0: 0.2, 1: 0.8}
            elif (alpha, beta) == (1, 1):
                return {0: 0.1, 1: 0.9}

        return parametrization

    def test_at_with_single_variable_parametrization(
        self, F, single_variable_parametrization
    ):
        """Test the at method with a single-variable parametrization."""

        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(
            single_variable_parametrization
        )
        expected_P0 = ProbabilityMeasure(sig_alg=F, name="P(theta=0)").from_dict(
            {0: 0.2, 1: 0.8}
        )
        expected_P1 = ProbabilityMeasure(sig_alg=F, name="P(theta=1)").from_dict(
            {0: 0.5, 1: 0.5}
        )

        assert isinstance(P.at(theta=0), ProbabilityMeasure)
        assert isinstance(P.at(theta=1), ProbabilityMeasure)
        assert P.at(theta=0) == expected_P0
        assert P.at(theta=1) == expected_P1

    def test_at_with_multi_variable_parametrization(
        self, F, multi_variable_parametrization
    ):
        """Test the at method with a multi-variable parametrization."""

        Q = ParametrizedProbabilityMeasure(sig_alg=F, name="Q").from_callable(
            multi_variable_parametrization
        )
        expected_Q00 = ProbabilityMeasure(
            sig_alg=F, name="Q(alpha=0, beta=0)"
        ).from_dict({0: 0.8, 1: 0.2})
        expected_Q10 = ProbabilityMeasure(
            sig_alg=F, name="Q(alpha=1, beta=0)"
        ).from_dict({0: 0.5, 1: 0.5})
        expected_Q01 = ProbabilityMeasure(
            sig_alg=F, name="Q(alpha=0, beta=1)"
        ).from_dict({0: 0.2, 1: 0.8})
        expected_Q11 = ProbabilityMeasure(
            sig_alg=F, name="Q(alpha=1, beta=1)"
        ).from_dict({0: 0.1, 1: 0.9})

        assert isinstance(Q.at(alpha=0, beta=0), ProbabilityMeasure)
        assert isinstance(Q.at(alpha=1, beta=0), ProbabilityMeasure)
        assert isinstance(Q.at(alpha=0, beta=1), ProbabilityMeasure)
        assert isinstance(Q.at(alpha=1, beta=1), ProbabilityMeasure)
        assert Q.at(alpha=0, beta=0) == expected_Q00
        assert Q.at(alpha=1, beta=0) == expected_Q10
        assert Q.at(alpha=0, beta=1) == expected_Q01
        assert Q.at(alpha=1, beta=1) == expected_Q11

    def test_at_with_unknown_parameters_raises(
        self, F, single_variable_parametrization
    ):
        """Test that the at method raises an error when unknown parameters are passed in."""
        P = ParametrizedProbabilityMeasure(sig_alg=F).from_callable(
            single_variable_parametrization
        )

        with pytest.raises(ValueError, match="unknown parameters"):
            P.at(theta=0, unknown_parameter=1)
