import numpy as np
import pytest
from sigalg.core import (
    Domain,
    Function,
    ParametrizedProbabilityMeasure,
    ProbabilityMeasure,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test data access methods --------------------- #


class TestCall:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F_1D(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: "a",
                1: "b",
                2: "b",
                3: "c",
            },
            variable_names=["F"],
        )

    @pytest.fixture
    def F_2D(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: ("a", "a"),
                1: ("a", "b"),
                2: ("b", "c"),
                3: ("b", "c"),
            },
            variable_names=["F_0", "F_1"],
        )

    @pytest.fixture
    def parameter_domain_1D(self):
        return Domain([0, 1], variable_names=["theta"])

    @pytest.fixture
    def parameter_domain_2D(self):
        return Domain([(0, 0), (0, 1), (1, 1)], variable_names=["alpha", "beta"])

    @pytest.fixture
    def P_func_2D(self):
        def P_func(*, theta, F):
            if (theta, F) == (0, "a"):
                return 0.75
            elif (theta, F) == (0, "b"):
                return 0.15
            elif (theta, F) == (0, "c"):
                return 0.10
            elif (theta, F) == (1, "a"):
                return 0.4
            elif (theta, F) == (1, "b"):
                return 0.2
            elif (theta, F) == (1, "c"):
                return 0.4

        return P_func

    @pytest.fixture
    def P_func_2D_parameter_domain_1D_sig_alg(self):
        def P_func(*, alpha, beta, F):
            if (alpha, beta, F) == (0, 0, "a"):
                return 0.1
            elif (alpha, beta, F) == (0, 0, "b"):
                return 0.7
            elif (alpha, beta, F) == (0, 0, "c"):
                return 0.2
            elif (alpha, beta, F) == (0, 1, "a"):
                return 0.3
            elif (alpha, beta, F) == (0, 1, "b"):
                return 0.5
            elif (alpha, beta, F) == (0, 1, "c"):
                return 0.2
            elif (alpha, beta, F) == (1, 1, "a"):
                return 0.5
            elif (alpha, beta, F) == (1, 1, "b"):
                return 0.1
            elif (alpha, beta, F) == (1, 1, "c"):
                return 0.4

        return P_func

    @pytest.fixture
    def P_func_1D_parameter_domain_2D_sig_alg(self):
        def P_func(*, theta, F_0, F_1):
            if (theta, F_0, F_1) == (0, "a", "a"):
                return 0.1
            elif (theta, F_0, F_1) == (0, "a", "b"):
                return 0.2
            elif (theta, F_0, F_1) == (0, "b", "c"):
                return 0.7
            elif (theta, F_0, F_1) == (1, "a", "a"):
                return 0.3
            elif (theta, F_0, F_1) == (1, "a", "b"):
                return 0.3
            elif (theta, F_0, F_1) == (1, "b", "c"):
                return 0.4

        return P_func

    @pytest.fixture
    def P_func_4D(self):
        def P_func(*, alpha, beta, F_0, F_1):
            if (alpha, beta, F_0, F_1) == (0, 0, "a", "a"):
                return 0.1
            elif (alpha, beta, F_0, F_1) == (0, 0, "a", "b"):
                return 0.2
            elif (alpha, beta, F_0, F_1) == (0, 0, "b", "c"):
                return 0.7
            elif (alpha, beta, F_0, F_1) == (0, 1, "a", "a"):
                return 0.3
            elif (alpha, beta, F_0, F_1) == (0, 1, "a", "b"):
                return 0.3
            elif (alpha, beta, F_0, F_1) == (0, 1, "b", "c"):
                return 0.4
            elif (alpha, beta, F_0, F_1) == (1, 1, "a", "a"):
                return 0.5
            elif (alpha, beta, F_0, F_1) == (1, 1, "a", "b"):
                return 0.3
            elif (alpha, beta, F_0, F_1) == (1, 1, "b", "c"):
                return 0.2

        return P_func

    def test_with_set_as_positional_arg(
        self,
        F_1D,
        F_2D,
        parameter_domain_1D,
        parameter_domain_2D,
        P_func_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_1D_parameter_domain_2D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with measurable_set as a positional argument."""

        A = F_1D.get_set([1, 2, 3])
        B = F_2D.get_set([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = Function(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert np.allclose(P(A), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = Function(
            domain=parameter_domain_2D,
            name="P(A)",
            mapping=lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert np.allclose(P(A), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = Function(
            domain=parameter_domain_1D,
            name="P(B)",
            mapping=lambda *, theta: 0.9 if theta == 0 else 0.7,
            output_name="probability",
        )
        assert np.allclose(P(B), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = Function(
            domain=parameter_domain_2D,
            name="P(B)",
            mapping=lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert np.allclose(P(B), expected_result)

    def test_with_list_as_positional_arg(
        self,
        F_1D,
        F_2D,
        parameter_domain_1D,
        parameter_domain_2D,
        P_func_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_1D_parameter_domain_2D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with list as a positional argument."""
        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = Function(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert np.allclose(P([1, 2, 3]), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = Function(
            domain=parameter_domain_2D,
            name="P(A)",
            mapping=lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert np.allclose(P([1, 2, 3]), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = Function(
            domain=parameter_domain_1D,
            name="P(B)",
            mapping=lambda *, theta: 0.9 if theta == 0 else 0.7,
            output_name="probability",
        )
        assert np.allclose(P([1, 2, 3]), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = Function(
            domain=parameter_domain_2D,
            name="P(B)",
            mapping=lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert np.allclose(P([1, 2, 3]), expected_result)

    def test_with_measurable_set_as_positional_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with measurable set as a positional argument and partial parameters."""
        A = F_1D.get_set([1, 2, 3])
        B = F_2D.get_set([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = Function(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert np.allclose(P(A, alpha=0), expected_result)
        assert np.allclose(P(A)(alpha=0), expected_result)
        assert np.allclose(P(alpha=0)(A), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = Function(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert np.allclose(P(B, alpha=0), expected_result)
        assert np.allclose(P(B)(alpha=0), expected_result)
        assert np.allclose(P(alpha=0)(B), expected_result)

    def test_with_list_as_positional_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with list as a positional argument and partial parameters."""
        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = Function(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert np.allclose(P([1, 2, 3], alpha=0), expected_result)
        assert np.allclose(P([1, 2, 3])(alpha=0), expected_result)
        assert np.allclose(P(alpha=0)([1, 2, 3]), expected_result)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = Function(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert np.allclose(P([1, 2, 3], alpha=0), expected_result)
        assert np.allclose(P([1, 2, 3])(alpha=0), expected_result)
        assert np.allclose(P(alpha=0)([1, 2, 3]), expected_result)

    def test_with_measurable_set_as_positional_and_all_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_1D,
        parameter_domain_2D,
        P_func_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_1D_parameter_domain_2D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with event as a positional argument and all parameters."""

        A = F_1D.get_set([1, 2, 3])
        B = F_2D.get_set([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P(A, theta=0) == pytest.approx(0.25)
        assert P(A)(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(A) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        assert P(A, alpha=0, beta=0) == pytest.approx(0.9)
        assert P(A)(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)(A) == pytest.approx(0.9)
        assert P(A, alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(A, alpha=0) == pytest.approx(0.9)
        assert P(A, beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(A, beta=0) == pytest.approx(0.9)
        assert P(A)(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(alpha=0)(A)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(A)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)(A) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)(A) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P(B, theta=0) == pytest.approx(0.9)
        assert P(B)(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(B) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        assert P(B, alpha=0, beta=0) == pytest.approx(0.9)
        assert P(B)(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)(B) == pytest.approx(0.9)
        assert P(B, alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(B, alpha=0) == pytest.approx(0.9)
        assert P(B, beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(B, beta=0) == pytest.approx(0.9)
        assert P(B)(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(alpha=0)(B)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(B)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)(B) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)(B) == pytest.approx(0.9)

    def test_with_list_as_positional_and_all_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_1D,
        parameter_domain_2D,
        P_func_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_1D_parameter_domain_2D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with list as a positional argument and all parameters."""
        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.25)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.25)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        assert P([1, 2, 3], alpha=0, beta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)([1, 2, 3]) == pytest.approx(0.9)
        assert P([1, 2, 3], alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)([1, 2, 3], alpha=0) == pytest.approx(0.9)
        assert P([1, 2, 3], beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)([1, 2, 3], beta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(alpha=0)([1, 2, 3])(beta=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)([1, 2, 3]) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)([1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.9)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        assert P([1, 2, 3], alpha=0, beta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)([1, 2, 3]) == pytest.approx(0.9)
        assert P([1, 2, 3], alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)([1, 2, 3], alpha=0) == pytest.approx(0.9)
        assert P([1, 2, 3], beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)([1, 2, 3], beta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(alpha=0)([1, 2, 3])(beta=0) == pytest.approx(0.9)
        assert P(beta=0)([1, 2, 3])(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)([1, 2, 3]) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)([1, 2, 3]) == pytest.approx(0.9)

    def test_with_partial_parameters_and_no_event(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with partial parameters and no event."""
        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )

        def expected_callable(*, beta, F):
            if (beta, F) == (0, "a"):
                return 0.1
            elif (beta, F) == (0, "b"):
                return 0.7
            elif (beta, F) == (0, "c"):
                return 0.2
            elif (beta, F) == (1, "a"):
                return 0.3
            elif (beta, F) == (1, "b"):
                return 0.5
            elif (beta, F) == (1, "c"):
                return 0.2

        parameter_domain = Domain([0, 1], variable_names=["beta"])

        expected_result = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain,
            name="P(alpha=0)",
            mapping=expected_callable,
        )
        assert P(alpha=0) == expected_result

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )

        def expected_callable(*, beta, F_0, F_1):
            if (beta, F_0, F_1) == (0, "a", "a"):
                return 0.1
            elif (beta, F_0, F_1) == (0, "a", "b"):
                return 0.2
            elif (beta, F_0, F_1) == (0, "b", "c"):
                return 0.7
            elif (beta, F_0, F_1) == (1, "a", "a"):
                return 0.3
            elif (beta, F_0, F_1) == (1, "a", "b"):
                return 0.3
            elif (beta, F_0, F_1) == (1, "b", "c"):
                return 0.4

        expected_result = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D,
            parameter_domain=parameter_domain,
            name="P(alpha=0)",
            mapping=expected_callable,
        )
        assert P(alpha=0) == expected_result

    def test_with_all_parameters_and_no_event(
        self,
        F_1D,
        F_2D,
        parameter_domain_1D,
        parameter_domain_2D,
        P_func_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_1D_parameter_domain_2D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with all parameters and no event."""
        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = ProbabilityMeasure(
            domain=F_1D,
            name="P(theta=0)",
            mapping={
                "a": 0.75,
                "b": 0.15,
                "c": 0.10,
            },
        )
        assert P(theta=0) == expected_result

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = ProbabilityMeasure(
            domain=F_1D,
            name="P(alpha=0, beta=0)",
            mapping={
                "a": 0.1,
                "b": 0.7,
                "c": 0.2,
            },
        )
        assert P(alpha=0, beta=0) == expected_result

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = ProbabilityMeasure(
            domain=F_2D,
            name="P(theta=0)",
            mapping={
                ("a", "a"): 0.1,
                ("a", "b"): 0.2,
                ("b", "c"): 0.7,
            },
        )
        assert P(theta=0) == expected_result

        P = ParametrizedProbabilityMeasure.from_domains(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = ProbabilityMeasure(
            domain=F_2D,
            name="P(alpha=0, beta=0)",
            mapping={
                ("a", "a"): 0.1,
                ("a", "b"): 0.2,
                ("b", "c"): 0.7,
            },
        )
        assert P(alpha=0, beta=0) == expected_result
