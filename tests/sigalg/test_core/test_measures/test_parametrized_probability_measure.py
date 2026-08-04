import inspect

import pandas as pd
import pytest

from sigalg.core import (
    Domain,
    MultivariateFunction,
    ParametrizedProbabilityMeasure,
    ProbabilityMeasure,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_with_no_parameters(self):
        """Test the constructor with no parameters."""
        P = ParametrizedProbabilityMeasure()

        assert P.sig_alg is None
        assert P.parameter_domain is None
        assert P.domain is None
        assert P.name == "P"

    def test_constructor_with_sig_alg_and_parameter_domain(self):
        """Test the constructor with sigma-algebra and parameter domain."""
        Omega = SampleSpace.from_sequence(size=2)
        G = SigmaAlgebra.power_set(Omega, name="G")
        parameter_domain = Domain([0, 1], variable_names=["theta"])
        Q = ParametrizedProbabilityMeasure(
            measure_domain=G,
            parameter_domain=parameter_domain,
            name="Q",
        )
        expected_domain = Domain(
            [(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "sample"]
        )

        assert Q.measure_domain is G.atom_space
        assert Q.parameter_domain is parameter_domain
        assert Q.domain == expected_domain
        assert Q.name == "Q"

    def test_constructor_with_tuples_for_parameter_domain(self):
        """Test the constructor with tuples for parameter domain."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        parameter_domain = Domain([(0, 1), (1, 2)], variable_names=["alpha", "beta"])
        P = ParametrizedProbabilityMeasure(
            measure_domain=F, parameter_domain=parameter_domain
        )
        expected_domain = Domain(
            [(0, 1, 0), (0, 1, 1), (1, 2, 0), (1, 2, 1)],
            variable_names=["alpha", "beta", "sample"],
        )

        assert P.sig_alg is F
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.name == "P"

    def test_constructor_with_domain_only_raises(self):
        """Test that the constructor with only domain raises an exception."""
        domain = Domain([(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"])

        with pytest.raises(
            ValueError,
            match="If domain is given",
        ):
            ParametrizedProbabilityMeasure(domain=domain)

    def test_constructor_with_parameter_domain_and_domain_raises(self):
        """Test that the constructor with both parameter domain and domain raises an exception."""
        parameter_domain = Domain([0, 1], variable_names=["theta"])
        domain = Domain([(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"])

        with pytest.raises(
            ValueError,
            match="If parameter_domain is given, the measure_domain must be given and domain must be None",
        ):
            ParametrizedProbabilityMeasure(
                parameter_domain=parameter_domain, domain=domain
            )

    def test_constructor_with_all_parameters_raises(self):
        """Test that the constructor with all parameters raises an exception."""
        Omega = SampleSpace.from_sequence(size=2)
        G = SigmaAlgebra.power_set(Omega, name="G")
        parameter_domain = Domain([0, 1], variable_names=["theta"])
        domain = Domain([(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"])

        with pytest.raises(
            ValueError,
            match="domain must be None.",
        ):
            ParametrizedProbabilityMeasure(
                measure_domain=G, parameter_domain=parameter_domain, domain=domain
            )

    def test_with_sig_alg_and_domain_at_construction(self):
        """Test the from_callable method with sigma-algebra and domain at construction."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra(
            domain=Omega,
            mapping={
                0: ("a", "b"),
                1: ("c", "d"),
            },
            variable_names=["F_0", "F_1"],
        )
        domain = Domain(
            [(0, "a", "b"), (0, "c", "d"), (1, "a", "b"), (1, "c", "d")],
            variable_names=["theta", "F_0", "F_1"],
        )

        def mapping(*, theta, F_0, F_1):
            if (theta, F_0, F_1) == (0, "a", "b"):
                return 0.75
            elif (theta, F_0, F_1) == (0, "c", "d"):
                return 0.25
            elif (theta, F_0, F_1) == (1, "a", "b"):
                return 0.4
            elif (theta, F_0, F_1) == (1, "c", "d"):
                return 0.6

        P = ParametrizedProbabilityMeasure(
            measure_domain=F, domain=domain, mapping=mapping
        )

        expected_data = pd.Series(
            [0.75, 0.25, 0.4, 0.6],
            index=domain.data,
            name="probability",
        )
        expected_dict = {
            (0, "a", "b"): 0.75,
            (0, "c", "d"): 0.25,
            (1, "a", "b"): 0.4,
            (1, "c", "d"): 0.6,
        }
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in ["theta", "F_0", "F_1"]
        ]
        expected_signature = inspect.Signature(parameters=expected_parameters)

        assert P.sig_alg is F
        assert P.parameter_domain is None
        assert P.domain == domain
        assert P.function is mapping
        pd.testing.assert_series_equal(P.data, expected_data)
        assert P.dict == expected_dict
        assert P.variable_names == ["theta", "F_0", "F_1"]
        assert P.signature == expected_signature
        assert P.num_variables == 3
        assert P.output_name == "probability"


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

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert P(A) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = MultivariateFunction(
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
        assert P(A) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
            mapping=lambda *, theta: 0.9 if theta == 0 else 0.7,
            output_name="probability",
        )
        assert P(B) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = MultivariateFunction(
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
        assert P(B) == expected_result

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
        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert P([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = MultivariateFunction(
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
        assert P([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
            mapping=lambda *, theta: 0.9 if theta == 0 else 0.7,
            output_name="probability",
        )
        assert P([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = MultivariateFunction(
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
        assert P([1, 2, 3]) == expected_result

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

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(A, alpha=0) == expected_result
        assert P(A)(alpha=0) == expected_result
        assert P(alpha=0)(A) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(B, alpha=0) == expected_result
        assert P(B)(alpha=0) == expected_result
        assert P(alpha=0)(B) == expected_result

    def test_with_list_as_positional_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with list as a positional argument and partial parameters."""
        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P([1, 2, 3], alpha=0) == expected_result
        assert P([1, 2, 3])(alpha=0) == expected_result
        assert P(alpha=0)([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P([1, 2, 3], alpha=0) == expected_result
        assert P([1, 2, 3])(alpha=0) == expected_result
        assert P(alpha=0)([1, 2, 3]) == expected_result

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

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P(A, theta=0) == pytest.approx(0.25)
        assert P(A)(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(A) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(
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
        assert P(A)(beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(A)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(A)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)(A) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)(A) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P(B, theta=0) == pytest.approx(0.9)
        assert P(B)(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(B) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
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
        assert P(B)(beta=0)(alpha=0) == pytest.approx(0.9)
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
        P = ParametrizedProbabilityMeasure(
            measure_domain=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.25)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.25)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(
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
        assert P([1, 2, 3])(beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)([1, 2, 3])(beta=0) == pytest.approx(0.9)
        assert P(beta=0)([1, 2, 3])(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)([1, 2, 3]) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)([1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
            measure_domain=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.9)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
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
        assert P([1, 2, 3])(beta=0)(alpha=0) == pytest.approx(0.9)
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
        P = ParametrizedProbabilityMeasure(
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

        expected_result = ParametrizedProbabilityMeasure(
            measure_domain=F_1D,
            domain=Domain(
                [
                    (0, "a"),
                    (0, "b"),
                    (0, "c"),
                    (1, "a"),
                    (1, "b"),
                    (1, "c"),
                ],
                variable_names=["beta", "F"],
            ),
            name="P(alpha=0)",
            mapping=expected_callable,
        )
        assert P(alpha=0) == expected_result

        P = ParametrizedProbabilityMeasure(
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

        expected_result = ParametrizedProbabilityMeasure(
            measure_domain=F_2D,
            domain=Domain(
                [
                    (0, "a", "a"),
                    (0, "a", "b"),
                    (0, "b", "c"),
                    (1, "a", "a"),
                    (1, "a", "b"),
                    (1, "b", "c"),
                ],
                variable_names=["beta", "F_0", "F_1"],
            ),
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
        P = ParametrizedProbabilityMeasure(
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

        P = ParametrizedProbabilityMeasure(
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

        P = ParametrizedProbabilityMeasure(
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

        P = ParametrizedProbabilityMeasure(
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
