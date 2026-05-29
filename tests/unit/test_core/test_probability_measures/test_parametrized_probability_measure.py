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


class TestBaseConstructor:
    def test_constructor_with_no_parameters(self):
        """Test the constructor with no parameters."""
        P = ParametrizedProbabilityMeasure()

        assert P.sig_alg is None
        assert P.parameter_domain is None
        assert P.domain is None
        assert P.name == "P"

    def test_constructor_with_sig_alg_and_parameter_domain(self):
        """Test the constructor with sigma-algebra and parameter domain."""
        Omega = SampleSpace().from_sequence(size=2)
        G = SigmaAlgebra.power_set(Omega, name="G")
        parameter_domain = Domain().from_list([0, 1], variable_names=["theta"])
        Q = ParametrizedProbabilityMeasure(
            sig_alg=G, parameter_domain=parameter_domain, name="Q"
        )
        expected_domain = Domain().from_list(
            [(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"]
        )

        assert Q.sig_alg is G
        assert Q.parameter_domain is parameter_domain
        assert Q.domain == expected_domain
        assert Q.name == "Q"

    def test_constructor_with_tuples_for_sig_alg(self):
        """Test the constructor with tuples for sigma-algebra atom identifiers."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: ("a", "b"),
                1: ("c", "d"),
            }
        )
        parameter_domain = Domain().from_list([0, 1], variable_names=["theta"])
        P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=parameter_domain)
        expected_domain = Domain().from_list(
            [(0, "a", "b"), (0, "c", "d"), (1, "a", "b"), (1, "c", "d")],
            variable_names=["theta", "F_0", "F_1"],
        )

        assert P.sig_alg is F
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.name == "P"

    def test_constructor_with_tuples_for_parameter_domain(self):
        """Test the constructor with tuples for parameter domain."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        parameter_domain = Domain().from_list(
            [(0, 1), (1, 2)], variable_names=["alpha", "beta"]
        )
        P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=parameter_domain)
        expected_domain = Domain().from_list(
            [(0, 1, 0), (0, 1, 1), (1, 2, 0), (1, 2, 1)],
            variable_names=["alpha", "beta", "F"],
        )

        assert P.sig_alg is F
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.name == "P"

    def test_constructor_with_tuples_for_sig_alg_and_parameter_domain(self):
        """Test the constructor with tuples for both sigma-algebra and parameter domain."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: ("a", "b"),
                1: ("c", "d"),
            }
        )
        parameter_domain = Domain().from_list(
            [(0, 1), (1, 2)], variable_names=["alpha", "beta"]
        )
        P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=parameter_domain)
        expected_domain = Domain().from_list(
            [(0, 1, "a", "b"), (0, 1, "c", "d"), (1, 2, "a", "b"), (1, 2, "c", "d")],
            variable_names=["alpha", "beta", "F_0", "F_1"],
        )

        assert P.sig_alg is F
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.name == "P"

    def test_constructor_with_sig_alg_and_domain(self):
        """Test the constructor with sigma-algebra and domain."""
        Omega = SampleSpace().from_sequence(size=2)
        G = SigmaAlgebra.power_set(Omega, name="G")
        domain = Domain().from_list(
            [(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"]
        )
        R = ParametrizedProbabilityMeasure(sig_alg=G, domain=domain, name="R")

        assert R.sig_alg is G
        assert R.parameter_domain is None
        assert R.domain == domain
        assert R.name == "R"

    def test_constructor_with_sig_alg_only_raises(self):
        """Test that the constructor with only sigma-algebra raises an exception."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)

        with pytest.raises(
            ValueError,
            match="If sig_alg is given, parameter_domain or domain must also be given",
        ):
            ParametrizedProbabilityMeasure(sig_alg=F)

    def test_constructor_with_domain_only_raises(self):
        """Test that the constructor with only domain raises an exception."""
        domain = Domain().from_list(
            [(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"]
        )

        with pytest.raises(
            ValueError,
            match="If domain is given, sig_alg must also be given",
        ):
            ParametrizedProbabilityMeasure(domain=domain)

    def test_constructor_with_parameter_domain_and_domain_raises(self):
        """Test that the constructor with both parameter domain and domain raises an exception."""
        parameter_domain = Domain().from_list([0, 1], variable_names=["theta"])
        domain = Domain().from_list(
            [(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"]
        )

        with pytest.raises(
            ValueError,
            match="If parameter_domain is given, sig_alg must be given and domain must be None",
        ):
            ParametrizedProbabilityMeasure(
                parameter_domain=parameter_domain, domain=domain
            )

    def test_constructor_with_all_parameters_raises(self):
        """Test that the constructor with all parameters raises an exception."""
        Omega = SampleSpace().from_sequence(size=2)
        G = SigmaAlgebra.power_set(Omega, name="G")
        parameter_domain = Domain().from_list([0, 1], variable_names=["theta"])
        domain = Domain().from_list(
            [(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"]
        )

        with pytest.raises(
            ValueError,
            match="If sig_alg and parameter_domain are given, domain must be None",
        ):
            ParametrizedProbabilityMeasure(
                sig_alg=G, parameter_domain=parameter_domain, domain=domain
            )


class TestFromCallable:
    def test_with_sig_alg_and_parameter_domain_at_construction(self):
        """Test the from_callable method with sigma-algebra and parameter domain at construction."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: ("a", "b"),
                1: ("c", "d"),
            }
        )
        parameter_domain = Domain().from_list([0, 1], variable_names=["theta"])

        def P_func(*, theta, F_0, F_1):
            if (theta, F_0, F_1) == (0, "a", "b"):
                return 0.75
            elif (theta, F_0, F_1) == (0, "c", "d"):
                return 0.25
            elif (theta, F_0, F_1) == (1, "a", "b"):
                return 0.4
            elif (theta, F_0, F_1) == (1, "c", "d"):
                return 0.6

        P = ParametrizedProbabilityMeasure(
            sig_alg=F, parameter_domain=parameter_domain
        ).from_callable(function=P_func)

        expected_domain = Domain().from_list(
            [(0, "a", "b"), (0, "c", "d"), (1, "a", "b"), (1, "c", "d")],
            variable_names=["theta", "F_0", "F_1"],
        )
        expected_data = pd.Series(
            [0.75, 0.25, 0.4, 0.6],
            index=expected_domain.data,
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
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.function is P_func
        pd.testing.assert_series_equal(P.data, expected_data)
        assert P.dict == expected_dict
        assert P.argument_names == ["theta", "F_0", "F_1"]
        assert P.signature == expected_signature
        assert P.num_arguments == 3
        assert P.output_name == "probability"

    def test_with_sig_alg_and_domain_at_construction(self):
        """Test the from_callable method with sigma-algebra and domain at construction."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: ("a", "b"),
                1: ("c", "d"),
            }
        )
        domain = Domain().from_list(
            [(0, "a", "b"), (0, "c", "d"), (1, "a", "b"), (1, "c", "d")],
            variable_names=["theta", "F_0", "F_1"],
        )

        def P_func(*, theta, F_0, F_1):
            if (theta, F_0, F_1) == (0, "a", "b"):
                return 0.75
            elif (theta, F_0, F_1) == (0, "c", "d"):
                return 0.25
            elif (theta, F_0, F_1) == (1, "a", "b"):
                return 0.4
            elif (theta, F_0, F_1) == (1, "c", "d"):
                return 0.6

        P = ParametrizedProbabilityMeasure(sig_alg=F, domain=domain).from_callable(
            function=P_func
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
        assert P.function is P_func
        pd.testing.assert_series_equal(P.data, expected_data)
        assert P.dict == expected_dict
        assert P.argument_names == ["theta", "F_0", "F_1"]
        assert P.signature == expected_signature
        assert P.num_arguments == 3
        assert P.output_name == "probability"

    def test_with_callable_missing_required_parameters_raises(self):
        """Test that from_callable raises an exception when the callable is missing atom identifier parameters."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: ("a", "b"),
                1: ("c", "d"),
            }
        )
        parameter_domain = Domain().from_list([0, 1], variable_names=["theta"])

        def P_func(*, theta, F_0):
            if (theta, F_0) == (0, "a"):
                return 0.75
            elif (theta, F_0) == (0, "c"):
                return 0.25
            elif (theta, F_0) == (1, "a"):
                return 0.4
            elif (theta, F_0) == (1, "c"):
                return 0.6

        P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=parameter_domain)

        with pytest.raises(
            ValueError,
            match="The provided callable must accept keyword-only arguments corresponding to the atom identifiers of the sigma-algebra",
        ):
            P.from_callable(function=P_func)


# --------------------- test data access methods --------------------- #


class TestCall:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F_1D(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: "a",
                1: "b",
                2: "b",
                3: "c",
            }
        )

    @pytest.fixture
    def F_2D(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: ("a", "a"),
                1: ("a", "b"),
                2: ("b", "c"),
                3: ("b", "c"),
            }
        )

    @pytest.fixture
    def parameter_domain_1D(self):
        return Domain().from_list([0, 1], variable_names=["theta"])

    @pytest.fixture
    def parameter_domain_2D(self):
        return Domain().from_list([(0, 0), (0, 1), (1, 1)], variable_names=["alpha", "beta"])

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

    def test_with_event_as_positional_arg(
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
        """Test the __call__ method with event as a positional argument."""

        A = F_1D.get_event([1, 2, 3])
        B = F_2D.get_event([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["theta"]), name="P(A)"
        ).from_callable(
            lambda *, theta: 0.25 if theta == 0 else 0.6, output_name="probability"
        )
        assert P(A) == expected_result

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(A)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P(A) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
        ).from_callable(
            lambda *, theta: 0.9 if theta == 0 else 0.7, output_name="probability"
        )
        assert P(B) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(B)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P(B) == expected_result

    def test_with_event_as_keyword_arg(
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
        """Test the __call__ method with event as a keyword argument."""

        A = F_1D.get_event([1, 2, 3])
        B = F_2D.get_event([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["theta"]), name="P(A)"
        ).from_callable(
            lambda *, theta: 0.25 if theta == 0 else 0.6, output_name="probability"
        )
        assert P(event=A) == expected_result

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(A)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P(event=A) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
        ).from_callable(
            lambda *, theta: 0.9 if theta == 0 else 0.7, output_name="probability"
        )
        assert P(event=B) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(B)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P(event=B) == expected_result

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
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["theta"]), name="P(A)"
        ).from_callable(
            lambda *, theta: 0.25 if theta == 0 else 0.6, output_name="probability"
        )
        assert P([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(A)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
        ).from_callable(
            lambda *, theta: 0.9 if theta == 0 else 0.7, output_name="probability"
        )
        assert P([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(B)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P([1, 2, 3]) == expected_result

    def test_with_list_as_keyword_arg(
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
        """Test the __call__ method with list as a keyword argument."""
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["theta"]), name="P(A)"
        ).from_callable(
            lambda *, theta: 0.25 if theta == 0 else 0.6, output_name="probability"
        )
        assert P(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(A)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
        ).from_callable(
            lambda *, theta: 0.9 if theta == 0 else 0.7, output_name="probability"
        )
        assert P(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_2D,
            name="P(B)",
        ).from_callable(
            lambda *, alpha, beta: (
                0.9
                if (alpha, beta) == (0, 0)
                else 0.7
                if (alpha, beta) == (0, 1)
                else 0.5
            ),
            output_name="probability",
        )
        assert P(event=[1, 2, 3]) == expected_result

    def test_with_event_as_positional_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with event as a positional argument and partial parameters."""
        A = F_1D.get_event([1, 2, 3])
        B = F_2D.get_event([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(A, alpha=0) == expected_result
        assert P(A)(alpha=0) == expected_result
        assert P(alpha=0)(A) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(B, alpha=0) == expected_result
        assert P(B)(alpha=0) == expected_result
        assert P(alpha=0)(B) == expected_result

    def test_with_event_as_keyword_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with event as a keyword argument and partial parameters."""
        A = F_1D.get_event([1, 2, 3])
        B = F_2D.get_event([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=A, alpha=0) == expected_result
        assert P(event=A)(alpha=0) == expected_result
        assert P(alpha=0)(event=A) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=B, alpha=0) == expected_result
        assert P(event=B)(alpha=0) == expected_result
        assert P(alpha=0)(event=B) == expected_result

    def test_with_list_as_positional_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with list as a positional argument and partial parameters."""
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P([1, 2, 3], alpha=0) == expected_result
        assert P([1, 2, 3])(alpha=0) == expected_result
        assert P(alpha=0)([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P([1, 2, 3], alpha=0) == expected_result
        assert P([1, 2, 3])(alpha=0) == expected_result
        assert P(alpha=0)([1, 2, 3]) == expected_result

    def test_with_list_as_keyword_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with list as a keyword argument and partial parameters."""
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=[1, 2, 3], alpha=0) == expected_result
        assert P(event=[1, 2, 3])(alpha=0) == expected_result
        assert P(alpha=0)(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain().from_list([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
        ).from_callable(
            lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=[1, 2, 3], alpha=0) == expected_result
        assert P(event=[1, 2, 3])(alpha=0) == expected_result
        assert P(alpha=0)(event=[1, 2, 3]) == expected_result

    def test_with_event_as_positional_and_all_parameters(
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

        A = F_1D.get_event([1, 2, 3])
        B = F_2D.get_event([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        assert P(A, theta=0) == pytest.approx(0.25)
        assert P(A)(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(A) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
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

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        assert P(B, theta=0) == pytest.approx(0.9)
        assert P(B)(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(B) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
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

    def test_with_event_as_keyword_and_all_parameters(
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
        """Test the __call__ method with event as a keyword argument and all parameters."""

        A = F_1D.get_event([1, 2, 3])
        B = F_2D.get_event([1, 2, 3], name="B")

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        assert P(event=A, theta=0) == pytest.approx(0.25)
        assert P(event=A)(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(event=A) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        assert P(event=A, alpha=0, beta=0) == pytest.approx(0.9)
        assert P(event=A)(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)(event=A) == pytest.approx(0.9)
        assert P(event=A, alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=A, alpha=0) == pytest.approx(0.9)
        assert P(event=A, beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=A, beta=0) == pytest.approx(0.9)
        assert P(event=A)(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(event=A)(beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=A)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=A)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)(event=A) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)(event=A) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        assert P(event=B, theta=0) == pytest.approx(0.9)
        assert P(event=B)(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(event=B) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        assert P(event=B, alpha=0, beta=0) == pytest.approx(0.9)
        assert P(event=B)(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)(event=B) == pytest.approx(0.9)
        assert P(event=B, alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=B, alpha=0) == pytest.approx(0.9)
        assert P(event=B, beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=B, beta=0) == pytest.approx(0.9)
        assert P(event=B)(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(event=B)(beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=B)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=B)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)(event=B) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)(event=B) == pytest.approx(0.9)

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
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.25)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.25)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
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

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.9)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
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

    def test_with_list_as_keyword_and_all_parameters(
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
        """Test the __call__ method with list as a keyword argument and all parameters."""
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        assert P(event=[1, 2, 3], theta=0) == pytest.approx(0.25)
        assert P(event=[1, 2, 3])(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(event=[1, 2, 3]) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        assert P(event=[1, 2, 3], alpha=0, beta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)(event=[1, 2, 3]) == pytest.approx(0.9)
        assert P(event=[1, 2, 3], alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=[1, 2, 3], alpha=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3], beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=[1, 2, 3], beta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=[1, 2, 3])(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=[1, 2, 3])(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)(event=[1, 2, 3]) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)(event=[1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        assert P(event=[1, 2, 3], theta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(event=[1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        assert P(event=[1, 2, 3], alpha=0, beta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(alpha=0, beta=0) == pytest.approx(0.9)
        assert P(alpha=0, beta=0)(event=[1, 2, 3]) == pytest.approx(0.9)
        assert P(event=[1, 2, 3], alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=[1, 2, 3], alpha=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3], beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=[1, 2, 3], beta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(alpha=0)(beta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(beta=0)(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(event=[1, 2, 3])(beta=0) == pytest.approx(0.9)
        assert P(beta=0)(event=[1, 2, 3])(alpha=0) == pytest.approx(0.9)
        assert P(alpha=0)(beta=0)(event=[1, 2, 3]) == pytest.approx(0.9)
        assert P(beta=0)(alpha=0)(event=[1, 2, 3]) == pytest.approx(0.9)

    def test_with_partial_parameters_and_no_event(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with partial parameters and no event."""
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
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
            F_1D,
            domain=Domain().from_list(
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
        ).from_callable(expected_callable, output_name="probability")
        assert P(alpha=0) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
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
            F_2D,
            domain=Domain().from_list(
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
        ).from_callable(expected_callable, output_name="probability")
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
        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_1D).from_callable(
            P_func_2D
        )
        expected_result = ProbabilityMeasure(F_1D, name="P(theta=0)").from_dict(
            {
                "a": 0.75,
                "b": 0.15,
                "c": 0.10,
            }
        )
        assert P(theta=0) == expected_result

        P = ParametrizedProbabilityMeasure(F_1D, parameter_domain_2D).from_callable(
            P_func_2D_parameter_domain_1D_sig_alg
        )
        expected_result = ProbabilityMeasure(F_1D, name="P(alpha=0, beta=0)").from_dict(
            {
                "a": 0.1,
                "b": 0.7,
                "c": 0.2,
            }
        )
        assert P(alpha=0, beta=0) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_1D).from_callable(
            P_func_1D_parameter_domain_2D_sig_alg
        )
        expected_result = ProbabilityMeasure(F_2D, name="P(theta=0)").from_dict(
            {
                ("a", "a"): 0.1,
                ("a", "b"): 0.2,
                ("b", "c"): 0.7,
            }
        )
        assert P(theta=0) == expected_result

        P = ParametrizedProbabilityMeasure(F_2D, parameter_domain_2D).from_callable(
            P_func_4D
        )
        expected_result = ProbabilityMeasure(F_2D, name="P(alpha=0, beta=0)").from_dict(
            {
                ("a", "a"): 0.1,
                ("a", "b"): 0.2,
                ("b", "c"): 0.7,
            }
        )
        assert P(alpha=0, beta=0) == expected_result
