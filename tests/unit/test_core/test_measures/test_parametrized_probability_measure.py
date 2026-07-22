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
            sig_alg=G, parameter_domain=parameter_domain, name="Q"
        )
        expected_domain = Domain(
            [(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "sample"]
        )

        assert Q.sig_alg is G
        assert Q.parameter_domain is parameter_domain
        assert Q.domain == expected_domain
        assert Q.name == "Q"

    def test_constructor_with_tuples_for_sig_alg(self):
        """Test the constructor with tuples for sigma-algebra atom identifiers."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "b"),
                1: ("c", "d"),
            },
            variable_names=["F_0", "F_1"],
        )
        parameter_domain = Domain([0, 1], variable_names=["theta"])
        P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=parameter_domain)
        expected_domain = Domain(
            [(0, "a", "b"), (0, "c", "d"), (1, "a", "b"), (1, "c", "d")],
            variable_names=["theta", "F_0", "F_1"],
        )

        assert P.sig_alg is F
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.name == "P"

    def test_constructor_with_tuples_for_parameter_domain(self):
        """Test the constructor with tuples for parameter domain."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        parameter_domain = Domain([(0, 1), (1, 2)], variable_names=["alpha", "beta"])
        P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=parameter_domain)
        expected_domain = Domain(
            [(0, 1, 0), (0, 1, 1), (1, 2, 0), (1, 2, 1)],
            variable_names=["alpha", "beta", "sample"],
        )

        assert P.sig_alg is F
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.name == "P"

    def test_constructor_with_tuples_for_sig_alg_and_parameter_domain(self):
        """Test the constructor with tuples for both sigma-algebra and parameter domain."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "b"),
                1: ("c", "d"),
            },
            variable_names=["F_0", "F_1"],
        )
        parameter_domain = Domain([(0, 1), (1, 2)], variable_names=["alpha", "beta"])
        P = ParametrizedProbabilityMeasure(sig_alg=F, parameter_domain=parameter_domain)
        expected_domain = Domain(
            [(0, 1, "a", "b"), (0, 1, "c", "d"), (1, 2, "a", "b"), (1, 2, "c", "d")],
            variable_names=["alpha", "beta", "F_0", "F_1"],
        )

        assert P.sig_alg is F
        assert P.parameter_domain is parameter_domain
        assert P.domain == expected_domain
        assert P.name == "P"

    def test_constructor_with_sig_alg_and_domain(self):
        """Test the constructor with sigma-algebra and domain."""
        Omega = SampleSpace.from_sequence(size=2)
        G = SigmaAlgebra.power_set(Omega, name="G")
        domain = Domain([(0, 0), (0, 1), (1, 0), (1, 1)], variable_names=["theta", "G"])
        R = ParametrizedProbabilityMeasure(sig_alg=G, domain=domain, name="R")

        assert R.sig_alg is G
        assert R.parameter_domain is None
        assert R.domain == domain
        assert R.name == "R"

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
            match="If parameter_domain is given, the space",
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
                sig_alg=G, parameter_domain=parameter_domain, domain=domain
            )

    def test_with_sig_alg_and_parameter_domain_at_construction(self):
        """Test the from_callable method with sigma-algebra and parameter domain at construction."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "b"),
                1: ("c", "d"),
            },
            variable_names=["F_0", "F_1"],
        )
        parameter_domain = Domain([0, 1], variable_names=["theta"])

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
            sig_alg=F, parameter_domain=parameter_domain, mapping=mapping
        )

        expected_domain = Domain(
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
        assert P.fun is mapping
        pd.testing.assert_series_equal(P.data, expected_data)
        assert P.dict == expected_dict
        assert P.argument_names == ["theta", "F_0", "F_1"]
        assert P.signature == expected_signature
        assert P.num_arguments == 3
        assert P.output_name == "probability"

    def test_with_sig_alg_and_domain_at_construction(self):
        """Test the from_callable method with sigma-algebra and domain at construction."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra(
            sample_space=Omega,
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

        P = ParametrizedProbabilityMeasure(sig_alg=F, domain=domain, mapping=mapping)

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
        assert P.fun is mapping
        pd.testing.assert_series_equal(P.data, expected_data)
        assert P.dict == expected_dict
        assert P.argument_names == ["theta", "F_0", "F_1"]
        assert P.signature == expected_signature
        assert P.num_arguments == 3
        assert P.output_name == "probability"


# --------------------- test properties --------------------- #


class TestSigAlg:
    def test_sig_alg_setter_with_1_dim_sig_alg_and_2_dim_parameter_domain(self):
        """Test the sig_alg setter with a 1-dimensional sigma-algebra and a 2-dimensional parameter domain."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 1,
                1: 0,
                2: 1,
                3: 2,
            },
            variable_names=["x"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 1,
                1: 1,
                2: 1,
                3: 2,
            },
            variable_names=["y"],
            name="G",
        )
        Theta = Domain(
            [(0, 0), (1, 1)], name="Theta", variable_names=["theta_0", "theta_1"]
        )

        def mapping(*, theta_0, theta_1, x):  # noqa: D103
            if (theta_0, theta_1, x) == (0, 0, 0):
                return 0.1
            elif (theta_0, theta_1, x) == (0, 0, 1):
                return 0.4
            elif (theta_0, theta_1, x) == (0, 0, 2):
                return 0.5
            elif (theta_0, theta_1, x) == (1, 1, 0):
                return 0.25
            elif (theta_0, theta_1, x) == (1, 1, 1):
                return 0.65
            elif (theta_0, theta_1, x) == (1, 1, 2):
                return 0.1

        P = ParametrizedProbabilityMeasure(
            sig_alg=F,
            parameter_domain=Theta,
            mapping=mapping,
        )
        P.sig_alg = G
        expected_domain = pd.MultiIndex.from_tuples(
            [(0, 0, 1), (0, 0, 2), (1, 1, 1), (1, 1, 2)],
            names=["theta_0", "theta_1", "y"],
        )
        expected_data = pd.Series(
            [0.5, 0.5, 0.9, 0.1], index=expected_domain, name="probability"
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_from_2_dim_to_1_dim_with_1_dim_parameter_domain(self):
        """Test the sig_alg setter from 2D to 1D sigma-algebra with 1D parameter domain."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "a"),
                1: ("a", "b"),
                2: ("b", "c"),
                3: ("b", "d"),
            },
            variable_names=["F_0", "F_1"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: "x",
                1: "x",
                2: "y",
                3: "y",
            },
            variable_names=["y"],
            name="G",
        )
        Theta = Domain([0, 1, 2], name="Theta", variable_names=["theta"])

        def mapping(*, theta, F_0, F_1):  # noqa: D103
            if (theta, F_0, F_1) == (0, "a", "a"):
                return 0.1
            elif (theta, F_0, F_1) == (0, "a", "b"):
                return 0.2
            elif (theta, F_0, F_1) == (0, "b", "c"):
                return 0.3
            elif (theta, F_0, F_1) == (0, "b", "d"):
                return 0.4
            elif (theta, F_0, F_1) == (1, "a", "a"):
                return 0.15
            elif (theta, F_0, F_1) == (1, "a", "b"):
                return 0.25
            elif (theta, F_0, F_1) == (1, "b", "c"):
                return 0.35
            elif (theta, F_0, F_1) == (1, "b", "d"):
                return 0.25
            elif (theta, F_0, F_1) == (2, "a", "a"):
                return 0.5
            elif (theta, F_0, F_1) == (2, "a", "b"):
                return 0.2
            elif (theta, F_0, F_1) == (2, "b", "c"):
                return 0.2
            elif (theta, F_0, F_1) == (2, "b", "d"):
                return 0.1

        P = ParametrizedProbabilityMeasure(
            sig_alg=F,
            parameter_domain=Theta,
            mapping=mapping,
        )
        P.sig_alg = G
        expected_domain = pd.MultiIndex.from_tuples(
            [
                (0, "x"),
                (0, "y"),
                (1, "x"),
                (1, "y"),
                (2, "x"),
                (2, "y"),
            ],
            names=["theta", "y"],
        )
        expected_data = pd.Series(
            [0.3, 0.7, 0.4, 0.6, 0.7, 0.3],
            index=expected_domain,
            name="probability",
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_from_2_dim_to_1_dim_with_2_dim_parameter_domain(self):
        """Test the sig_alg setter from 2D to 1D sigma-algebra with 2D parameter domain."""
        Omega = SampleSpace.from_sequence(size=6)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("x", "a"),
                1: ("x", "b"),
                2: ("y", "c"),
                3: ("y", "d"),
                4: ("z", "e"),
                5: ("z", "f"),
            },
            variable_names=["F_0", "F_1"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: "u",
                1: "u",
                2: "v",
                3: "v",
                4: "w",
                5: "w",
            },
            variable_names=["z"],
            name="G",
        )
        Theta = Domain(
            [(0, 0), (0, 1), (1, 1)],
            name="Theta",
            variable_names=["alpha", "beta"],
        )

        def mapping(*, alpha, beta, F_0, F_1):  # noqa: D103
            if (alpha, beta, F_0, F_1) == (0, 0, "x", "a"):
                return 0.1
            elif (alpha, beta, F_0, F_1) == (0, 0, "x", "b"):
                return 0.15
            elif (alpha, beta, F_0, F_1) == (0, 0, "y", "c"):
                return 0.2
            elif (alpha, beta, F_0, F_1) == (0, 0, "y", "d"):
                return 0.25
            elif (alpha, beta, F_0, F_1) == (0, 0, "z", "e"):
                return 0.15
            elif (alpha, beta, F_0, F_1) == (0, 0, "z", "f"):
                return 0.15
            elif (alpha, beta, F_0, F_1) == (0, 1, "x", "a"):
                return 0.05
            elif (alpha, beta, F_0, F_1) == (0, 1, "x", "b"):
                return 0.05
            elif (alpha, beta, F_0, F_1) == (0, 1, "y", "c"):
                return 0.3
            elif (alpha, beta, F_0, F_1) == (0, 1, "y", "d"):
                return 0.3
            elif (alpha, beta, F_0, F_1) == (0, 1, "z", "e"):
                return 0.15
            elif (alpha, beta, F_0, F_1) == (0, 1, "z", "f"):
                return 0.15
            elif (alpha, beta, F_0, F_1) == (1, 1, "x", "a"):
                return 0.2
            elif (alpha, beta, F_0, F_1) == (1, 1, "x", "b"):
                return 0.3
            elif (alpha, beta, F_0, F_1) == (1, 1, "y", "c"):
                return 0.1
            elif (alpha, beta, F_0, F_1) == (1, 1, "y", "d"):
                return 0.05
            elif (alpha, beta, F_0, F_1) == (1, 1, "z", "e"):
                return 0.2
            elif (alpha, beta, F_0, F_1) == (1, 1, "z", "f"):
                return 0.15

        P = ParametrizedProbabilityMeasure(
            sig_alg=F,
            parameter_domain=Theta,
            mapping=mapping,
        )
        P.sig_alg = G
        expected_domain = pd.MultiIndex.from_tuples(
            [
                (0, 0, "u"),
                (0, 0, "v"),
                (0, 0, "w"),
                (0, 1, "u"),
                (0, 1, "v"),
                (0, 1, "w"),
                (1, 1, "u"),
                (1, 1, "v"),
                (1, 1, "w"),
            ],
            names=["alpha", "beta", "z"],
        )
        expected_data = pd.Series(
            [0.25, 0.45, 0.3, 0.1, 0.6, 0.3, 0.5, 0.15, 0.35],
            index=expected_domain,
            name="probability",
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_from_3_dim_to_1_dim_with_2_dim_parameter_domain(self):
        """Test the sig_alg setter from 3D to 1D sigma-algebra with 2D parameter domain."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "x", 1),
                1: ("a", "y", 2),
                2: ("b", "x", 1),
                3: ("b", "y", 3),
            },
            variable_names=["F_0", "F_1", "F_2"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: "A",
                1: "A",
                2: "B",
                3: "B",
            },
            variable_names=["w"],
            name="G",
        )
        Theta = Domain(
            [(0, 0), (1, 0), (1, 1)],
            name="Theta",
            variable_names=["p", "q"],
        )

        def mapping(*, p, q, F_0, F_1, F_2):  # noqa: D103
            if (p, q, F_0, F_1, F_2) == (0, 0, "a", "x", 1):
                return 0.3
            elif (p, q, F_0, F_1, F_2) == (0, 0, "a", "y", 2):
                return 0.4
            elif (p, q, F_0, F_1, F_2) == (0, 0, "b", "x", 1):
                return 0.2
            elif (p, q, F_0, F_1, F_2) == (0, 0, "b", "y", 3):
                return 0.1
            elif (p, q, F_0, F_1, F_2) == (1, 0, "a", "x", 1):
                return 0.15
            elif (p, q, F_0, F_1, F_2) == (1, 0, "a", "y", 2):
                return 0.25
            elif (p, q, F_0, F_1, F_2) == (1, 0, "b", "x", 1):
                return 0.35
            elif (p, q, F_0, F_1, F_2) == (1, 0, "b", "y", 3):
                return 0.25
            elif (p, q, F_0, F_1, F_2) == (1, 1, "a", "x", 1):
                return 0.2
            elif (p, q, F_0, F_1, F_2) == (1, 1, "a", "y", 2):
                return 0.3
            elif (p, q, F_0, F_1, F_2) == (1, 1, "b", "x", 1):
                return 0.25
            elif (p, q, F_0, F_1, F_2) == (1, 1, "b", "y", 3):
                return 0.25

        P = ParametrizedProbabilityMeasure(
            sig_alg=F,
            parameter_domain=Theta,
            mapping=mapping,
        )
        P.sig_alg = G
        expected_domain = pd.MultiIndex.from_tuples(
            [
                (0, 0, "A"),
                (0, 0, "B"),
                (1, 0, "A"),
                (1, 0, "B"),
                (1, 1, "A"),
                (1, 1, "B"),
            ],
            names=["p", "q", "w"],
        )
        expected_data = pd.Series(
            [0.7, 0.3, 0.4, 0.6, 0.5, 0.5],
            index=expected_domain,
            name="probability",
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_with_1_dim_sig_alg_and_1_dim_parameter_domain(self):
        """Test the sig_alg setter with a 1-dimensional sigma-algebra and 1-dimensional parameter domain."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: "a",
                1: "b",
                2: "c",
                3: "d",
            },
            variable_names=["x"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: "a",
                1: "a",
                2: "b",
                3: "b",
            },
            variable_names=["y"],
            name="G",
        )
        Theta = Domain([0.25, 0.5, 0.75], name="Theta", variable_names=["theta"])

        def mapping(*, theta, x):  # noqa: D103
            if (theta, x) == (0.25, "a"):
                return 0.1
            elif (theta, x) == (0.25, "b"):
                return 0.2
            elif (theta, x) == (0.25, "c"):
                return 0.3
            elif (theta, x) == (0.25, "d"):
                return 0.4
            elif (theta, x) == (0.5, "a"):
                return 0.15
            elif (theta, x) == (0.5, "b"):
                return 0.35
            elif (theta, x) == (0.5, "c"):
                return 0.25
            elif (theta, x) == (0.5, "d"):
                return 0.25
            elif (theta, x) == (0.75, "a"):
                return 0.4
            elif (theta, x) == (0.75, "b"):
                return 0.3
            elif (theta, x) == (0.75, "c"):
                return 0.2
            elif (theta, x) == (0.75, "d"):
                return 0.1

        P = ParametrizedProbabilityMeasure(
            sig_alg=F,
            parameter_domain=Theta,
            mapping=mapping,
        )
        P.sig_alg = G
        expected_domain = pd.MultiIndex.from_tuples(
            [
                (0.25, "a"),
                (0.25, "b"),
                (0.5, "a"),
                (0.5, "b"),
                (0.75, "a"),
                (0.75, "b"),
            ],
            names=["theta", "y"],
        )
        expected_data = pd.Series(
            [0.3, 0.7, 0.5, 0.5, 0.7, 0.3],
            index=expected_domain,
            name="probability",
        )

        pd.testing.assert_series_equal(P.data, expected_data)


# --------------------- test data access methods --------------------- #


class TestCall:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F_1D(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
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
            sample_space=Omega,
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert P(A) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
            sig_alg=F_2D,
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
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert P(event=A) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
        assert P(event=A) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
            mapping=lambda *, theta: 0.9 if theta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=B) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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
        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert P([1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
            sig_alg=F_2D,
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
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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
        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["theta"]),
            name="P(A)",
            mapping=lambda *, theta: 0.25 if theta == 0 else 0.6,
            output_name="probability",
        )
        assert P(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
        assert P(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=parameter_domain_1D,
            name="P(B)",
            mapping=lambda *, theta: 0.9 if theta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=A, alpha=0) == expected_result
        assert P(event=A)(alpha=0) == expected_result
        assert P(alpha=0)(event=A) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
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
        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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

    def test_with_list_as_keyword_and_partial_parameters(
        self,
        F_1D,
        F_2D,
        parameter_domain_2D,
        P_func_2D_parameter_domain_1D_sig_alg,
        P_func_4D,
    ):
        """Test the __call__ method with list as a keyword argument and partial parameters."""
        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(A)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
            output_name="probability",
        )
        assert P(event=[1, 2, 3], alpha=0) == expected_result
        assert P(event=[1, 2, 3])(alpha=0) == expected_result
        assert P(alpha=0)(event=[1, 2, 3]) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = MultivariateFunction(
            domain=Domain([0, 1], variable_names=["beta"]),
            name="P(B)(alpha=0)",
            mapping=lambda *, beta: 0.9 if beta == 0 else 0.7,
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P(A, theta=0) == pytest.approx(0.25)
        assert P(A)(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(A) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
            sig_alg=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P(B, theta=0) == pytest.approx(0.9)
        assert P(B)(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(B) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P(event=A, theta=0) == pytest.approx(0.25)
        assert P(event=A)(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(event=A) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P(event=B, theta=0) == pytest.approx(0.9)
        assert P(event=B)(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(event=B) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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
        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.25)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.25)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
            sig_alg=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P([1, 2, 3], theta=0) == pytest.approx(0.9)
        assert P([1, 2, 3])(theta=0) == pytest.approx(0.9)
        assert P(theta=0)([1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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
        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        assert P(event=[1, 2, 3], theta=0) == pytest.approx(0.25)
        assert P(event=[1, 2, 3])(theta=0) == pytest.approx(0.25)
        assert P(theta=0)(event=[1, 2, 3]) == pytest.approx(0.25)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
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

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        assert P(event=[1, 2, 3], theta=0) == pytest.approx(0.9)
        assert P(event=[1, 2, 3])(theta=0) == pytest.approx(0.9)
        assert P(theta=0)(event=[1, 2, 3]) == pytest.approx(0.9)

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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
        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
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
            sig_alg=F_1D,
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
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
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
            sig_alg=F_2D,
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
            sig_alg=F_1D, parameter_domain=parameter_domain_1D, mapping=P_func_2D
        )
        expected_result = ProbabilityMeasure(
            sig_alg=F_1D,
            name="P(theta=0)",
            mapping={
                "a": 0.75,
                "b": 0.15,
                "c": 0.10,
            },
        )
        assert P(theta=0) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_1D,
            parameter_domain=parameter_domain_2D,
            mapping=P_func_2D_parameter_domain_1D_sig_alg,
        )
        expected_result = ProbabilityMeasure(
            sig_alg=F_1D,
            name="P(alpha=0, beta=0)",
            mapping={
                "a": 0.1,
                "b": 0.7,
                "c": 0.2,
            },
        )
        assert P(alpha=0, beta=0) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D,
            parameter_domain=parameter_domain_1D,
            mapping=P_func_1D_parameter_domain_2D_sig_alg,
        )
        expected_result = ProbabilityMeasure(
            sig_alg=F_2D,
            name="P(theta=0)",
            mapping={
                ("a", "a"): 0.1,
                ("a", "b"): 0.2,
                ("b", "c"): 0.7,
            },
        )
        assert P(theta=0) == expected_result

        P = ParametrizedProbabilityMeasure(
            sig_alg=F_2D, parameter_domain=parameter_domain_2D, mapping=P_func_4D
        )
        expected_result = ProbabilityMeasure(
            sig_alg=F_2D,
            name="P(alpha=0, beta=0)",
            mapping={
                ("a", "a"): 0.1,
                ("a", "b"): 0.2,
                ("b", "c"): 0.7,
            },
        )
        assert P(alpha=0, beta=0) == expected_result
