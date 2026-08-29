from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra


class TestEquality:
    def test_reflexivity(self):
        """Test that a measure equals itself."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )

        assert mu == mu

    def test_identical_power_set_1d(self):
        """Test equality with identical measures using 1D power-set sigma-algebras."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
            name="nu",
        )

        assert mu == nu

    def test_power_set_2d_domain_same_structure(self):
        """Test equality with 2D power-set sigma-algebras with identical structure."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1, 2): 1,
                (0, 1, 3, 4): 2,
                (0, 1, 5, 6): 3,
                (2, 3, 1, 2): 1,
                (2, 3, 3, 4): 2,
                (2, 3, 5, 6): 3,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1, 2): 1,
                (0, 1, 3, 4): 2,
                (0, 1, 5, 6): 3,
                (2, 3, 1, 2): 1,
                (2, 3, 3, 4): 2,
                (2, 3, 5, 6): 3,
            },
            name="nu",
        )

        assert mu == nu

    def test_power_set_reordered_parameters(self):
        """Test equality with reordered parameter domains."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])
        Phi = Domain([(3, 2), (1, 0)], variable_names=["theta_1", "theta_0"])

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1, 2): 1,
                (0, 1, 3, 4): 2,
                (0, 1, 5, 6): 3,
                (2, 3, 1, 2): 1,
                (2, 3, 3, 4): 2,
                (2, 3, 5, 6): 3,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Phi,
            mapping={
                (1, 0, 1, 2): 1,
                (1, 0, 3, 4): 2,
                (1, 0, 5, 6): 3,
                (3, 2, 1, 2): 1,
                (3, 2, 3, 4): 2,
                (3, 2, 5, 6): 3,
            },
            name="nu",
        )

        assert mu == nu

    def test_power_set_reordered_domains(self):
        """Test inequality with reordered measure domains."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Y = Domain([(4, 3), (2, 1), (6, 5)], variable_names=["b", "a"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1, 2): 1,
                (0, 1, 3, 4): 2,
                (0, 1, 5, 6): 3,
                (2, 3, 1, 2): 1,
                (2, 3, 3, 4): 2,
                (2, 3, 5, 6): 3,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Theta,
            mapping={
                (0, 1, 2, 1): 1,
                (0, 1, 4, 3): 2,
                (0, 1, 6, 5): 3,
                (2, 3, 2, 1): 1,
                (2, 3, 4, 3): 2,
                (2, 3, 6, 5): 3,
            },
            name="nu",
        )

        assert mu != nu

    def test_power_set_all_reordered(self):
        """Test inequality with both domains and parameters reordered."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Y = Domain([(4, 3), (2, 1), (6, 5)], variable_names=["b", "a"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])
        Phi = Domain([(3, 2), (1, 0)], variable_names=["theta_1", "theta_0"])

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1, 2): 1,
                (0, 1, 3, 4): 2,
                (0, 1, 5, 6): 3,
                (2, 3, 1, 2): 1,
                (2, 3, 3, 4): 2,
                (2, 3, 5, 6): 3,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Phi,
            mapping={
                (1, 0, 2, 1): 1,
                (1, 0, 4, 3): 2,
                (1, 0, 6, 5): 3,
                (3, 2, 2, 1): 1,
                (3, 2, 4, 3): 2,
                (3, 2, 6, 5): 3,
            },
            name="nu",
        )

        assert mu != nu

    def test_custom_sig_alg_different_atom_labels_1d(self):
        """Test equality with custom sigma-algebras using different 1D atom labels."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])

        F = SigmaAlgebra(
            domain=X,
            mapping={
                (1, 2): 1,
                (3, 4): 2,
                (5, 6): 2,
            },
            variable_names=["u"],
        )

        H = SigmaAlgebra(
            domain=X,
            mapping={
                (1, 2): 4,
                (3, 4): -2,
                (5, 6): -2,
            },
            variable_names=["u"],
            name="H",
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1): 2,
                (0, 1, 2): 3,
                (2, 3, 1): 4,
                (2, 3, 2): 0,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=H,
            parameter_domain=Theta,
            mapping={
                (0, 1, 4): 2,
                (0, 1, -2): 3,
                (2, 3, 4): 4,
                (2, 3, -2): 0,
            },
            name="nu",
        )

        assert mu == nu

    def test_custom_sig_alg_different_atom_labels_2d(self):
        """Test inequality with custom sigma-algebras: 1D vs 2D atom labels."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Y = Domain([(4, 3), (2, 1), (6, 5)], variable_names=["b", "a"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])
        Phi = Domain([(3, 2), (1, 0)], variable_names=["theta_1", "theta_0"])

        F = SigmaAlgebra(
            domain=X,
            mapping={
                (1, 2): 1,
                (3, 4): 2,
                (5, 6): 2,
            },
            variable_names=["u"],
        )

        G = SigmaAlgebra(
            domain=Y,
            mapping={
                (2, 1): ("r", 1),
                (4, 3): ("s", 1),
                (6, 5): ("s", 1),
            },
            variable_names=["v", "w"],
            name="G",
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1): 2,
                (0, 1, 2): 3,
                (2, 3, 1): 4,
                (2, 3, 2): 0,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=G,
            parameter_domain=Phi,
            mapping={
                (1, 0, "r", 1): 2,
                (1, 0, "s", 1): 3,
                (3, 2, "r", 1): 4,
                (3, 2, "s", 1): 0,
            },
            name="nu",
        )

        assert mu != nu

    def test_different_partitions(self):
        """Test that measures on different partitions are not equal."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])

        F = SigmaAlgebra(
            domain=X,
            mapping={
                (1, 2): 1,
                (3, 4): 2,
                (5, 6): 2,
            },
            variable_names=["u"],
        )

        H = SigmaAlgebra(
            domain=X,
            mapping={
                (1, 2): 4,
                (3, 4): 4,
                (5, 6): -2,
            },
            variable_names=["u"],
            name="H",
        )

        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1): 2,
                (0, 1, 2): 3,
                (2, 3, 1): 4,
                (2, 3, 2): 0,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=H,
            parameter_domain=Theta,
            mapping={
                (0, 1, 4): 2,
                (0, 1, -2): 3,
                (2, 3, 4): 4,
                (2, 3, -2): 0,
            },
            name="nu",
        )

        assert mu != nu

    def test_same_partition_different_values(self):
        """Test that measures with same partition but different values are not equal."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 99,
                (1, 2): 6,
            },
            name="nu",
        )

        assert mu != nu

    def test_different_number_of_atoms(self):
        """Test that measures with different numbers of atoms are not equal."""
        X = Domain.from_sequence(size=4, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        F = SigmaAlgebra(
            domain=X,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            },
            variable_names=["u"],
        )

        G = SigmaAlgebra(
            domain=X,
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            },
            variable_names=["u"],
            name="G",
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (1, 0): 3,
                (1, 1): 4,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=G,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 1,
                (0, 2): 1,
                (1, 0): 2,
                (1, 1): 2,
                (1, 2): 2,
            },
            name="nu",
        )

        assert mu != nu

    def test_different_parameter_dimensions(self):
        """Test that measures with different parameter dimensions are not equal."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta_1d = Domain.from_sequence(size=2, variable_name="theta")
        Theta_2d = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta_1d,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta_2d,
            mapping={
                (0, 1, 0): 1,
                (0, 1, 1): 2,
                (0, 1, 2): 3,
                (2, 3, 0): 4,
                (2, 3, 1): 5,
                (2, 3, 2): 6,
            },
            name="nu",
        )

        assert mu != nu

    def test_different_parameter_values(self):
        """Test that measures with different parameter sets are not equal."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")
        Phi = Domain.from_sequence(size=3, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Phi,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
                (2, 0): 7,
                (2, 1): 8,
                (2, 2): 9,
            },
            name="nu",
        )

        assert mu != nu

    def test_completely_different_domains(self):
        """Test that measures on completely different domains are not equal."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Y = Domain.from_sequence(size=4, variable_name="y")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (0, 3): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
                (1, 3): 6,
            },
            name="nu",
        )

        assert mu != nu

    def test_symmetry_simple(self):
        """Test that equality is symmetric: mu == nu implies nu == mu."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
            name="nu",
        )

        assert (mu == nu) == (nu == mu)

    def test_symmetry_reordered_complex(self):
        """Test symmetry with complex reordering."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Y = Domain([(4, 3), (2, 1), (6, 5)], variable_names=["b", "a"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])
        Phi = Domain([(3, 2), (1, 0)], variable_names=["theta_1", "theta_0"])

        F = SigmaAlgebra(
            domain=X,
            mapping={
                (1, 2): 1,
                (3, 4): 2,
                (5, 6): 2,
            },
            variable_names=["u"],
        )
        G = SigmaAlgebra(
            domain=Y,
            mapping={
                (2, 1): ("r", 1),
                (4, 3): ("s", 1),
                (6, 5): ("s", 1),
            },
            variable_names=["v", "w"],
            name="G",
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1): 2,
                (0, 1, 2): 3,
                (2, 3, 1): 4,
                (2, 3, 2): 0,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=G,
            parameter_domain=Phi,
            mapping={
                (1, 0, "r", 1): 2,
                (1, 0, "s", 1): 3,
                (3, 2, "r", 1): 4,
                (3, 2, "s", 1): 0,
            },
            name="nu",
        )

        assert (mu == nu) == (nu == mu)

    def test_symmetry_inequality(self):
        """Test symmetry for inequality: mu != nu implies nu != mu."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 99,
                (1, 2): 6,
            },
            name="nu",
        )

        assert (mu == nu) == (nu == mu)
        assert mu != nu

    def test_single_parameter_value(self):
        """Test equality with single parameter value."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=1, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
            },
            name="nu",
        )

        assert mu == nu

    def test_single_atom_sigma_algebra(self):
        """Test equality with sigma-algebra having single atom."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        F = SigmaAlgebra(
            domain=X,
            mapping={
                0: 0,
                1: 0,
                2: 0,
            },
            variable_names=["u"],
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 0): 5,
                (1, 0): 10,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 0): 5,
                (1, 0): 10,
            },
            name="nu",
        )

        assert mu == nu

    def test_single_point_domain(self):
        """Test equality with domain containing single point."""
        X = Domain.from_sequence(size=1, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 7,
                (1, 0): 14,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 7,
                (1, 0): 14,
            },
            name="nu",
        )

        assert mu == nu

    def test_all_single_elements(self):
        """Test equality with single parameter, single atom, and single point."""
        X = Domain.from_sequence(size=1, variable_name="x")
        Theta = Domain.from_sequence(size=1, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 42},
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 42},
            name="nu",
        )

        assert mu == nu

    def test_single_element_inequality(self):
        """Test inequality with single parameter but different values."""
        X = Domain.from_sequence(size=1, variable_name="x")
        Theta = Domain.from_sequence(size=1, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 42},
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 99},
            name="nu",
        )

        assert mu != nu

    def test_measure_with_zero_values(self):
        """Test equality when measures have zero values."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 0,
                (0, 1): 2,
                (0, 2): 0,
                (1, 0): 4,
                (1, 1): 0,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 0,
                (0, 1): 2,
                (0, 2): 0,
                (1, 0): 4,
                (1, 1): 0,
                (1, 2): 6,
            },
            name="nu",
        )

        assert mu == nu

    def test_measure_all_zeros(self):
        """Test equality when all measure values are zero."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 0,
                (0, 1): 0,
                (0, 2): 0,
                (1, 0): 0,
                (1, 1): 0,
                (1, 2): 0,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 0,
                (0, 1): 0,
                (0, 2): 0,
                (1, 0): 0,
                (1, 1): 0,
                (1, 2): 0,
            },
            name="nu",
        )

        assert mu == nu

    def test_zero_vs_nonzero_inequality(self):
        """Test inequality when one has zero and other has nonzero for same atom."""
        X = Domain.from_sequence(size=2, variable_name="x")
        Theta = Domain.from_sequence(size=1, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 0,
                (0, 1): 5,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 5,
            },
            name="nu",
        )

        assert mu != nu

    def test_trivial_sigma_algebra_equality(self):
        """Test equality with trivial sigma-algebra."""
        X = Domain.from_sequence(size=5, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        F = SigmaAlgebra(
            domain=X,
            mapping={0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            variable_names=["u"],
        )
        G = SigmaAlgebra(
            domain=X,
            mapping={0: 99, 1: 99, 2: 99, 3: 99, 4: 99},
            variable_names=["u"],
            name="G",
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={(0, 0): 10, (1, 0): 20},
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=G,
            parameter_domain=Theta,
            mapping={(0, 99): 10, (1, 99): 20},
            name="nu",
        )

        assert mu == nu

    def test_minimal_measure_1d(self):
        """Test minimal measure: 1 parameter, 1 domain point."""
        X = Domain.from_sequence(size=1, variable_name="x")
        Theta = Domain.from_sequence(size=1, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 1},
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 1},
            name="nu",
        )

        assert mu == nu
        assert mu == mu

    def test_singleton_domain_different_values(self):
        """Test inequality with singleton domain but different measure values."""
        X = Domain.from_sequence(size=1, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 10, (1, 0): 20},
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={(0, 0): 10, (1, 0): 99},
            name="nu",
        )

        assert mu != nu

    def test_non_involutive_permutation_3cycle_parameters(self):
        """Test equality with 3-cycle permutation on parameters."""
        X = Domain.from_sequence(size=2, variable_name="x")
        Theta = Domain(
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            variable_names=["theta_0", "theta_1", "theta_2"],
        )
        Phi = Domain(
            [(2, 3, 1), (5, 6, 4), (8, 9, 7)],
            variable_names=["theta_1", "theta_2", "theta_0"],
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (1, 2, 3, 0): 10,
                (1, 2, 3, 1): 20,
                (4, 5, 6, 0): 30,
                (4, 5, 6, 1): 40,
                (7, 8, 9, 0): 50,
                (7, 8, 9, 1): 60,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Phi,
            mapping={
                (2, 3, 1, 0): 10,
                (2, 3, 1, 1): 20,
                (5, 6, 4, 0): 30,
                (5, 6, 4, 1): 40,
                (8, 9, 7, 0): 50,
                (8, 9, 7, 1): 60,
            },
            name="nu",
        )

        assert mu == nu

    def test_non_involutive_permutation_3cycle_domains(self):
        """Test inequality with 3-cycle permutation on domain variables."""
        X = Domain(
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            variable_names=["a", "b", "c"],
        )
        Y = Domain(
            [(2, 3, 1), (5, 6, 4), (8, 9, 7)],
            variable_names=["b", "c", "a"],
        )
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 1, 2, 3): 10,
                (0, 4, 5, 6): 20,
                (0, 7, 8, 9): 30,
                (1, 1, 2, 3): 40,
                (1, 4, 5, 6): 50,
                (1, 7, 8, 9): 60,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Theta,
            mapping={
                (0, 2, 3, 1): 10,
                (0, 5, 6, 4): 20,
                (0, 8, 9, 7): 30,
                (1, 2, 3, 1): 40,
                (1, 5, 6, 4): 50,
                (1, 8, 9, 7): 60,
            },
            name="nu",
        )

        assert mu != nu

    def test_non_involutive_permutation_both_3cycle(self):
        """Test inequality with 3-cycle permutations on both domains and parameters."""
        X = Domain(
            [(1, 2, 3), (4, 5, 6)],
            variable_names=["a", "b", "c"],
        )
        Y = Domain(
            [(2, 3, 1), (5, 6, 4)],
            variable_names=["b", "c", "a"],
        )
        Theta = Domain(
            [(10, 20, 30), (40, 50, 60)],
            variable_names=["t0", "t1", "t2"],
        )
        Phi = Domain(
            [(20, 30, 10), (50, 60, 40)],
            variable_names=["t1", "t2", "t0"],
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (10, 20, 30, 1, 2, 3): 100,
                (10, 20, 30, 4, 5, 6): 200,
                (40, 50, 60, 1, 2, 3): 300,
                (40, 50, 60, 4, 5, 6): 400,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Phi,
            mapping={
                (20, 30, 10, 2, 3, 1): 100,
                (20, 30, 10, 5, 6, 4): 200,
                (50, 60, 40, 2, 3, 1): 300,
                (50, 60, 40, 5, 6, 4): 400,
            },
            name="nu",
        )

        assert mu != nu

    def test_same_size_disjoint_domain_labels(self):
        """Test inequality with same size but completely disjoint domain variable names."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Y = Domain.from_sequence(size=3, variable_name="y")
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
            name="nu",
        )

        assert mu != nu

    def test_same_size_disjoint_parameter_labels(self):
        """Test inequality with same size but completely disjoint parameter variable names."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Theta = Domain.from_sequence(size=2, variable_name="theta")
        Phi = Domain.from_sequence(size=2, variable_name="phi")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Phi,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
            name="nu",
        )

        assert mu != nu

    def test_same_size_disjoint_both(self):
        """Test inequality with same size but disjoint variable names on both domains and parameters."""
        X = Domain.from_sequence(size=2, variable_name="x")
        Y = Domain.from_sequence(size=2, variable_name="y")
        Theta = Domain.from_sequence(size=2, variable_name="theta")
        Phi = Domain.from_sequence(size=2, variable_name="phi")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (1, 0): 3,
                (1, 1): 4,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Phi,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (1, 0): 3,
                (1, 1): 4,
            },
            name="nu",
        )

        assert mu != nu

    def test_same_name_disjoint_domain_values(self):
        """Test inequality when domain shares a variable name but values never overlap."""
        X = Domain.from_sequence(size=3, variable_name="x")
        Y = Domain([10, 11, 12], variable_names=["x"])
        Theta = Domain.from_sequence(size=2, variable_name="theta")

        mu = ParametrizedMeasure.from_domains(
            measure_domain=X,
            parameter_domain=Theta,
            mapping={
                (0, 0): 1,
                (0, 1): 2,
                (0, 2): 3,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 6,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=Y,
            parameter_domain=Theta,
            mapping={
                (0, 10): 1,
                (0, 11): 2,
                (0, 12): 3,
                (1, 10): 4,
                (1, 11): 5,
                (1, 12): 6,
            },
            name="nu",
        )

        assert mu != nu

    def test_different_partition_2d_relabeled_reordered(self):
        """Test inequality with different partition under 2D relabeled and reordered setup."""
        X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["a", "b"])
        Y = Domain([(4, 3), (2, 1), (6, 5)], variable_names=["b", "a"])
        Theta = Domain([(0, 1), (2, 3)], variable_names=["theta_0", "theta_1"])
        Phi = Domain([(3, 2), (1, 0)], variable_names=["theta_1", "theta_0"])

        F = SigmaAlgebra(
            domain=X,
            mapping={
                (1, 2): 1,
                (3, 4): 2,
                (5, 6): 2,
            },
            variable_names=["u"],
        )

        G_diff = SigmaAlgebra(
            domain=Y,
            mapping={
                (2, 1): ("r", 1),
                (4, 3): ("r", 1),
                (6, 5): ("s", 1),
            },
            variable_names=["v", "w"],
            name="G_diff",
        )

        mu = ParametrizedMeasure.from_domains(
            measure_domain=F,
            parameter_domain=Theta,
            mapping={
                (0, 1, 1): 2,
                (0, 1, 2): 3,
                (2, 3, 1): 4,
                (2, 3, 2): 0,
            },
        )
        nu = ParametrizedMeasure.from_domains(
            measure_domain=G_diff,
            parameter_domain=Phi,
            mapping={
                (1, 0, "r", 1): 2,
                (1, 0, "s", 1): 3,
                (3, 2, "r", 1): 4,
                (3, 2, "s", 1): 0,
            },
            name="nu",
        )

        assert mu != nu
