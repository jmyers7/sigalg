import pytest

import sigalg as sa


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def trivial_algebra(self, sample_space):
        return sa.SigmaAlgebra.trivial(sample_space)

    @pytest.fixture
    def middle_algebra(self, sample_space):
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        return sa.SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    @pytest.fixture
    def power_set_algebra(self, sample_space):
        return sa.SigmaAlgebra.power_set(sample_space)

    def test_construction_with_discrete_time(
        self, trivial_algebra, middle_algebra, power_set_algebra
    ):
        time = sa.Time.discrete(start=0, length=3)
        filtration = sa.Filtration(
            sigma_algebras=[trivial_algebra, middle_algebra, power_set_algebra],
            time=time,
            name="F",
        )
        assert len(filtration.sigma_algebras) == 3
        assert filtration.name == "F"

    def test_construction_with_continuous_time(
        self, trivial_algebra, middle_algebra, power_set_algebra
    ):
        time = sa.Time.continuous(start=0.0, stop=1.0, step=0.5)
        filtration = sa.Filtration(
            sigma_algebras=[trivial_algebra, middle_algebra, power_set_algebra],
            time=time,
            name="F",
        )
        assert len(filtration.sigma_algebras) == 3
        assert filtration.name == "F"

    def test_construction_stores_sigma_algebras(
        self, trivial_algebra, power_set_algebra
    ):
        time = sa.Time.discrete(start=0, length=2)
        filtration = sa.Filtration(
            sigma_algebras=[trivial_algebra, power_set_algebra], time=time, name="F"
        )
        assert filtration.sigma_algebras[0] == trivial_algebra
        assert filtration.sigma_algebras[1] == power_set_algebra


class TestValidation:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def other_sample_space(self):
        return sa.SampleSpace(["a", "b", "c"])

    def test_construction_with_empty_list_raises_error(self):
        time = sa.Time.discrete(start=0, length=1)
        with pytest.raises(ValueError, match="non-empty list"):
            sa.Filtration(sigma_algebras=[], time=time, name="F")

    def test_construction_with_non_sigma_algebra_raises_error(self, sample_space):
        time = sa.Time.discrete(start=0, length=2)
        alg = sa.SigmaAlgebra.trivial(sample_space)
        with pytest.raises(ValueError, match="instances of SigmaAlgebra"):
            sa.Filtration(sigma_algebras=[alg, "not an algebra"], time=time, name="F")

    def test_construction_with_non_time_raises_error(self, sample_space):
        alg = sa.SigmaAlgebra.trivial(sample_space)
        with pytest.raises(TypeError, match="must be a Time object"):
            sa.Filtration(sigma_algebras=[alg], time=[0, 1, 2], name="F")

    def test_construction_with_non_string_name_raises_error(self, sample_space):
        time = sa.Time.discrete(start=0, length=1)
        alg = sa.SigmaAlgebra.trivial(sample_space)
        with pytest.raises(TypeError, match="must be a string"):
            sa.Filtration(sigma_algebras=[alg], time=time, name=123)

    def test_construction_with_mismatched_lengths_raises_error(self, sample_space):
        time = sa.Time.discrete(start=0, length=3)
        alg1 = sa.SigmaAlgebra.trivial(sample_space)
        alg2 = sa.SigmaAlgebra.power_set(sample_space)
        with pytest.raises(ValueError, match="must match the length"):
            sa.Filtration(sigma_algebras=[alg1, alg2], time=time, name="F")

    def test_construction_with_different_sample_spaces_raises_error(
        self, sample_space, other_sample_space
    ):
        time = sa.Time.discrete(start=0, length=2)
        alg1 = sa.SigmaAlgebra.trivial(sample_space)
        alg2 = sa.SigmaAlgebra.trivial(other_sample_space)
        with pytest.raises(ValueError, match="same sample space"):
            sa.Filtration(sigma_algebras=[alg1, alg2], time=time, name="F")

    def test_construction_with_non_increasing_algebras_raises_error(self, sample_space):
        time = sa.Time.discrete(start=0, length=2)
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        with pytest.raises(ValueError, match="do not form a valid filtration"):
            sa.Filtration(sigma_algebras=[power_set, trivial], time=time, name="F")


class TestProperties:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        time = sa.Time.discrete(start=0, length=3)
        return sa.Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    def test_sigma_algebras_property_returns_list(self, filtration):
        assert isinstance(filtration.sigma_algebras, list)
        assert len(filtration.sigma_algebras) == 3

    def test_sigma_algebras_property_returns_copy(self, filtration):
        algs1 = filtration.sigma_algebras
        algs1.append("something")
        algs2 = filtration.sigma_algebras
        assert len(algs2) == 3

    def test_name_property_returns_string(self, filtration):
        assert filtration.name == "F"

    def test_time_property_returns_time_object(self, filtration):
        assert isinstance(filtration.time, sa.Time)

    def test_sample_space_property(self, filtration, sample_space):
        assert filtration.sample_space == sample_space

    def test_coarsest_property(self, filtration, sample_space):
        coarsest = filtration.coarsest
        assert coarsest.num_atoms == 1
        assert coarsest == sa.SigmaAlgebra.trivial(sample_space)

    def test_finest_property(self, filtration, sample_space):
        finest = filtration.finest
        assert finest.num_atoms == 4
        assert finest == sa.SigmaAlgebra.power_set(sample_space)


class TestSetters:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        time = sa.Time.discrete(start=0, length=2)
        return sa.Filtration(sigma_algebras=[trivial, power_set], time=time, name="F")

    def test_name_setter_changes_name(self, filtration):
        filtration.name = "NewName"
        assert filtration.name == "NewName"

    def test_name_setter_with_non_string_raises_error(self, filtration):
        with pytest.raises(TypeError, match="must be a string"):
            filtration.name = 123


class TestDataAccess:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def discrete_filtration(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        time = sa.Time.discrete(start=0, length=3)
        return sa.Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    @pytest.fixture
    def continuous_filtration(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        time = sa.Time.continuous(start=0.0, stop=1.0, step=0.5)
        return sa.Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    def test_at_exact_time_discrete(self, discrete_filtration, sample_space):
        alg = discrete_filtration.at[0]
        assert alg == sa.SigmaAlgebra.trivial(sample_space)

    def test_at_exact_time_continuous(self, continuous_filtration, sample_space):
        alg = continuous_filtration.at[0.5]
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        expected = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert alg == expected

    def test_at_interpolated_time_discrete(self, discrete_filtration, sample_space):
        alg = discrete_filtration.at[1.7]
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        expected = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert alg == expected

    def test_at_interpolated_time_continuous(self, continuous_filtration, sample_space):
        alg = continuous_filtration.at[0.3]
        assert alg == sa.SigmaAlgebra.trivial(sample_space)

    def test_at_time_before_start_raises_error(self, discrete_filtration):
        with pytest.raises(ValueError, match="before the start"):
            discrete_filtration.at[-1]

    def test_at_time_after_end_raises_error(self, discrete_filtration):
        with pytest.raises(ValueError, match="after the end"):
            discrete_filtration.at[10]

    def test_at_returns_largest_time_less_than_or_equal(
        self, continuous_filtration, sample_space
    ):
        alg = continuous_filtration.at[0.99]
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        expected = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert alg == expected


class TestSequenceMethods:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = sa.SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        power_set = sa.SigmaAlgebra.power_set(sample_space)
        time = sa.Time.discrete(start=0, length=3)
        return sa.Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    def test_len_returns_number_of_algebras(self, filtration):
        assert len(filtration) == 3

    def test_iteration_yields_sigma_algebras(self, filtration, sample_space):
        algebras = list(filtration)
        assert len(algebras) == 3
        assert algebras[0] == sa.SigmaAlgebra.trivial(sample_space)
        assert algebras[2] == sa.SigmaAlgebra.power_set(sample_space)
