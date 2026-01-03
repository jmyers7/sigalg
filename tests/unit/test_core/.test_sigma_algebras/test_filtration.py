import pytest

from sigalg.core import (
    FilteredSigmaAlgebra,
    Filtration,
    SampleSpace,
    SigmaAlgebra,
    Time,
)


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def trivial_algebra(self, sample_space):
        return SigmaAlgebra.trivial(sample_space)

    @pytest.fixture
    def middle_algebra(self, sample_space):
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    @pytest.fixture
    def power_set_algebra(self, sample_space):
        return SigmaAlgebra.power_set(sample_space)

    @pytest.mark.parametrize(
        "time_type,time_params,name,expected_num_algebras",
        [
            pytest.param(
                "discrete",
                {"start": 0, "length": 3},
                "F",
                3,
                id="discrete_time_custom_name",
            ),
            pytest.param(
                "continuous",
                {"start": 0.0, "stop": 1.0, "num_points": 3},
                "G",
                3,
                id="continuous_time_custom_name",
            ),
            pytest.param(
                "discrete",
                {"start": 0, "length": 2},
                "default_name_flag",
                2,
                id="discrete_time_default_name",
            ),
            pytest.param(
                "continuous",
                {"start": 0.0, "stop": 2.0, "num_points": 2},
                None,
                2,
                id="continuous_time_none_name",
            ),
        ],
    )
    def test_constructor(
        self,
        sample_space,
        trivial_algebra,
        middle_algebra,
        power_set_algebra,
        time_type,
        time_params,
        name,
        expected_num_algebras,
    ):
        """Test constructor with various time types and names."""
        if time_type == "discrete":
            time = Time.discrete(**time_params)
        else:
            time = Time.continuous(**time_params)

        if expected_num_algebras == 3:
            sigma_algebras = [trivial_algebra, middle_algebra, power_set_algebra]
        else:
            sigma_algebras = [trivial_algebra, power_set_algebra]

        if name == "default_name_flag":
            filtration = Filtration(sigma_algebras=sigma_algebras, time=time)
            name = "Ft"
        else:
            filtration = Filtration(sigma_algebras=sigma_algebras, time=time, name=name)

        assert len(filtration.sigma_algebras) == expected_num_algebras
        assert filtration.name == name
        assert filtration.time == time

    def test_constructor_stores_sigma_algebras(
        self, trivial_algebra, power_set_algebra
    ):
        """Test that constructor correctly stores sigma algebras."""
        time = Time.discrete(start=0, length=2)
        filtration = Filtration(
            sigma_algebras=[trivial_algebra, power_set_algebra], time=time, name="F"
        )
        assert filtration.sigma_algebras[0] == trivial_algebra
        assert filtration.sigma_algebras[1] == power_set_algebra

    def test_constructor_stores_sample_space(
        self, sample_space, trivial_algebra, power_set_algebra
    ):
        """Test that constructor correctly stores sample space."""
        time = Time.discrete(start=0, length=2)
        filtration = Filtration(
            sigma_algebras=[trivial_algebra, power_set_algebra], time=time, name="F"
        )
        assert filtration.sample_space == sample_space


class TestValidation:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def other_sample_space(self):
        return SampleSpace(["a", "b", "c"])

    @pytest.mark.parametrize(
        "invalid_sigma_algebras,error_type,error_match",
        [
            pytest.param(
                [],
                ValueError,
                "non-empty list",
                id="empty_list",
            ),
            pytest.param(
                "not a list",
                ValueError,
                "non-empty list",
                id="not_a_list",
            ),
        ],
    )
    def test_invalid_sigma_algebras_list_raises_error(
        self, invalid_sigma_algebras, error_type, error_match
    ):
        """Test that invalid sigma_algebras list raises appropriate errors."""
        time = Time.discrete(start=0, length=1)
        with pytest.raises(error_type, match=error_match):
            Filtration(sigma_algebras=invalid_sigma_algebras, time=time, name="F")

    def test_non_sigma_algebra_element_raises_error(self, sample_space):
        """Test that non-SigmaAlgebra elements in list raise ValueError."""
        time = Time.discrete(start=0, length=2)
        alg = SigmaAlgebra.trivial(sample_space)
        with pytest.raises(ValueError, match="instances of SigmaAlgebra"):
            Filtration(sigma_algebras=[alg, "not an algebra"], time=time, name="F")

    @pytest.mark.parametrize(
        "invalid_time,error_match",
        [
            pytest.param([0, 1, 2], "must be a Time object", id="list_instead_of_time"),
            pytest.param(
                {"start": 0}, "must be a Time object", id="dict_instead_of_time"
            ),
            pytest.param(
                "time_string", "must be a Time object", id="string_instead_of_time"
            ),
        ],
    )
    def test_invalid_time_raises_error(self, sample_space, invalid_time, error_match):
        """Test that invalid time parameter raises TypeError."""
        alg = SigmaAlgebra.trivial(sample_space)
        with pytest.raises(TypeError, match=error_match):
            Filtration(sigma_algebras=[alg], time=invalid_time, name="F")

    @pytest.mark.parametrize(
        "invalid_name",
        [
            pytest.param(["list", "name"], id="list_name"),
            pytest.param({"key": "value"}, id="dict_name"),
        ],
    )
    def test_invalid_name_raises_error(self, sample_space, invalid_name):
        """Test that invalid name parameter raises TypeError."""
        time = Time.discrete(start=0, length=1)
        alg = SigmaAlgebra.trivial(sample_space)
        with pytest.raises(TypeError, match="must be hashable"):
            Filtration(sigma_algebras=[alg], time=time, name=invalid_name)

    def test_mismatched_lengths_raises_error(self, sample_space):
        """Test that mismatched lengths between sigma_algebras and time raise ValueError."""
        time = Time.discrete(start=0, length=3)
        alg1 = SigmaAlgebra.trivial(sample_space)
        alg2 = SigmaAlgebra.power_set(sample_space)
        with pytest.raises(ValueError, match="must match the length"):
            Filtration(sigma_algebras=[alg1, alg2], time=time, name="F")

    def test_different_sample_spaces_raises_error(
        self, sample_space, other_sample_space
    ):
        """Test that sigma algebras with different sample spaces raise ValueError."""
        time = Time.discrete(start=0, length=2)
        alg1 = SigmaAlgebra.trivial(sample_space)
        alg2 = SigmaAlgebra.trivial(other_sample_space)
        with pytest.raises(ValueError, match="same sample space"):
            Filtration(sigma_algebras=[alg1, alg2], time=time, name="F")

    def test_non_increasing_algebras_raises_error(self, sample_space):
        """Test that non-increasing sigma algebras raise ValueError."""
        time = Time.discrete(start=0, length=2)
        trivial = SigmaAlgebra.trivial(sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        with pytest.raises(ValueError, match="do not form a valid filtration"):
            Filtration(sigma_algebras=[power_set, trivial], time=time, name="F")


class TestProperties:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=3)
        return Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    def test_sigma_algebras_property(self, filtration):
        """Test that sigma_algebras property returns correct list."""
        assert isinstance(filtration.sigma_algebras, list)
        assert len(filtration.sigma_algebras) == 3

    def test_name_property(self, filtration):
        """Test that name property returns correct value."""
        assert filtration.name == "F"

    def test_time_property(self, filtration):
        """Test that time property returns Time object."""
        assert isinstance(filtration.time, Time)
        assert len(filtration.time) == 3

    def test_sample_space_property(self, filtration, sample_space):
        """Test that sample_space property returns correct sample space."""
        assert filtration.sample_space == sample_space

    def test_coarsest_property(self, filtration, sample_space):
        """Test that coarsest property returns the first sigma algebra."""
        coarsest = filtration.coarsest
        assert coarsest.num_atoms == 1
        assert coarsest == SigmaAlgebra.trivial(sample_space)

    def test_finest_property(self, filtration, sample_space):
        """Test that finest property returns the last sigma algebra."""
        finest = filtration.finest
        assert finest.num_atoms == 4
        assert finest == SigmaAlgebra.power_set(sample_space)


class TestSetters:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=2)
        return Filtration(sigma_algebras=[trivial, power_set], time=time, name="F")

    @pytest.mark.parametrize(
        "new_name",
        [
            pytest.param("NewName", id="string_name"),
            pytest.param("G", id="single_char_name"),
            pytest.param(42, id="int_name"),
            pytest.param(("tuple", "name"), id="tuple_name"),
            pytest.param(None, id="none_name"),
        ],
    )
    def test_name_setter(self, filtration, new_name):
        """Test that name setter correctly updates name."""
        filtration.name = new_name
        assert filtration.name == new_name

    @pytest.mark.parametrize(
        "invalid_name",
        [
            pytest.param(["list", "name"], id="list_name"),
            pytest.param({"dict": "name"}, id="dict_name"),
        ],
    )
    def test_name_setter_with_unhashable_raises_error(self, filtration, invalid_name):
        """Test that setting unhashable name raises TypeError."""
        with pytest.raises(TypeError, match="must be a hashable or None"):
            filtration.name = invalid_name


class TestDataAccess:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def discrete_filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=3)
        return Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    @pytest.fixture
    def continuous_filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.continuous(start=0.0, stop=1.0, num_points=3)
        return Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    @pytest.mark.parametrize(
        "time_value,expected_algebra_type",
        [
            pytest.param(0, "trivial", id="first_time_point"),
            pytest.param(1, "middle", id="second_time_point"),
            pytest.param(2, "power_set", id="last_time_point"),
        ],
    )
    def test_at_exact_time_discrete(
        self, discrete_filtration, sample_space, time_value, expected_algebra_type
    ):
        """Test accessing sigma algebra at exact time points in discrete filtration."""
        alg = discrete_filtration.at[time_value]
        if expected_algebra_type == "trivial":
            expected = SigmaAlgebra.trivial(sample_space)
        elif expected_algebra_type == "middle":
            atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
            expected = SigmaAlgebra(
                sample_id_to_atom_id=atom_ids, sample_space=sample_space
            )
        else:  # power_set
            expected = SigmaAlgebra.power_set(sample_space)
        assert alg == expected

    @pytest.mark.parametrize(
        "time_value,expected_algebra_type",
        [
            pytest.param(0.0, "trivial", id="first_time_point"),
            pytest.param(0.5, "middle", id="middle_time_point"),
            pytest.param(1.0, "power_set", id="last_time_point"),
        ],
    )
    def test_at_exact_time_continuous(
        self, continuous_filtration, sample_space, time_value, expected_algebra_type
    ):
        """Test accessing sigma algebra at exact time points in continuous filtration."""
        alg = continuous_filtration.at[time_value]
        if expected_algebra_type == "trivial":
            expected = SigmaAlgebra.trivial(sample_space)
        elif expected_algebra_type == "middle":
            atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
            expected = SigmaAlgebra(
                sample_id_to_atom_id=atom_ids, sample_space=sample_space
            )
        else:  # power_set
            expected = SigmaAlgebra.power_set(sample_space)
        assert alg == expected

    @pytest.mark.parametrize(
        "time_value,expected_algebra_type",
        [
            pytest.param(0.5, "trivial", id="between_first_and_second"),
            pytest.param(1.7, "middle", id="between_second_and_third"),
        ],
    )
    def test_at_interpolated_time_discrete(
        self, discrete_filtration, sample_space, time_value, expected_algebra_type
    ):
        """Test accessing sigma algebra at interpolated times in discrete filtration."""
        alg = discrete_filtration.at[time_value]
        if expected_algebra_type == "trivial":
            expected = SigmaAlgebra.trivial(sample_space)
        else:  # middle
            atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
            expected = SigmaAlgebra(
                sample_id_to_atom_id=atom_ids, sample_space=sample_space
            )
        assert alg == expected

    @pytest.mark.parametrize(
        "time_value,expected_algebra_type",
        [
            pytest.param(0.3, "trivial", id="between_first_and_second"),
            pytest.param(0.75, "middle", id="between_second_and_third"),
            pytest.param(0.99, "middle", id="just_before_last"),
        ],
    )
    def test_at_interpolated_time_continuous(
        self, continuous_filtration, sample_space, time_value, expected_algebra_type
    ):
        """Test accessing sigma algebra at interpolated times in continuous filtration."""
        alg = continuous_filtration.at[time_value]
        if expected_algebra_type == "trivial":
            expected = SigmaAlgebra.trivial(sample_space)
        else:  # middle
            atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
            expected = SigmaAlgebra(
                sample_id_to_atom_id=atom_ids, sample_space=sample_space
            )
        assert alg == expected

    @pytest.mark.parametrize(
        "time_value",
        [
            pytest.param(-1, id="negative_time"),
            pytest.param(-0.5, id="small_negative_time"),
        ],
    )
    def test_at_time_before_start_raises_error(self, discrete_filtration, time_value):
        """Test that accessing time before filtration start raises ValueError."""
        with pytest.raises(ValueError, match="before the start"):
            discrete_filtration.at[time_value]

    @pytest.mark.parametrize(
        "time_value",
        [
            pytest.param(10, id="large_time"),
            pytest.param(3, id="slightly_after_end"),
        ],
    )
    def test_at_time_after_end_raises_error(self, discrete_filtration, time_value):
        """Test that accessing time after filtration end raises ValueError."""
        with pytest.raises(ValueError, match="after the end"):
            discrete_filtration.at[time_value]


class TestSequenceMethods:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=3)
        return Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    def test_len_returns_length(self, filtration):
        """Test that len returns number of sigma algebras minus one."""
        assert len(filtration) == 2
        assert len(filtration.sigma_algebras) == 3

    def test_iteration_yields_all_sigma_algebras(self, filtration, sample_space):
        """Test that iteration yields all sigma algebras in order."""
        algebras = list(filtration)
        assert len(algebras) == 3
        assert algebras[0] == SigmaAlgebra.trivial(sample_space)
        assert algebras[2] == SigmaAlgebra.power_set(sample_space)

    def test_iteration_order(self, filtration):
        """Test that iteration order matches the filtration order."""
        algebras = list(filtration)
        for i in range(len(algebras)):
            assert algebras[i] == filtration.sigma_algebras[i]


class TestRepresentation:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=2)
        return Filtration(sigma_algebras=[trivial, power_set], time=time, name="F")

    def test_repr(self, filtration):
        """Test the __repr__ method."""
        result = repr(filtration)
        assert "Filtration" in result
        assert "name='F'" in result
        assert "length=1" in result

    def test_str_contains_name(self, filtration):
        """Test that __str__ contains filtration name."""
        result = str(filtration)
        assert "Filtration (F)" in result

    def test_str_contains_time(self, filtration):
        """Test that __str__ contains time information."""
        result = str(filtration)
        assert "Time" in result

    def test_str_contains_sigma_algebras(self, filtration):
        """Test that __str__ contains all sigma algebras."""
        result = str(filtration)
        assert "At time 0:" in result
        assert "At time 1:" in result


class TestFilteredSigmaAlgebraConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=3)
        return Filtration(
            sigma_algebras=[trivial, middle, power_set], time=time, name="F"
        )

    def test_constructor_with_sigma_algebra(self, filtration):
        """Test FilteredSigmaAlgebra constructor with explicit sigma_algebra."""
        fsa = FilteredSigmaAlgebra(
            filtration=filtration, sigma_algebra=filtration.finest
        )
        assert fsa.filtration == filtration
        assert fsa.sigma_algebra == filtration.finest

    def test_constructor_without_sigma_algebra(self, filtration):
        """Test FilteredSigmaAlgebra constructor without sigma_algebra."""
        fsa = FilteredSigmaAlgebra(filtration=filtration)
        assert fsa.filtration == filtration
        assert fsa.sigma_algebra == filtration.finest
