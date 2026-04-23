import pytest

from sigalg.core import (
    Filtration,
    SampleSpace,
    SigmaAlgebra,
    Time,
)


class TestConstructor:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_sequence(size=4, prefix="s", initial_index=0)

    @pytest.fixture
    def trivial_algebra(self, sample_space):
        return SigmaAlgebra.trivial(sample_space)

    @pytest.fixture
    def middle_algebra(self, sample_space):
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        return SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

    @pytest.fixture
    def power_set_algebra(self, sample_space):
        return SigmaAlgebra.power_set(sample_space)

    def test_constructor_discrete_time_custom_name(
        self,
        sample_space,
        trivial_algebra,
        middle_algebra,
        power_set_algebra,
    ):
        """Test constructor with discrete time and custom name."""
        time = Time.discrete(start=0, length=2)
        sigma_algebras = [trivial_algebra, middle_algebra, power_set_algebra]
        name = "F"

        filtration = Filtration(time=time, name=name).from_list(sigma_algebras)

        assert len(filtration.sigma_algebras) == 3
        assert filtration.name == name
        assert filtration.time == time

    def test_constructor_continuous_time_custom_name(
        self,
        sample_space,
        trivial_algebra,
        middle_algebra,
        power_set_algebra,
    ):
        """Test constructor with continuous time and custom name."""
        time = Time.continuous(start=0.0, stop=1.0, num_points=3)
        sigma_algebras = [trivial_algebra, middle_algebra, power_set_algebra]
        name = "G"

        filtration = Filtration(time=time, name=name).from_list(sigma_algebras)

        assert len(filtration.sigma_algebras) == 3
        assert filtration.name == name
        assert filtration.time == time

    def test_constructor_discrete_time_default_name(
        self,
        sample_space,
        trivial_algebra,
        power_set_algebra,
    ):
        """Test constructor with discrete time and default name."""
        time = Time.discrete(start=0, length=1)
        sigma_algebras = [trivial_algebra, power_set_algebra]

        filtration = Filtration(time=time).from_list(sigma_algebras)
        name = "F"

        assert len(filtration.sigma_algebras) == 2
        assert filtration.name == name
        assert filtration.time == time

    def test_constructor_continuous_time_none_name(
        self,
        sample_space,
        trivial_algebra,
        power_set_algebra,
    ):
        """Test constructor with continuous time and None name."""
        time = Time.continuous(start=0.0, stop=2.0, num_points=2)
        sigma_algebras = [trivial_algebra, power_set_algebra]
        name = None

        filtration = Filtration(time=time, name=name).from_list(sigma_algebras)

        assert len(filtration.sigma_algebras) == 2
        assert filtration.name == name
        assert filtration.time == time

    def test_constructor_stores_sigma_algebras(
        self, trivial_algebra, power_set_algebra
    ):
        """Test that constructor correctly stores sigma algebras."""
        time = Time.discrete(start=0, length=1)
        filtration = Filtration(time=time, name="F").from_list(
            [trivial_algebra, power_set_algebra]
        )
        assert filtration.sigma_algebras[0] == trivial_algebra
        assert filtration.sigma_algebras[1] == power_set_algebra

    def test_constructor_stores_sample_space(
        self, sample_space, trivial_algebra, power_set_algebra
    ):
        """Test that constructor correctly stores sample space."""
        time = Time.discrete(start=0, length=1)
        filtration = Filtration(time=time, name="F").from_list(
            [trivial_algebra, power_set_algebra]
        )
        assert filtration.sample_space == sample_space


class TestValidation:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_sequence(size=4, prefix="s", initial_index=0)

    @pytest.fixture
    def other_sample_space(self):
        return SampleSpace().from_list(["a", "b", "c"])

    def test_invalid_sigma_algebras_empty_list_raises_error(self):
        """Test that empty sigma_algebras list raises ValueError."""
        time = Time.discrete(start=0, length=1)
        invalid_sigma_algebras = []
        error_type = ValueError
        error_match = "non-empty list"

        with pytest.raises(error_type, match=error_match):
            Filtration(time=time, name="F").from_list(invalid_sigma_algebras)

    def test_invalid_sigma_algebras_not_a_list_raises_error(self):
        """Test that non-list sigma_algebras raises ValueError."""
        time = Time.discrete(start=0, length=1)
        invalid_sigma_algebras = "not a list"
        error_type = ValueError
        error_match = "non-empty list"

        with pytest.raises(error_type, match=error_match):
            Filtration(time=time, name="F").from_list(invalid_sigma_algebras)

    def test_non_sigma_algebra_element_raises_error(self, sample_space):
        """Test that non-SigmaAlgebra elements in list raise ValueError."""
        time = Time.discrete(start=0, length=2)
        alg = SigmaAlgebra.trivial(sample_space)
        with pytest.raises(ValueError, match="instances of SigmaAlgebra"):
            Filtration(time=time, name="F").from_list([alg, "not an algebra"])

    def test_invalid_time_list_instead_of_time_raises_error(self, sample_space):
        """Test that list instead of Time raises TypeError."""
        alg = SigmaAlgebra.trivial(sample_space)
        invalid_time = [0, 1, 2]
        error_match = "must be an Index object"

        with pytest.raises(TypeError, match=error_match):
            Filtration(time=invalid_time, name="F").from_list([alg])

    def test_invalid_time_dict_instead_of_time_raises_error(self, sample_space):
        """Test that dict instead of Time raises TypeError."""
        alg = SigmaAlgebra.trivial(sample_space)
        invalid_time = {"start": 0}
        error_match = "must be an Index object"

        with pytest.raises(TypeError, match=error_match):
            Filtration(time=invalid_time, name="F").from_list([alg])

    def test_invalid_time_string_instead_of_time_raises_error(self, sample_space):
        """Test that string instead of Time raises TypeError."""
        alg = SigmaAlgebra.trivial(sample_space)
        invalid_time = "time_string"
        error_match = "must be an Index object"

        with pytest.raises(TypeError, match=error_match):
            Filtration(time=invalid_time, name="F").from_list([alg])

    def test_invalid_name_list_name_raises_error(self, sample_space):
        """Test that list name raises TypeError."""
        time = Time.discrete(start=0, length=1)
        invalid_name = ["list", "name"]

        with pytest.raises(TypeError, match="must be a hashable"):
            Filtration(time=time, name=invalid_name)

    def test_invalid_name_dict_name_raises_error(self, sample_space):
        """Test that dict name raises TypeError."""
        time = Time.discrete(start=0, length=1)
        invalid_name = {"key": "value"}

        with pytest.raises(TypeError, match="must be a hashable"):
            Filtration(time=time, name=invalid_name)

    def test_mismatched_lengths_raises_error(self, sample_space):
        """Test that mismatched lengths between sigma_algebras and time raise ValueError."""
        time = Time.discrete(start=0, length=3)
        alg1 = SigmaAlgebra.trivial(sample_space)
        alg2 = SigmaAlgebra.power_set(sample_space)
        with pytest.raises(ValueError, match="must match the length"):
            Filtration(time=time, name="F").from_list([alg1, alg2])

    def test_different_sample_spaces_raises_error(
        self, sample_space, other_sample_space
    ):
        """Test that sigma algebras with different sample spaces raise ValueError."""
        time = Time.discrete(start=0, length=1)
        alg1 = SigmaAlgebra.trivial(sample_space)
        alg2 = SigmaAlgebra.trivial(other_sample_space)
        with pytest.raises(ValueError, match="same sample space"):
            Filtration(time=time, name="F").from_list([alg1, alg2])

    def test_non_increasing_algebras_raises_error(self, sample_space):
        """Test that non-increasing sigma algebras raise ValueError."""
        time = Time.discrete(start=0, length=1)
        trivial = SigmaAlgebra.trivial(sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        with pytest.raises(ValueError, match="do not form a valid filtration"):
            Filtration(time=time, name="F").from_list([power_set, trivial])


class TestFromPandas:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_sequence(size=5, prefix="s", initial_index=0)

    def test_from_pandas_basic(self):
        """Test from_pandas with basic DataFrame."""
        import pandas as pd

        df = pd.DataFrame(
            {
                0: [0, 0, 0, 0, 0],  # Trivial
                1: [0, 0, 0, 1, 1],  # Two atoms
                2: [0, 1, 2, 3, 4],  # Power set
            }
        )

        filtration = Filtration().from_pandas(df)

        assert len(filtration.sigma_algebras) == 3
        assert filtration.name == "F"
        assert len(filtration.time) == 3

    def test_from_pandas_with_time_index(self):
        """Test from_pandas when time is pre-specified."""
        import pandas as pd

        df = pd.DataFrame(
            {
                0: [0, 0, 0, 0, 0],  # Trivial
                1: [0, 0, 0, 1, 1],  # Two atoms
            }
        )

        time = Time.discrete(start=0, length=1)
        filtration = Filtration(time=time, name="F").from_pandas(df)

        assert len(filtration.sigma_algebras) == 2
        assert filtration.name == "F"
        assert filtration.time == time

    def test_from_pandas_creates_correct_sigma_algebras(self):
        """Test that from_pandas creates correct sigma algebras."""
        import pandas as pd

        df = pd.DataFrame(
            {
                0: [0, 0, 0],  # Trivial
                1: [0, 0, 1],  # Middle
                2: [0, 1, 2],  # Power set
            },
            index=["s_0", "s_1", "s_2"],
        )

        filtration = Filtration().from_pandas(df)

        # Check first sigma algebra is trivial
        assert filtration.sigma_algebras[0].num_atoms == 1

        # Check second sigma algebra has 2 atoms
        assert filtration.sigma_algebras[1].num_atoms == 2

        # Check third sigma algebra is power set
        assert filtration.sigma_algebras[2].num_atoms == 3

    def test_from_pandas_invalid_not_dataframe_raises_error(self):
        """Test that non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="must be a `pd.DataFrame`"):
            Filtration().from_pandas([[0, 1], [0, 2]])

    def test_from_pandas_invalid_filtration_raises_error(self):
        """Test that invalid filtration raises ValueError."""
        import pandas as pd

        # Non-increasing: second column is not a refinement of first
        df = pd.DataFrame(
            {
                0: [0, 0, 1],
                1: [0, 1, 0],  # Invalid: atom 1 in col 0 maps to both 0 and 1 in col 1
            }
        )

        with pytest.raises(ValueError, match="does not represent a valid filtration"):
            Filtration().from_pandas(df)

    def test_from_pandas_time_mismatch_raises_error(self):
        """Test that mismatched time index raises ValueError."""
        import pandas as pd

        df = pd.DataFrame(
            {
                0: [0, 0, 0],
                1: [0, 0, 1],
            }
        )

        # Time has length 3 but df has only 2 columns
        time = Time.discrete(start=0, length=3)

        with pytest.raises(ValueError, match="must match the columns"):
            Filtration(time=time).from_pandas(df)

    def test_from_pandas_with_custom_columns(self):
        """Test from_pandas with custom column names."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "t0": [0, 0, 0],
                "t1": [0, 0, 1],
                "t2": [0, 1, 2],
            }
        )

        filtration = Filtration().from_pandas(df, is_time=False)

        assert len(filtration.sigma_algebras) == 3
        assert list(filtration.time.data) == ["t0", "t1", "t2"]


class TestProperties:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_sequence(size=4, prefix="s", initial_index=0)

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        middle = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=2)
        return Filtration(time=time, name="F").from_list([trivial, middle, power_set])

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
        return SampleSpace().from_sequence(size=3, prefix="s", initial_index=0)

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=1)
        return Filtration(time=time, name="F").from_list([trivial, power_set])

    def test_name_setter_string_name(self, filtration):
        """Test that name setter correctly updates name with string."""
        new_name = "NewName"
        filtration.name = new_name
        assert filtration.name == new_name

    def test_name_setter_single_char_name(self, filtration):
        """Test that name setter correctly updates name with single char."""
        new_name = "G"
        filtration.name = new_name
        assert filtration.name == new_name

    def test_name_setter_int_name(self, filtration):
        """Test that name setter correctly updates name with int."""
        new_name = 42
        filtration.name = new_name
        assert filtration.name == new_name

    def test_name_setter_tuple_name(self, filtration):
        """Test that name setter correctly updates name with tuple."""
        new_name = ("tuple", "name")
        filtration.name = new_name
        assert filtration.name == new_name

    def test_name_setter_none_name(self, filtration):
        """Test that name setter correctly updates name with None."""
        new_name = None
        filtration.name = new_name
        assert filtration.name == new_name

    def test_name_setter_with_unhashable_list_name_raises_error(self, filtration):
        """Test that setting unhashable list name raises TypeError."""
        invalid_name = ["list", "name"]

        with pytest.raises(TypeError, match="must be a hashable or None"):
            filtration.name = invalid_name

    def test_name_setter_with_unhashable_dict_name_raises_error(self, filtration):
        """Test that setting unhashable dict name raises TypeError."""
        invalid_name = {"dict": "name"}

        with pytest.raises(TypeError, match="must be a hashable or None"):
            filtration.name = invalid_name


class TestDataAccess:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_sequence(size=4, prefix="s", initial_index=0)

    @pytest.fixture
    def discrete_filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        middle = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=2)
        return Filtration(time=time, name="F").from_list([trivial, middle, power_set])

    @pytest.fixture
    def continuous_filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        middle = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.continuous(start=0.0, stop=1.0, num_points=3)
        return Filtration(time=time, name="F").from_list([trivial, middle, power_set])

    def test_at_exact_time_discrete_first_time_point(
        self, discrete_filtration, sample_space
    ):
        """Test accessing sigma algebra at first time point in discrete filtration."""
        time_value = 0
        alg = discrete_filtration.at[time_value]
        expected = SigmaAlgebra.trivial(sample_space)
        assert alg == expected

    def test_at_exact_time_discrete_second_time_point(
        self, discrete_filtration, sample_space
    ):
        """Test accessing sigma algebra at second time point in discrete filtration."""
        time_value = 1
        alg = discrete_filtration.at[time_value]
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        expected = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        assert alg == expected

    def test_at_exact_time_discrete_last_time_point(
        self, discrete_filtration, sample_space
    ):
        """Test accessing sigma algebra at last time point in discrete filtration."""
        time_value = 2
        alg = discrete_filtration.at[time_value]
        expected = SigmaAlgebra.power_set(sample_space)
        assert alg == expected

    def test_at_exact_time_continuous_first_time_point(
        self, continuous_filtration, sample_space
    ):
        """Test accessing sigma algebra at first time point in continuous filtration."""
        time_value = 0.0
        alg = continuous_filtration.at[time_value]
        expected = SigmaAlgebra.trivial(sample_space)
        assert alg == expected

    def test_at_exact_time_continuous_middle_time_point(
        self, continuous_filtration, sample_space
    ):
        """Test accessing sigma algebra at middle time point in continuous filtration."""
        time_value = 0.5
        alg = continuous_filtration.at[time_value]
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        expected = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        assert alg == expected

    def test_at_exact_time_continuous_last_time_point(
        self, continuous_filtration, sample_space
    ):
        """Test accessing sigma algebra at last time point in continuous filtration."""
        time_value = 1.0
        alg = continuous_filtration.at[time_value]
        expected = SigmaAlgebra.power_set(sample_space)
        assert alg == expected

    def test_at_interpolated_time_discrete_between_first_and_second(
        self, discrete_filtration, sample_space
    ):
        """Test accessing sigma algebra at interpolated time between first and second in discrete filtration."""
        time_value = 0.5
        alg = discrete_filtration.at[time_value]
        expected = SigmaAlgebra.trivial(sample_space)
        assert alg == expected

    def test_at_interpolated_time_discrete_between_second_and_third(
        self, discrete_filtration, sample_space
    ):
        """Test accessing sigma algebra at interpolated time between second and third in discrete filtration."""
        time_value = 1.7
        alg = discrete_filtration.at[time_value]
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        expected = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        assert alg == expected

    def test_at_interpolated_time_continuous_between_first_and_second(
        self, continuous_filtration, sample_space
    ):
        """Test accessing sigma algebra at interpolated time between first and second in continuous filtration."""
        time_value = 0.3
        alg = continuous_filtration.at[time_value]
        expected = SigmaAlgebra.trivial(sample_space)
        assert alg == expected

    def test_at_interpolated_time_continuous_between_second_and_third(
        self, continuous_filtration, sample_space
    ):
        """Test accessing sigma algebra at interpolated time between second and third in continuous filtration."""
        time_value = 0.75
        alg = continuous_filtration.at[time_value]
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        expected = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        assert alg == expected

    def test_at_interpolated_time_continuous_just_before_last(
        self, continuous_filtration, sample_space
    ):
        """Test accessing sigma algebra at interpolated time just before last in continuous filtration."""
        time_value = 0.99
        alg = continuous_filtration.at[time_value]
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        expected = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        assert alg == expected

    def test_at_time_before_start_negative_time_raises_error(self, discrete_filtration):
        """Test that accessing negative time before filtration start raises ValueError."""
        time_value = -1
        with pytest.raises(ValueError, match="before the start"):
            discrete_filtration.at[time_value]

    def test_at_time_before_start_small_negative_time_raises_error(
        self, discrete_filtration
    ):
        """Test that accessing small negative time before filtration start raises ValueError."""
        time_value = -0.5
        with pytest.raises(ValueError, match="before the start"):
            discrete_filtration.at[time_value]

    def test_at_time_after_end_large_time_raises_error(self, discrete_filtration):
        """Test that accessing large time after filtration end raises ValueError."""
        time_value = 10
        with pytest.raises(ValueError, match="after the end"):
            discrete_filtration.at[time_value]

    def test_at_time_after_end_slightly_after_end_raises_error(
        self, discrete_filtration
    ):
        """Test that accessing time slightly after filtration end raises ValueError."""
        time_value = 3
        with pytest.raises(ValueError, match="after the end"):
            discrete_filtration.at[time_value]


class TestSequenceMethods:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_sequence(size=4, prefix="s", initial_index=0)

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        middle = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=2)
        return Filtration(time=time, name="F").from_list([trivial, middle, power_set])

    def test_len_returns_length(self, filtration):
        """Test that len returns number of sigma algebras minus one."""
        assert len(filtration) == 3
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
        return SampleSpace().from_sequence(size=3, prefix="s", initial_index=0)

    @pytest.fixture
    def filtration(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space)
        power_set = SigmaAlgebra.power_set(sample_space)
        time = Time.discrete(start=0, length=1)
        return Filtration(time=time, name="F").from_list([trivial, power_set])

    def test_repr(self, filtration):
        """Test the __repr__ method."""
        result = repr(filtration)
        assert "Filtration" in result
        assert "name='F'" in result
        assert "length=2" in result

    def test_str_contains_name(self, filtration):
        """Test that __str__ contains filtration name."""
        result = str(filtration)
        assert "Filtration 'F'" in result

    def test_str_contains_time(self, filtration):
        """Test that __str__ contains time information."""
        result = str(filtration)
        assert "Time" in result

    def test_str_contains_sigma_algebras(self, filtration):
        """Test that __str__ contains all sigma algebras."""
        result = str(filtration)
        assert "At index 0:" in result
        assert "At index 1:" in result


