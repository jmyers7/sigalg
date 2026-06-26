from numbers import Real

import pandas as pd
import pytest

from sigalg.core import Time

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_no_parameters(self):
        """Test constructor with no parameters."""
        time = Time()

        assert time.name == "T"
        assert time.variable_names is None
        assert time.indices is None
        assert time.data is None
        assert time.is_discrete is None

    def test_from_list_with_default_parameters(self):
        """Test constructor from list with default parameters."""
        T = Time(indices=[0, 1, 2])
        expected_data = pd.Index([0, 1, 2], name="time")

        assert T.name == "T"
        assert T.variable_names == ["time"]
        assert T.indices == [0, 1, 2]
        pd.testing.assert_index_equal(T.data, expected_data)
        assert T.is_discrete is True

    def test_from_list_with_custom_parameters(self):
        """Test constructor from list with custom parameters."""
        S = Time(name="S", indices=[0, 1, 2], variable_names=["time_idx"])
        expected_data = pd.Index([0, 1, 2], name="time_idx")

        assert S.name == "S"
        assert S.variable_names == ["time_idx"]
        assert S.indices == [0, 1, 2]
        pd.testing.assert_index_equal(S.data, expected_data)
        assert S.is_discrete is True

    def test_non_monotonically_increasing_indices_raises(self):
        """Test that non-monotonically increasing indices raise ValidationError."""
        indices = [2, 1]

        with pytest.raises(ValueError, match="Time index must be in ascending order"):
            Time(indices=indices)

    def test_from_pandas_with_default_parameters(self):
        """Test constructor from pandas with default parameters."""
        indices = pd.Index([0, 1, 2])
        T = Time(indices=indices)
        expected_data = pd.Index([0, 1, 2], name="time")

        assert T.name == "T"
        assert T.variable_names == ["time"]
        assert T.indices == [0, 1, 2]
        pd.testing.assert_index_equal(T.data, expected_data)
        assert T.is_discrete is True

    def test_from_pandas_with_custom_parameters(self):
        """Test constructor from pandas with custom parameters."""
        indices = pd.Index([0, 1, 2])
        S = Time(indices=indices, name="S", variable_names=["time_idx"])
        expected_data = pd.Index([0, 1, 2], name="time_idx")

        assert S.name == "S"
        assert S.variable_names == ["time_idx"]
        assert S.indices == [0, 1, 2]
        pd.testing.assert_index_equal(S.data, expected_data)
        assert S.is_discrete is True


class TestDiscrete:
    def test_discrete_with_custom_start_and_length_and_names(self):
        """Test discrete constructor with custom start and length and names."""
        S = Time.discrete(start=5, length=3, name="S", variable_name="time_idx")
        expected_indices = [5, 6, 7, 8]
        expected_data = pd.Index(expected_indices, name="time_idx")

        assert S.name == "S"
        assert S.variable_names == ["time_idx"]
        assert S.indices == expected_indices
        pd.testing.assert_index_equal(S.data, expected_data)
        assert S.is_discrete is True

    def test_discrete_with_default_start_and_length(self):
        """Test discrete constructor with default start and length."""
        T = Time.discrete(start=0, length=5)
        expected_indices = [0, 1, 2, 3, 4, 5]
        expected_data = pd.Index(expected_indices, name="time")

        assert T.name == "T"
        assert T.variable_names == ["time"]
        assert T.indices == expected_indices
        pd.testing.assert_index_equal(T.data, expected_data)
        assert T.is_discrete is True

    def test_discrete_with_custom_start_and_stop(self):
        """Test discrete constructor with custom start and stop."""
        T = Time.discrete(start=2, stop=4)
        expected_indices = [2, 3, 4]
        expected_data = pd.Index(expected_indices, name="time")

        assert T.name == "T"
        assert T.variable_names == ["time"]
        assert T.indices == expected_indices
        pd.testing.assert_index_equal(T.data, expected_data)
        assert T.is_discrete is True

    def test_discrete_with_default_start_and_stop(self):
        """Test discrete constructor with default start and stop."""
        T = Time.discrete(start=0, stop=3)
        expected_indices = [0, 1, 2, 3]
        expected_data = pd.Index(expected_indices, name="time")

        assert T.name == "T"
        assert T.variable_names == ["time"]
        assert T.indices == expected_indices
        pd.testing.assert_index_equal(T.data, expected_data)
        assert T.is_discrete is True


class TestContinuous:
    def test_continuous_with_num_points_and_custom_names(self):
        """Test continuous constructor with num_points and custom names."""
        S = Time.continuous(
            start=0.0, stop=1.0, num_points=5, name="S", variable_name="time_idx"
        )
        expected_indices = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected_data = pd.Index(expected_indices, name="time_idx")

        assert S.name == "S"
        assert S.variable_names == ["time_idx"]
        assert S.indices == pytest.approx(expected_indices)
        pd.testing.assert_index_equal(S.data, expected_data, check_exact=False)
        assert S.is_discrete is False

    def test_continuous_with_dt(self):
        """Test continuous constructor with dt."""
        T = Time.continuous(start=0.0, stop=1.0, dt=0.2, variable_name="t")
        expected_indices = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        expected_data = pd.Index(expected_indices, name="t")

        assert T.name == "T"
        assert T.variable_names == ["t"]
        assert T.indices == pytest.approx(expected_indices)
        pd.testing.assert_index_equal(T.data, expected_data, check_exact=False)
        assert T.is_discrete is False

    def test_invalid_start_greater_than_stop_raises(self):
        """Test that start > stop raises ValueError."""
        with pytest.raises(ValueError, match="start must be less than stop"):
            Time.continuous(start=1.0, stop=0.0, num_points=5)

    def test_invalid_neither_dt_nor_num_points_raises(self):
        """Test that neither dt nor num_points raises ValueError."""
        with pytest.raises(ValueError, match="Specify exactly one of dt or num_points"):
            Time.continuous(start=0.0, stop=1.0)

    def test_invalid_both_dt_and_num_points_raises(self):
        """Test that both dt and num_points raises ValueError."""
        with pytest.raises(ValueError, match="Specify exactly one of dt or num_points"):
            Time.continuous(start=0.0, stop=1.0, num_points=5, dt=0.1)

    def test_invalid_num_points_less_than_2_raises(self):
        """Test that num_points < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_points must be an integer >= 2"):
            Time.continuous(start=0.0, stop=1.0, num_points=1)

    def test_invalid_negative_dt_raises(self):
        """Test that negative dt raises ValueError."""
        with pytest.raises(ValueError, match="dt must be a positive real number"):
            Time.continuous(start=0.0, stop=1.0, dt=-0.1)

    def test_invalid_non_real_start_raises(self):
        """Test that non-real start raises TypeError."""
        with pytest.raises(TypeError, match="start and stop must be real numbers"):
            Time.continuous(start="zero", stop=1.0, num_points=5)

    def test_invalid_non_real_stop_raises(self):
        """Test that non-real stop raises TypeError."""
        with pytest.raises(TypeError, match="start and stop must be real numbers"):
            Time.continuous(start=0.0, stop="one", num_points=5)

    def test_invalid_non_real_dt_raises(self):
        """Test that non-real dt raises ValueError."""
        with pytest.raises(ValueError, match="dt must be a positive real number"):
            Time.continuous(start=0.0, stop=1.0, dt="point_one")


# --------------------- test data access --------------------- #


class TestGetItem:
    def test_getitem(self):
        """Test __getitem__ method with various position types."""
        time = Time.discrete(start=0, length=5)
        single_item = time[2]
        slice_item = time[1:4]
        list_item = time[[0, 3, 4]]

        assert isinstance(single_item, Real)
        assert single_item == 2
        assert isinstance(slice_item, Time)
        assert list(slice_item) == [1, 2, 3]
        assert isinstance(list_item, Time)
        assert list(list_item) == [0, 3, 4]


class TestFindNearestTime:
    @pytest.fixture
    def discrete_time(self):
        return Time.discrete(start=0, length=5)

    @pytest.fixture
    def continuous_time(self):
        return Time.continuous(start=0.0, stop=1.0, num_points=5)

    @pytest.fixture
    def negative_time(self):
        return Time.discrete(start=-5, length=10)

    def test_exact_match_discrete(self, discrete_time):
        """Test finding exact match in discrete time."""
        result = discrete_time.find_nearest_time(3)
        assert result == 3

    def test_exact_match_continuous(self, continuous_time):
        """Test finding exact match in continuous time."""
        result = continuous_time.find_nearest_time(0.5)
        assert result == pytest.approx(0.5)

    def test_rounding_down_discrete(self, discrete_time):
        """Test rounding down to nearest time point."""
        result = discrete_time.find_nearest_time(2.3)
        assert result == 2

    def test_rounding_up_discrete(self, discrete_time):
        """Test rounding up to nearest time point."""
        result = discrete_time.find_nearest_time(4.7)
        assert result == 5

    def test_rounding_continuous(self, continuous_time):
        """Test rounding to nearest time point in continuous time."""
        result = continuous_time.find_nearest_time(0.6)
        assert result == pytest.approx(0.5)

    def test_single_element_time(self):
        """Test finding nearest time with single-element Time index."""
        time = Time([10])
        result = time.find_nearest_time(10)
        assert result == 10

    def test_negative_time_values(self, negative_time):
        """Test finding nearest time with negative time values."""
        result = negative_time.find_nearest_time(-2.7)
        assert result == -3

    def test_negative_exact_match(self, negative_time):
        """Test exact match with negative time values."""
        result = negative_time.find_nearest_time(-1)
        assert result == -1

    def test_invalid_time_point_before_start_raises(self, discrete_time):
        """Test that time_point before start raises ValueError."""
        with pytest.raises(ValueError, match="is before the start of the Time index"):
            discrete_time.find_nearest_time(-1)

    def test_invalid_time_point_after_end_raises(self, discrete_time):
        """Test that time_point after end raises ValueError."""
        with pytest.raises(ValueError, match="is after the end of the Time index"):
            discrete_time.find_nearest_time(10)


class TestInsertTime:
    @pytest.fixture
    def discrete_time(self):
        return Time.discrete(start=0, length=5)

    @pytest.fixture
    def continuous_time(self):
        return Time.continuous(start=0.0, stop=1.0, num_points=5)

    def test_insert_at_beginning_discrete(self, discrete_time):
        """Test inserting time point at the beginning of discrete time."""
        new_time = discrete_time.insert_time(-1)
        expected_indices = [-1, 0, 1, 2, 3, 4, 5]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "insert(T)"

    def test_insert_at_middle_discrete(self, discrete_time):
        """Test inserting time point in the middle of discrete time."""
        time = Time([0, 2, 4, 6, 8, 10])
        new_time = time.insert_time(5)
        expected_indices = [0, 2, 4, 5, 6, 8, 10]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "insert(T)"

    def test_insert_at_end_discrete(self, discrete_time):
        """Test inserting time point at the end of discrete time."""
        new_time = discrete_time.insert_time(6)
        expected_indices = [0, 1, 2, 3, 4, 5, 6]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "insert(T)"

    def test_insert_at_beginning_continuous(self, continuous_time):
        """Test inserting time point at the beginning of continuous time."""
        new_time = continuous_time.insert_time(-0.5)
        expected_indices = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0]

        assert new_time.indices == pytest.approx(expected_indices)
        assert new_time.is_discrete is False
        assert new_time.name == "insert(T)"

    def test_insert_at_middle_continuous(self, continuous_time):
        """Test inserting time point in the middle of continuous time."""
        new_time = continuous_time.insert_time(0.6)
        expected_indices = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0]

        assert new_time.indices == pytest.approx(expected_indices)
        assert new_time.is_discrete is False
        assert new_time.name == "insert(T)"

    def test_insert_at_end_continuous(self, continuous_time):
        """Test inserting time point at the end of continuous time."""
        new_time = continuous_time.insert_time(1.5)
        expected_indices = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

        assert new_time.indices == pytest.approx(expected_indices)
        assert new_time.is_discrete is False
        assert new_time.name == "insert(T)"

    def test_invalid_non_real_raises(self, discrete_time):
        """Test that non-real number raises TypeError."""
        with pytest.raises(TypeError, match="time must be a real number"):
            discrete_time.insert_time("not_a_number")

    def test_invalid_empty_index_raises(self):
        """Test that empty Time index raises ValueError."""
        time = Time()
        with pytest.raises(ValueError, match="Time index is empty"):
            time.insert_time(5)

    def test_invalid_duplicate_time_raises(self, discrete_time):
        """Test that duplicate time point raises ValueError."""
        with pytest.raises(ValueError, match="already exists in the Time index"):
            discrete_time.insert_time(3)


class TestRemoveTime:
    @pytest.fixture
    def discrete_time(self):
        return Time.discrete(start=0, length=5)

    @pytest.fixture
    def continuous_time(self):
        return Time.continuous(start=0.0, stop=1.0, num_points=5)

    def test_remove_by_time_at_beginning(self, discrete_time):
        """Test removing time point by value at the beginning."""
        new_time = discrete_time.remove_time(time=0)
        expected_indices = [1, 2, 3, 4, 5]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "remove(T)"

    def test_remove_by_time_at_middle(self, discrete_time):
        """Test removing time point by value in the middle."""
        new_time = discrete_time.remove_time(time=3)
        expected_indices = [0, 1, 2, 4, 5]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "remove(T)"

    def test_remove_by_time_at_end(self, discrete_time):
        """Test removing time point by value at the end."""
        new_time = discrete_time.remove_time(time=5)
        expected_indices = [0, 1, 2, 3, 4]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "remove(T)"

    def test_remove_by_pos_at_beginning(self, discrete_time):
        """Test removing time point by position at the beginning."""
        new_time = discrete_time.remove_time(pos=0)
        expected_indices = [1, 2, 3, 4, 5]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "remove(T)"

    def test_remove_by_pos_at_middle(self, discrete_time):
        """Test removing time point by position in the middle."""
        new_time = discrete_time.remove_time(pos=3)
        expected_indices = [0, 1, 2, 4, 5]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "remove(T)"

    def test_remove_by_pos_at_end(self, discrete_time):
        """Test removing time point by position at the end."""
        new_time = discrete_time.remove_time(pos=5)
        expected_indices = [0, 1, 2, 3, 4]

        assert new_time.indices == expected_indices
        assert new_time.is_discrete is True
        assert new_time.name == "remove(T)"

    def test_remove_continuous_by_time(self, continuous_time):
        """Test removing from continuous time by value."""
        new_time = continuous_time.remove_time(time=0.5)
        expected_indices = [0.0, 0.25, 0.75, 1.0]

        assert new_time.indices == pytest.approx(expected_indices)
        assert new_time.is_discrete is False
        assert new_time.name == "remove(T)"

    def test_remove_continuous_by_pos(self, continuous_time):
        """Test removing from continuous time by position."""
        new_time = continuous_time.remove_time(pos=2)
        expected_indices = [0.0, 0.25, 0.75, 1.0]

        assert new_time.indices == pytest.approx(expected_indices)
        assert new_time.is_discrete is False
        assert new_time.name == "remove(T)"

    def test_invalid_empty_index_raises(self):
        """Test that empty Time index raises ValueError."""
        time = Time()
        with pytest.raises(ValueError, match="Time index is empty"):
            time.remove_time(time=5)

    def test_invalid_non_real_time_raises(self, discrete_time):
        """Test that non-real time raises TypeError."""
        with pytest.raises(TypeError, match="time must be a real number"):
            discrete_time.remove_time(time="not_real")

    def test_invalid_non_int_pos_raises(self, discrete_time):
        """Test that non-integer pos raises TypeError."""
        with pytest.raises(TypeError, match="pos must be an integer"):
            discrete_time.remove_time(pos=1.5)

    def test_invalid_time_not_in_index_raises(self, discrete_time):
        """Test that time not in index raises ValueError."""
        with pytest.raises(ValueError, match="does not exist in the Time index"):
            discrete_time.remove_time(time=10)

    def test_invalid_both_time_and_pos_raises(self, discrete_time):
        """Test that specifying both time and pos raises ValueError."""
        with pytest.raises(
            ValueError, match="Only one of time or pos must be specified"
        ):
            discrete_time.remove_time(time=2, pos=2)

    def test_invalid_neither_time_nor_pos_raises(self, discrete_time):
        """Test that specifying neither time nor pos raises ValueError."""
        with pytest.raises(ValueError, match="Either time or pos must be specified"):
            discrete_time.remove_time()

    def test_invalid_pos_out_of_bounds_negative_raises(self, discrete_time):
        """Test that negative pos raises ValueError."""
        with pytest.raises(ValueError, match="is out of bounds"):
            discrete_time.remove_time(pos=-1)

    def test_invalid_pos_out_of_bounds_too_large_raises(self, discrete_time):
        """Test that pos >= length raises ValueError."""
        with pytest.raises(ValueError, match="is out of bounds"):
            discrete_time.remove_time(pos=10)


# --------------------- test equality --------------------- #


class TestEquality:
    def test_equality_identical_discrete_times(self):
        """Test equality with identical discrete times."""
        time1 = Time.discrete(start=0, length=5, name="time1")
        time2 = Time.discrete(start=0, length=5, name="time2")
        assert time1 == time2

    def test_equality_identical_continuous_times(self):
        """Test equality with identical continuous times."""
        time1 = Time.continuous(start=0.0, stop=1.0, num_points=5, name="time1")
        time2 = Time.continuous(start=0.0, stop=1.0, num_points=5, name="time2")
        assert time1 == time2

    def test_equality_same_values_different_names(self):
        """Test equality when times have same values but different names."""
        time1 = Time.discrete(start=0, length=5, name="First")
        time2 = Time.discrete(start=0, length=5, name="Second")
        assert time1 == time2

    def test_non_equality_different_time_points(self):
        """Test inequality when time points differ."""
        time1 = Time.discrete(start=0, length=5)
        time2 = Time.discrete(start=0, length=10)
        assert time1 != time2

    def test_non_equality_different_is_discrete(self):
        """Test inequality when is_discrete values differ."""
        time1 = Time.discrete(start=0, length=5)
        time2 = Time.continuous(start=0.0, stop=5.0, num_points=6)
        assert time1 != time2

    def test_non_equality_wrong_type(self):
        """Test inequality when comparing with non-Time object."""
        time = Time.discrete(start=0, length=5)
        other = [0, 1, 2, 3, 4, 5]
        assert time != other

    def test_non_equality_with_string(self):
        """Test inequality when comparing with string."""
        time = Time.discrete(start=0, length=5)
        assert time != "not_a_time"


# --------------------- test set-theoretic operations --------------------- #


class TestAnd:
    def test_partial_overlap_discrete(self):
        """Test intersection with partial overlap in discrete time."""
        time1 = Time.discrete(start=0, length=5, name="time1")
        time2 = Time.discrete(start=3, length=5, name="time2")
        intersection = time1 & time2
        expected_indices = [3, 4, 5]

        assert intersection.indices == expected_indices
        assert intersection.name == "time1 intersect time2"
        assert intersection.is_discrete is True

    def test_partial_overlap_continuous(self):
        """Test intersection with partial overlap in continuous time."""
        time1 = Time.continuous(start=0.0, stop=1.0, num_points=5, name="time1")
        time2 = Time.continuous(start=0.5, stop=1.5, num_points=5, name="time2")
        intersection = time1 & time2
        expected_indices = [0.5, 0.75, 1.0]

        assert intersection.indices == pytest.approx(expected_indices)
        assert intersection.name == "time1 intersect time2"
        assert intersection.is_discrete is False

    def test_full_overlap(self):
        """Test intersection with full overlap."""
        time1 = Time.discrete(start=0, length=5, name="time1")
        time2 = Time.discrete(start=0, length=5, name="time2")
        intersection = time1 & time2
        expected_indices = [0, 1, 2, 3, 4, 5]

        assert intersection.indices == expected_indices
        assert intersection.name == "time1 intersect time2"
        assert intersection.is_discrete is True

    def test_single_element_overlap(self):
        """Test intersection with single overlapping point."""
        time1 = Time.discrete(start=0, length=5, name="time1")
        time2 = Time.discrete(start=5, length=5, name="time2")
        intersection = time1 & time2

        assert intersection.indices == [5]
        assert intersection.name == "time1 intersect time2"
        assert intersection.is_discrete is True

    def test_intersection_preserves_discrete_flag(self):
        """Test that is_discrete flag is preserved in intersection."""
        time1 = Time.discrete(start=0, length=5, name="time1")
        time2 = Time.discrete(start=3, length=5, name="time2")
        intersection = time1 & time2

        assert intersection.is_discrete is True

    def test_invalid_mismatched_is_discrete_raises(self):
        """Test that mismatched is_discrete values raise ValueError."""
        time1 = Time.discrete(start=0, length=5, name="time1")
        time2 = Time.continuous(start=0.0, stop=5.0, num_points=6, name="time2")

        with pytest.raises(
            ValueError,
            match="Cannot intersect Time indices with different is_discrete values",
        ):
            time1 & time2
