from numbers import Real

import pytest
from pydantic import ValidationError

from sigalg.core import Time


class TestConstructor:
    def test_constructor_custom_names(self):
        """Test Time constructor with custom names."""
        indices = [0.0, 1.0, 2.0, 3.0]
        is_discrete = False
        name = "TimeIndex"
        data_name = "DataIndex"
        time = Time(name=name, data_name=data_name).from_list(
            indices=indices, is_discrete=is_discrete
        )

        assert time.is_discrete == is_discrete
        assert list(time) == indices
        assert time.name == name
        assert time.data.name == data_name

    def test_constructor_none_names_with_int_indices(self):
        """Test Time constructor with None names and integer indices."""
        indices = [0, 1, 2, 3, 4]
        is_discrete = True
        name = None
        data_name = None
        time = Time(name=name, data_name=data_name).from_list(
            indices=indices, is_discrete=is_discrete
        )

        assert time.is_discrete == is_discrete
        assert list(time) == indices
        assert time.name == name
        assert time.data.name == data_name

    def test_constructor_custom_names_with_float_indices(self):
        """Test Time constructor with custom names and float indices."""
        indices = [0.0, 0.5, 1.0, 1.5]
        is_discrete = False
        name = "CustomTime"
        data_name = "CustomData"
        time = Time(name=name, data_name=data_name).from_list(
            indices=indices, is_discrete=is_discrete
        )

        assert time.is_discrete == is_discrete
        assert list(time) == indices
        assert time.name == name
        assert time.data.name == data_name

    def test_constructor_default_name(self):
        """Test Time constructor with default name."""
        indices = [10, 11, 12]
        is_discrete = True
        data_name = "custom_data_name"
        time = Time(data_name=data_name).from_list(indices, is_discrete)
        name = "T"

        assert time.is_discrete == is_discrete
        assert list(time) == indices
        assert time.name == name
        assert time.data.name == data_name

    def test_constructor_default_data_name(self):
        """Test Time constructor with default data_name."""
        indices = [0.0, 0.1, 0.2]
        is_discrete = False
        name = "custom_name"
        time = Time(name=name).from_list(indices, is_discrete)
        data_name = "time"

        assert time.is_discrete == is_discrete
        assert list(time) == indices
        assert time.name == name
        assert time.data.name == data_name

    def test_invalid_mixed_type_indices_raises(self):
        """Test that mixed type indices raise ValidationError."""
        with pytest.raises(ValidationError):
            Time().from_list(indices=[0, 1, "two", 3], is_discrete=True)

    def test_invalid_empty_indices_raises(self):
        """Test that empty indices raise ValidationError."""
        with pytest.raises(ValidationError):
            Time().from_list(indices=[], is_discrete=True)

    def test_invalid_float_indices_for_discrete_raises(self):
        """Test that float indices for discrete time raise ValidationError."""
        with pytest.raises(ValidationError):
            Time().from_list(indices=[0.0, 1.0, 2.0], is_discrete=True)

    def test_invalid_non_monotonic_int_indices_raises(self):
        """Test that non-monotonic integer indices raise ValidationError."""
        with pytest.raises(ValidationError):
            Time().from_list(indices=[2, 1], is_discrete=True)

    def test_invalid_non_monotonic_float_indices_raises(self):
        """Test that non-monotonic float indices raise ValidationError."""
        with pytest.raises(ValidationError):
            Time().from_list(indices=[0.0, 1.0, 0.5], is_discrete=False)


class TestDiscrete:
    def test_discrete_default_names(self):
        """Test discrete factory method with default names."""
        start = 5
        length = 3
        expected_indices = [5, 6, 7, 8]
        time = Time.discrete(start=start, length=length)
        name = "T"
        data_name = "time"

        assert time.is_discrete is True
        assert list(time) == expected_indices
        assert time.name == name
        assert time.data.name == data_name

    def test_discrete_custom_names(self):
        """Test discrete factory method with custom names."""
        start = 0
        length = 3
        name = "CustomTime"
        data_name = "CustomData"
        expected_indices = [0, 1, 2, 3]
        time = Time.discrete(start=start, length=length, name=name, data_name=data_name)

        assert time.is_discrete is True
        assert list(time) == expected_indices
        assert time.name == name
        assert time.data.name == data_name

    def test_invalid_non_integer_start_raises(self):
        """Test that non-integer start raises TypeError."""
        with pytest.raises(TypeError, match="start must be an integer"):
            Time.discrete(start=0.5, length=5)

    def test_invalid_negative_length_raises(self):
        """Test that negative length raises ValueError."""
        with pytest.raises(ValueError, match="length must be a positive integer"):
            Time.discrete(start=0, length=-3)


class TestContinuous:
    def test_continuous_default_names_with_num_points(self):
        """Test continuous factory method with default names and num_points."""
        start = 0.0
        stop = 1.0
        num_points = 5
        dt = None
        expected_indices = [0.0, 0.25, 0.5, 0.75, 1.0]
        time = Time.continuous(
            start=start,
            stop=stop,
            num_points=num_points,
            dt=dt,
        )
        name = "T"
        data_name = "time"

        assert time.is_discrete is False
        assert list(time) == pytest.approx(expected_indices)
        assert time.name == name
        assert time.data.name == data_name

    def test_continuous_custom_names_with_dt(self):
        """Test continuous factory method with custom names and dt."""
        start = 1.0
        stop = 2.0
        num_points = None
        dt = 0.2
        name = "CustomTime"
        data_name = "CustomData"
        expected_indices = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        time = Time.continuous(
            start=start,
            stop=stop,
            num_points=num_points,
            dt=dt,
            name=name,
            data_name=data_name,
        )

        assert time.is_discrete is False
        assert list(time) == pytest.approx(expected_indices)
        assert time.name == name
        assert time.data.name == data_name

    def test_invalid_start_greater_than_stop_raises(self):
        """Test that start > stop raises ValueError."""
        with pytest.raises(ValueError, match="start must be less than stop"):
            Time.continuous(start=1.0, stop=0.0, num_points=5, dt=None)

    def test_invalid_neither_dt_nor_num_points_raises(self):
        """Test that neither dt nor num_points raises ValueError."""
        with pytest.raises(ValueError, match="Specify exactly one of dt or num_points"):
            Time.continuous(start=0.0, stop=1.0, num_points=None, dt=None)

    def test_invalid_both_dt_and_num_points_raises(self):
        """Test that both dt and num_points raises ValueError."""
        with pytest.raises(ValueError, match="Specify exactly one of dt or num_points"):
            Time.continuous(start=0.0, stop=1.0, num_points=5, dt=0.1)

    def test_invalid_num_points_less_than_2_raises(self):
        """Test that num_points < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_points must be an integer >= 2"):
            Time.continuous(start=0.0, stop=1.0, num_points=1, dt=None)

    def test_invalid_negative_dt_raises(self):
        """Test that negative dt raises ValueError."""
        with pytest.raises(ValueError, match="dt must be a positive real number"):
            Time.continuous(start=0.0, stop=1.0, num_points=None, dt=-0.1)

    def test_invalid_non_real_start_raises(self):
        """Test that non-real start raises TypeError."""
        with pytest.raises(TypeError, match="start and stop must be real numbers"):
            Time.continuous(start="zero", stop=1.0, num_points=5, dt=None)

    def test_invalid_non_real_stop_raises(self):
        """Test that non-real stop raises TypeError."""
        with pytest.raises(TypeError, match="start and stop must be real numbers"):
            Time.continuous(start=0.0, stop="one", num_points=5, dt=None)

    def test_invalid_non_real_dt_raises(self):
        """Test that non-real dt raises ValueError."""
        with pytest.raises(ValueError, match="dt must be a positive real number"):
            Time.continuous(start=0.0, stop=1.0, num_points=None, dt="point_one")


def test_getitem():
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
