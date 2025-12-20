from numbers import Real

import pytest
from pydantic import ValidationError

from sigalg.core import Time


class TestConstructor:

    @pytest.mark.parametrize(
        "indices, is_discrete, name, data_name",
        [
            pytest.param(
                [0, 1, 2, 3, 4], True, None, None, id="default_names_with_int_indices"
            ),
            pytest.param(
                [0.0, 0.5, 1.0, 1.5],
                False,
                "CustomTime",
                "CustomData",
                id="custom_names_with_float_indices",
            ),
        ],
    )
    def test_constructor(self, indices, is_discrete, name, data_name):
        """Test Time constructor with various indices and is_discrete values."""
        time = Time(
            indices=indices, is_discrete=is_discrete, name=name, data_name=data_name
        )
        expected_name = name if name is not None else "T"
        expected_data_name = data_name if data_name is not None else "time"
        assert time.is_discrete == is_discrete
        assert list(time) == indices
        assert time.name == expected_name
        assert time.data.name == expected_data_name

    @pytest.mark.parametrize(
        "indices, is_discrete",
        [
            pytest.param([0, 1, "two", 3], True, id="mixed_type_indices"),
            pytest.param([], True, id="empty_indices"),
            pytest.param([0.0, 1.0, 2.0], True, id="float_indices_for_discrete"),
            pytest.param([2, 1], True, id="non_monotonic_int_indices"),
            pytest.param([0.0, 1.0, 0.5], False, id="non_monotonic_float_indices"),
        ],
    )
    def test_invalid_indices_raises(self, indices, is_discrete):
        """Test that invalid indices raise a TypeError."""
        with pytest.raises(ValidationError):
            Time(indices=indices, is_discrete=is_discrete)


class TestDiscrete:

    @pytest.mark.parametrize(
        "start, length, name, data_name, expected_indices",
        [
            pytest.param(5, 4, None, None, [5, 6, 7, 8], id="default_names"),
            pytest.param(
                0, 3, "CustomTime", "CustomData", [0, 1, 2], id="custom_names"
            ),
        ],
    )
    def test_discrete(self, start, length, name, data_name, expected_indices):
        """Test the discrete factory method."""
        time = Time.discrete(start=start, length=length, name=name, data_name=data_name)
        expected_name = name if name is not None else "T"
        expected_data_name = data_name if data_name is not None else "time"

        assert time.is_discrete is True
        assert list(time) == expected_indices
        assert time.name == expected_name
        assert time.data.name == expected_data_name

    @pytest.mark.parametrize(
        "start, length",
        [
            pytest.param(0.5, 5, id="non_integer_start"),
            pytest.param(0, -3, id="negative_length"),
        ],
    )
    def test_invalid_discrete_parameters_raises(self, start, length):
        """Test that invalid parameters to discrete factory raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            Time.discrete(start=start, length=length)


class TestContinuous:

    @pytest.mark.parametrize(
        "start, stop, num_points, dt, name, data_name, expected_indices",
        [
            pytest.param(
                0.0,
                1.0,
                5,
                None,
                None,
                None,
                [0.0, 0.25, 0.5, 0.75, 1.0],
                id="default_names_with_num_points",
            ),
            pytest.param(
                1.0,
                2.0,
                None,
                0.2,
                "CustomTime",
                "CustomData",
                [1.0, 1.2, 1.4, 1.6, 1.8],
                id="custom_names_with_dt",
            ),
        ],
    )
    def test_continuous(
        self, start, stop, num_points, dt, name, data_name, expected_indices
    ):
        """Test the continuous factory method."""
        time = Time.continuous(
            start=start,
            stop=stop,
            num_points=num_points,
            dt=dt,
            name=name,
            data_name=data_name,
        )
        expected_name = name if name is not None else "T"
        expected_data_name = data_name if data_name is not None else "time"

        assert time.is_discrete is False
        assert list(time) == pytest.approx(expected_indices)
        assert time.name == expected_name
        assert time.data.name == expected_data_name

    @pytest.mark.parametrize(
        "start, stop, num_points, dt",
        [
            pytest.param(1.0, 0.0, 5, None, id="start_greater_than_stop"),
            pytest.param(0.0, 1.0, None, None, id="neither_dt_nor_num_points"),
            pytest.param(0.0, 1.0, 5, 0.1, id="both_dt_and_num_points"),
            pytest.param(0.0, 1.0, 1, None, id="num_points_less_than_2"),
            pytest.param(0.0, 1.0, None, -0.1, id="negative_dt"),
            pytest.param("zero", 1.0, 5, None, id="non_real_start"),
            pytest.param(0.0, "one", 5, None, id="non_real_stop"),
            pytest.param(0.0, 1.0, None, "point_one", id="non_real_dt"),
        ],
    )
    def test_invalid_input_raises(self, start, stop, num_points, dt):
        """Test that invalid parameters to continuous factory raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            Time.continuous(start=start, stop=stop, num_points=num_points, dt=dt)


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
