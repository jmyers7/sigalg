import pytest
import sigalg as sa
import pandas as pd
import numpy as np


class TestEventInitialization:

    def test_init_event_from_sample_space_and_data_list(self):
        """Event can be initialized from SampleSpace and data as a list of sample point indices"""

        sample_point_indices = sa.SamplePointsIndex(["s1", "s2", "s3"], name="points")
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6]], sample_point_indices=sample_point_indices
        )
        event = sa.Event(space_rep=sample_space, data=["s1", "s3"])

        assert event.sample_space is sample_space
        assert len(event) == 2

        right_df = pd.DataFrame(
            [[1, 2], [5, 6]],
            index=["s1", "s3"],
            columns=[0, 1],
        )
        right_df.index.name = "points"

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            right_df,
        )

    def test_init_event_from_sample_space_and_data_df(self):
        """Event can be initialized from SampleSpace and data as a DataFrame"""

        sample_point_indices = sa.SamplePointsIndex(["s1", "s2", "s3"], name="points")
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6]], sample_point_indices=sample_point_indices
        )
        data_df = pd.DataFrame(
            [[3, 4], [5, 6]],
            index=["s2", "s3"],
            columns=[0, 1],
        )
        data_df.index.name = "points"

        event = sa.Event(space_rep=sample_space, data=data_df)

        assert event.sample_space is sample_space
        assert len(event) == 2

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            data_df,
        )

    def test_init_event_from_sample_space_only(self):
        """Event can be initialized from SampleSpace only (data=None)"""

        sample_point_indices = sa.SamplePointsIndex(["s1", "s2", "s3"], name="points")
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6]], sample_point_indices=sample_point_indices
        )
        event = sa.Event(space_rep=sample_space, data=None)

        assert event.sample_space is sample_space
        assert len(event) == 3

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            sample_space.to_pandas(),
        )

    def test_init_event_from_dataframe_only(self):
        """Event can be initialized directly from a DataFrame"""

        data_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["s1", "s2", "s3"],
            columns=[0, 1],
        )
        data_df.index.name = "points"

        event = sa.Event(space_rep=data_df, data=None)

        assert len(event) == 3

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            data_df,
        )

    def test_init_event_from_dataframe_and_data_list(self):
        """Event can be initialized directly from a DataFrame and data as a list of sample point indices"""

        data_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["s1", "s2", "s3"],
            columns=[0, 1],
        )
        data_df.index.name = "points"

        event = sa.Event(space_rep=data_df, data=["s1", "s3"])

        assert len(event) == 2

        right_df = pd.DataFrame(
            [[1, 2], [5, 6]],
            index=["s1", "s3"],
            columns=[0, 1],
        )
        right_df.index.name = "points"

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            right_df,
        )

    def test_init_event_from_dataframe_and_data_df(self):
        """Event can be initialized directly from a DataFrame and data as a DataFrame"""

        data_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["s1", "s2", "s3"],
            columns=[0, 1],
        )
        data_df.index.name = "points"

        event_data_df = pd.DataFrame(
            [[3, 4], [5, 6]],
            index=["s2", "s3"],
            columns=[0, 1],
        )
        event_data_df.index.name = "points"

        event = sa.Event(space_rep=data_df, data=event_data_df)

        assert len(event) == 2

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            event_data_df,
        )

    def test_init_event_from_sample_space_and_data_series(self):
        """Event can be initialized from SampleSpace and data as a Series"""

        sample_point_indices = sa.SamplePointsIndex(["s1", "s2", "s3"], name="points")
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6]], sample_point_indices=sample_point_indices
        )
        data_series = pd.Series([3, 4], index=[0, 1], name="s2")

        event = sa.Event(space_rep=sample_space, data=data_series)

        assert event.sample_space is sample_space
        assert len(event) == 1

        right_df = pd.DataFrame(
            [[3, 4]],
            index=["s2"],
            columns=[0, 1],
        )
        right_df.index.name = "points"

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            right_df,
        )

    def test_init_event_from_dataframe_and_data_series(self):
        """Event can be initialized directly from a DataFrame and data as a Series"""

        data_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["s1", "s2", "s3"],
            columns=[0, 1],
        )
        data_df.index.name = "points"

        data_series = pd.Series([5, 6], index=[0, 1], name="s3")

        event = sa.Event(space_rep=data_df, data=data_series)

        assert len(event) == 1

        right_df = pd.DataFrame(
            [[5, 6]],
            index=["s3"],
            columns=[0, 1],
        )
        right_df.index.name = "points"

        pd.testing.assert_frame_equal(
            event.to_pandas(),
            right_df,
        )


class TestEventProperties:

    @pytest.fixture
    def event(self):
        sample_point_indices = sa.SamplePointsIndex(["s1", "s2", "s3"], name="points")
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6]], sample_point_indices=sample_point_indices
        )
        event = sa.Event(space_rep=sample_space, data=["s1", "s3"])
        return event

    def test_sample_point_indices(self, event):
        spi = event.sample_point_indices
        assert isinstance(spi, sa.SamplePointsIndex)
        assert list(spi) == ["s1", "s3"]
        assert spi.name == "points"

    def test_sample_space(self, event):
        ss = event.sample_space
        assert isinstance(ss, sa.SampleSpace)
        assert list(ss.sample_point_indices) == ["s1", "s2", "s3"]

    def test_num_features(self, event):
        assert event.num_features == 2


class TestEventMethods:

    @pytest.fixture
    def event(self):
        sample_point_indices = sa.SamplePointsIndex(["s1", "s2", "s3"], name="points")
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6]], sample_point_indices=sample_point_indices
        )
        event = sa.Event(space_rep=sample_space, data=["s1", "s3"])
        return event

    def test_to_numpy(self, event):
        np_array = event.to_numpy()
        expected_array = np.array([[1, 2], [5, 6]])
        assert np.array_equal(np_array, expected_array)

    def test___array__(self, event):
        np_array = np.array(event)
        expected_array = np.array([[1, 2], [5, 6]])
        assert np.array_equal(np_array, expected_array)

    def test___len__(self, event):
        assert len(event) == 2

    def test_name_of_sample_point_indices(self, event):
        assert event.sample_point_indices.name == "points"

    def test__getitem__(self, event):
        sub_event = event["s1"]
        assert isinstance(sub_event, sa.Event)
        assert len(sub_event) == 1
        expected_df = pd.DataFrame(
            [[1, 2]],
            index=["s1"],
            columns=[0, 1],
        )
        expected_df.index.name = "points"
        pd.testing.assert_frame_equal(
            sub_event.to_pandas(),
            expected_df,
        )

    def test__getitem__multiple_keys(self, event):
        sample_point_indices = sa.SamplePointsIndex(
            ["s1", "s2", "s3", "s4"], name="points"
        )
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6], [7, 8]],
            sample_point_indices=sample_point_indices,
        )
        event = sa.Event(space_rep=sample_space, data=["s1", "s3", "s4"])
        sub_event = event[["s3", "s4"]]
        assert isinstance(sub_event, sa.Event)
        assert len(sub_event) == 2
        expected_df = pd.DataFrame(
            [[5, 6], [7, 8]],
            index=["s3", "s4"],
            columns=[0, 1],
        )
        expected_df.index.name = "points"
        pd.testing.assert_frame_equal(
            sub_event.to_pandas(),
            expected_df,
        )

    def test_apply(self, event):
        """Test applying a function to the event data"""

        def add_one(x):
            return x + 1

        modified_event = event.apply(add_one)
        expected_df = pd.DataFrame(
            [[2, 3], [6, 7]],
            index=["s1", "s3"],
            columns=[0, 1],
        )
        expected_df.index.name = "points"

        pd.testing.assert_frame_equal(
            modified_event,
            expected_df,
        )

    def test_sum(self, event):
        """Test summing the event data across features"""
        summed_series = event.sum()
        expected_series = pd.Series(
            [3, 11],
            index=["s1", "s3"],
        )
        expected_series.index.name = "points"

        pd.testing.assert_series_equal(
            summed_series,
            expected_series,
        )

    def test_eq(self, event):
        """Test equality comparison between events"""

        sample_point_indices = sa.SamplePointsIndex(["s1", "s2", "s3"], name="points")
        sample_space = sa.SampleSpace(
            data=[[1, 2], [3, 4], [5, 6]], sample_point_indices=sample_point_indices
        )
        same_event = sa.Event(space_rep=sample_space, data=["s1", "s3"])

        different_event = sa.Event(space_rep=sample_space, data=["s2"])

        assert event == same_event
        assert event != different_event
