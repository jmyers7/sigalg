import sigalg as sa


class TestConstruction:

    def test_construction(self):

        process = sa.IIDBernoulli(
            probability=0.25, n_trajectories=1, length=40, initial_time=3
        )
        assert process.probability == 0.25
        assert process.n_trajectories == 1
        assert process.length == 40
        assert process.initial_time == 3
