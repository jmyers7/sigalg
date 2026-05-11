import pandas as pd

from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra


def test_changing_all_parameters_of_prob_measure():
    """Test that changing all parameters of a probability measure updates the data and all related properties accordingly."""
    # Build a probability measure
    Omega1 = SampleSpace(name="Omega1").from_sequence(size=4)
    atom_ids = {
        0: 0,
        1: 0,
        2: 1,
        3: 2,
    }
    F1 = SigmaAlgebra(sample_space=Omega1, name="F1").from_dict(atom_ids)
    probs = {1: 0.15, 0: 0.75, 2: 0.1}
    P = ProbabilityMeasure(sig_alg=F1).from_dict(probs)
    expected_data = pd.Series(
        [0.75, 0.15, 0.1],
        index=pd.Index([0, 1, 2], name="atom ID"),
        name="probability",
    )

    assert P.sig_alg == F1
    assert P.atom_probs == probs
    assert P.point_probs is None
    pd.testing.assert_series_equal(P.data, expected_data)
    assert P.point_data is None

    # Change the sigma-algebra and check that the data is updated accordingly
    atom_ids2 = {
        0: 1,
        1: 1,
        2: 0,
        3: 0,
    }
    F2 = SigmaAlgebra(sample_space=Omega1, name="F2").from_dict(atom_ids2)
    P.sig_alg = F2
    expected_data = pd.Series(
        [0.75, 0.25],
        index=pd.Index([1, 0], name="atom ID"),
        name="probability",
    )

    assert P.sig_alg == F2
    assert P.atom_probs == {0: 0.25, 1: 0.75}
    assert P.point_probs is None
    pd.testing.assert_series_equal(P.data, expected_data)
    assert P.point_data is None

    # Force overwrite the sigma-algebra using from_dict with `type='point'` and check that the data is updated accordingly
    point_probs1 = {
        0: 0.1,
        1: 0.2,
        2: 0.7,
    }
    P.from_dict(point_probs1, type="point", overwrite_sig_alg=True)
    expected_sample_space = SampleSpace().from_list(list(point_probs1.keys()))
    expected_sig_alg = SigmaAlgebra.power_set(sample_space=expected_sample_space)
    expected_data = pd.Series(
        [0.1, 0.2, 0.7],
        index=pd.Index([0, 1, 2], name="atom ID"),
        name="probability",
    )
    expected_point_data = pd.Series(
        [0.1, 0.2, 0.7],
        index=pd.Index([0, 1, 2], name="sample"),
        name="probability",
    )

    assert P.sig_alg == expected_sig_alg
    assert P.atom_probs == {0: 0.1, 1: 0.2, 2: 0.7}
    assert P.point_probs == point_probs1
    pd.testing.assert_series_equal(P.data, expected_data)
    pd.testing.assert_series_equal(P.point_data, expected_point_data)

    # Force overwrite the sigma-algebra using from_pandas with `type='point'` and check that the data is updated accordingly
    point_data1 = pd.Series([0.4, 0.3, 0.2, 0.1], index=["a", "b", "c", "d"])
    P.from_pandas(point_data1, type="point", overwrite_sig_alg=True)
    expected_sample_space = SampleSpace().from_list(list(point_data1.index))
    expected_sig_alg = SigmaAlgebra.power_set(sample_space=expected_sample_space)
    expected_data = pd.Series(
        [0.4, 0.3, 0.2, 0.1],
        index=pd.Index(["a", "b", "c", "d"], name="atom ID"),
        name="probability",
    )
    expected_point_data = pd.Series(
        [0.4, 0.3, 0.2, 0.1],
        index=pd.Index(["a", "b", "c", "d"], name="sample"),
        name="probability",
    )

    assert P.sig_alg == expected_sig_alg
    assert P.atom_probs == {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    assert P.point_probs == {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    pd.testing.assert_series_equal(P.data, expected_data)
    pd.testing.assert_series_equal(P.point_data, expected_point_data)

    # Set the sigma-algebra and change the atom probabilities using from_dict with `type='atom'` and check that the data is updated accordingly
    atom_ids3 = {
        "a": "cat",
        "b": "cat",
        "c": "dog",
        "d": "dog",
    }
    F3 = SigmaAlgebra(sample_space=expected_sample_space, name="F3").from_dict(
        atom_ids3
    )
    P.sig_alg = F3
    atom_probs3 = {"cat": 0.35, "dog": 0.65}
    P.from_dict(atom_probs3, type="atom", overwrite_sig_alg=False)
    expected_data = pd.Series(
        [0.35, 0.65],
        index=pd.Index(["cat", "dog"], name="atom ID"),
        name="probability",
    )

    assert P.sig_alg == F3
    assert P.atom_probs == atom_probs3
    assert P.point_probs is None
    pd.testing.assert_series_equal(P.data, expected_data)
    assert P.point_data is None
