import pandas as pd

from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra


def test_probability_measure_progressive_modification_workflow():
    """End-to-end integration test simulating a realistic ProbabilityMeasure workflow.

    Tests state consistency and internal invariants when progressively modifying
    a ProbabilityMeasure through a sequence of operations: setting sig_alg via setter,
    then using from_dict/from_pandas with various type (atom/point) and overwrite
    parameters. Verifies that the object maintains consistency across chained
    mutations as would occur in real notebook usage.
    """

    """Build a probability measure."""

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

    """Test the sig_alg setter. The new sigma-algebra F2 is a coarsening of F1
    (fewer atoms). Setting the sig_alg should update the following:

    * The sig_alg changes to F2.
    * The atom_probs are recalculated by summing probabilities from F1's atoms
      that map to the same atom in F2.
    * The data is updated with new atom probabilities.

    This will not change the following:

    * The point_probs remains None (not computing point probabilities).
    * The point_data remains None.
    """

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

    """Test the from_dict method with type='point' and overwrite_sig_alg=True.
    This completely replaces the sigma-algebra with a new power-set sigma-algebra
    built from the dictionary keys, and assigns point probabilities.

    This should update everything:

    * The sig_alg changes to a power-set on a new sample space.
    * The atom_probs are set to the provided point probabilities (since power-set).
    * The point_probs are set to the provided dictionary.
    * The data is updated.
    * The point_data is created (new).
    """

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

    """Test the from_pandas method with type='point' and overwrite_sig_alg=True.
    This completely replaces the sigma-algebra with a new power-set sigma-algebra
    built from the Series index, and assigns point probabilities.

    This should update everything:

    * The sig_alg changes to a power-set on a new sample space.
    * The atom_probs are set to the provided point probabilities.
    * The point_probs are set from the Series.
    * The data is updated.
    * The point_data is updated.
    """

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

    """Test setting sig_alg via setter followed by from_dict with type='atom' and
    overwrite_sig_alg=False. First we coarsen the sigma-algebra (from power-set to
    a partition), then update the atom probabilities while preserving the sigma-algebra.

    Setting sig_alg to F3 should update:

    * The sig_alg changes to F3.
    * The atom_probs are recalculated by summing point probabilities.
    * The data is updated.
    * The point_probs becomes None (no longer tracking point probabilities).
    * The point_data becomes None.

    Then calling from_dict with type='atom' and overwrite_sig_alg=False should update:

    * The atom_probs are set to the provided dictionary.
    * The data is updated.

    This will not change:

    * The sig_alg stays as F3.
    * The point_probs remains None.
    * The point_data remains None.
    """

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
