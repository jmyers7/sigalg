import pandas as pd

from sigalg.core import Event, SampleSpace, SigmaAlgebra


def test_sigma_algebra_progressive_modification_workflow():
    """End-to-end integration test simulating a realistic SigmaAlgebra workflow.

    Tests state consistency and internal invariants when progressively modifying
    a SigmaAlgebra through a sequence of operations: setting sample_space via setter,
    then using from_dict/from_pandas with various overwrite parameters. Verifies
    that the object maintains consistency across chained mutations as would occur
    in real notebook usage.
    """

    """Build a sigma-algebra."""

    # Build a sigma-algebra
    Omega1 = SampleSpace(name="Omega1").from_sequence(size=3)
    atom_ids = {
        0: 0,
        1: 0,
        2: 1,
    }
    F = SigmaAlgebra(sample_space=Omega1).from_dict(atom_ids)
    expected_data = pd.Series([0, 0, 1], index=Omega1.data, name="atom ID")

    assert F.sample_space == Omega1
    assert F.sample_id_to_atom_id == atom_ids
    pd.testing.assert_series_equal(F.data, expected_data)

    """Test the sample_space setter. The new sample space Omega2 = {a, b, c} has
    the same number of elements as Omega1 = {0, 1, 2}. Setting the sample_space
    should update the following:

    * The sample_space changes to Omega2.
    * The sample_id_to_atom_id dictionary keys are updated to match Omega2.
    * The data index is updated.
    * All derived properties (atom_id_to_sample_ids, atom_id_to_event, etc.).

    This will not change the following:

    * The atom structure (atom_ids, num_atoms, atom_space).
    * The is_power_set property.
    """

    Omega2 = SampleSpace(name="Omega2").from_list(["a", "b", "c"])
    F.sample_space = Omega2
    expected_data = pd.Series([0, 0, 1], index=Omega2.data, name="atom ID")

    assert F.sample_space == Omega2
    assert F.sample_id_to_atom_id == {"a": 0, "b": 0, "c": 1}
    pd.testing.assert_series_equal(F.data, expected_data)
    assert F.atom_space == SampleSpace(name="atom_space").from_list([0, 1])
    assert F.num_atoms == 2
    assert F.atom_ids == [0, 1]
    assert F.atom_id_to_sample_ids == {0: ["a", "b"], 1: ["c"]}
    assert F.atom_id_to_event == {
        0: Event(sig_alg=F).from_list(["a", "b"]),
        1: Event(sig_alg=F).from_list(["c"]),
    }
    assert F.atom_id_to_cardinality == {0: 2, 1: 1}
    assert F.is_power_set is False
    assert F.to_atoms == [
        Event(sig_alg=F).from_list(["a", "b"]),
        Event(sig_alg=F).from_list(["c"]),
    ]

    """Test the from_dict method with overwrite_sample_space=True. This completely
    replaces the existing sample space with the keys from the provided dictionary
    and updates the atom ID mapping.

    This should update everything:

    * The sample_space changes to a new SampleSpace built from dictionary keys.
    * The sample_id_to_atom_id dictionary.
    * The data.
    * The atom_space (new atom IDs).
    * The num_atoms and atom_ids.
    * All derived properties (atom_id_to_sample_ids, atom_id_to_event, etc.).
    * The is_power_set property (if applicable).
    """

    new_atom_ids = {
        "blue": "cat",
        "green": "cat",
        "red": "dog",
        "orange": "bird",
    }
    F.from_dict(new_atom_ids, overwrite_sample_space=True)
    expected_sample_space = SampleSpace(name="Omega3").from_list(
        ["blue", "green", "red", "orange"]
    )
    expected_data = pd.Series(
        ["cat", "cat", "dog", "bird"], index=expected_sample_space.data, name="atom ID"
    )

    assert F.sample_space == expected_sample_space
    assert F.sample_id_to_atom_id == new_atom_ids
    pd.testing.assert_series_equal(F.data, expected_data)
    assert F.atom_space == SampleSpace(name="atom_space").from_list(
        ["cat", "dog", "bird"]
    )
    assert F.num_atoms == 3
    assert F.atom_ids == ["cat", "dog", "bird"]
    assert F.atom_id_to_sample_ids == {
        "cat": ["blue", "green"],
        "dog": ["red"],
        "bird": ["orange"],
    }
    assert F.atom_id_to_event == {
        "cat": Event(sig_alg=F).from_list(["blue", "green"]),
        "dog": Event(sig_alg=F).from_list(["red"]),
        "bird": Event(sig_alg=F).from_list(["orange"]),
    }
    assert F.atom_id_to_cardinality == {"cat": 2, "dog": 1, "bird": 1}
    assert F.is_power_set is False
    assert F.to_atoms == [
        Event(sig_alg=F).from_list(["blue", "green"]),
        Event(sig_alg=F).from_list(["red"]),
        Event(sig_alg=F).from_list(["orange"]),
    ]

    """Test the from_pandas method with overwrite_sample_space=True. This completely
    replaces the existing sample space with the index from the provided Series
    and updates the atom ID mapping.

    This should update everything:

    * The sample_space changes to a new SampleSpace built from the Series index.
    * The sample_id_to_atom_id dictionary.
    * The data.
    * The atom_space (new atom IDs from Series values).
    * The num_atoms and atom_ids.
    * All derived properties (atom_id_to_sample_ids, atom_id_to_event, etc.).
    * The is_power_set property (becomes True in this case as each sample maps to unique atom).
    """

    new_data = pd.Series(["car", "plane", "bike"], index=["purple", "brown", "indigo"])
    F.from_pandas(new_data, overwrite_sample_space=True)
    expected_sample_space = SampleSpace(name="Omega4").from_list(
        ["purple", "brown", "indigo"]
    )
    expected_data = pd.Series(
        ["car", "plane", "bike"], index=["purple", "brown", "indigo"], name="atom ID"
    )

    assert F.sample_space == expected_sample_space
    assert F.sample_id_to_atom_id == {
        "purple": "car",
        "brown": "plane",
        "indigo": "bike",
    }
    pd.testing.assert_series_equal(F.data, expected_data)
    assert F.atom_space == SampleSpace(name="atom_space").from_list(
        ["car", "plane", "bike"]
    )
    assert F.num_atoms == 3
    assert F.atom_ids == ["car", "plane", "bike"]
    assert F.atom_id_to_sample_ids == {
        "car": ["purple"],
        "plane": ["brown"],
        "bike": ["indigo"],
    }
    assert F.atom_id_to_event == {
        "car": Event(sig_alg=F).from_list(["purple"]),
        "plane": Event(sig_alg=F).from_list(["brown"]),
        "bike": Event(sig_alg=F).from_list(["indigo"]),
    }
    assert F.atom_id_to_cardinality == {"car": 1, "plane": 1, "bike": 1}
    assert F.is_power_set is True
    assert F.to_atoms == [
        Event(sig_alg=F).from_list(["purple"]),
        Event(sig_alg=F).from_list(["brown"]),
        Event(sig_alg=F).from_list(["indigo"]),
    ]

    """Test the from_dict method with overwrite_sample_space=False (default).
    This preserves the existing sample space but updates the atom ID mapping.
    The dictionary keys must align with the existing sample space.

    This should update the following:

    * The sample_id_to_atom_id dictionary (new atom ID mapping).
    * The data values.
    * The atom_space (new atom IDs).
    * The num_atoms and atom_ids.
    * All derived properties (atom_id_to_sample_ids, atom_id_to_event, etc.).
    * The is_power_set property (becomes False in this case).

    This will not change the following:

    * The sample_space stays as expected_sample_space (purple, brown, indigo).
    """

    new_atom_ids = {
        "purple": "apple",
        "brown": "apple",
        "indigo": "orange",
    }
    F.from_dict(new_atom_ids, overwrite_sample_space=False)
    expected_sample_space = SampleSpace(name="Omega4").from_list(
        ["purple", "brown", "indigo"]
    )
    expected_data = pd.Series(
        ["apple", "apple", "orange"],
        index=["purple", "brown", "indigo"],
        name="atom ID",
    )

    assert F.sample_space == expected_sample_space
    assert F.sample_id_to_atom_id == new_atom_ids
    pd.testing.assert_series_equal(F.data, expected_data)
    assert F.atom_space == SampleSpace(name="atom_space").from_list(["apple", "orange"])
    assert F.num_atoms == 2
    assert F.atom_ids == ["apple", "orange"]
    assert F.atom_id_to_sample_ids == {
        "apple": ["purple", "brown"],
        "orange": ["indigo"],
    }
    assert F.atom_id_to_event == {
        "apple": Event(sig_alg=F).from_list(["purple", "brown"]),
        "orange": Event(sig_alg=F).from_list(["indigo"]),
    }
    assert F.atom_id_to_cardinality == {"apple": 2, "orange": 1}
    assert F.is_power_set is False
    assert F.to_atoms == [
        Event(sig_alg=F).from_list(["purple", "brown"]),
        Event(sig_alg=F).from_list(["indigo"]),
    ]

    """Test the from_pandas method with overwrite_sample_space=False (default).
    This preserves the existing sample space but updates the atom ID mapping.
    The Series index must align with the existing sample space.

    This should update the following:

    * The sample_id_to_atom_id dictionary (new atom ID mapping).
    * The data values.
    * The atom_space (new atom IDs from Series values).
    * The num_atoms and atom_ids.
    * All derived properties (atom_id_to_sample_ids, atom_id_to_event, etc.).
    * The is_power_set property remains False.

    This will not change the following:

    * The sample_space stays as expected_sample_space (purple, brown, indigo).
    """

    new_data = pd.Series(
        ["dog", "dog", "cat"], index=["purple", "brown", "indigo"], name="atom ID"
    )
    F.from_pandas(new_data, overwrite_sample_space=False)
    expected_sample_space = SampleSpace(name="Omega4").from_list(
        ["purple", "brown", "indigo"]
    )
    expected_data = pd.Series(
        ["dog", "dog", "cat"], index=["purple", "brown", "indigo"], name="atom ID"
    )

    assert F.sample_space == expected_sample_space
    assert F.sample_id_to_atom_id == {
        "purple": "dog",
        "brown": "dog",
        "indigo": "cat",
    }
    pd.testing.assert_series_equal(F.data, expected_data)
    assert F.atom_space == SampleSpace(name="atom_space").from_list(["dog", "cat"])
    assert F.num_atoms == 2
    assert F.atom_ids == ["dog", "cat"]
    assert F.atom_id_to_sample_ids == {
        "dog": ["purple", "brown"],
        "cat": ["indigo"],
    }
    assert F.atom_id_to_event == {
        "dog": Event(sig_alg=F).from_list(["purple", "brown"]),
        "cat": Event(sig_alg=F).from_list(["indigo"]),
    }
    assert F.atom_id_to_cardinality == {"dog": 2, "cat": 1}
    assert F.is_power_set is False
    assert F.to_atoms == [
        Event(sig_alg=F).from_list(["purple", "brown"]),
        Event(sig_alg=F).from_list(["indigo"]),
    ]
