import pandas as pd

from sigalg.core import (
    Index,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)


def test_random_vector_progressive_modification_workflow():
    """End-to-end integration test simulating a realistic RandomVector workflow.

    Tests state consistency and internal invariants when progressively modifying
    a RandomVector through a sequence of operations: setting domain, sigma-algebra,
    and probability measure via setters, then using from_dict/from_pandas with
    various overwrite parameters. Verifies that the object maintains consistency
    across chained mutations as would occur in real notebook usage.
    """

    """Build a random vector."""

    Omega1 = SampleSpace(name="Omega1").from_sequence(size=5)
    atom_ids1 = {
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 3,
    }
    F1 = SigmaAlgebra(sample_space=Omega1, name="F1").from_dict(atom_ids1)
    atom_probs1 = {
        0: 0.25,
        1: 0.65,
        2: 0.05,
        3: 0.05,
    }
    P1 = ProbabilityMeasure(sig_alg=F1, name="P1").from_dict(atom_probs1)
    point_outputs1 = {
        0: (1, 2),
        1: (3, 4),
        2: (3, 4),
        3: (3, 4),
        4: (3, 4),
    }
    prob_space1 = ProbabilitySpace(Omega1, F1, P1)
    X = RandomVector(*prob_space1).from_dict(point_outputs1)

    expected_atom_outputs = {
        0: (1, 2),
        1: (3, 4),
        2: (3, 4),
        3: (3, 4),
    }
    expected_index = Index().from_list(
        [
            "X_0",
            "X_1",
        ],
        data_name="feature",
    )
    expected_data = pd.DataFrame.from_dict(
        {
            0: (1, 2),
            1: (3, 4),
            2: (3, 4),
            3: (3, 4),
            4: (3, 4),
        },
        orient="index",
        columns=pd.Index(["X_0", "X_1"], name="feature"),
    )
    expected_data.index.name = "sample"
    expected_atom_data = pd.DataFrame.from_dict(
        {
            0: (1, 2),
            1: (3, 4),
            2: (3, 4),
            3: (3, 4),
        },
        orient="index",
        columns=pd.Index(["X_0", "X_1"], name="feature"),
    )
    expected_atom_data.index.name = "atom ID"
    expected_components = [
        RandomVariable(*prob_space1, name="X_0").from_dict(
            {
                0: 1,
                1: 3,
                2: 3,
                3: 3,
                4: 3,
            }
        ),
        RandomVariable(*prob_space1, name="X_1").from_dict(
            {
                0: 2,
                1: 4,
                2: 4,
                3: 4,
                4: 4,
            }
        ),
    ]
    expected_generated_sig_alg = SigmaAlgebra(
        sample_space=Omega1, name="sigma(X)"
    ).from_dict(
        {
            0: (1, 2),
            1: (3, 4),
            2: (3, 4),
            3: (3, 4),
            4: (3, 4),
        }
    )
    expected_range_sample_space = SampleSpace(name="X_range").from_list(
        [
            (1, 2),
            (3, 4),
        ]
    )
    expected_range_sig_alg = SigmaAlgebra.power_set(expected_range_sample_space)
    expected_range_prob_measure = ProbabilityMeasure(
        sig_alg=expected_range_sig_alg
    ).from_dict(
        {
            (1, 2): 0.25,
            (3, 4): 0.75,
        }
    )
    expected_range = ProbabilitySpace(
        sample_space=expected_range_sample_space,
        sig_alg=expected_range_sig_alg,
        prob_measure=expected_range_prob_measure,
    )

    assert X.domain == Omega1
    assert X.sig_alg == F1
    assert X.prob_measure == P1
    assert X.prob_space == prob_space1
    assert X.point_outputs == point_outputs1
    assert X.atom_outputs == expected_atom_outputs
    assert X.index == expected_index
    pd.testing.assert_frame_equal(X.data, expected_data)
    pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 2
    assert X.components == expected_components
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "P1_X"

    """Test the domain setter. The new domain Omega2 = {a, b, c, d, e} has the same number of elements as the first domain Omega1 = {0, 1, 2, 3, 4}. Setting the domain should update the following:

    * The domain changes to Omega2.
    * The sigma-algebra on the underlying probability space.
    * The underlying probability space.
    * The point_outputs dictionary.
    * The data.
    * The component random variables.
    * The generated sigma-algebra.

    This will not change the following:

    * The probability measure on the underlying probability space should stay as P1.
    * The atom_outputs dictionary.
    * The index.
    * The atom_data.
    * The dimension.
    * The range.
    """

    Omega2 = SampleSpace(name="Omega2").from_list(
        ["a", "b", "c", "d", "e"], data_name="letter"
    )
    X.domain = Omega2

    expected_sig_alg = SigmaAlgebra(Omega2).from_dict(
        {
            "a": 0,
            "b": 1,
            "c": 1,
            "d": 2,
            "e": 3,
        }
    )
    expected_prob_space = ProbabilitySpace(Omega2, expected_sig_alg, P1)
    expected_point_outputs = {
        "a": (1, 2),
        "b": (3, 4),
        "c": (3, 4),
        "d": (3, 4),
        "e": (3, 4),
    }
    expected_data = pd.DataFrame.from_dict(
        {
            "a": (1, 2),
            "b": (3, 4),
            "c": (3, 4),
            "d": (3, 4),
            "e": (3, 4),
        },
        orient="index",
        columns=pd.Index(["X_0", "X_1"], name="feature"),
    )
    expected_data.index.name = "letter"
    expected_components = [
        RandomVariable(*expected_prob_space, name="X_0").from_dict(
            {
                "a": 1,
                "b": 3,
                "c": 3,
                "d": 3,
                "e": 3,
            }
        ),
        RandomVariable(*expected_prob_space, name="X_1").from_dict(
            {
                "a": 2,
                "b": 4,
                "c": 4,
                "d": 4,
                "e": 4,
            }
        ),
    ]
    expected_generated_sig_alg = SigmaAlgebra(
        sample_space=Omega2, name="sigma(X)"
    ).from_dict(
        {
            "a": (1, 2),
            "b": (3, 4),
            "c": (3, 4),
            "d": (3, 4),
            "e": (3, 4),
        }
    )

    assert X.domain == Omega2
    assert X.sig_alg == expected_sig_alg
    assert X.prob_measure == P1
    assert X.prob_space == expected_prob_space
    assert X.point_outputs == expected_point_outputs
    assert X.atom_outputs == expected_atom_outputs
    assert X.index == expected_index
    pd.testing.assert_frame_equal(X.data, expected_data)
    pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 2
    assert X.components == expected_components
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "P1_X"

    """Test the sigma-algebra setter. The new sigma-algebra F2 is a sub-sigma-algebra of the existing one. Setting the sigma-algebra should update the following:

    * The sigma-algebra changes to F2.
    * The probability measure changes to the restriction of the existing probability measure to the new sigma-algebra.
    * The underlying probability space.
    * The component random variables.
    * The atom_outputs dictionary.
    * The atom_data.

    This will not change the following:

    * The domain stays as Omega2.
    * The point_outputs dictionary.
    * The generated sigma-algebra.
    * The index.
    * The dimension.
    * The range.
    """

    F2 = SigmaAlgebra(Omega2, name="F2").from_dict(
        {
            "a": "cat",
            "b": "dog",
            "c": "dog",
            "d": "bird",
            "e": "bird",
        }
    )
    X.sig_alg = F2

    expected_prob_measure = ProbabilityMeasure(F2).from_dict(
        {
            "cat": 0.25,
            "dog": 0.65,
            "bird": 0.1,
        }
    )
    expected_prob_space = ProbabilitySpace(Omega2, F2, expected_prob_measure)
    expected_atom_outputs = {
        "cat": (1, 2),
        "dog": (3, 4),
        "bird": (3, 4),
    }
    expected_atom_data = pd.DataFrame.from_dict(
        {
            "cat": (1, 2),
            "dog": (3, 4),
            "bird": (3, 4),
        },
        orient="index",
        columns=pd.Index(["X_0", "X_1"], name="feature"),
    )
    expected_atom_data.index.name = "atom ID"
    expected_components = [
        RandomVariable(*expected_prob_space, name="X_0").from_dict(
            {
                "a": 1,
                "b": 3,
                "c": 3,
                "d": 3,
                "e": 3,
            }
        ),
        RandomVariable(*expected_prob_space, name="X_1").from_dict(
            {
                "a": 2,
                "b": 4,
                "c": 4,
                "d": 4,
                "e": 4,
            }
        ),
    ]

    assert X.domain == Omega2
    assert X.sig_alg == F2
    assert X.prob_measure == expected_prob_measure
    assert X.prob_space == expected_prob_space
    assert X.point_outputs == expected_point_outputs
    assert X.atom_outputs == expected_atom_outputs
    assert X.index == expected_index
    pd.testing.assert_frame_equal(X.data, expected_data)
    pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 2
    assert X.components == expected_components
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "P1_X"

    """Test the probability measure setter. The new probability measure P2 is defined on a sub-sigma-algebra F3 of the existing one. Setting the probability measure should update the following:

    * The probability measure changes P2.
    * The sigma-algebra changes to F3.
    * The underlying probability space.
    * The atom_outputs dictionary. (Since the sigma-algebra changes.)
    * The atom_data. (Since the sigma-algebra changes.)
    * The component random variables.
    * The range.

    This will not change the following:

    * The domain stays as Omega2.
    * The point_outputs dictionary.
    * The generated sigma-algebra.
    * The index.
    * The dimension.
    """

    F3 = SigmaAlgebra(Omega2, name="F3").from_dict(
        {
            "a": "cat",
            "b": "dog",
            "c": "dog",
            "d": "dog",
            "e": "dog",
        }
    )
    P2 = ProbabilityMeasure(F3, name="P2").from_dict(
        {
            "cat": 0.1,
            "dog": 0.9,
        }
    )
    X.prob_measure = P2

    expected_prob_space = ProbabilitySpace(Omega2, F3, P2)
    expected_atom_outputs = {
        "cat": (1, 2),
        "dog": (3, 4),
    }
    expected_atom_data = pd.DataFrame.from_dict(
        {
            "cat": (1, 2),
            "dog": (3, 4),
        },
        orient="index",
        columns=pd.Index(["X_0", "X_1"], name="feature"),
    )
    expected_atom_data.index.name = "atom ID"
    expected_components = [
        RandomVariable(*expected_prob_space, name="X_0").from_dict(
            {
                "a": 1,
                "b": 3,
                "c": 3,
                "d": 3,
                "e": 3,
            }
        ),
        RandomVariable(*expected_prob_space, name="X_1").from_dict(
            {
                "a": 2,
                "b": 4,
                "c": 4,
                "d": 4,
                "e": 4,
            }
        ),
    ]
    expected_range_sample_space = SampleSpace(name="X_range").from_list(
        [
            (1, 2),
            (3, 4),
        ]
    )
    expected_range_sig_alg = SigmaAlgebra.power_set(expected_range_sample_space)
    expected_range_prob_measure = ProbabilityMeasure(
        sig_alg=expected_range_sig_alg
    ).from_dict(
        {
            (1, 2): 0.1,
            (3, 4): 0.9,
        }
    )
    expected_range = ProbabilitySpace(
        sample_space=expected_range_sample_space,
        sig_alg=expected_range_sig_alg,
        prob_measure=expected_range_prob_measure,
    )

    assert X.domain == Omega2
    assert X.sig_alg == F3
    assert X.prob_measure == P2
    assert X.prob_space == expected_prob_space
    assert X.point_outputs == expected_point_outputs
    assert X.atom_outputs == expected_atom_outputs
    assert X.index == expected_index
    pd.testing.assert_frame_equal(X.data, expected_data)
    pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 2
    assert X.components == expected_components
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "P2_X"

    """Test the `overwrite` parameters of the `from_dict` method:

    * We set `overwrite_domain=False`, its default value. This means the method will attempt to align the keys of the dictionary with the existing sample space Omega2.
    * We set `overwrite_index=True` since we are going to change the dimension of the random vector to 3, so we need to replace the existing "2-dimensional" index.

    This should update the following:

    * The point_outputs dictionary.
    * The generated sigma-algebra.
    * The index.
    * The dimension.
    * The atom_outputs dictionary.
    * The atom_data.
    * The component random variables.
    * The range.

    This will not change the following:

    * The domain stays as Omega2.
    * The probability measure stays as P2.
    * The sigma-algebra stays as F3.
    * The underlying probability space.
    """

    point_outputs2 = {
        "a": (0, 1, 2),
        "b": (3, 4, 5),
        "c": (3, 4, 5),
        "d": (3, 4, 5),
        "e": (3, 4, 5),
    }
    X.from_dict(point_outputs2, overwrite_index=True)

    expected_atom_outputs = {
        "cat": (0, 1, 2),
        "dog": (3, 4, 5),
    }
    expected_index = Index().from_list(["X_0", "X_1", "X_2"], data_name="feature")
    expected_data = pd.DataFrame.from_dict(
        {
            "a": (0, 1, 2),
            "b": (3, 4, 5),
            "c": (3, 4, 5),
            "d": (3, 4, 5),
            "e": (3, 4, 5),
        },
        orient="index",
        columns=pd.Index(["X_0", "X_1", "X_2"], name="feature"),
    )
    expected_data.index.name = "letter"
    expected_atom_data = pd.DataFrame.from_dict(
        {
            "cat": (0, 1, 2),
            "dog": (3, 4, 5),
        },
        orient="index",
        columns=pd.Index(["X_0", "X_1", "X_2"], name="feature"),
    )
    expected_atom_data.index.name = "atom ID"
    expected_components = [
        RandomVariable(*expected_prob_space, name="X_0").from_dict(
            {
                "a": 0,
                "b": 3,
                "c": 3,
                "d": 3,
                "e": 3,
            }
        ),
        RandomVariable(*expected_prob_space, name="X_1").from_dict(
            {
                "a": 1,
                "b": 4,
                "c": 4,
                "d": 4,
                "e": 4,
            }
        ),
        RandomVariable(*expected_prob_space, name="X_2").from_dict(
            {
                "a": 2,
                "b": 5,
                "c": 5,
                "d": 5,
                "e": 5,
            }
        ),
    ]
    expected_generated_sig_alg = SigmaAlgebra(
        sample_space=Omega2, name="sigma(X)"
    ).from_dict(
        {
            "a": (0, 1, 2),
            "b": (3, 4, 5),
            "c": (3, 4, 5),
            "d": (3, 4, 5),
            "e": (3, 4, 5),
        }
    )
    expected_range_sample_space = SampleSpace(name="X_range").from_list(
        [
            (0, 1, 2),
            (3, 4, 5),
        ]
    )
    expected_range_sig_alg = SigmaAlgebra.power_set(expected_range_sample_space)
    expected_range_prob_measure = ProbabilityMeasure(
        sig_alg=expected_range_sig_alg
    ).from_dict(
        {
            (0, 1, 2): 0.1,
            (3, 4, 5): 0.9,
        }
    )
    expected_range = ProbabilitySpace(
        sample_space=expected_range_sample_space,
        sig_alg=expected_range_sig_alg,
        prob_measure=expected_range_prob_measure,
    )

    assert X.domain == Omega2
    assert X.sig_alg == F3
    assert X.prob_measure == P2
    assert X.prob_space == expected_prob_space
    assert X.point_outputs == point_outputs2
    assert X.atom_outputs == expected_atom_outputs
    assert X.index == expected_index
    pd.testing.assert_frame_equal(X.data, expected_data)
    pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 3
    assert X.components == expected_components
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "P2_X"

    """Again test the `overwrite` parameters of the `from_dict` method. This time:

    * We set `overwrite_domain=True`. This will completely replace the existing domain with the keys of the provided dictionary. The sigma-algebra will be set as the power-set on the new domain, and the probability measure as the uniform measure.
    * We set `overwrite_index=True` since we are going to change the dimension of the random vector to 1, so we need to replace the existing "3-dimensional" index.

    This should update everything:

    * The domain changes to the keys of the provided dictionary.
    * The probability measure changes to the uniform measure.
    * The sigma-algebra changes to the power-set on the new domain.
    * The underlying probability space.
    * The point_outputs dictionary.
    * The generated sigma-algebra.
    * The index.
    * The dimension.
    * The atom_outputs dictionary.
    * The atom_data.
    * The component random variables.
    * The range.
    """

    point_outputs3 = {
        "pen": 10,
        "pencil": 42,
        "marker": 42,
    }
    X.from_dict(point_outputs3, overwrite_domain=True, overwrite_index=True)

    expected_domain = SampleSpace().from_list(["pen", "pencil", "marker"])
    expected_sig_alg = SigmaAlgebra.power_set(expected_domain)
    expected_prob_measure = ProbabilityMeasure.uniform(expected_sig_alg)
    expected_prob_space = ProbabilitySpace(
        expected_domain, expected_sig_alg, expected_prob_measure
    )
    expected_data = pd.Series(
        {
            "pen": 10,
            "pencil": 42,
            "marker": 42,
        },
        name="X",
    )
    expected_data.index.name = "sample"
    expected_atom_data = pd.Series(
        {
            "pen": 10,
            "pencil": 42,
            "marker": 42,
        },
        name="X",
    )
    expected_atom_data.index.name = "atom ID"
    expected_generated_sig_alg = SigmaAlgebra(
        sample_space=expected_domain, name="sigma(X)"
    ).from_dict(
        {
            "pen": 10,
            "pencil": 42,
            "marker": 42,
        }
    )
    expected_range_sample_space = SampleSpace(name="X_range").from_list([10, 42])
    expected_range_sig_alg = SigmaAlgebra.power_set(expected_range_sample_space)
    expected_range_prob_measure = ProbabilityMeasure(
        sig_alg=expected_range_sig_alg
    ).from_dict(
        {
            10: 1 / 3,
            42: 2 / 3,
        }
    )
    expected_range = ProbabilitySpace(
        sample_space=expected_range_sample_space,
        sig_alg=expected_range_sig_alg,
        prob_measure=expected_range_prob_measure,
    )

    assert X.domain == expected_domain
    assert X.sig_alg == expected_sig_alg
    assert X.prob_measure == expected_prob_measure
    assert X.prob_space == expected_prob_space
    assert X.point_outputs == point_outputs3
    assert X.atom_outputs == point_outputs3
    assert X.index is None
    pd.testing.assert_series_equal(X.data, expected_data)
    pd.testing.assert_series_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 1
    assert X.components == [X]
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "uniform_X"

    """Test the `overwrite` parameters of the `from_pandas` method:

    * We set `overwrite_domain=False`, its default value. This means the method will attempt to align the keys of the data frame with the existing sample space `expected_domain = {"pen", "pencil", "marker"}.
    * We set `overwrite_index=False`, its default value, since the current index is `None`.

    This should update the following:

    * The point_outputs dictionary.
    * The generated sigma-algebra.
    * The index.
    * The dimension.
    * The atom_outputs dictionary.
    * The atom_data.
    * The component random variables.
    * The range.

    This will not change the following:

    * The domain stays as expected_domain.
    * The probability measure stays as expected_prob_measure.
    * The sigma-algebra stays as expected_sig_alg.
    * The underlying probability space stays as expected_prob_space.
    """

    data1 = pd.DataFrame(
        [(1, 2), (3, 4), (3, 4)],
        index=["pen", "pencil", "marker"],
        columns=pd.Index(["vec1", "vec2"], name="vecs"),
    )
    X.from_pandas(data1)

    expected_point_outputs = {
        "pen": (1, 2),
        "pencil": (3, 4),
        "marker": (3, 4),
    }
    expected_index = Index().from_list(["vec1", "vec2"], data_name="vecs")
    expected_atom_data = pd.DataFrame(
        [
            (1, 2),
            (3, 4),
            (3, 4),
        ],
        index=pd.Index(["pen", "pencil", "marker"], name="atom ID"),
        columns=pd.Index(["vec1", "vec2"], name="vecs"),
    )
    expected_components = [
        RandomVariable(*expected_prob_space, name="vec1").from_dict(
            {
                "pen": 1,
                "pencil": 3,
                "marker": 3,
            }
        ),
        RandomVariable(*expected_prob_space, name="vec2").from_dict(
            {
                "pen": 2,
                "pencil": 4,
                "marker": 4,
            }
        ),
    ]
    expected_generated_sig_alg = SigmaAlgebra(
        sample_space=expected_domain, name="sigma(X)"
    ).from_dict(
        {
            "pen": (1, 2),
            "pencil": (3, 4),
            "marker": (3, 4),
        }
    )
    expected_range_sample_space = SampleSpace(name="X_range").from_list(
        [(1, 2), (3, 4)]
    )
    expected_range_sig_alg = SigmaAlgebra.power_set(expected_range_sample_space)
    expected_range_prob_measure = ProbabilityMeasure(
        sig_alg=expected_range_sig_alg
    ).from_dict({(1, 2): 1 / 3, (3, 4): 2 / 3})
    expected_range = ProbabilitySpace(
        sample_space=expected_range_sample_space,
        sig_alg=expected_range_sig_alg,
        prob_measure=expected_range_prob_measure,
    )

    assert X.domain == expected_domain
    assert X.sig_alg == expected_sig_alg
    assert X.prob_measure == expected_prob_measure
    assert X.prob_space == expected_prob_space
    assert X.point_outputs == expected_point_outputs
    assert X.atom_outputs == expected_point_outputs
    assert X.index == expected_index
    pd.testing.assert_frame_equal(X.data, data1)
    pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 2
    assert X.components == expected_components
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "uniform_X"

    """Again test the `overwrite` parameters of the `from_pandas` method. This time:

    * We set `overwrite_domain=True`. This should overwrite the existing domain with the index of the provided data frame.
    * We set `overwrite_index=True`. This should overwrite the existing index with the columns of the provided data frame.

    This should update all components of the random vector.
    """

    data2 = pd.DataFrame(
        [(1, 2), (3, 4), (5, 6)],
        index=["tv", "phone", "microwave"],
        columns=pd.Index(["G1", "G2"], name="G"),
    )
    X.from_pandas(data2, overwrite_domain=True, overwrite_index=True)

    expected_domain = SampleSpace().from_list(["tv", "phone", "microwave"])
    expected_sig_alg = SigmaAlgebra.power_set(expected_domain)
    expected_prob_measure = ProbabilityMeasure.uniform(expected_sig_alg)
    expected_prob_space = ProbabilitySpace(
        expected_domain, expected_sig_alg, expected_prob_measure
    )
    expected_point_outputs = {
        "tv": (1, 2),
        "phone": (3, 4),
        "microwave": (5, 6),
    }
    expected_atom_outputs = {
        "tv": (1, 2),
        "phone": (3, 4),
        "microwave": (5, 6),
    }
    expected_index = Index().from_list(["G1", "G2"], data_name="G")
    expected_atom_data = pd.DataFrame(
        [
            (1, 2),
            (3, 4),
            (5, 6),
        ],
        index=pd.Index(["tv", "phone", "microwave"], name="atom ID"),
        columns=pd.Index(["G1", "G2"], name="G"),
    )
    expected_components = [
        RandomVariable(*expected_prob_space, name="vec1").from_dict(
            {
                "tv": 1,
                "phone": 3,
                "microwave": 5,
            }
        ),
        RandomVariable(*expected_prob_space, name="vec2").from_dict(
            {
                "tv": 2,
                "phone": 4,
                "microwave": 6,
            }
        ),
    ]
    expected_generated_sig_alg = SigmaAlgebra(
        sample_space=expected_domain, name="sigma(X)"
    ).from_dict(
        {
            "tv": (1, 2),
            "phone": (3, 4),
            "microwave": (5, 6),
        }
    )
    expected_range_sample_space = SampleSpace(name="X_range").from_list(
        [(1, 2), (3, 4), (5, 6)]
    )
    expected_range_sig_alg = SigmaAlgebra.power_set(expected_range_sample_space)
    expected_range_prob_measure = ProbabilityMeasure(
        sig_alg=expected_range_sig_alg
    ).from_dict(
        {
            (1, 2): 1 / 3,
            (3, 4): 1 / 3,
            (5, 6): 1 / 3,
        }
    )
    expected_range = ProbabilitySpace(
        sample_space=expected_range_sample_space,
        sig_alg=expected_range_sig_alg,
        prob_measure=expected_range_prob_measure,
    )

    assert X.domain == expected_domain
    assert X.sig_alg == expected_sig_alg
    assert X.prob_measure == expected_prob_measure
    assert X.prob_space == expected_prob_space
    assert X.point_outputs == expected_point_outputs
    assert X.atom_outputs == expected_atom_outputs
    assert X.index == expected_index
    pd.testing.assert_frame_equal(X.data, data2)
    pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
    assert X.dimension == 2
    assert X.components == expected_components
    assert X.generated_sig_alg == expected_generated_sig_alg
    assert X.range == expected_range
    assert X.range.sample_space.name == "X_range"
    assert X.range.sig_alg.name == "power_set"
    assert X.range.prob_measure.name == "uniform_X"
