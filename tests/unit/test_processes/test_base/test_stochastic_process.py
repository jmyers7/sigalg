from sigalg.core import (
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
    Time,
)
from sigalg.processes import StochasticProcess

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_with_no_parameters(self):
        """Test constructing an empty stochastic process."""
        X = StochasticProcess()
        prob_space = ProbabilitySpace()

        assert X.point_outputs is None
        assert X.atom_outputs is None
        assert X.data is None
        assert X.atom_data is None
        assert X.components is None
        assert X.time is None
        assert X.generated_sig_alg is None
        assert X.prob_space == prob_space
        assert X.domain is None
        assert X.sig_alg is None
        assert X.prob_measure is None
        assert X.range is None
        assert X.is_discrete_time is None
        assert X.n_trajectories is None
        assert X.natural_filtration is None
        assert X.last_rv is None

    def test_with_all_parameters(self):
        """Test constructing stochastic process with all parameters."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 3,
                1: 3,
                2: 1,
                3: 2,
            }
        )
        P = ProbabilityMeasure(sig_alg=F).from_dict(
            {
                1: 0.2,
                2: 0.35,
                3: 0.45,
            }
        )
        T = Time.discrete(length=3)
        Y = StochasticProcess(
            sample_space=Omega,
            sig_alg=F,
            prob_measure=P,
            index=T,
            name="Y",
        )
        prob_space = ProbabilitySpace(Omega, F, P)

        assert Y.point_outputs is None
        assert Y.atom_outputs is None
        assert Y.data is None
        assert Y.atom_data is None
        assert Y.components is None
        assert Y.time is T
        assert Y.generated_sig_alg is None
        assert Y.prob_space == prob_space
        assert Y.domain is Omega
        assert Y.sig_alg is F
        assert Y.prob_measure is P
        assert Y.range is None
        assert Y.is_discrete_time is True
        assert Y.n_trajectories is None
        assert Y.natural_filtration is None
        assert Y.last_rv is None


# class TestConstructor:
#     def test_constructor_with_time_and_domain(self):
#         """Test StochasticProcess constructor with time and domain."""
#         domain = SampleSpace().from_sequence(size=3)
#         time = Time.discrete(length=4)
#         X = StochasticProcess(domain=domain, time=time)

#         assert X.domain == domain
#         assert X.time == time
#         assert X.name == "X"

#     def test_constructor_with_time_only(self):
#         """Test StochasticProcess constructor with time only."""
#         time = Time.discrete(length=5)
#         X = StochasticProcess(time=time)

#         assert X.domain is None
#         assert X.time == time
#         assert X.name == "X"

#     def test_constructor_with_custom_name(self):
#         """Test StochasticProcess constructor with custom name."""
#         time = Time.discrete(length=3)
#         Y = StochasticProcess(time=time, name="Y")

#         assert Y.name == "Y"
#         assert Y.time == time

#     def test_constructor_with_none_name(self):
#         """Test StochasticProcess constructor with None name."""
#         time = Time.discrete(length=3)
#         X = StochasticProcess(time=time, name=None)

#         assert X.name is None
#         assert X.time == time

#     def test_constructor_with_custom_is_discrete_state(self):
#         """Test StochasticProcess constructor with custom is_discrete_state."""
#         time = Time.discrete(length=3)
#         X = StochasticProcess(time=time, is_discrete_state=True)

#         assert X.is_discrete_state is True

#     def test_constructor_with_none_is_discrete_state(self):
#         """Test StochasticProcess constructor with None is_discrete_state."""
#         time = Time.discrete(length=3)
#         X = StochasticProcess(time=time, is_discrete_state=None)

#         assert X.is_discrete_state is None


# class TestProperties:
#     def test_time_property(self):
#         """Test time property returns the time index."""
#         time = Time.discrete(length=4)
#         X = StochasticProcess(time=time)

#         assert X.time == time

#     def test_time_setter_invalid_type_raises(self):
#         """Test that time setter raises TypeError for non-Time objects."""
#         X = StochasticProcess(is_discrete_time=True)
#         with pytest.raises(TypeError):
#             X.time = "not a Time instance"

#     def test_n_trajectories_with_data(self):
#         """Test n_trajectories property with data."""
#         domain = SampleSpace().from_sequence(size=3)
#         time = Time.discrete(length=3)
#         X = StochasticProcess(domain=domain, time=time).from_dict(
#             {
#                 0: (1, 2, 3, 4),
#                 1: (5, 6, 7, 8),
#                 2: (9, 10, 11, 12),
#             }
#         )

#         assert X.n_trajectories == 3

#     def test_n_trajectories_without_data(self):
#         """Test n_trajectories property without data."""
#         time = Time.discrete(length=4)
#         X = StochasticProcess(time=time)

#         assert X.n_trajectories is None

#     def test_probability_measure_with_non_generated_data_returns_uniform(self):
#         """Test that probability_measure returns uniform measure with non-generated data."""
#         domain = SampleSpace().from_sequence(size=3)
#         time = Time.discrete(length=2)
#         X = StochasticProcess(domain=domain, time=time).from_dict(
#             {
#                 0: (1, 2, 3),
#                 1: (3, 4, 5),
#                 2: (5, 6, 7),
#             }
#         )

#         expected_measure = ProbabilityMeasure.uniform(sig_alg=SigmaAlgebra.power_set(domain))
#         assert X.prob_measure == expected_measure


# class TestFromConstant:
#     def test_from_constant_with_domain_and_time(self):
#         """Test from_constant method with domain and time."""
#         domain = SampleSpace().from_sequence(size=2)
#         time = Time.discrete(length=2)
#         X = StochasticProcess(domain=domain, time=time).from_constant(value=5)

#         expected_data = pd.DataFrame(
#             [[5, 5, 5], [5, 5, 5]],
#             index=domain.data,
#             columns=time.data,
#         )

#         pd.testing.assert_frame_equal(X.data, expected_data)

#     def test_from_constant_sets_uniform_probability_measure(self):
#         """Test that from_constant sets uniform probability measure."""
#         domain = SampleSpace().from_sequence(size=2)
#         time = Time.discrete(length=3)
#         X = StochasticProcess(domain=domain, time=time).from_constant(value=1)

#         expected_measure = ProbabilityMeasure.uniform(sig_alg=SigmaAlgebra.power_set(domain))
#         assert X.prob_measure == expected_measure

#     def test_from_constant_without_domain_raises(self):
#         """Test that from_constant raises ValueError without domain."""
#         time = Time.discrete(length=3)
#         X = StochasticProcess(time=time)

#         with pytest.raises(ValueError):
#             X.from_constant(value=1)

#     def test_from_constant_non_numeric_value_raises(self):
#         """Test that non-numeric value raises TypeError."""
#         domain = SampleSpace().from_sequence(size=2)
#         time = Time.discrete(length=3)
#         X = StochasticProcess(domain=domain, time=time)

#         with pytest.raises(TypeError):
#             X.from_constant(value="not a number")


# class TestDataAccess:
#     @pytest.fixture
#     def process(self):
#         domain = SampleSpace().from_sequence(size=3)
#         time = Time.discrete(start=1, length=3)
#         return StochasticProcess(domain=domain, time=time, name="X").from_dict(
#             {
#                 0: (1, 2, 3, 4),
#                 1: (5, 6, 7, 8),
#                 2: (9, 10, 11, 12),
#             }
#         )

#     def test_getitem_returns_random_variable(self, process):
#         """Test __getitem__ returns a RandomVariable."""
#         rv = process[2]
#         expected_data = pd.Series(
#             [2, 6, 10],
#             index=process.domain.data,
#             name="X_2",
#         )

#         pd.testing.assert_series_equal(rv.data, expected_data)

#     def test_getitem_without_time_raises(self):
#         """Test that __getitem__ raises ValueError without time."""
#         X = StochasticProcess(is_discrete_time=True)

#         with pytest.raises(ValueError):
#             _ = X[0]

#     def test_iter_returns_random_variables(self, process):
#         """Test __iter__ yields RandomVariable instances."""
#         rvs = list(process)

#         assert len(rvs) == 4
#         assert all(isinstance(rv, RandomVariable) for rv in rvs)


# class TestEquality:
#     def test_equality_same_processes(self):
#         """Test equality for identical processes."""
#         domain = SampleSpace().from_sequence(size=2)
#         time = Time.discrete(length=2)
#         outputs = {0: (1, 2, 3), 1: (4, 5, 6)}

#         X1 = StochasticProcess(domain=domain, time=time, name="X1").from_dict(outputs)
#         X2 = StochasticProcess(domain=domain, time=time, name="X2").from_dict(outputs)

#         assert X1 == X2

#     def test_inequality_different_data(self):
#         """Test inequality for processes with different data."""
#         domain = SampleSpace().from_sequence(size=2)
#         time = Time.discrete(length=2)
#         outputs1 = {0: (1, 2, 3), 1: (4, 5, 6)}
#         outputs2 = {0: (1, 2, 3), 1: (4, 5, 7)}

#         X1 = StochasticProcess(domain=domain, time=time).from_dict(outputs1)
#         X2 = StochasticProcess(domain=domain, time=time).from_dict(outputs2)

#         assert X1 != X2

#     def test_inequality_with_non_stochastic_process(self):
#         """Test inequality comparison with non-StochasticProcess object."""
#         domain = SampleSpace().from_sequence(size=2)
#         time = Time.discrete(length=2)
#         outputs = {0: (1, 2, 3), 1: (4, 5, 6)}
#         X = StochasticProcess(domain=domain, time=time).from_dict(outputs)

#         assert X != "not a stochastic process"
#         assert X != 42


# class TestValidationHelpers:
#     def test_validate_and_initialize_domain_creates_domain(self):
#         """Test _validate_and_initialize_domain creates domain."""
#         X = StochasticProcess(is_discrete_time=True)
#         X._validate_and_initialize_domain(n_trajectories=3)

#         assert X.domain is not None
#         assert len(X.domain) == 3

#     def test_validate_and_initialize_domain_with_existing_domain(self):
#         """Test _validate_and_initialize_domain with existing domain."""
#         domain = SampleSpace().from_sequence(size=3)
#         X = StochasticProcess(domain=domain, is_discrete_time=True)
#         X._validate_and_initialize_domain(n_trajectories=3)

#         assert X.domain == domain

#     def test_validate_and_initialize_domain_mismatched_size_raises(self):
#         """Test that _validate_and_initialize_domain raises with mismatched size."""
#         domain = SampleSpace().from_sequence(size=2)
#         X = StochasticProcess(domain=domain, is_discrete_time=True)

#         with pytest.raises(ValueError):
#             X._validate_and_initialize_domain(n_trajectories=3)


# class TestPlotTrajectories:
#     def test_plot_trajectories_returns_axes(self):
#         """Test that plot_trajectories returns a matplotlib Axes object."""
#         T = Time.discrete(length=2)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()
#         ax = X.plot_trajectories()

#         assert isinstance(ax, Axes)

#     def test_plot_trajectories_over_discrete_time(self):
#         """Test plot_trajectories with discrete time process."""
#         T = Time.discrete(length=2)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()
#         ax = X.plot_trajectories()

#         assert isinstance(ax, Axes)
#         assert ax.get_xlabel() == "time"
#         assert ax.get_ylabel() == "state"
#         assert ax.get_title() == "Random walk process 'X'"
#         n_lines = len(ax.get_lines())
#         assert n_lines == X.n_trajectories

#         plt.close()

#     def test_plot_trajectories_over_continuous_time(self):
#         """Test plot_trajectories with continuous time process."""
#         T = Time.continuous(start=1, stop=2, dt=0.13)
#         X = BrownianMotion(time=T).from_simulation(n_trajectories=3, random_state=42)
#         ax = X.plot_trajectories()

#         assert isinstance(ax, Axes)
#         assert ax.get_xlabel() == "time"
#         assert ax.get_ylabel() == "state"
#         n_lines = len(ax.get_lines())
#         assert n_lines == 3

#         plt.close()

#     def test_plot_trajectories_with_custom_labels(self):
#         """Test plot_trajectories with custom axis labels."""
#         T = Time.discrete(length=2)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()
#         ax = X.plot_trajectories(
#             x_label="Custom Time",
#             y_label="Custom State",
#             title="Custom Title",
#         )

#         assert ax.get_xlabel() == "Custom Time"
#         assert ax.get_ylabel() == "Custom State"
#         assert ax.get_title() == "Custom Title"

#         plt.close()

#     def test_plot_trajectories_with_custom_axes(self):
#         """Test plot_trajectories with provided axes object."""
#         _, custom_ax = plt.subplots()
#         T = Time.discrete(length=2)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()
#         ax = X.plot_trajectories(ax=custom_ax)

#         assert ax is custom_ax
#         assert len(ax.get_lines()) == X.n_trajectories

#         plt.close()

#     def test_plot_trajectories_with_plot_kwargs(self):
#         """Test plot_trajectories with custom plot kwargs."""
#         T = Time.discrete(length=2)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()
#         ax = X.plot_trajectories(plot_kwargs={"linewidth": 3, "alpha": 0.5})

#         lines = ax.get_lines()
#         for line in lines:
#             assert line.get_linewidth() == 3
#             assert line.get_alpha() == 0.5

#         plt.close()

#     def test_plot_trajectories_without_data_raises(self):
#         """Test that plot_trajectories raises ValueError without data."""
#         T = Time.discrete(length=2)
#         X = StochasticProcess(time=T)

#         with pytest.raises(ValueError):
#             X.plot_trajectories()

#     def test_plot_trajectories_invalid_axes_raises(self):
#         """Test that plot_trajectories raises TypeError with invalid axes."""
#         T = Time.discrete(length=2)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()

#         with pytest.raises(TypeError):
#             X.plot_trajectories(ax="not an axes object")


# class TestMartingaleMethods:
#     def test_enumerated_symmetric_random_walk_is_martingale(self):
#         """Test that an enumerated symmetric random walk is a martingale."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()

#         assert X.is_martingale()
#         assert X.is_submartingale()
#         assert X.is_supermartingale()

#     def test_enumerated_random_walk_with_positive_drift_is_submartingale(self):
#         """Test that an enumerated random walk with positive drift is a submartingale."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()

#         assert X.is_submartingale()
#         assert not X.is_supermartingale()
#         assert not X.is_martingale()

#     def test_enumerated_random_walk_with_negative_drift_is_supermartingale(self):
#         """Test that an enumerated random walk with negative drift is a supermartingale."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.3, time=T).from_enumeration()

#         assert X.is_supermartingale()
#         assert not X.is_submartingale()
#         assert not X.is_martingale()

#     # @pytest.mark.filterwarnings("ignore::UserWarning")
#     # def test_simulated_symmetric_random_walk_is_martingale(self):
#     #     """Test that a simulated symmetric random walk is a martingale."""
#     #     T = Time.discrete(length=2)
#     #     X = RandomWalk(p=0.5, time=T).from_simulation(
#     #         n_trajectories=10_000, random_state=42
#     #     )

#     #     assert X.is_martingale(atol=0.5)
#     #     assert X.is_submartingale(atol=0.5)
#     #     assert X.is_supermartingale(atol=0.5)

#     # @pytest.mark.filterwarnings("ignore::UserWarning")
#     # def test_simulated_random_walk_with_positive_drift_is_submartingale(self):
#     #     """Test that a simulated random walk with positive drift is a submartingale."""
#     #     T = Time.discrete(length=2)
#     #     X = RandomWalk(p=0.7, time=T).from_simulation(
#     #         n_trajectories=10_000, random_state=42
#     #     )

#     #     assert not X.is_martingale()
#     #     assert X.is_submartingale()
#     #     assert not X.is_supermartingale()

#     # @pytest.mark.filterwarnings("ignore::UserWarning")
#     # def test_simulated_random_walk_with_negative_drift_is_supermartingale(self):
#     #     """Test that a simulated random walk with negative drift is a supermartingale."""
#     #     T = Time.discrete(length=2)
#     #     X = RandomWalk(p=0.3, time=T).from_simulation(
#     #         n_trajectories=10_000, random_state=42
#     #     )

#     #     assert not X.is_martingale()
#     #     assert not X.is_submartingale()
#     #     assert X.is_supermartingale()

#     # def test_martingale_checks_raise_for_non_discrete_state(self):
#     #     """Test that martingale checks raise ValueError for non-discrete-state processes."""
#     #     T = Time.continuous(start=0, stop=1, dt=0.1)
#     #     X = BrownianMotion(time=T).from_simulation(n_trajectories=3, random_state=42)

#     #     with pytest.raises(ValueError):
#     #         X.is_martingale()
#     #     with pytest.raises(ValueError):
#     #         X.is_submartingale()
#     #     with pytest.raises(ValueError):
#     #         X.is_supermartingale()

#     def test_process_is_adapted(self):
#         """Test that a process is adapted."""
#         T = Time.discrete(start=0, stop=2)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()

#         # Define a process Y for which each Y_t is a function of X_0, ..., X_t
#         def f0(X: StochasticProcess) -> RandomVariable:
#             return X[0]

#         def f1(X: StochasticProcess) -> RandomVariable:
#             return 2 * X[0] + X[1]

#         def f2(X: StochasticProcess) -> RandomVariable:
#             return X[2] - X[1] + X[0]

#         Y = X.transform(functions=[f0, f1, f2], time=T, name="Y")

#         assert Y.is_adapted(filtration=X.natural_filtration)

#     def test_process_is_not_adapted(self):
#         """Test that a process is not adapted."""
#         T = Time.discrete(start=0, stop=2)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()

#         # Define a process Y for which each Y_t is not a function of X_0, ..., X_t
#         def f0(X: StochasticProcess) -> RandomVariable:
#             return X[1]

#         def f1(X: StochasticProcess) -> RandomVariable:
#             return X[2]

#         def f2(X: StochasticProcess) -> RandomVariable:
#             return X[0]

#         Y = X.transform(functions=[f0, f1, f2], time=T, name="Y")

#         assert not Y.is_adapted(filtration=X.natural_filtration)

#     # def test_process_is_predictable(self):
#     #     """Test that a process is predictable."""
#     #     T = Time.discrete(start=0, stop=3)
#     #     X = RandomWalk(p=0.7, time=T).from_enumeration()

#     #     # Define a process Y for which each Y_t is a function of X_0, ..., X_{t-1}
#     #     def f1(X: StochasticProcess) -> RandomVariable:
#     #         return 2 * X[0]

#     #     def f2(X: StochasticProcess) -> RandomVariable:
#     #         return X[1] + X[0]

#     #     def f3(X: StochasticProcess) -> RandomVariable:
#     #         return X[2] - 5 * X[1]

#     #     S = Time.discrete(start=1, stop=3)
#     #     Y = X.transform(functions=[f1, f2, f3], time=S, name="Y")

#     #     assert Y.is_predictable(filtration=X.natural_filtration)

#     # def test_process_is_not_predictable(self):
#     #     """Test that a process is not predictable."""
#     #     T = Time.discrete(start=0, stop=3)
#     #     X = RandomWalk(p=0.7, time=T).from_enumeration()

#     #     # Define a process Y for which each Y_t is not a function of X_0, ..., X_{t-1}
#     #     def f1(X: StochasticProcess) -> RandomVariable:
#     #         return 2 * X[0]

#     #     def f2(X: StochasticProcess) -> RandomVariable:
#     #         return X[1] + X[0]

#     #     def f3(X: StochasticProcess) -> RandomVariable:
#     #         return X[3]

#     #     S = Time.discrete(start=1, stop=3)
#     #     Y = X.transform(functions=[f1, f2, f3], time=S, name="Y")

#     #     assert not Y.is_predictable(filtration=X.natural_filtration)

#     def test_is_martingale_without_data_raises(self):
#         """Test that is_martingale raises ValueError without data."""
#         T = Time.discrete(length=5)
#         X = StochasticProcess(time=T)

#         with pytest.raises(ValueError, match="Data must be generated"):
#             X.is_martingale()

#     def test_is_martingale_invalid_filtration_type_raises(self):
#         """Test that is_martingale raises TypeError for invalid filtration type."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()

#         with pytest.raises(TypeError, match="must be an instance of Filtration"):
#             X.is_martingale(filtration="not a filtration")

#     def test_is_martingale_filtration_mismatched_sample_space_raises(self):
#         """Test that is_martingale raises TypeError for mismatched sample space."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()

#         different_domain = SampleSpace().from_sequence(size=10)
#         wrong_filtration = Filtration(time=T).from_pandas(
#             pd.DataFrame(
#                 {t: range(10) for t in T.data},
#                 index=different_domain.data,
#             )
#         )

#         with pytest.raises(
#             TypeError, match="sample space must match the domain of the process"
#         ):
#             X.is_martingale(filtration=wrong_filtration)

#     def test_is_martingale_filtration_mismatched_time_raises(self):
#         """Test that is_martingale raises TypeError for mismatched time index."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.5, time=T).from_enumeration()

#         different_time = Time.discrete(length=3)
#         wrong_filtration = Filtration(time=different_time).from_pandas(
#             pd.DataFrame(
#                 {t: range(len(X.domain)) for t in different_time.data},
#                 index=X.domain.data,
#             )
#         )

#         with pytest.raises(
#             TypeError, match="time index must match the time index of the process"
#         ):
#             X.is_martingale(filtration=wrong_filtration)

#     def test_is_submartingale_without_data_raises(self):
#         """Test that is_submartingale raises ValueError without data."""
#         T = Time.discrete(length=5)
#         X = StochasticProcess(time=T)

#         with pytest.raises(ValueError, match="Data must be generated"):
#             X.is_submartingale()

#     def test_is_submartingale_invalid_filtration_type_raises(self):
#         """Test that is_submartingale raises TypeError for invalid filtration type."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()

#         with pytest.raises(TypeError, match="must be an instance of Filtration"):
#             X.is_submartingale(filtration=42)

#     def test_is_submartingale_filtration_mismatched_sample_space_raises(self):
#         """Test that is_submartingale raises TypeError for mismatched sample space."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()

#         different_domain = SampleSpace().from_sequence(size=10)
#         wrong_filtration = Filtration(time=T).from_pandas(
#             pd.DataFrame(
#                 {t: range(10) for t in T.data},
#                 index=different_domain.data,
#             )
#         )

#         with pytest.raises(
#             TypeError, match="sample space must match the domain of the process"
#         ):
#             X.is_submartingale(filtration=wrong_filtration)

#     def test_is_submartingale_filtration_mismatched_time_raises(self):
#         """Test that is_submartingale raises TypeError for mismatched time index."""
#         T = Time.discrete(length=5)
#         X = RandomWalk(p=0.7, time=T).from_enumeration()

#         different_time = Time.discrete(length=3)
#         wrong_filtration = Filtration(time=different_time).from_pandas(
#             pd.DataFrame(
#                 {t: range(len(X.domain)) for t in different_time.data},
#                 index=X.domain.data,
#             )
#         )

#         with pytest.raises(
#             TypeError, match="time index must match the time index of the process"
#         ):
#             X.is_submartingale(filtration=wrong_filtration)

#     def test_is_supermartingale_without_data_raises(self):
#         """Test that is_supermartingale raises ValueError without data."""
#         T = Time.discrete(length=5)
#         X = StochasticProcess(time=T)

#         with pytest.raises(ValueError, match="Data must be generated"):
#             X.is_supermartingale()


# class TestComparisonOperators:
#     def test_lt_two_stochastic_processes(self):
#         """Test less than comparison of two StochasticProcesses."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
#         )
#         result = X < Y
#         expected_data = pd.DataFrame(
#             [[False, True], [False, True], [False, False]],
#             index=Omega.data,
#             columns=T.data,
#         )

#         assert isinstance(result, StochasticProcess)
#         assert result.name == "(X < Y)"
#         assert result.domain == Omega
#         assert result.time == T
#         pd.testing.assert_frame_equal(result.data, expected_data)

#     def test_le_two_stochastic_processes(self):
#         """Test less than or equal comparison of two StochasticProcesses."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (1, 3), 1: (2, 4), 2: (3, 4)}
#         )
#         result = X <= Y
#         expected_data = pd.DataFrame(
#             [[True, True], [True, True], [True, True]],
#             index=Omega.data,
#             columns=T.data,
#         )

#         assert isinstance(result, StochasticProcess)
#         assert result.name == "(X <= Y)"
#         assert result.domain == Omega
#         assert result.time == T
#         pd.testing.assert_frame_equal(result.data, expected_data)

#     def test_gt_two_stochastic_processes(self):
#         """Test greater than comparison of two StochasticProcesses."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (5, 6), 1: (3, 4), 2: (1, 2)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (3, 5), 1: (3, 3), 2: (2, 3)}
#         )
#         result = X > Y
#         expected_data = pd.DataFrame(
#             [[True, True], [False, True], [False, False]],
#             index=Omega.data,
#             columns=T.data,
#         )

#         assert isinstance(result, StochasticProcess)
#         assert result.name == "(X > Y)"
#         assert result.domain == Omega
#         assert result.time == T
#         pd.testing.assert_frame_equal(result.data, expected_data)

#     def test_ge_two_stochastic_processes(self):
#         """Test greater than or equal comparison of two StochasticProcesses."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (5, 6), 1: (3, 4), 2: (1, 2)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (5, 5), 1: (3, 4), 2: (2, 3)}
#         )
#         result = X >= Y
#         expected_data = pd.DataFrame(
#             [[True, True], [True, True], [False, False]],
#             index=Omega.data,
#             columns=T.data,
#         )

#         assert isinstance(result, StochasticProcess)
#         assert result.name == "(X >= Y)"
#         assert result.domain == Omega
#         assert result.time == T
#         pd.testing.assert_frame_equal(result.data, expected_data)

#     def test_comparison_at_time_point(self):
#         """Test comparison of individual time slices returns RandomVariable."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=2)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (0, 3, 2), 1: (1, 4, 3), 2: (2, 5, 4)}
#         )
#         result = X[1] < Y[1]
#         expected_data = pd.Series(
#             [True, True, True],
#             index=Omega.data,
#             name="(X_1 < Y_1)",
#         )

#         assert isinstance(result, RandomVariable)
#         assert result.name == "(X_1 < Y_1)"
#         pd.testing.assert_series_equal(result.data, expected_data)

#     def test_lt_with_different_domains_raises(self):
#         """Test that comparing StochasticProcesses with different domains raises ValueError."""
#         Omega1 = SampleSpace().from_sequence(size=3)
#         Omega2 = SampleSpace().from_sequence(size=4)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega1, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega2, time=T, name="Y").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (4, 5)}
#         )

#         with pytest.raises(ValueError, match="must have the same domain"):
#             _ = X < Y

#     def test_lt_with_different_dimensions_raises(self):
#         """Test that comparing StochasticProcesses with different dimensions raises ValueError."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T1 = Time.discrete(start=0, stop=1)
#         T2 = Time.discrete(start=0, stop=2)
#         X = StochasticProcess(domain=Omega, time=T1, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T2, name="Y").from_dict(
#             {0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5)}
#         )

#         with pytest.raises(ValueError, match="must have the same dimension"):
#             _ = X < Y

#     def test_lt_with_non_stochastic_process_raises(self):
#         """Test that comparing StochasticProcess with non-StochasticProcess raises TypeError."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )

#         with pytest.raises(TypeError, match="must be a RandomVector"):
#             _ = X < "not a stochastic process"


# class TestBooleanMethods:
#     def test_all_returns_true_when_all_true(self):
#         """Test that all() returns True when all values are True."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (2, 3), 1: (3, 4), 2: (4, 5)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         result = X > Y

#         assert result.all() is True

#     def test_all_returns_false_when_some_false(self):
#         """Test that all() returns False when some values are False."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
#         )
#         result = X < Y

#         assert result.all() is False

#     def test_any_returns_true_when_some_true(self):
#         """Test that any() returns True when at least one value is True."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
#         )
#         result = X < Y

#         assert result.any() is True

#     def test_any_returns_false_when_all_false(self):
#         """Test that any() returns False when all values are False."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (5, 6), 1: (7, 8), 2: (9, 10)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (1, 2), 1: (3, 4), 2: (5, 6)}
#         )
#         result = X < Y

#         assert result.any() is False

#     def test_bool_raises_value_error(self):
#         """Test that __bool__() raises ValueError to prevent ambiguous boolean conversion."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
#         )
#         result = X < Y

#         with pytest.raises(
#             ValueError, match="truth value of a RandomVector is ambiguous"
#         ):
#             bool(result)

#     def test_bool_in_if_statement_raises(self):
#         """Test that using StochasticProcess in if statement raises ValueError."""
#         Omega = SampleSpace().from_sequence(size=3)
#         T = Time.discrete(start=0, stop=1)
#         X = StochasticProcess(domain=Omega, time=T, name="X").from_dict(
#             {0: (1, 2), 1: (2, 3), 2: (3, 4)}
#         )
#         Y = StochasticProcess(domain=Omega, time=T, name="Y").from_dict(
#             {0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
#         )
#         result = X < Y

#         with pytest.raises(
#             ValueError, match="truth value of a RandomVector is ambiguous"
#         ):
#             if result:
#                 pass
