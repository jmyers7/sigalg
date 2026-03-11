from sigalg.core import Time
from sigalg.processes import RandomWalk, StochasticProcess


class TestItoIntegral:
    def test_fundamental_theorem_of_calculus(self):
        """Integrating the increments of a process against time should return the original process."""
        time = Time().discrete(length=2)
        X = RandomWalk(p=0.6, time=time, initial_state=1).from_enumeration()
        T = StochasticProcess(domain=X.domain, time=time, name="T").from_time()
        integral = X.increments().ito_integral(T)

        assert X[0] + integral == X
        assert integral.name == "int X_increments dT"
