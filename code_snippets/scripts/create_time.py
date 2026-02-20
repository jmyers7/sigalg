"""Create discrete and continuoue Time instances."""

from sigalg.core import Time

T_discrete = Time.discrete(start=1, stop=4, name="T_discrete")  # (1)!

T_discrete_from_length = Time.discrete(  # (2)!
    start=1, length=5, name="T_discrete_from_length"
)

T_continuous = Time.continuous(  # (3)!
    start=0, stop=1, num_points=10, name="T_continuous"
)


print(T_discrete)
print("\n", T_discrete_from_length)
print("\n", T_continuous)
