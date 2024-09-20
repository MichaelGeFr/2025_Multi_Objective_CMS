import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter


# Define the optimization problem
class HardeningOvenProblem(Problem):

    def __init__(self):
        super().__init__(
            n_var=3,      # 3 decision variables: T (temperature), t (time), Q (capacity)
            n_obj=2,      # 2 objectives: productivity and emissions
            n_constr=1,   # 1 constraint on maximum emissions
            xl=np.array([500, 1, 50]),   # Lower bounds: T_min, t_min, Q_min
            xu=np.array([800, 10, 500])  # Upper bounds: T_max, t_max, Q_max
        )

    def _evaluate(self, X, out, *args, **kwargs):
        T = X[:, 0]  # Temperature
        t = X[:, 1]  # Time
        Q = X[:, 2]  # Load capacity

        # Objective 1: Maximize productivity (we minimize -productivity to make it compatible with pymoo's minimization)
        f1 = -Q / t

        # Objective 2: Minimize emissions (energy consumption modeled as E(T, t) = alpha * T^beta * t)
        alpha = 0.01
        beta = 1.5
        f2 = alpha * T ** beta * t

        # Constraint: Ensure emissions do not exceed a certain threshold (E_max)
        E_max = 5000
        g1 = f2 - E_max

        # Assign objectives and constraints
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1])


# Instantiate the problem
problem = HardeningOvenProblem()

# Choose an algorithm (NSGA-II)
algorithm = NSGA2(pop_size=100)

# Define a termination condition (number of generations)
termination = get_termination("n_gen", 10)

# Run the optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

# Plot the Pareto front (trade-off between productivity and emissions)
plot = Scatter()
plot.add(res.F, facecolor="red")
plot.show()

# Print the Pareto-optimal solutions
print("Pareto-optimal solutions:")
print(res.X)

# Print the corresponding objective values
print("Objective values:")
print(res.F)
