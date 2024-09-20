# result
# after algorithm execution a result object is returned

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize

# SOO 
# no constraints
problem = get_problem("sphere")
algorithm = GA(pop_size=5)
res = minimize(problem,
               algorithm,
               ('n_gen', 30),
               seed=1)

# possible attributes of the result object
# res.X: Design space values are

# res.F: Objective spaces values

# res.G: Constraint values

# res.CV: Aggregated constraint violation

# res.algorithm: Algorithm object which has been iterated over

# res.opt: The solutions as a Population object.

# res.pop: The final Population

# res.history: The history of the algorithm. (only if save_history has been enabled during the algorithm initialization)

# res.exec_time: The time required to run the algorithm



problem = get_problem("g1")
algorithm = GA(pop_size=5)
res = minimize(problem,
               algorithm,
               ('n_gen', 5),
               verbose=True,
               seed=1)