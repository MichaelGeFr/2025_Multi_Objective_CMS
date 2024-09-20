# minimize
# function/ method

def minimize(problem,
             algorithm,
             termination=None,
             seed=None,
             verbose=False,
             display=None,
             callback=None,
             return_least_infeasible=False,
             save_history=False
             )
    

# problem formulation

# unconstrained
import numpy as np
from pymoo.core.problem import Problem

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, #two objectives
                         n_obj=2, #two variables
                         xl=-2.0, #lower bound
                         xu=2.0) #upper bound

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[:, 0]**2 + x[:, 1]**2) #vectorized with input X as a (N,2) matrix
        f2 = (x[:, 0]-1)**2 + x[:, 1]**2
        out["F"] = np.column_stack([f1, f2])

problem=MyProblem()

# constrained
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2, #two variables
                         n_obj=2, #two objectives
                         n_ieq_constr=2, #two inequality constraints
                         xl=np.array([-2,-2]), #lower bound as a vector with the length of the number of variables
                         xu=np.array([2,2])) 

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)  #input x is one dimensional array of length 2
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18 #normalized inequality constraint
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8 #normalized inequality constraint

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

problem = MyProblem()

# algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2

algorithm = NSGA2()

