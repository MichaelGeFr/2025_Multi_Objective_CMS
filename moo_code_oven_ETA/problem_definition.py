# definition of the problem
# problem is not just a function, it is defined with meta data

import numpy as np
from pymoo.core.problem import Problem

# MOO problem definition for the oven in the ETA Reserach Factory
class EtaMooOvenProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, #two variables
                         n_obj=2, #two objectives
                         n_ieq_constr=2, #two inequality constraints
                         xl=np.ndarray[0,0], #lower bound
                         xu=np.ndarray[5000,5000] #upper bound
                         vtype=np.float64) #variable type

    def _evaluate(self, x, out, *args, **kwargs): #actual function evaluation takes place in the _evaluate method, which fills the out dictionary
        f1 = 100 * (x[:, 0]**2 + x[:, 1]**2) #vectorized with input X as a (N,2) matrix
        f2 = (x[:, 0]-1)**2 + x[:, 1]**2 #x is a matrix: each row is an individual, each column is a variable
        out["F"] = np.column_stack([f1, f2]) #function values are supposed to be written into out["F"]
        out["G"] = np.row_stack(F_list_of_lists) #constraints are written into out["G"]
