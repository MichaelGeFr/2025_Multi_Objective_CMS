# each objective function has to be minimized ->maximization objectives need to be multiplied by -1
# each constraint has to be formulated as <= 0 ->change sign by multiplying by -1
# normalize constraints to make them operating on the same scale and give them equal importance: divide by the product of the coefficients

# element wise problem formulation

# problem formulation
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([-2,-2]), xu=np.array([2,2])) # set the correct attributes

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]
problem = MyProblem()

# algorithm is set up as an object to enable its usage for solving the problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# instead of using default hyper parameters I adapt the algorithm to create my own version of it
# further hyper parameter adaptions can be done to enable a faster convergence
algorithm = NSGA2(
    pop_size=40, # makes the algorithm greedier
    n_offsprings=10, # makes the algorithm greedier
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# define a termination criterion to start the optimization procedure

from pymoo.termination import get_termination

termination = get_termination("n_gen", 40) # limit of iterations is set to 40 iterations of the algorithm

# solve the problem with the algorithm and the termination criterion

from pymoo.optimize import minimize

res = minimize (problem, algorithm, termination, seed=1, save_history=True, verbose=True) # verbose delivers some printouts during algorithm execution

X=res.X
F=res.F

# visualization of the results

import matplotlib.pyplot as plt

# design space
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.title("Design Space")
plt.show()

# objective space
# pareto optimal solutions 
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()

# post processing
# analysis of scales 
# we assume the ideal and nadir points (also referred to as boundary points) and the Pareto-front are not known
fl = F.min(axis=0)
fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

# nadir point
# ideal point
approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

# plot boundary points
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
plt.title("Objective Space")
plt.legend()
plt.show()

# normalize objective values regarding the boundary points
nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

plt.figure(figsize=(7, 5))
plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()