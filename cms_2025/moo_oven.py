#basic imports for building the problem
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem 
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2  import NSGA2
from pymoo.visualization.scatter import Scatter

# imports for algorithm instantiation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

#imports for termination criterion
from pymoo.termination import get_termination

#for compromise programming
from pymoo.decomposition.asf import ASF

#global font parameters for plots
plt.rcParams['font.family'] = 'Times New Roman'

#parameters of the oven
m_batch = 200 #mass of batch in kg

#time relevant parameters of the heat treatment process
t_ref_h = 0.6 #reference time for heating up at full load in h
t_n = 2 #reference time for the nitriding process in s
t_ref_c = 1.3 #reference time for cooling at full load in h
t_shift = 8 #shift hours in h

#power related parameters
P_h = 360e6 #gas heater power in J/h
P_a = 36e6 #stationary gas heater power during annealing in J/h
P_c = 27e6 #cooling fan power in  J/h

#multi-objective optimization problem
class HeatTreatmentAnnealingOven(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, 
                         n_obj=2, 
                         n_ieq_constr=1, 
                         xl=np.array([0.1, 0]), 
                         xu=np.array([1, 1]))

    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[0]
        x2 = x[1]
        
        #objective functions (in pymoo only minimization problems solvable)
        f1 = -(m_batch * x1)/(t_ref_h * x1 + t_n + t_ref_c * (x1/x2**2))
        f2 = P_h * t_ref_h * x1 + P_a * t_n + P_c * t_ref_c * x1 * x2**3
        out["F"] = [f1, f2]
        
        #calculate the constraints (in pymoo only <= constraints)
        g1 = t_ref_h * x1 + t_n + t_ref_c * (x1/x2**2) - t_shift
        out["G"] = [g1]

#create an instance of the problem
problem = HeatTreatmentAnnealingOven()

#set up the NSGA-II algorithm
algorithm = NSGA2(pop_size=40,
                  n_offsprings=100,
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(eta=20),
                  eliminate_duplicates=True
)

#set termination criterion
termination = get_termination("n_gen", 40)

#optimize the problem
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
#if verbose = True: 
# each line = one iteration; 
# n_gen=current generation counter; 
# n_eval=number of evaluations; 
# cv_min=minimum constraint violation; 
# cv_avg=average constraint violation,
# n_nds=number of non-dominated solutions;

X=res.X
F=res.F

#visualize the results
xl, xu = problem.bounds()
#design space
plt.figure(figsize=(3.36, 3.36))
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.xlabel('Batch mass', fontsize=10)
plt.ylabel('Fan speed', fontsize=10)
plt.title('Design space',  fontsize=10)
plt.savefig('design_space.pdf', bbox_inches='tight')
plt.show()

#objective space normalized
fl = F.min(axis=0)
fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")
approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

#normalization of the objective values with regards to the boundary points
nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

plt.figure(figsize=(3.36, 3.36))
plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.xlabel('Batch mass', fontsize=10)
plt.ylabel('Fan speed', fontsize=10)
#plt.title('Pareto front', fontsize=10)
plt.savefig('pareto_front.pdf', bbox_inches='tight')
plt.show()

#compromise programming
#weights: first objective is less important then the second
weights=np.array([0.2, 0.8])
weights_even=np.array([0.5,0.5])
decomp = ASF()

i = decomp.do(nF, 1/weights).argmin()
j = decomp.do(nF, 1/weights_even).argmin()

#visualize weighted results
print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))
print("Best regarding ASF: Point \ni = %s\nF = %s" % (j, F[j]))

plt.figure(figsize=(3.36, 3.36))
plt.scatter(F[:, 0]*(-1), F[:, 1]/3.6e6, s=30, facecolors='none', edgecolors='blue') #divide by 3.6e6 for kWh
plt.scatter(F[i, 0]*(-1), F[i, 1]/3.6e6, marker="x", color="red", s=200)
plt.scatter(F[j, 0]*(-1), F[j, 1]/3.6e6, marker="x", color="green", s=200)
plt.xlabel('output in kg / h', fontsize=10)
plt.ylabel('energy consumption in kWh', fontsize=10)
#plt.title('Objective space with best solution', fontsize=10)
plt.savefig('objective_space.pdf', bbox_inches='tight')
plt.show()

#print variabels for the best weighted solutions
best_solution_variables = X[i]
print(f"Best Solution Design Variables: x1 = {best_solution_variables[0]:.4f}, x2 = {best_solution_variables[1]:.4f}")

best_solution_variables = X[j]
print(f"Best Solution Design Variables even: x1 = {best_solution_variables[0]:.4f}, x2 = {best_solution_variables[1]:.4f}")
