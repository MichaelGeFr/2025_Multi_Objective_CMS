# test case
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class ConstrainedProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_ieq_constr=1,
                         n_eq_constr=0,
                         xl=0,
                         xu=2,
                         **kwargs)
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0] ** 2 + x[1] ** 2 #objective function
        out["G"] = 1.0 - (x[0] + x[1]) #inequality constraint formulated as smaller equal 0

import numpy as np
import matplotlib.pyplot as plt

X1, X2 = np.meshgrid(np.linspace(0, 2, 500), np.linspace(0, 2, 500))

F = X1**2 + X2**2
plt.rc('font', family='serif')

levels = 5 * np.linspace(0, 1, 10)
plt.figure(figsize=(7, 5))
CS = plt.contour(X1, X2, F, levels, colors='black', alpha=0.5)
CS.collections[0].set_label("$f(x)$")

X = np.linspace(0, 1, 500)
plt.plot(X, 1-X, linewidth=2.0, color="green", linestyle='dotted', label="g(x)")

plt.scatter([0.5], [0.5], marker="*", color="red", s=200, label="Optimum (0.5, 0.5)")

plt.xlim(0, 2)
plt.ylim(0, 2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=4, fancybox=True, shadow=False)

plt.tight_layout()
plt.show()