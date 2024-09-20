# callback
# receive a notification of the algorithm object after each generation

# a posteriori analysis by save_history=True -> stores a deep copy of the algorithm in each iteration
# -> therefore configure callback to receive only reduced data

import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())


problem = get_problem("sphere")

algorithm = GA(pop_size=100)

# res = minimize(problem,
#                algorithm,
#                ('n_gen', 20),
#                seed=1,
#                callback=MyCallback(),
#                verbose=True)

# val = res.algorithm.callback.data["best"]
# plt.plot(np.arange(len(val)), val)
# plt.show()

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               save_history=True)

val = [e.opt.get("F")[0] for e in res.history]
plt.plot(np.arange(len(val)), val)
plt.show()