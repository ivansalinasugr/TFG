from hyperopt import tpe, fmin, hp

objective = lambda x: (x-3)**2 + 2

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = objective(x)

fig = plt.figure()
plt.plot(x, y)
plt.show()

# Define the search space of x between -10 and 10.
space = hp.uniform('x', -10, 10)

best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm
    max_evals=1000 # Number of optimization attempts
)
print(best)
