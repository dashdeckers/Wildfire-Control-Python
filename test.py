import timeit

setup = """
import numpy as np
"""

code = """
m = np.zeros((10000, 10000))
m.fill(3.14)
"""

num_runs = 100
total = timeit.timeit(setup=setup, stmt=code, number=num_runs)
r1, r2 = round(total, 4), round(total / num_runs, 4)
print("Total:", r1, "Average per run", r2)