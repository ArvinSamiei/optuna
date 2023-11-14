import ctypes as ct
import math

import numpy as np
from collections import Counter

from optuna import create_study, visualization
from optuna.samplers import NSGAIISampler


# Reconstruct the structure from the .h file


# Load the compiled library
# mylib = npct.load_library('./libuntitled1.so', './libuntitled1.so')


class CircularBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.current = 0

    def add(self, item):
        self.buffer[self.current] = item
        self.current = (self.current + 1) % self.capacity

    def get_buffer(self):
        res = []
        for i in self.buffer:
            if i is not None:
                res.append(i)
            else:
                break
        return res

    def __iter__(self):
        for item in self.buffer:
            yield item

    def __repr__(self):
        return str(self.buffer)


class Function:
    def __init__(self):
        mylib = ct.CDLL(
            './libuntitled1.so')

        self.iteration = mylib.while_iteration
        # Define the return type of the C function
        self.iteration.restype = ct.c_long

        # Define arguments of the C function
        self.iteration.argtypes = [
            ct.c_int32,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double)
        ]

        self.create_objects = mylib.create_objects


population_size = 100

fitness_values = CircularBuffer(population_size)

function = Function()

function.create_objects()


def objective(trial):
    inputs = [[] for _ in range(2)]
    inputs_scaled = [[] for _ in range(2)]
    constraints_list = []
    for i in range(2):
        for j in range(24):
            sug_f = trial.suggest_float(f"inputs{i}{j}", 0, 0.01)
            inputs_scaled[i].append(sug_f * 10000)
            inputs[i].append(sug_f)
            constraints_list.append((inputs[i][j] >= 0) and (inputs[i][j] <= 1))
    arr1 = (ct.c_double * 24)(*(inputs[0]))
    arr2 = (ct.c_double * 24)(*(inputs[1]))
    trial.set_user_attr("constraint", constraints_list)

    v0 = function.iteration(3, arr1, arr2)
    v1 = abs(100000 - v0)
    v2 = 0
    return v0, v1, v2


def calc_determinant():
    fitness_list = fitness_values.get_buffer()
    if len(fitness_list) == 0:
        return 0
    matrix = np.array(fitness_list)
    transpose = matrix.T
    product = np.dot(transpose, matrix)
    (sign, logabsdet) = np.linalg.slogdet(product)
    return sign * np.exp(logabsdet)


def constraints(trial):
    return trial.user_attrs["constraint"]


sampler = NSGAIISampler(population_size=population_size, constraints_func=constraints)
study = create_study(directions=["maximize", "maximize", "maximize"], sampler=sampler)
study.optimize(objective, n_trials=1000)

fig = visualization.plot_pareto_front(study)
fig.show()

print(study.best_trials)
