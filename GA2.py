import ctypes as ct
import itertools
import math
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cupy as cp
from collections import Counter, defaultdict

from optuna import create_study, visualization
from optuna.samplers import NSGAIISampler
from numba import njit

import torch


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


print(torch.cuda.is_available())
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
    matrix = cp.array(fitness_list)
    transpose = matrix.T
    product = cp.dot(matrix, transpose)
    cp.linalg.det(product)


def constraints(trial):
    return trial.user_attrs["constraint"]


# sampler = NSGAIISampler(population_size=population_size, constraints_func=constraints)
# study = create_study(directions=["maximize", "maximize", "maximize"], sampler=sampler)
# study.optimize(objective, n_trials=1000)
#
# fig = visualization.plot_pareto_front(study)
# fig.show()
#
# print(study.best_trials)

total_inputs = CircularBuffer(population_size)
counter = 0


class Trial:
    def __init__(self, v0, v1, inputs):
        global counter
        self.trial_id = counter
        counter += 1
        self.values = [v0, v1, 0]
        self.inputs = inputs


def _dominates2(population,
                trial0, trial1
                ) -> bool:
    values0 = trial0.values
    pop_without_0 = [x for x in population if x.trial_id != trial0.trial_id]
    matrix0 = cp.array([list(obj.inputs) for obj in pop_without_0], dtype=cp.float32)
    values1 = trial1.values
    pop_without_1 = [x for x in population if x.trial_id != trial1.trial_id]
    matrix1 = cp.array([list(obj.inputs) for obj in pop_without_1], dtype=cp.float32)
    matrix = cp.array([list(obj.inputs) for obj in population], dtype=cp.float32)

    griddim = 10, 20
    blockdim = 3, 4

    det = calc_det(matrix)

    det0 = calc_det(matrix0)

    det1 = calc_det(matrix1)

    values0[2] = abs(det) - abs(det0)
    values1[2] = abs(det) - abs(det1)

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    return all(v0 <= v1 for v0, v1 in zip(values0, values1))


@njit
def calc_det(matrix):
    transpose = matrix.T
    product = cp.dot(transpose, matrix)
    (sign, logabsdet) = cp.linalg.slogdet(product)
    det = sign * cp.exp(logabsdet)
    return det


res = None


def check_dominance(population, pair):
    p, q = pair
    if _dominates2(population, p, q):
        return ('p_dominates', p.trial_id, q.trial_id)
    elif _dominates2(population, q, p):
        return ('q_dominates', q.trial_id, p.trial_id)
    return ('none', None, None)


def fast_non_dominated_sort(
        population
):
    global res
    dominated_count = defaultdict(int)
    dominates_list = defaultdict(list)

    pairs = list(itertools.combinations(population, 2))
    partial_check_dominance = partial(check_dominance, population)
    with ProcessPoolExecutor() as executor:
        results = executor.map(partial_check_dominance, pairs)

    for result in results:
        status, winner_id, loser_id = result
        if status == 'p_dominates':
            dominates_list[winner_id].append(loser_id)
            dominated_count[loser_id] += 1
        elif status == 'q_dominates':
            dominates_list[winner_id].append(loser_id)
            dominated_count[loser_id] += 1

    population_per_rank = []
    while population:
        non_dominated_population = []
        i = 0
        while i < len(population):
            if dominated_count[population[i].trial_id] == 0:
                individual = population[i]
                if i == len(population) - 1:
                    population.pop()
                else:
                    population[i] = population.pop()
                non_dominated_population.append(individual)
            else:
                i += 1

        for x in non_dominated_population:
            for y in dominates_list[x.trial_id]:
                dominated_count[y] -= 1

        assert non_dominated_population
        population_per_rank.append(non_dominated_population)

    res = []
    ret_val = []
    for lst in population_per_rank:
        res.append([])
        for el in lst:
            ret_val.append(el)
            res[-1].append(el)
            if len(ret_val) >= 100:
                break
        if len(ret_val) >= 100:
            break
    return ret_val


trials = []


def random_search():
    global trials
    for k in range(2000):
        new_input = []
        scaled_inputs = []
        for i in range(2):
            new_input.append([])
            for j in range(24):
                sug_f = random.uniform(0, 0.01)

                new_input[i].append(sug_f)
                scaled_inputs.append(sug_f * 1000)

        arr1 = (ct.c_double * 24)(*(new_input[0]))
        arr2 = (ct.c_double * 24)(*(new_input[1]))

        v0 = function.iteration(3, arr1, arr2)
        v1 = abs(100000 - v0)
        trials.append(Trial(v0, v1, scaled_inputs))
        if k != 0 and k % 100 == 0:
            trials = fast_non_dominated_sort(trials)

    return fast_non_dominated_sort(trials)


for j in range(10):
    with open(f'result{j}.txt', 'a') as file:
        for i in random_search():
            for inp in i.inputs:
                file.write(str(inp / 1000) + ' ')
            file.write('\n')
            for val in i.values:
                file.write(str(val) + ' ')
            file.write('\n')
