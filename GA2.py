import ctypes as ct
import datetime
import itertools
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import numpy as cp

from optuna import create_study
from optuna.samplers import NSGAIISampler


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

        self.iteration = mylib.start_collision_detection
        # Define the return type of the C function
        self.iteration.restype = ct.c_long

        # Define arguments of the C function
        self.iteration.argtypes = [
            ct.c_int32,
            ct.POINTER(ct.c_double)
        ]


def get_objective(iter_func):
    def objective(trial):
        inputs = []
        for i in range(6):
            sug_f = trial.suggest_float(f"inputs{i}", 0, 0.01)
            inputs.append(sug_f)
        for i in range(6, 15):
            sug_f = trial.suggest_float(f"inputs{i}", 0, 3)
            inputs.append(sug_f)
        arr = (ct.c_double * 15)(*inputs)

        v0 = iter_func(3, arr)
        v1 = abs(100000 - v0)
        v2 = 0
        return v1, v2

    return objective


total_inputs = CircularBuffer(100)
counter = 0


class Trial:
    def __init__(self, v0, v1, inputs):
        global counter
        self.trial_id = counter
        counter += 1
        self.values = [v0, v1, 0]
        self.inputs = inputs


def scale_actions(lst):
    for i in range(6):
        lst[i] *= 30


def _dominates2(population,
                trial0, trial1
                ) -> bool:
    values0 = trial0.values
    pop_without_0 = [x for x in population if x.trial_id != trial0.trial_id]
    matrix0 = cp.array([scale_actions(list(obj.inputs)) for obj in pop_without_0], dtype=cp.float32)
    values1 = trial1.values
    pop_without_1 = [x for x in population if x.trial_id != trial1.trial_id]
    matrix1 = cp.array([scale_actions(list(obj.inputs)) for obj in pop_without_1], dtype=cp.float32)
    matrix = cp.array([scale_actions(list(obj.inputs)) for obj in population], dtype=cp.float32)

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


def calc_det(matrix):
    transpose = matrix.T
    product = cp.dot(transpose, matrix)
    return cp.linalg.det(product)


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
    print('line 208')
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
    print('line 299')
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
    print('line 241')
    return ret_val


def random_search(iter_func):
    trials = []
    for k in range(2000):
        new_input = []
        inputs = []
        for i in range(6):
            rand_num = random.uniform(0, 0.01)
            inputs.append(rand_num)
        for i in range(6, 15):
            rand_num = random.uniform(0, 3)
            inputs.append(rand_num)
        arr = (ct.c_double * 15)(*inputs)

        v0 = iter_func(3, arr)
        # print(v0)
        v1 = abs(100000 - v0)
        trials.append(Trial(v0, v1, inputs))
        if k != 0 and k % 100 == 0:
            # print(k)
            trials = fast_non_dominated_sort(trials)

    return fast_non_dominated_sort(trials)


def exec_random_search(j):
    population_size = 100
    function = Function()
    fitness_values = CircularBuffer(population_size)

    # function.create_objects()
    result = random_search(function.iteration)
    with open(f'result_random{j}.txt', 'a') as file:
        for i in result:
            for inp in i.inputs:
                file.write(str(inp / 1000) + ' ')
            file.write('\n')
            for val in i.values:
                file.write(str(val) + ' ')
            file.write('\n')


# for i in range(10):
#     exec_random_search(i)


directory = f'NSGA_res_{datetime.datetime.now()}'
if not os.path.exists(directory):
    os.makedirs(directory)

for j in range(10):
    sampler = NSGAIISampler(population_size=200)
    study = create_study(directions=["maximize", "maximize"], sampler=sampler)
    study.optimize(get_objective(Function().iteration), n_trials=10000)
    with open(f'{directory}/result_NSGA{j}.txt', 'a') as file:
        for i in study.best_trials:
            file.write(str(list(i.params.values())))
            file.write('\n')
            file.write(str(i.values))
            file.write('\n')
