import ctypes as ct
import itertools
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as cp

from optuna.study._multi_objective import dominates_facade
from utils import fitness_combination, population_size, run_iter_func, n_trials, return_objectives, calc_det


class Trial:
    def __init__(self, exec_time, inputs):
        self.exec_time = exec_time
        self.inputs = inputs


class CircularBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.size = 0

    def add(self, trial):
        if self.size < self.capacity:
            self.buffer.append(trial)
            self.size = self.size + 1
            self.buffer = sorted(self.buffer, key=lambda x: x.exec_time)
        else:
            if self.buffer[0].exec_time < trial.exec_time:
                self.buffer[0] = trial
                self.buffer = sorted(self.buffer, key=lambda x: x.exec_time)

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


def random_search():
    trials = []
    buffer = CircularBuffer(population_size)
    for k in range(n_trials):
        inputs = []
        for _ in range(6):
            rand_num = random.uniform(0, 0.01)
            inputs.append(rand_num)
        for _ in range(6, 15):
            rand_num = random.uniform(0, 3)
            inputs.append(rand_num)

        v0 = run_iter_func(inputs)
        trial = Trial(v0, inputs)
        buffer.add(trial)

    return buffer.get_buffer()
