import ctypes as ct
import multiprocessing
import random
from multiprocessing import Process

import numpy as np

import utils
from utils import return_objectives


class CasesFacade:
    def __init__(self):
        if utils.case_study == utils.CaseStudy.First:
            self.case = InitCase()
        elif utils.case_study == utils.CaseStudy.DOF6:
            self.case = DOF6()
        self.execute_c_code = self.case.execute_c_code

    def get_objective(self):
        return self.case.get_objective(self.run_iter_func)

    def run_iter_func(self, total_inputs):
        results = multiprocessing.Queue()
        p = Process(target=self.execute_c_code, args=(total_inputs, results))
        p.start()
        p.join()

        exec_times = []
        while not results.empty():
            result = results.get()
            exec_times.append(result)
        return exec_times

    def create_random_nums(self):
        return self.case.create_random_nums()

    def add_points_of_population(self, population):
        self.case.add_points_of_population(population)

    def get_points(self):
        return self.case.points_covered_set

    def reset(self):
        self.case.points_covered_set = [[] for _ in range(self.case.no_sets)]


class InitCase:
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
        self.no_sets = 3
        self.points_covered_set = [[] for _ in range(self.no_sets)]

    def get_objective(self, run_iter_func):
        def objective(trials):
            total_inputs = []
            for trial in trials:
                inputs = []
                for i in range(6):
                    sug_f = trial.suggest_float(f"inputs{i}", 0, 0.01)
                    inputs.append(sug_f)
                for i in range(6, 15):
                    sug_f = trial.suggest_float(f"inputs{i}", 0, 3)
                    inputs.append(sug_f)

                total_inputs.append(inputs)

            execution_times = run_iter_func(total_inputs)

            total_objectives = []
            for exec_time in execution_times:
                v0 = exec_time
                v1 = abs(100000 - v0)
                v2 = 0
                total_objectives.append(return_objectives(v0, v1, v2))

            return total_objectives

        return objective

    def add_points_of_population(self, population):
        for trial in population:
            inputs = list(trial.params.values())
            self.points_covered_set[0].append(inputs[6:9])
            self.points_covered_set[1].append(inputs[9:12])
            self.points_covered_set[2].append(inputs[12:15])

    def execute_c_code(self, total_inputs, results_q):
        for inputs in total_inputs:
            arr = (ct.c_double * 15)(*inputs)
            exec_times = []
            for i in range(50):
                exec_time = self.iteration(3, arr)
                if exec_time <= 0:
                    results_q.put(-1)
                    break
                exec_times.append(exec_time)
            if len(exec_times) == 0:
                continue
            p75, p25 = np.percentile(exec_times, [75, 25])
            np_arr_times = np.array(exec_times)
            np_arr_times = np_arr_times[np_arr_times < p75]
            np_arr_times = np_arr_times[np_arr_times > p25]
            results_q.put(np.mean(np_arr_times))

    def create_random_nums(self):
        inputs = {}
        key = 'inputs'
        for j in range(6):
            rand_num = random.uniform(0, 0.01)
            inputs[f'{key}{j}'] = rand_num
        for j in range(6, 15):
            rand_num = random.uniform(0, 3)
            inputs[f'{key}{j}'] = rand_num

        return inputs


class DOF6:
    def __init__(self):
        mylib = ct.CDLL(
            './lib6dof.so')

        self.iteration = mylib.start_collision_detection
        # Define the return type of the C function
        self.iteration.restype = ct.c_long

        # Define arguments of the C function
        self.iteration.argtypes = [
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.c_bool
        ]

        self.no_sets = 2
        self.points_covered_set = [[] for _ in range(self.no_sets)]
        self.limits = [[-2.9671, 2.9671],
                       [-1.7453, 2.3562],
                       [-2.0769, 2.8972],
                       [-3.3161, 3.3161],
                       [-2.0944, 2.0944],
                       [-6.2832, 6.2832]]

    def get_objective(self, run_iter_func):
        def objective(trials):
            total_inputs = []
            for trial in trials:
                inputs = []
                for i in range(6):
                    sug_f = trial.suggest_float(f"degree{i}", self.limits[i][0], self.limits[i][1])
                    inputs.append(sug_f)

                total_inputs.append(inputs)

            execution_times = run_iter_func(total_inputs)

            total_objectives = []
            for exec_time in execution_times:
                v0 = exec_time
                v1 = abs(100000 - v0)
                v2 = 0
                total_objectives.append(return_objectives(v0, v1, v2))

            return total_objectives

        return objective

    def execute_c_code(self, total_inputs, results_q):
        for inputs in total_inputs:
            arr = (ct.c_double * 6)(*inputs)
            arr2 = (ct.c_double * 6)(*inputs)
            exec_times = []
            for i in range(10):
                exec_time = self.iteration(arr, arr2, ct.c_bool(False))
                if exec_time <= 0:
                    exec_time *= -1
                exec_times.append(exec_time)
            if len(exec_times) == 0:
                continue
            p75, p25 = np.percentile(exec_times, [75, 25])
            np_arr_times = np.array(exec_times)
            np_arr_times = np_arr_times[np_arr_times < p75]
            np_arr_times = np_arr_times[np_arr_times > p25]
            results_q.put(np.mean(np_arr_times))

    def create_random_nums(self):
        inputs = {}
        key = 'degree'
        for j in range(6):
            rand_num = random.uniform(0, 0.01)
            inputs[f'{key}{j}'] = rand_num
        return inputs

    def add_points_of_population(self, population):
        for trial in population:
            inputs = list(trial.params.values())
            self.points_covered_set[0].append(inputs[:3])
            self.points_covered_set[1].append(inputs[3:])


cases_facade = CasesFacade()
