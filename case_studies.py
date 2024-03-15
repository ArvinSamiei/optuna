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
            ct.c_int32,
            ct.POINTER(ct.c_double)
        ]

    def get_objective(self, run_iter_func):
        def objective(trials):
            total_inputs = []
            for trial in trials:
                inputs = []
                for i in range(6):
                    sug_f = trial.suggest_float(f"degree{i}", 0, 0.01)
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
        return inputs


cases_facade = CasesFacade()
