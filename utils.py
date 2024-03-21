import ctypes as ct
import multiprocessing
from enum import Enum
from multiprocessing import Process

import numpy as np
from scipy.stats import qmc

import utils


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


class FitnessCombination(Enum):
    EXEC = 1,
    JIT = 2,
    DIV = 3
    EXEC_DIV = 4,
    JIT_DIV = 5,
    EXEC_JIT_DIV = 6


class Algorithm(Enum):
    RANDOM = 1,
    GA = 2


class CaseStudy(Enum):
    First = 1,
    DOF6 = 2


def calc_det(matrix):
    transpose = matrix.T
    product = np.dot(transpose, matrix)
    sign, Ldet = np.linalg.slogdet(product)
    return Ldet


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        next_task = self.task_queue.get()
        answer = next_task()
        self.task_queue.task_done()
        self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, inputs):
        self.inputs = inputs

    def __call__(self):
        arr = (ct.c_double * 15)(*self.inputs)
        function = Function()
        function.iteration(3, arr)
        v0 = function.iteration(3, arr)
        return v0


def execute_c_code(total_inputs, results_q):
    for inputs in total_inputs:
        arr = (ct.c_double * 15)(*inputs)
        function = Function()
        exec_times = []
        for i in range(10):
            exec_time = function.iteration(3, arr)
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


def run_iter_func(total_inputs):
    results = multiprocessing.Queue()
    p = Process(target=execute_c_code, args=(total_inputs, results))
    p.start()
    p.join()

    exec_times = []
    while not results.empty():
        result = results.get()
        exec_times.append(result)
    # arr = (ct.c_double * 15)(*inputs)
    # function = Function()
    # function.iteration(3, arr)
    # result = function.iteration(3, arr)
    return exec_times


def single_run_iter_func(inputs):
    results = multiprocessing.Queue()
    p = Process(target=execute_c_code, args=(inputs, results))
    p.start()
    p.join()

    result = results.get()
    return result


def return_objectives(v0, v1, v2):
    if fitness_combination == FitnessCombination.EXEC:
        return [v0]
    elif fitness_combination == FitnessCombination.DIV:
        return [v2]
    elif fitness_combination == FitnessCombination.JIT:
        return [v1]
    elif fitness_combination == FitnessCombination.EXEC_DIV:
        return [v0, v2]
    elif fitness_combination == FitnessCombination.JIT_DIV:
        return [v1, v2]
    elif fitness_combination == FitnessCombination.EXEC_JIT_DIV:
        return [v0, v1, v2]


def get_num_objectives():
    if fitness_combination == FitnessCombination.EXEC or fitness_combination == FitnessCombination.JIT or fitness_combination == FitnessCombination.DIV:
        return 1
    elif fitness_combination == FitnessCombination.EXEC_DIV or fitness_combination == FitnessCombination.JIT_DIV:
        return 2
    elif fitness_combination == FitnessCombination.EXEC_JIT_DIV:
        return 3


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class PopulationStore(metaclass=SingletonMeta):
    def __init__(self, no_sets=0):
        self.population = []
        self.max_exec = []
        self.max_div = []

    def set_population(self, population):
        if len(population) > 0:
            self.population = population.copy()

    def set_max_exec(self, max_exec):
        self.max_exec.append(max_exec)

    def set_max_div(self, max_div):
        self.max_div.append(max_div)

    def get_population(self):
        return self.population

    def reset(self):
        self.population = []
        self.max_exec = []
        self.max_div = []


def calc_diversity(population, trial_id):
    if utils.diversity_mode == DiversityMode.MATRIX:
        matrix = np.array([scale_motions(list(obj.params.values())) for obj in population], dtype=np.float32)

        trial_popped = [x for x in population if x._trial_id != trial_id]
        matrix_wo_trial = np.array([scale_motions(list(obj.params.values())) for obj in trial_popped], dtype=np.float32)

        det = calc_det(matrix)
        det_wo_trial = calc_det(matrix_wo_trial)
        return abs(det) - abs(det_wo_trial)
    elif utils.diversity_mode == DiversityMode.DISCREPANCY:
        all = []
        all_wo_trial = []
        for obj in population:
            all.append(list(obj.params.values()))
            if obj._trial_id != trial_id:
                all_wo_trial.append(list(obj.params.values()))

        discrepancy_all = qmc.discrepancy(np.array(all), iterative=True)
        discrepancy_wo_trial = qmc.discrepancy(np.array(all_wo_trial), iterative=True)

        return discrepancy_wo_trial - discrepancy_all


def scale_motions(lst):
    return [x * 30 for x in lst[:6]] + lst[6:]


algorithm = Algorithm.GA
fitness_combination = FitnessCombination.EXEC_DIV
population_size = 100
n_trials = 100000
GA_rand_ratio = 0.2
case_study = CaseStudy.DOF6


class DiversityMode(Enum):
    DISCREPANCY = 1,
    MATRIX = 2


diversity_mode = DiversityMode.DISCREPANCY
