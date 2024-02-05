import ctypes as ct
import multiprocessing
import numpy as np
from enum import Enum
from multiprocessing import Process


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


def execute_c_code(inputs, results_q):
    arr = (ct.c_double * 15)(*inputs)
    function = Function()
    exec_times = []
    for i in range(10):
        exec_time = function.iteration(3, arr)
        exec_times.append(exec_time)
    p75, p25 = np.percentile(exec_times, [75, 25])
    np_arr_times = np.array(exec_times)
    np_arr_times = np_arr_times[np_arr_times < p75]
    np_arr_times = np_arr_times[np_arr_times > p25]
    results_q.put(np.mean(np_arr_times))


def run_iter_func(inputs):
    results = multiprocessing.Queue()
    p = Process(target=execute_c_code, args=(inputs, results))
    p.start()
    p.join()

    result = results.get()
    # arr = (ct.c_double * 15)(*inputs)
    # function = Function()
    # function.iteration(3, arr)
    # result = function.iteration(3, arr)
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


algorithm = Algorithm.GA
fitness_combination = FitnessCombination.EXEC_DIV
population_size = 200
n_trials = 20000


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
    def set_population(self, population):
        if len(population) > 0:
            self.population = population.copy()

    def get_population(self):
        return self.population
