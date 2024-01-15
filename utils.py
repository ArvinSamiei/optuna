import ctypes as ct
import multiprocessing
from enum import Enum

import numpy


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
    product = numpy.dot(transpose, matrix)
    return numpy.linalg.det(product)


function = Function()


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
        function.iteration(3, arr)
        v0 = function.iteration(3, arr)
        return v0


def run_iter_func(inputs):
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    consumer = Consumer(tasks, results)
    consumer.start()
    tasks.put(Task(inputs))
    tasks.join()

    result = results.get()
    return result


algorithm = Algorithm.RANDOM
fitness_combination = FitnessCombination.EXEC
population_size = 200
