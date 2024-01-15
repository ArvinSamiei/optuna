import ctypes as ct
from enum import Enum


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
    product = cp.dot(transpose, matrix)
    return cp.linalg.det(product)


algorithm = Algorithm.RANDOM
fitness_combination = FitnessCombination.EXEC
population_size = 200
counter = 0