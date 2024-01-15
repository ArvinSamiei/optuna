import ctypes as ct


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


function = Function()
inputs = [0.00963025500894449, 0.006603285330547511, 0.004025194061833527, 0.003342589993477564, 0.007381862448325963, 0.0074141419546683365, 1.1308307129490813, 2.2624077821477355, 2.909106722289755, 1.052591306761617, 2.170251619492708, 1.8850048228839018, 0.7363520731648506, 2.5210123489730534, 2.201172534904283]
arr = (ct.c_double * 15)(*inputs)

res = []
for i in range(10):
    exec_time = function.iteration(3, arr)
    res.append(exec_time)

print(res)
