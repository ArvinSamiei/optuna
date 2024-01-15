import ctypes as ct


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

        iter_func(3, arr)
        v0 = iter_func(3, arr)
        v1 = abs(100000 - v0)
        v2 = 0
        return v1, v2

    return objective
