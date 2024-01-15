import ctypes as ct

from utils import run_iter_func


def get_objective(iter_func):
    def objective(trial):
        inputs = []
        for i in range(6):
            sug_f = trial.suggest_float(f"inputs{i}", 0, 0.01)
            inputs.append(sug_f)
        for i in range(6, 15):
            sug_f = trial.suggest_float(f"inputs{i}", 0, 3)
            inputs.append(sug_f)

        v0 = run_iter_func(inputs)
        v1 = abs(100000 - v0)
        v2 = 0
        return v1, v2

    return objective
