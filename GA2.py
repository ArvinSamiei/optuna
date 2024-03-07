import ctypes as ct

from utils import run_iter_func, fitness_combination, FitnessCombination, return_objectives


def get_objective():
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
