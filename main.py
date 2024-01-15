import os
from datetime import datetime

from GA2 import get_objective
from optuna import create_study
from optuna.samplers import NSGAIISampler
from random_alg import random_search
from utils import algorithm, Algorithm, Function

if __name__ == "__main__":
    directory = ''
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if algorithm == Algorithm.RANDOM:
        directory = f'Random_res_{now}'
    elif algorithm == Algorithm.GA:
        directory = f'NSGA_res_{now}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    if algorithm == Algorithm.RANDOM:
        for i in range(10):
            function = Function()

            result = random_search(function.iteration)
            with open(f'{directory}/random_res{i}.txt', 'a') as file:
                for res in result:
                    for inp in res.inputs:
                        file.write(str(inp / 1000) + ' ')
                    file.write('\n')
                    for val in res.values:
                        file.write(str(val) + ' ')
                    file.write('\n')

    elif algorithm == Algorithm.GA:
        for i in range(10):
            sampler = NSGAIISampler(population_size=200)
            study = create_study(directions=["maximize", "maximize"], sampler=sampler)
            study.optimize(get_objective(Function().iteration), n_trials=10000)
            with open(f'{directory}/NSGA_res{i}.txt', 'a') as file:
                for t in study.best_trials:
                    file.write(str(list(t.params.values())))
                    file.write('\n')
                    file.write(str(t.values))
                    file.write('\n')
