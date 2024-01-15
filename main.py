import os
from datetime import datetime

from GA2 import get_objective
from optuna import create_study
from optuna.samplers import NSGAIISampler
from random_alg import random_search
from utils import algorithm, Algorithm, Function

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
        with open(f'{directory}/random_res{j}.txt', 'a') as file:
            for i in result:
                for inp in i.inputs:
                    file.write(str(inp / 1000) + ' ')
                file.write('\n')
                for val in i.values:
                    file.write(str(val) + ' ')
                file.write('\n')

elif algorithm == Algorithm.GA:
    for j in range(10):
        sampler = NSGAIISampler(population_size=200)
        study = create_study(directions=["maximize", "maximize"], sampler=sampler)
        study.optimize(get_objective(Function().iteration), n_trials=10000)
        with open(f'{directory}/NSGA_res{j}.txt', 'a') as file:
            for i in study.best_trials:
                file.write(str(list(i.params.values())))
                file.write('\n')
                file.write(str(i.values))
                file.write('\n')
