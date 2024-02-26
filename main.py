import os
from datetime import datetime

import matplotlib.pyplot as plt

import utils
from GA2 import get_objective
from optuna import create_study
from optuna.samplers import NSGAIISampler
from pure_random import random_search
from utils import algorithm, Algorithm, Function, population_size, n_trials, get_num_objectives

population_store = []

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
            result = random_search()
            with open(f'{directory}/random_res{i}.txt', 'a') as file:
                for res in result:
                    file.write(str(res.inputs()))
                    file.write('\n')
                    file.write(str(res.exec_time))
                    file.write('\n')

    elif algorithm == Algorithm.GA:
        for i in range(10):
            sampler = NSGAIISampler(population_size=population_size)
            study = create_study(directions=['maximize'] * get_num_objectives(), sampler=sampler)
            study.optimize(get_objective(Function().iteration), n_trials=n_trials)
            with open(f'{directory}/NSGA_res{i}.txt', 'a') as file:
                for t in study.best_trials:
                    file.write(str(list(t.params.values())))
                    file.write('\n')
                    file.write(str(t.values))
                    file.write('\n')

                file.write('population is: \n')
                for t in utils.PopulationStore().get_population():
                    file.write(str(list(t.params.values())))
                    file.write('\n')
                    file.write(str(t.values))
                    file.write('\n')

            y = utils.PopulationStore().max_exec
            x = list(range(0, len(y)))
            plt.plot(x, y)
            plt.title('exec times')
            plt.savefig(f'{directory}/exec_plot{i}.pdf')
            plt.clf()
            utils.PopulationStore().max_exec = []

            if utils.fitness_combination == utils.FitnessCombination.EXEC_DIV:
                y = utils.PopulationStore().max_div
                x = list(range(0, len(y)))
                plt.plot(x, y)
                plt.title('diversities')
                plt.savefig(f'{directory}/div_plot{i}.pdf')
                plt.clf()
                utils.PopulationStore().max_div = []
