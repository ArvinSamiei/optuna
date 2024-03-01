import os
from datetime import datetime

import matplotlib.pyplot as plt

import utils
from GA2 import get_objective
from optuna import create_study
from optuna.samplers import NSGAIISampler
from pure_random import random_search
from utils import algorithm, Algorithm, Function, population_size, n_trials, get_num_objectives

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
            study.optimize(get_objective(), n_trials=n_trials)
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

            if utils.fitness_combination == utils.FitnessCombination.EXEC_DIV:
                y = utils.PopulationStore().max_div
                x = list(range(0, len(y)))
                plt.plot(x, y)
                plt.title('diversities')
                plt.savefig(f'{directory}/div_plot{i}.pdf')
                plt.clf()
                plt.close()

            points_set = utils.PopulationStore().points_covered_set

            set_no = 0
            for points in points_set:
                x = []
                y = []
                z = []
                for p in points:
                    x.append(p[0])
                    y.append(p[1])
                    z.append(p[2])

                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.grid()

                ax.scatter(x, y, z, c='r')
                ax.set_title(f'3D Scatter Plot for set {i}')

                # Set axes label
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.savefig(f'{directory}/points_plot_{i}_set{set_no}.pdf')
                plt.clf()
                plt.close()

                set_no += 1
            utils.PopulationStore().reset()
