import matplotlib.pyplot as plt
import numpy as np

from extractor import NSGA_exec_div, extract_rand_exec_times, NSGA_exec


def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    filtered_data = [x for x in data if x >= lower_bound and x <= upper_bound]

    return filtered_data


def make_boxplot_exec():
    ga_data = []
    rand_data = []
    ga_div_data = []

    for i in range(10):
        rand_data += extract_rand_exec_times(i)
        ga_data += NSGA_exec(i)
        ga_div_data += NSGA_exec_div(i)[0]

    scale = 0.001
    rand_data = np.array(rand_data) * scale
    ga_data = np.array(ga_data) * scale
    ga_div_data = np.array(ga_div_data) * scale

    data = [rand_data, ga_data, ga_div_data]
    draw_plots(data, 'fitness_exec.pdf')


def make_boxplot_exec_div():
    ga_data = []
    rand_data = []
    for i in range(10):
        rand_data += extract_rand_exec_times(i)

    for i in range(10):
        ga_data += NSGA_exec_div(i)[0]

    data = [rand_data, ga_data]
    draw_plots(data, 'fitness_exec_div.pdf')


def draw_plots(data, file_name):
    plt.figure(figsize=(10, 7))
    bp = plt.boxplot(data, patch_artist=True, showfliers=False)

    colors = ['blue', 'red', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (µs)')
    plt.xticks([1, 2, 3], ['Rand', 'GA Exec', 'GA Exec Div'])

    plt.savefig(file_name)
    plt.clf()


def save_box_plot():
    for i in range(3):
        values = NSGA_exec_div(i)

        exec_Q3, exec_Q1 = np.percentile(values[0], [75, 25])
        div_Q3, div_Q1 = np.percentile(values[1], [75, 25])
        exec_IQR = exec_Q3 - exec_Q1
        div_IQR = div_Q3 - div_Q1

        draw_plot(exec_IQR, exec_Q1, exec_Q3, f'plots/ga_exec_div/exec{i}.png', i, values[0])
        draw_plot(div_IQR, div_Q1, div_Q3, f'plots/ga_exec_div/div{i}.png', i, values[1])


def draw_plot(IQR, Q1, Q3, file_name, i, values):
    plt.boxplot(values, showfliers=False)
    plt.title(f'boxplot for {file_name}')
    plt.ylabel('Values')
    plt.ylim(Q1 - (1.5 * IQR), Q3 + (1.5 * IQR))
    plt.savefig(file_name)
    plt.clf()


make_boxplot_exec()
