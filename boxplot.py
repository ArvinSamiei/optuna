import matplotlib.pyplot as plt
import numpy as np

from extractor import NSGA_exec_div


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


save_box_plot()
