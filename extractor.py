import ast
from statistics import mean, stdev

GA_exec_dir_num = 6
GA_div_exec_dir_num = 5
rand_dir_num = 2
num_iterations = 10


def NSGA_exec(i):
    exec_times = []
    with open(f'/home/arvins/Desktop/results_wo_outliers/GA_exec/{GA_exec_dir_num}/NSGA_res{i}.txt', 'r') as file:
        lines = file.readlines()
        counter = 1
        for j in range(3, len(lines)):
            if ',' not in lines[j]:
                values = ast.literal_eval(lines[j].strip())
                exec_times.append(values[0])
            counter += 1

    rand_exec_times = extract_rand_exec_times(i)

    exec_times.sort(reverse=True)

    print(exec_times)
    print(f'Max for NSGA is: {max(exec_times):,}')
    print(f'Average for NSGA is: {mean(exec_times):,}')
    print(f'Max for random algorithm is: {max(rand_exec_times):,}')
    print(f'Average for random algorithm is: {mean(rand_exec_times):,}')

    print('---------------------------------------------')
    print('\n\n\n\n\n')
    return exec_times


def extract_rand_exec_times(i):
    with open(f'/home/arvins/Desktop/results_wo_outliers/random_exec/{rand_dir_num}/random_res{i}.txt', 'r') as file:
        lines = file.readlines()
        counter = 1
        rand_exec_times = []
        for line in lines:
            if counter % 2 == 0:
                rand_exec_times.append(float(line.strip()))
            counter += 1
    return rand_exec_times


def NSGA_exec_div(i):
    exec_times = []
    diversities = []
    with open(f'/home/arvins/Desktop/results_wo_outliers/GA_exec_div/{GA_div_exec_dir_num}/NSGA_res{i}.txt',
              'r') as file:
        lines = file.readlines()
        for j in range(0, len(lines)):
            if 'population' in lines[j]:
                break
        counter = 1
        for k in range(j + 1, len(lines)):
            line = lines[k]
            if line.count(',') == 1:
                values = ast.literal_eval(line.strip())
                exec_times.append(values[0])
                diversities.append(values[1])
            counter += 1

    rand_exec_times = extract_rand_exec_times(i)

    min_size = min(len(exec_times), len(rand_exec_times))

    exec_times.sort(reverse=True)
    exec_times = exec_times[:min_size]
    diversities = diversities[:min_size]
    rand_exec_times = rand_exec_times[:min_size]

    print(exec_times)
    print(f'average execution time for NSGA is: {mean(exec_times)}')
    print(f'STD of execution time for NSGA is: {stdev(exec_times)}')
    print(f'average execution time for random algorithm is: {mean(rand_exec_times)}')

    print(f'average execution ')

    return exec_times, diversities


latex_code = r'''
\begin{table}[h]
\centering
\caption{Comparative Performance Analysis across Algorithms and Fitness Functions}
\label{table:comprehensive_analysis}
\begin{tabular}{|c|c|c|c|c|c|c|c|c|}
\hline
\rowcolor{gray}
\multirow{3}{*}{Iteration} & \multicolumn{4}{c|}{\textbf{Rand Alg}} & \multicolumn{4}{c|}{\textbf{Genetic Alg}} \\
\cline{2-9} 
\rowcolor{gray}
\textbf{Iteration} & \multicolumn{2}{c|}{\textbf{Exe (ns)}} & \multicolumn{2}{c|}{\textbf{Exe \& Div (ns)}} & \multicolumn{2}{c|}{\textbf{Exe (ns)}} & \multicolumn{2}{c|}{\textbf{Exe \& Div (ns)}} \\
\cline{2-9} 
\rowcolor{gray}
 & \textbf{Max} & \textbf{Avg} & \textbf{Max} & \textbf{Avg} & \textbf{Max} & \textbf{Avg} & \textbf{Max} & \textbf{Avg} \\
\hline
'''

GA_exec_res = []
GA_div_exec_res = []
rand_exec_res = []
rand_div_exec_res = []
for i in range(num_iterations):
    GA_exec_res = NSGA_exec(i)
    GA_div_exec_res = NSGA_exec_div(i)
    rand_exec_res = extract_rand_exec_times(i)
    rand_max = max(rand_exec_res) / 1000
    rand_mean = mean(rand_exec_res) / 1000
    GA_e_max = max(GA_exec_res) / 1000
    GA_e_mean = mean(GA_exec_res) / 1000
    GA_de_max = max(GA_div_exec_res[0]) / 1000
    GA_de_mean = mean(GA_div_exec_res[0]) / 1000
    latex_code += f"{i} & {rand_max: .2f} & {rand_mean: .2f} & - & - & {GA_e_max: .2f} & {GA_e_mean: .2f} & {GA_de_max: .2f} & {GA_de_mean: .2f} \\\\\n"

latex_code += r'''
\hline
\multicolumn{9}{|l|}{\cellcolor{gray}\textbf{Note:} Exe = Execution Time, Div = Diversity} \\
\hline
\end{tabular}
\end{table}
'''

print(latex_code)
