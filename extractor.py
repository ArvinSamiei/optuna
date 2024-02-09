import ast
from statistics import mean, stdev


def NSGA_exec(i):
    exec_times = []
    with open(f'/home/arvins/Desktop/results_wo_outliers/GA_exec/4/NSGA_res{i}.txt', 'r') as file:
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
    with open(f'/home/arvins/Desktop/results_wo_outliers/random_exec/2/random_res{i}.txt', 'r') as file:
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
    with open(f'/home/arvins/Desktop/results_wo_outliers/GA_exec_div/2/NSGA_res{i}.txt', 'r') as file:
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
    with open(f'/home/arvins/Desktop/results_wo_outliers/random_exec/2/random_res{i}.txt', 'r') as file:
        lines = file.readlines()
        counter = 1
        rand_exec_times = []
        for line in lines:
            if counter % 2 == 0:
                rand_exec_times.append(float(line.strip()))
            counter += 1

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

    return mean(exec_times), mean(diversities)

#
# a = []
# b = []
# for i in range(0, 10):
#     values = NSGA_exec_div(i)
#     a.append(values[0])
#     b.append(values[1])
# print(mean(a))
# print(stdev(a))
# print(mean(b))
# print(stdev(b))
