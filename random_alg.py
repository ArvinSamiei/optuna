import itertools
from utils import fitness_combination, population_size, n_trials, return_objectives, calc_det, \
    single_run_iter_func

counter = 0


class Trial:
    def __init__(self, v0, v1, inputs):
        global counter
        self._trial_id = counter
        counter += 1
        self.values = return_objectives(v0, v1, 0)
        self.inputs = inputs
        self.params = {}
        for i in range(len(inputs)):
            self.params[str(i)] = inputs[i]


def scale_actions(lst):
    for i in range(6):
        lst[i] = lst[i] * 30
    return lst


def _dominates2(population,
                trial0, trial1
                ) -> bool:
    values0 = trial0.values
    pop_without_0 = [x for x in population if x._trial_id != trial0._trial_id]
    matrix0 = cp.array([scale_actions(list(obj.inputs)) for obj in pop_without_0], dtype=cp.float32)

    values1 = trial1.values
    pop_without_1 = [x for x in population if x._trial_id != trial1._trial_id]
    matrix1 = cp.array([scale_actions(list(obj.inputs)) for obj in pop_without_1], dtype=cp.float32)
    matrix = cp.array([scale_actions(list(obj.inputs)) for obj in population], dtype=cp.float32)

    det = calc_det(matrix)

    det0 = calc_det(matrix0)

    det1 = calc_det(matrix1)

    values0[2] = abs(det) - abs(det0)
    values1[2] = abs(det) - abs(det1)

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    return values0[1] <= values1[1] and values0[2] <= values1[2]


def check_dominance(population, pair):
    p, q = pair
    if dominates_facade(population, p, q, None, fitness_combination):
        return ('p_dominates', p._trial_id, q._trial_id)
    elif dominates_facade(population, q, p, None, fitness_combination):
        return ('q_dominates', q._trial_id, p._trial_id)
    return ('none', None, None)


def fast_non_dominated_sort(
        population
):
    dominated_count = defaultdict(int)
    dominates_list = defaultdict(list)

    results = []
    for p, q in itertools.combinations(population, 2):
        if dominates_facade(population, p, q, None, fitness_combination):
            results.append(('p_dominates', p._trial_id, q._trial_id))
        elif dominates_facade(population, q, p, None, fitness_combination):
            results.append(('q_dominates', p._trial_id, q._trial_id))

    for result in results:
        status, winner_id, loser_id = result
        if status == 'p_dominates':
            dominates_list[winner_id].append(loser_id)
            dominated_count[loser_id] += 1
        elif status == 'q_dominates':
            dominates_list[winner_id].append(loser_id)
            dominated_count[loser_id] += 1

    population_per_rank = []
    print('line 208')
    while population:
        non_dominated_population = []
        i = 0
        while i < len(population):
            if dominated_count[population[i]._trial_id] == 0:
                individual = population[i]
                if i == len(population) - 1:
                    population.pop()
                else:
                    population[i] = population.pop()
                non_dominated_population.append(individual)
            else:
                i += 1

        for x in non_dominated_population:
            for y in dominates_list[x._trial_id]:
                dominated_count[y] -= 1

        assert non_dominated_population
        population_per_rank.append(non_dominated_population)
    print('line 299')
    res = []
    ret_val = []
    for lst in population_per_rank:
        res.append([])
        for el in lst:
            ret_val.append(el)
            res[-1].append(el)
            if len(ret_val) >= population_size:
                break
        if len(ret_val) >= population_size:
            break
    print('line 241')
    return ret_val


def random_search():
    trials = []
    for k in range(n_trials):
        inputs = []
        for _ in range(6):
            rand_num = random.uniform(0, 0.01)
            inputs.append(rand_num)
        for _ in range(6, 15):
            rand_num = random.uniform(0, 3)
            inputs.append(rand_num)

        v0 = single_run_iter_func(inputs)
        v1 = abs(100000 - v0)
        trials.append(Trial(v0, v1, inputs))
        if k != 0 and k % population_size == 0:
            trials = fast_non_dominated_sort(trials)

    return fast_non_dominated_sort(trials)
