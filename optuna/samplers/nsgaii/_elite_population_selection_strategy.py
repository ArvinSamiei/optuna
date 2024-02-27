from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import random

import optuna
import utils
from GA2 import get_objective
from optuna.samplers.nsgaii._dominates import _constrained_dominates
from optuna.samplers.nsgaii._dominates import _validate_constraints
from optuna.study import Study
from optuna.study._multi_objective import _dominates, dominates_facade
from optuna.trial import FrozenTrial
from utils import fitness_combination


class NSGAIIElitePopulationSelectionStrategy:
    def __init__(
        self,
        *,
        population_size: int,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        self._population_size = population_size
        self._constraints_func = constraints_func

    def __call__(self, study: Study, population: list[FrozenTrial]) -> list[FrozenTrial]:
        """Select elite population from the given trials by NSGA-II algorithm.

        Args:
            study:
                Target study object.
            population:
                Trials in the study.

        Returns:
            A list of trials that are selected as elite population.
        """
        _validate_constraints(population, self._constraints_func)
        dominates = _dominates if self._constraints_func is None else _constrained_dominates
        population_per_rank = _fast_non_dominated_sort(population, study.directions, dominates)

        elite_population: list[FrozenTrial] = []
        for individuals in population_per_rank:
            if len(elite_population) + len(individuals) < self._population_size:
                elite_population.extend(individuals)
            else:
                n = self._population_size - len(elite_population)
                _crowding_distance_sort(individuals)
                elite_population.extend(individuals[:n])
                break


        max_exec = max(t.values[0] for t in study.best_trials)
        utils.PopulationStore().set_max_exec(max_exec)
        if utils.algorithm == utils.Algorithm.GA and fitness_combination == utils.FitnessCombination.EXEC_DIV:
            max_div = max(t.values[1] for t in study.best_trials)
            utils.PopulationStore().set_max_div(max_div)
        utils.PopulationStore().set_population(elite_population)

        # add random trials
        selection_size = int(self._population_size * (1 - utils.GA_rand_ratio))
        random_size = self._population_size - selection_size
        random_indices = random.sample(range(len(elite_population)), random_size)

        for i in range(random_size):
            index = random_indices[i]
            trial = elite_population[index]

            inputs = {}
            key = 'inputs'
            for j in range(6):
                rand_num = random.uniform(0, 0.01)
                inputs[f'{key}{j}'] = rand_num
            for j in range(6, 15):
                rand_num = random.uniform(0, 3)
                inputs[f'{key}{j}'] = rand_num

            exec_time = utils.run_iter_func(list(inputs.values()))
            trial.params = inputs
            if utils.algorithm == utils.Algorithm.GA and fitness_combination == utils.FitnessCombination.EXEC_DIV:
                trial.values = [exec_time, 0]
            else:
                trial.values = [exec_time]

        return elite_population


def _calc_crowding_distance(population: list[FrozenTrial]) -> defaultdict[int, float]:
    """Calculates the crowding distance of population.

    We define the crowding distance as the summation of the crowding distance of each dimension
    of value calculated as follows:

    * If all values in that dimension are the same, i.e., [1, 1, 1] or [inf, inf],
      the crowding distances of all trials in that dimension are zero.
    * Otherwise, the crowding distances of that dimension is the difference between
      two nearest values besides that value, one above and one below, divided by the difference
      between the maximal and minimal finite value of that dimension. Please note that:
        * the nearest value below the minimum is considered to be -inf and the
          nearest value above the maximum is considered to be inf, and
        * inf - inf and (-inf) - (-inf) is considered to be zero.
    """

    manhattan_distances: defaultdict[int, float] = defaultdict(float)
    if len(population) == 0:
        return manhattan_distances

    for i in range(len(population[0].values)):
        population.sort(key=lambda x: x.values[i])

        # If all trials in population have the same value in the i-th dimension, ignore the
        # objective dimension since it does not make difference.
        if population[0].values[i] == population[-1].values[i]:
            continue

        vs = [-float("inf")] + [trial.values[i] for trial in population] + [float("inf")]

        # Smallest finite value.
        v_min = next(x for x in vs if x != -float("inf"))

        # Largest finite value.
        v_max = next(x for x in reversed(vs) if x != float("inf"))

        width = v_max - v_min
        if width <= 0:
            # width == 0 or width == -inf
            width = 1.0

        for j in range(len(population)):
            # inf - inf and (-inf) - (-inf) is considered to be zero.
            gap = 0.0 if vs[j] == vs[j + 2] else vs[j + 2] - vs[j]
            manhattan_distances[population[j].number] += gap / width
    return manhattan_distances


def _crowding_distance_sort(population: list[FrozenTrial]) -> None:
    manhattan_distances = _calc_crowding_distance(population)
    population.sort(key=lambda x: manhattan_distances[x.number])
    population.reverse()


def _fast_non_dominated_sort(
    population: list[FrozenTrial],
    directions: list[optuna.study.StudyDirection],
    dominates: Callable[[FrozenTrial, FrozenTrial, list[optuna.study.StudyDirection]], bool],
) -> list[list[FrozenTrial]]:
    dominated_count: defaultdict[int, int] = defaultdict(int)
    dominates_list = defaultdict(list)

    for p, q in itertools.combinations(population, 2):
        if dominates_facade(population, p, q, directions, fitness_combination):
            dominates_list[p.number].append(q.number)
            dominated_count[q.number] += 1
        elif dominates_facade(population, q, p, directions, fitness_combination):
            dominates_list[q.number].append(p.number)
            dominated_count[p.number] += 1

    population_per_rank = []
    while population:
        non_dominated_population = []
        i = 0
        while i < len(population):
            if dominated_count[population[i].number] == 0:
                individual = population[i]
                if i == len(population) - 1:
                    population.pop()
                else:
                    population[i] = population.pop()
                non_dominated_population.append(individual)
            else:
                i += 1

        for x in non_dominated_population:
            for y in dominates_list[x.number]:
                dominated_count[y] -= 1

        assert non_dominated_population
        population_per_rank.append(non_dominated_population)
    return population_per_rank
