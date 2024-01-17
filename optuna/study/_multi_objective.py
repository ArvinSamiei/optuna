import numpy as np
from typing import List
from typing import Optional
from typing import Sequence

import optuna
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from utils import FitnessCombination


def _get_pareto_front_trials_2d(
        trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    trials = [trial for trial in trials if trial.state == TrialState.COMPLETE]

    n_trials = len(trials)
    if n_trials == 0:
        return []

    trials.sort(
        key=lambda trial: (
            _normalize_value(trial.values[0], directions[0]),
            _normalize_value(trial.values[1], directions[1]),
        ),
    )

    last_nondominated_trial = trials[0]
    pareto_front = [last_nondominated_trial]
    for i in range(1, n_trials):
        trial = trials[i]
        if _dominates(last_nondominated_trial, trial, directions):
            continue
        pareto_front.append(trial)
        last_nondominated_trial = trial

    pareto_front.sort(key=lambda trial: trial.number)
    return pareto_front


def _get_pareto_front_trials_nd(
        trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    pareto_front = []
    trials = [t for t in trials if t.state == TrialState.COMPLETE]

    # TODO(vincent): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
    for trial in trials:
        dominated = False
        for other in trials:
            if _dominates(other, trial, directions):
                dominated = True
                break

        if not dominated:
            pareto_front.append(trial)

    return pareto_front


def _get_pareto_front_trials_by_trials(
        trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    if len(directions) == 2:
        return _get_pareto_front_trials_2d(trials, directions)  # Log-linear in number of trials.
    return _get_pareto_front_trials_nd(trials, directions)  # Quadratic in number of trials.


def _get_pareto_front_trials(study: "optuna.study.Study") -> List[FrozenTrial]:
    return _get_pareto_front_trials_by_trials(study.trials, study.directions)


value_str = 'value'
state_str = 'state'


def _dominates_value_state(
        value_state0, value_state1, directions: Sequence[StudyDirection]
) -> bool:
    values0 = value_state0[value_str]
    values1 = value_state1[value_str]
    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    if value_state0[state_str] != TrialState.COMPLETE:
        return False

    if value_state0[state_str] != TrialState.COMPLETE:
        return True

    normalized_values0 = [_normalize_value(v, d) for v, d in zip(values0, directions)]
    normalized_values1 = [_normalize_value(v, d) for v, d in zip(values1, directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def _dominates(
        trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    values0 = trial0.values
    values1 = trial1.values

    assert values0 is not None
    assert values1 is not None

    try:
        value_state0 = {value_str: values0, state_str: trial0.state}
        value_state1 = {value_str: values1, state_str: trial1.state}
    except AttributeError:
        value_state0 = {value_str: values0, state_str: TrialState.COMPLETE}
        value_state1 = {value_str: values1, state_str: TrialState.COMPLETE}
    return _dominates_value_state(value_state0, value_state1, directions)


def scale_motions(lst):
    for i in range(6):
        lst[i] = lst[i] * 30
    return lst


def _dominates_with_diversity(population,
                              trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection], values_size
                              ) -> bool:
    values0 = trial0.values
    pop_without_0 = [x for x in population if x.trial_id != trial0.trial_id]
    matrix0 = np.array([scale_motions(list(obj.inputs)) for obj in pop_without_0], dtype=np.float32)

    values1 = trial1.values
    pop_without_1 = [x for x in population if x.trial_id != trial1.trial_id]
    matrix1 = np.array([scale_motions(list(obj.inputs)) for obj in pop_without_1], dtype=np.float32)
    matrix = np.array([scale_motions(list(obj.inputs)) for obj in population], dtype=np.float32)

    transpose = matrix.T
    product = np.dot(transpose, matrix)
    det = np.linalg.det(product)

    transpose = matrix0.T
    product = np.dot(transpose, matrix0)
    det0 = np.linalg.det(product)

    transpose = matrix1.T
    product = np.dot(transpose, matrix1)
    det1 = np.linalg.det(product)

    values0[values_size - 1] = abs(det) - abs(det0)
    values1[values_size - 1] = abs(det) - abs(det1)

    try:
        value_state0 = {value_str: values0, state_str: trial0.state}
        value_state1 = {value_str: values1, state_str: trial1.state}
    except AttributeError:
        value_state0 = {value_str: values0, state_str: TrialState.COMPLETE}
        value_state1 = {value_str: values1, state_str: TrialState.COMPLETE}

    _dominates_value_state(value_state0, value_state1, directions)


def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value


def dominates_facade(population,
                     trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection], case
                     ) -> bool:
    num_objectives = 0
    if case == FitnessCombination.EXEC or case == FitnessCombination.JIT or case == FitnessCombination.DIV:
        num_objectives = 1
    elif case == FitnessCombination.EXEC_DIV or case == FitnessCombination.JIT_DIV:
        num_objectives = 2
    elif case == FitnessCombination.EXEC_JIT_DIV:
        num_objectives = 3

    if directions is None:
        directions = ["maximize"] * num_objectives

    if num_objectives == 1:

        return _dominates(trial0, trial1, directions)
    else:
        _dominates_with_diversity(population, trial0, trial1, directions, num_objectives)