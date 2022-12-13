"""Models utils"""
from typing import Any, Dict, List


def generate_hyperparams_from_keys(
    params: Any,
    current_set: Dict[str, Any],
    hyperparams: List[Any],
) -> List[Dict[str, Any]]:
    """Recursive function to generate hyperparams set from keys"""
    if hyperparams:
        hyperparam = hyperparams.pop(0)
        hyperparam_sets = []
        for value in params[hyperparam]:
            my_set = current_set.copy()
            my_set[hyperparam] = value
            hyperparam_sets.extend(
                generate_hyperparams_from_keys(params, my_set, hyperparams[:])
            )
        return hyperparam_sets
    return [current_set]


def sanitize_params(params: Any) -> Any:
    """Sanitize params as it is the optimal set, and the algorithm will expect
    different options"""
    for key in params:
        params[key] = [params[key]]
    return params
