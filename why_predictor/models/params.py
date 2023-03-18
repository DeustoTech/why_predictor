"""Group hyperparams"""
from typing import List, Literal, Tuple, TypedDict, Union

KNNHyperParamKeys = Literal[
    "n_neighbors",
    "weights",
    # "algorithm",
    # "leaf_size",
    # "p"
]


class KNNHyperParams(TypedDict):
    """KNN HyperParams type"""

    n_neighbors: List[int]
    weights: List[str]
    # algorithm: List[str]
    # leaf_size: Dict[str, List[int]]
    # p: List[int]


MLPHyperParamKeys = Literal[
    "hidden_layer_sizes",
    "activation",
]


class MLPHyperParams(TypedDict):
    """MLP HyperParams type"""

    hidden_layer_sizes: List[Tuple[int]]
    activation: List[str]


RFHyperParamKeys = Literal[
    "n_estimators",
    "max_depth",
    # "criterion",
    # "min_samples_split",
    # "min_samples_leaf",
    # "min_weight_fraction_leaf",
    # "max_features",
    # "max_leaf_nodes",
    # "min_impurity_decrease",
    # "bootstrap",
    # "oob_score",
    # "ccp_alpha",
]


class RFHyperParams(TypedDict):
    """Random Forest HyperParams Type"""

    n_estimators: List[int]
    max_depth: List[Union[None, int]]
    # criterion: List[str]
    # min_samples_split: List[int]
    # min_samples_leaf: List[float]
    # min_weight_fraction_leaf: List[Union[int, float]]
    # max_features: List[Union[int, float, str]]
    # max_leaf_nodes: List[Union[None, int]]
    # min_impurity_decrease: List[float]
    # bootstrap: List[bool]
    # oob_score: Dict[str, List[bool]]
    # ccp_alpha: List[float]


SGDHyperParamKeys = Literal["alpha",]


class SGDHyperParams(TypedDict):
    """SGD HyperParams type"""

    alpha: List[float]


SVRHyperParamKeys = Literal[
    "epsilon",
    "tol",
    "C",
    "loss",
]


class SVRHyperParams(TypedDict):
    """SVR HyperParams type"""

    epsilon: List[float]
    tol: List[float]
    C: List[float]
    loss: List[str]


DTHyperParamKeys = Literal[
    "max_depth",
    # "criterion",
    # "splitter",
    # "min_samples_split",
    # "min_samples_leaf",
    # "max_features",
    # "max_leaf_nodes",
    # "min_weight_fraction_leaf",
    # "min_impurity_decrease",
    # "ccp_alpha",
]


class DTHyperParams(TypedDict):
    """Decission Tree HyperParams type"""

    max_depth: List[Union[None, int]]
    # criterion: List[str]
    # splitter: List[str]
    # min_samples_split: List[int]
    # min_samples_leaf: List[float]
    # max_features: List[Union[int, float, str]]
    # max_leaf_nodes: List[Union[None, int]]
    # min_weight_fraction_leaf: List[Union[int, float]]
    # min_impurity_decrease: List[float]
    # ccp_alpha: List[float]
