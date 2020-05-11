"""
Function to modify datasets in different ways. 
DoWhy.causal_refuter does somewhat similar job
It is a slighly modifed version of linear_dataset from 
https://github.com/microsoft/dowhy/blob/master/dowhy/datasets.py
Full code is availible at 
"""
import random
import numpy as np
import pandas as pd
from dowhy.datasets import stochastically_convert_to_binary
from dowhy.datasets import create_dot_graph
from dowhy.datasets import create_gml_graph


def modified_linear_dataset(
    beta,
    num_common_causes,
    num_samples,
    num_instruments=0,
    num_effect_modifiers=0,
    num_treatments=1,
    effect_on_w=0,
    num_w_affected=0,
    effect_on_z=0,
    num_z_affected=0,
    effect_on_t=0,
    num_t_affected=0,
    effect_on_y=0,
    treatment_is_binary=True,
    outcome_is_binary=False,
):

    W, X, Z, c1, c2, ce, cz = [None] * 7
    beta = float(beta)
    # Making beta an array
    if type(beta) not in [list, np.ndarray]:
        beta = np.repeat(beta, num_treatments)
    if num_common_causes > 0:
        range_c1 = max(beta) * 0.5
        range_c2 = max(beta) * 0.5
        means = np.random.uniform(-1, 1, num_common_causes)
        cov_mat = np.diag(np.ones(num_common_causes))
        W = np.random.multivariate_normal(means, cov_mat, num_samples)
        c1 = np.random.uniform(0, range_c1, (num_common_causes, num_treatments))
        c2 = np.random.uniform(0, range_c2, num_common_causes)

    if num_instruments > 0:
        range_cz = beta
        Z = np.zeros((num_samples, num_instruments))
        for i in range(num_instruments):
            Z[:, i] = np.random.uniform(0, 1, size=num_samples)
        # TODO Ensure that we do not generate weak instruments
        cz = np.random.uniform(
            range_cz - (range_cz * 0.05),
            range_cz + (range_cz * 0.05),
            (num_instruments, num_treatments),
        )

    if num_effect_modifiers > 0:
        range_ce = beta * 0.5
        means = np.random.uniform(-1, 1, num_effect_modifiers)
        cov_mat = np.diag(np.ones(num_effect_modifiers))
        X = np.random.multivariate_normal(means, cov_mat, num_samples)
        ce = np.random.uniform(0, range_ce, num_effect_modifiers)
    # TODO - test all our methods with random noise added to covariates (instead of the stochastic treatment assignment)

    # Creating random variable
    U = np.random.normal(0, 1, size=num_samples)
    # Modeling U -> X effect
    if num_w_affected > 0:
        affected_w = random.sample(list(range(0, W.shape[1])), num_w_affected)
        for var in affected_w:
            W[:, var] = W[:, var] + U * effect_on_w
    # Modeling U -> Z effect
    if num_z_affected > 0:
        affected_z = random.sample(list(range(0, Z.shape[1])), num_z_affected)
        for var in affected_z:
            Z[:, var] = Z[:, var] + U * effect_on_z

    t = np.random.normal(0, 1, (num_samples, num_treatments))
    # Modeling U -> t effect
    if num_t_affected > 0:
        affected_t = random.sample(list(range(0, t.shape[1])), num_t_affected)
        for var in affected_t:
            t[:, var] = t[:, var] + U * effect_on_t
    if num_common_causes > 0:
        t += W @ c1  # + np.random.normal(0, 0.01)
    if num_instruments > 0:
        t += Z @ cz
    # Converting treatment to binary if required
    if treatment_is_binary:
        t = np.vectorize(stochastically_convert_to_binary)(t)

    def _compute_y(t, W, X, beta, c2, ce):
        y = t @ beta  # + np.random.normal(0,0.01)
        if num_common_causes > 0:
            y += W @ c2
        if num_effect_modifiers > 0:
            y += (X @ ce) * np.prod(t, axis=1)
        # Modeling U -> Y effect
        y += U * effect_on_y
        return y

    y = _compute_y(t, W, X, beta, c2, ce)
    if outcome_is_binary:
        y = np.vectorize(stochastically_convert_to_binary)(t)

    data = np.column_stack((t, y))
    if num_common_causes > 0:
        data = np.column_stack((W, data))
    if num_instruments > 0:
        data = np.column_stack((Z, data))
    if num_effect_modifiers > 0:
        data = np.column_stack((X, data))

    treatments = [("v" + str(i)) for i in range(0, num_treatments)]
    outcome = "y"
    common_causes = [("W" + str(i)) for i in range(0, num_common_causes)]
    ate = np.mean(
        _compute_y(np.ones((num_samples, num_treatments)), W, X, beta, c2, ce)
        - _compute_y(np.zeros((num_samples, num_treatments)), W, X, beta, c2, ce)
    )
    instruments = [("Z" + str(i)) for i in range(0, num_instruments)]
    effect_modifiers = [("X" + str(i)) for i in range(0, num_effect_modifiers)]
    # other_variables = None
    col_names = effect_modifiers + instruments + common_causes + treatments + [outcome]
    data = pd.DataFrame(data, columns=col_names)
    # Specifying the correct dtypes
    if treatment_is_binary:
        data = data.astype({tname: "bool" for tname in treatments}, copy=False)
    if outcome_is_binary:
        data = data.astype({outcome: "bool"}, copy=False)

    # Now specifying the corresponding graph strings
    dot_graph = create_dot_graph(
        treatments, outcome, common_causes, instruments, effect_modifiers
    )
    # Now writing the gml graph
    gml_graph = create_gml_graph(
        treatments, outcome, common_causes, instruments, effect_modifiers
    )
    ret_dict = {
        "df": data,
        "treatment_name": treatments,
        "outcome_name": outcome,
        "common_causes_names": common_causes,
        "instrument_names": instruments,
        "effect_modifier_names": effect_modifiers,
        "dot_graph": dot_graph,
        "gml_graph": gml_graph,
        "ate": ate,
    }

    return ret_dict
