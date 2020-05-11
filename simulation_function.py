"""
Function to run simulations on different versions of the original dataset from
https://microsoft.github.io/dowhy/dowhy_estimation_methods.html 
modified_linear_dataset is a slighly modifed version of linear_dataset from 
https://github.com/microsoft/dowhy/blob/master/dowhy/datasets.py
Full code is availible at 
"""
import numpy as np
import pandas as pd
from dowhy import CausalModel
import dowhy.datasets
from modified_linear_dataset import modified_linear_dataset


def simulate_dag_violations(
    methods,  # estimators to use
    beta,  # true treatment effect
    num_w_affected,  # number of common causes affected
    effect_on_w,  # effect of U on common causes
    num_z_affected,  # number of common causes affected
    effect_on_z,  # effect of U on instruments
    num_t_affected,  # number of treatments affected
    effect_on_t,  # effect of U on treatment
    effect_on_y,  # effect of U on outcomes
    times,  # number of simulation
):

    output = []
    for _ in range(times):
        # beta, num_common_causes, num_instruments, num_samples, etc. are as in the tutorial
        data = modified_linear_dataset(
            beta=beta,
            # u -> common causes
            num_w_affected=num_w_affected,
            effect_on_w=effect_on_w,
            # u -> instruments
            num_z_affected=num_z_affected,
            effect_on_z=effect_on_z,
            # u -> treatment
            num_t_affected=num_t_affected,
            effect_on_t=effect_on_t,
            # u -> outcome
            effect_on_y=effect_on_y,
            num_common_causes=5,
            num_instruments=2,
            num_samples=10000,
            treatment_is_binary=True,
        )

        df = data["df"]

        model = CausalModel(
            data=df,
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            instruments=data["instrument_names"],
            proceed_when_unidentifiable=True,
        )

        identified_estimand = model.identify_effect()

        estimates = [
            model.estimate_effect(
                identified_estimand, method_name=i[0], method_params=i[1]
            ).value
            for i in methods
        ]

        tmp_output = list(zip(estimates, [item[0] for item in methods]))

        output = output + tmp_output

    return output
