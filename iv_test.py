"""
Simulating performance of the IV estimators against unobserved covariates.
"""
import numpy as np
import pandas as pd
from modified_linear_dataset import modified_linear_dataset
from simulation_function import simulate_dag_violations

# Number of simulations
N = 100
# True treatment effect
treatment_effect = 10

# Choosing IV estimators
methods = [["iv.instrumental_variable", {"iv_instrument_name": "Z0"}]]

# Simulating U -> outcome
u_outcome = simulate_dag_violations(
    methods=methods,
    beta=treatment_effect,
    num_w_affected=0,
    effect_on_w=0,
    num_z_affected=0,
    effect_on_z=0,
    num_t_affected=0,
    effect_on_t=0,
    effect_on_y=treatment_effect * 0.5,
    times=N,
)

df_u_outcome = pd.DataFrame(u_outcome, columns=["value", "method"])
df_u_outcome["affected"] = pd.Series(
    ["outcome" for x in range(len(df_u_outcome.index))]
)

# Simulating U -> outcome and treatment
u_outcome_and_treatment = simulate_dag_violations(
    methods=methods,
    beta=treatment_effect,
    num_w_affected=0,
    effect_on_w=0,
    num_z_affected=0,
    effect_on_z=0,
    num_t_affected=1,
    effect_on_t=treatment_effect * 0.5,
    effect_on_y=treatment_effect * 0.5,
    times=N,
)

df_u_outcome_and_treatment = pd.DataFrame(
    u_outcome_and_treatment, columns=["value", "method"]
)
df_u_outcome_and_treatment["affected"] = pd.Series(
    ["outcome_and_treatment" for x in range(len(df_u_outcome_and_treatment.index))]
)

# Simulating U -> outcome and a random instrument
u_outcome_and_instrument = simulate_dag_violations(
    methods=methods,
    beta=treatment_effect,
    num_w_affected=0,
    effect_on_w=0,
    num_z_affected=1,
    effect_on_z=treatment_effect * 0.5,
    num_t_affected=0,
    effect_on_t=0,
    effect_on_y=treatment_effect * 0.5,
    times=N,
)

df_u_outcome_and_instrument = pd.DataFrame(
    u_outcome_and_instrument, columns=["value", "method"]
)
df_u_outcome_and_instrument["affected"] = pd.Series(
    ["outcome_and_instrument" for x in range(len(df_u_outcome_and_instrument.index))]
)

# Simulating U -> treatment and a random instrument
u_treatment_and_instrument = simulate_dag_violations(
    methods=methods,
    beta=treatment_effect,
    num_w_affected=0,
    effect_on_w=0,
    num_z_affected=1,
    effect_on_z=treatment_effect * 0.5,
    num_t_affected=1,
    effect_on_t=treatment_effect * 0.5,
    effect_on_y=0,
    times=N,
)

df_u_treatment_and_instrument = pd.DataFrame(
    u_treatment_and_instrument, columns=["value", "method"]
)
df_u_treatment_and_instrument["affected"] = pd.Series(
    [
        "treatment_and_instrument"
        for x in range(len(df_u_treatment_and_instrument.index))
    ]
)

# Combining all datasets
df_list = [
    df_u_outcome,
    df_u_outcome_and_treatment,
    df_u_outcome_and_instrument,
    df_u_treatment_and_instrument,
]
df_all = pd.concat(df_list)
df_all.to_csv("Data/iv_tests.csv", sep="\t")
