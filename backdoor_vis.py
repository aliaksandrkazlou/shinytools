import plotly.express as px
import pandas as pd

df_backdoor = pd.read_csv(filepath_or_buffer="Data/backdoor_tests.csv", sep="\t")
df_backdoor["method"] = df_backdoor["method"].str.replace("backdoor.", "")
df_backdoor["affected"] = df_backdoor["affected"].str.replace("_", " ")

fig = px.box(df_backdoor, x="affected", y="value", color="method",)

fig.update_layout(
    xaxis_title="Affected variables",
    yaxis_title="Estimated treatment effect",
    legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(family="sans-serif", size=14, color="black"),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2,
    ),
)

fig.add_shape(
    type="line",
    opacity=0.5,
    x0=-0.5,
    y0=10,
    x1=3.5,
    y1=10,
    line=dict(color="Grey", width=2,),
)

fig.show()
