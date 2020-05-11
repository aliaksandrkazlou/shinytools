import plotly.express as px
import pandas as pd

df_iv = pd.read_csv(filepath_or_buffer="Data/iv_tests.csv", sep="\t")
df_iv["method"] = df_iv["method"].str.replace("backdoor.", "").str.replace("iv.", "")
df_iv["affected"] = df_iv["affected"].str.replace("_", " ")

fig = px.box(df_iv, x="affected", y="value", color="method")

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
    # Line Horizontal
    type="line",
    opacity=0.5,
    x0=-0.5,
    y0=10,
    x1=3.5,
    y1=10,
    line=dict(color="Grey", width=2,),
)

fig.show()
