import pandas as pd
import plotly.express as px

df = pd.read_csv("/home/gaboloth/D/fisica/comp/ising-gpu/gpu2.5/legrp.txt")

fig = px.scatter(df, x = 'T', y = 'M', title="Curva di magnetizzazione", error_y="err")
fig.show()