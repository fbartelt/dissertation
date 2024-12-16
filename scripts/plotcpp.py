#%%
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

path_ = '/home/fbartelt/Documents/Projetos/dissertation/scripts'
file_name = 'cpp_adaptive.csv'
file_path = os.path.join(path_, file_name)

df = pd.read_csv(file_path, header=None)
df = df.dropna(axis=1)

near_R_cols = np.array(df.iloc[:, :9]).reshape(-1, 3, 3)
near_p =np.array(df.iloc[:, 9:12]).reshape(-1, 3)
R_cols = np.array(df.iloc[:, 12:21]).reshape(-1, 3, 3)
p = np.array(df.iloc[:, 21:24]).reshape(-1, 3)
xi_t = np.array(df.iloc[:, 24:30]).reshape(-1, 6)
xi_n = np.array(df.iloc[:, 30:36]).reshape(-1, 6)
dq = np.array(df.iloc[:, 36:42]).reshape(-1, 6)
s_norm = np.array(df.iloc[:, 42])

px.line(s_norm)
# %%
# Imports necessary plotly function to make subplots
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
time = np.arange(0, len(s_norm)) * 1e-3
fig.add_trace(go.Scatter(x=time, y=np.linalg.norm(p - near_p, axis=1)*100), row=1, col=1)
ori_errs = []

for i, R in enumerate(R_cols):
        R_star = near_R_cols[i]
        trace_ = np.trace(R_star @ np.linalg.inv(R))
        acos = np.arccos((trace_ - 1) / 2)
        # checks if acos is nan
        if np.isnan(acos):
            acos = 0
        ori_errs.append(acos * 180 / np.pi)
fig.add_trace(go.Scatter(x=time, y=ori_errs), row=2, col=1)
fig.update_layout(margin=dict(l=10, r=0, t=0, b=10))
fig.show()
# %%
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
time = np.arange(0, len(s_norm)) * 1e-3
fig.add_trace(go.Scatter(x=time, y=np.linalg.norm((xi_t+xi_n - dq)[:, 3:], axis=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=np.linalg.norm((xi_t+xi_n - dq)[:, :3], axis=1)), row=2, col=1)
fig.show()
# %%
