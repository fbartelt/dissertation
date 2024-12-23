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
distances = np.array(df.iloc[:, 43])
distances_approx = np.array(df.iloc[:, 44])

px.line(s_norm)
# %%
# Imports necessary plotly function to make subplots
from plotly.subplots import make_subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)
time = np.arange(0, len(s_norm)) * 1e-3
fig.add_trace(go.Scatter(x=time, y=distances, showlegend=False, line_width=4), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=np.linalg.norm(p - near_p, axis=1)*100, showlegend=False, line_width=4), row=2, col=1)
ori_errs = []

for i, R in enumerate(R_cols):
        R_star = near_R_cols[i]
        trace_ = np.trace(R_star @ np.linalg.inv(R))
        acos = np.arccos((trace_ - 1) / 2)
        # checks if acos is nan
        if np.isnan(acos):
            acos = 0
        ori_errs.append(acos * 180 / np.pi)
fig.add_trace(go.Scatter(x=time, y=ori_errs, showlegend=False, line_width=4), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1, tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
                 zerolinecolor='rgba(0.0, 0, 0, 0.5)', )
fig.update_xaxes(title_text=None, row=1, col=1, gridcolor='rgba(0.0, 0, 0, 0.5)', zerolinecolor='rgba(0.0, 0, 0, 0.5)', )
fig.update_xaxes(title_text=None, row=2, col=1, gridcolor='rgba(0.0, 0, 0, 0.5)', zerolinecolor='rgba(0.0, 0, 0, 0.5)', )
fig.update_yaxes(title_text="Distance D", row=1, col=1, tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
                 zerolinecolor='rgba(0.0, 0, 0, 0.5)', title_standoff=20)
fig.update_yaxes(title_text="Pos. error (cm)", row=2, col=1, tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
                 zerolinecolor='rgba(0.0, 0, 0, 0.5)', title_standoff=20)
fig.update_yaxes(title_text="Ori. error (deg)", row=3, col=1, tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
                 zerolinecolor='rgba(0.0, 0, 0, 0.5)', title_standoff=20)
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', width=1200, height=600, margin=dict(l=10, r=0, t=0, b=10, pad=0))
fig.show()
# %%
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
time = np.arange(0, len(s_norm)) * 1e-3
fig.add_trace(go.Scatter(x=time, y=np.linalg.norm((xi_t+xi_n - dq)[:, :3], axis=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=np.linalg.norm((xi_t+xi_n - dq)[:, 3:], axis=1)), row=2, col=1)
fig.show()
# %%
""" Kinematic data """
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

path_ = '/home/fbartelt/Documents/Projetos/dissertation/scripts'
file_name = 'kinematic.csv'
file_path = os.path.join(path_, file_name)

df = pd.read_csv(file_path, header=None)
df = df.dropna(axis=1)

near_R_cols = np.array(df.iloc[:, :9]).reshape(-1, 3, 3)
near_p =np.array(df.iloc[:, 9:12]).reshape(-1, 3)
R_cols = np.array(df.iloc[:, 12:21]).reshape(-1, 3, 3)
p = np.array(df.iloc[:, 21:24]).reshape(-1, 3)
xi_t = np.array(df.iloc[:, 24:30]).reshape(-1, 6)
xi_n = np.array(df.iloc[:, 30:36]).reshape(-1, 6)
distances = np.array(df.iloc[:, 36])
distances_approx = np.array(df.iloc[:, 37])
min_indexes = np.array(df.iloc[:, 38])

dt = 1e-3
time = np.arange(0, len(distances)) * dt

fig = go.Figure(go.Scatter(x=time, y=distances, name='distances'))
fig.add_trace(go.Scatter(x=time, y=distances_approx, name='distances approx'))
fig.show()

#%%
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=time, y=np.linalg.norm(p - near_p, axis=1)*100), row=1, col=1)
ori_errs = []

for i, R in enumerate(R_cols):
        R_star = near_R_cols[i]
        Z = R_star @ np.linalg.inv(R)
        trace_ = np.trace(Z)
        cost = (trace_ - 1) / 2
        sint = np.linalg.norm(Z - np.linalg.inv(Z), 'fro') / (2 * np.sqrt(2))
        theta = np.arctan2(sint, cost)
        # checks if acos is nan
        # if np.isnan(acos):
        #     acos = 0
        ori_errs.append(theta * 180 / np.pi)
        # ori_errs.append(np.trace(R_star))
fig.add_trace(go.Scatter(x=time, y=ori_errs), row=2, col=1)
fig.update_layout(margin=dict(l=10, r=0, t=0, b=10))
# Change yaxis title
fig.update_yaxes(title_text="Pos. error (cm)", row=1, col=1)
fig.update_yaxes(title_text="Ori. error (deg)", row=2, col=1)
fig.show()
# %%
def hds(s, c1, h0):
    H = np.eye(7)
    s = 2 * np.pi * s
    p = np.array([
         c1 * (np.sin(s) + 2 * np.sin(2 * s)),
         c1 * (np.cos(s) - 2 * np.cos(2 * s)),
         h0 + c1 * (-np.sin(3 * s))
    ])
    
    
    Rx = np.array([
      [1, 0, 0],
      [0, np.cos(2 * s), -np.sin(2 * s)],
      [0, np.sin(2 * s), np.cos(2 * s)]   
    ])

    Rz = np.array([
        [np.cos(s), -np.sin(s), 0],
        [np.sin(s), np.cos(s), 0],
        [0, 0, 1]
    ])

    R = Rz @ Rx

    H[:3, :3] = R
    H[3:6, 6] = p

    return H

def EEdist(V, W):
    Z = np.linalg.inv(V) @ W
    Q = Z[:3, :3]
    u = Z[3:6, 6]
    costheta = (np.trace(Q) - 1) / 2
    sintheta = np.linalg.norm(Q - np.linalg.inv(Q), 'fro') / (2 * np.sqrt(2))
    theta = np.arctan2(sintheta, costheta)
    return np.sqrt(2 * theta ** 2 + np.linalg.norm(u) ** 2)

def average_dist(curve):
    n_points = len(curve)
    dists = []
    for i in range(n_points - 1):
        dists.append(EEdist(curve[i], curve[i + 1]))
    return np.mean(dists)

n_points = 2000
c1, h0 = 0.7, 0.4
curve = [hds(s, c1, h0) for s in np.linspace(0, 1, n_points)]
p_curve = np.array([h[3:6, 6] for h in curve]).reshape(-1, 3)
R_curve = np.array([h[:3, :3] for h in curve]).reshape(-1, 3, 3)

fig = go.Figure(go.Scatter3d(x=p_curve[:, 0], y=p_curve[:, 1], z=p_curve[:, 2], mode='lines', line=dict(color='blue', width=2), name='curve'))
max_ = len(p)
fig.add_trace(go.Scatter3d(x=p[:max_, 0], y=p[:max_, 1], z=p[:max_, 2], mode='lines', line=dict(color='red', width=4), name='trajectory'))
fig.show()
# %%
import plotly.graph_objects as go

# Sample data
x = [i for i in range(100)]
y = [i**2 for i in range(100)]

# Create the main figure
fig = go.Figure()

# Main plot
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Main Plot'))

# Add a rectangle to highlight the zoomed region
fig.add_shape(
    type="rect",
    x0=20, x1=30, y0=400, y1=1000,  # Coordinates of the zoomed region
    line=dict(color="gray", dash="dash"),  # Dashed red line
)

# Zoomed-in plot (as a subplot)
zoomed_x = [i for i in range(20, 31)]
zoomed_y = [i**2 for i in zoomed_x]

fig.add_trace(go.Scatter(x=zoomed_x, y=zoomed_y, mode='lines', name='Zoomed Region',
                         xaxis='x2', yaxis='y2'))

# Add secondary axes for the zoomed-in plot
fig.update_layout(
    xaxis2=dict(domain=[0.6, 0.95], anchor='y2'),  # Adjust position
    yaxis2=dict(domain=[0.6, 0.95], anchor='x2'),
)

fig.add_shape(
    type="line",
    x0=20, y0=1000, x1=0.6*x[-1], y1=0.95*y[-1],  # Adjust x1, y1 to match zoom plot position
    xref="x", yref="y",
    line=dict(color="gray", dash="dot"),
)
fig.add_shape(
    type="line",
    x0=30, y0=400, x1=0.95*x[-1], y1=0.6*y[-1],  # Adjust x1, y1 to match zoom plot position
    xref="x", yref="y",
    line=dict(color="gray", dash="dot"),
)

fig.update_layout(
    xaxis2=dict(showline=True, linecolor="gray", linewidth=2, mirror=True),  # Gray border for x-axis of zoomed graph
    yaxis2=dict(showline=True, linecolor="gray", linewidth=2, mirror=True)   # Gray border for y-axis of zoomed graph
)
# %%
import plotly.graph_objects as go
import plotly.express as px

def scatter_with_zoom(x, y, x_zoom, y_zoom, zoom_x0, zoom_y0, zoom_x1, zoom_y1, 
                      box_color=None, mode_plot1='lines', mode_plot2='lines', 
                      plot_color=None, plot_width=2, box_width=1, 
                      box_dash='dot', draw_rectangle=True, zoom_pos='NE'):
    """Creates a Scatter plot Figure using Plotly with a zoom-in figure. For
    the moment, it only supports 2D data (y must be 1-dimensional).
    Parameters:
    -----------
    x : list
        x-axis data.
    y : list
        y-axis data.
    x_zoom : list
        x-axis data for the zoomed plot.
    y_zoom : list
        y-axis data for the zoomed plot
    zoom_x0 : float
        The percentage of the x-axis where the box starts with respect to the 
        true x-axis. It should be a value between 0 and 1.
    zoom_y0 : float
        The percentage of the y-axis where the box starts with respect to the 
        true y-axis. It should be a value between 0 and 1.
    zoom_x1 : float
        The percentage of the x-axis where the box ends with respect to the 
        true x-axis. It should be a value between 0 and 1.
    zoom_y1 : float
        The percentage of the y-axis where the box ends with respect to the 
        true y-axis. It should be a value between 0 and 1.
    box_color : str
        The color of the box. Default is None, which means a gray color with
        opacity 0.5. It is also the color of the dashed lines that connect both 
        plots.
    mode_plot1 : str
        The mode of the main plot. Default is 'lines'. This is passed to the
        go.Scatter() function as the `mode` paramer.
    mode_plot2 : str
        The mode of the zoomed plot. Default is 'lines'. This is passed to the
        go.Scatter() function as the `mode` paramer.
    plot_color : str
        The color of the plot. Default is None. If None, it uses the first color
        of the Plotly palette.
    box_dash : str
        The dash of the box. Default is 'dot'. It is also the dash of the
        dashed lines that connect both plots.
    draw_rectangle : bool
        Whether to draw the rectangle or not. Default is True.
    zoom_pos : str
        The position of the zoomed plot. Default is 'NE' (North-East). It can be
        'NE', 'NW', 'SE', 'SW'.
    """
    diff_y = abs(y[1] - y[0])
    x_max, y_max = max(x), max(y)
    x_min, y_min = min(x), min(y)
    box_x1, box_y1 = max(x_zoom), max(y_zoom)
    box_x0, box_y0 = min(x_zoom), min(y_zoom)
    if plot_color is None:
        plot_color = px.colors.qualitative.Plotly[0]
    if box_color is None:
        box_color = 'rgba(133, 132, 131, 0.6)'
    # Create the main figure
    fig = go.Figure()
    # Main plot
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode_plot1, showlegend=False, 
                             line=dict(color=plot_color, width=plot_width)))
    
    if draw_rectangle:
        # Add a rectangle to highlight the zoomed region
        fig.add_shape(
            type="rect",
            x0=box_x0, x1=box_x1, y0=box_y0, y1=box_y1,
            line=dict(color=box_color, dash=box_dash, width=box_width),
        )

    # Add zoom plot
    fig.add_trace(go.Scatter(x=x_zoom, y=y_zoom, mode=mode_plot2,  xaxis='x2', 
                             yaxis='y2', showlegend=False, 
                             line=dict(color=plot_color, width=plot_width)))
    # Add secondary axes for the zoomed-in plot
    fig.update_layout(
        xaxis2=dict(domain=[zoom_x0, zoom_x1], anchor='y2'),  # Adjust position
        yaxis2=dict(domain=[zoom_y0, zoom_y1], anchor='x2'),
    )
    
    # Add dashed lines that connect both plots
    match zoom_pos:
        case 'NE':
            lc_x0, lc_x1 = box_x0, zoom_x0 * x_max
            lc_y0, lc_y1 = box_y1, zoom_y1 * (y_max + 0.1 * diff_y)
            rc_x0, rc_x1 = box_x1, zoom_x1 * x_max
            rc_y0, rc_y1 = box_y0, zoom_y0 * (y_max - 0.1 * diff_y)
        case 'NW':
            lc_x0, lc_x1 = zoom_x0 * x_max, box_x0
            lc_y0, lc_y1 = zoom_y0 * (y_max - 0.1 * diff_y), box_y0
            rc_x0, rc_x1 = zoom_x1 * x_max, box_x1
            rc_y0, rc_y1 = zoom_y1 * (y_max + 0.1 * diff_y), box_y1
        case 'SE':
            lc_x0, lc_x1 = box_x0, zoom_x0 * x_max
            lc_y0, lc_y1 = box_y0, zoom_y0 * (y_max - 0.1 * diff_y)
            rc_x0, rc_x1 = box_x1, zoom_x1 * x_max
            rc_y0, rc_y1 = box_y1, zoom_y1 * (y_max + 0.1 * diff_y)
        case 'SW':
            lc_x0, lc_x1 = zoom_x0 * x_max, box_x0
            lc_y0, lc_y1 = zoom_y1 * (y_max + 0.1 * diff_y), box_y1
            rc_x0, rc_x1 = zoom_x1 * x_max, box_x1
            rc_y0, rc_y1 = zoom_y0 * (y_max - 0.1 * diff_y), box_y0
        case _:
            raise ValueError('Invalid zoom_pos. It should be "NE", "NW", "SE" or "SW"')
    
    fig.add_shape(
        type="line",
        x0=lc_x0, y0=lc_y0, x1=lc_x1, y1=lc_y1,
        xref="x", yref="y",
        line=dict(color=box_color, dash=box_dash, width=box_width),
    )
    fig.add_shape(
        type="line",
        x0=rc_x0, y0=rc_y0, x1=rc_x1, y1=rc_y1,
        xref="x", yref="y",
        line=dict(color=box_color, dash=box_dash, width=box_width),
    )

    fig.update_layout(
        xaxis2=dict(showline=True, linecolor=box_color, linewidth=2, mirror=True),
        yaxis2=dict(showline=True, linecolor=box_color, linewidth=2, mirror=True),
        xaxis_range=[x_min, x_max],
        yaxis_range=[y_min - 0.1 * diff_y, y_max + 0.1 * diff_y], 
    )

    return fig
         
# %%
time = np.arange(0, len(p)) * 0.001
max_idex = 1500
fig = scatter_with_zoom(time, list(s_norm.ravel()), time[:max_idex], list(s_norm.ravel())[:max_idex], 0.4, 0.4, 0.95, 
                        0.95, draw_rectangle=True, plot_width=4, box_width=2, box_dash='dot', zoom_pos='NE')

fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', width=1200, height=600)
fig.update_xaxes(title_text="Time (s)", gridcolor='rgba(0.0, 0, 0, 0.5)', zerolinecolor='rgba(0.0, 0, 0, 0.5)', tickprefix="£", ticksuffix="£")
fig.update_yaxes(title_text="Norm of s", gridcolor='rgba(0.0, 0, 0, 0.5)', zerolinecolor='rgba(0.0, 0, 0, 0.5)', tickprefix="£", ticksuffix="£")
fig.update_layout(xaxis2_title=None, yaxis2_title=None, margin=dict(t=10, b=10, l=10, r=10))
fig.show()
# %%
import plotly.graph_objects as go
import plotly.colors as pc

def plot_cylinder(p_hist, R_hist, radius, height, center, nt=10, nv=10, 
                  init_color=None, final_color=None, mid_color=None):
    """
    Plot a cylinder with the given position, orientation, radius, and height.

    Parameters:
    -----------
    p_hist : np.ndarray
        The position history of the cylinder.
    R_hist : np.ndarray
        The orientation history of the cylinder.
    radius : float
        The radius of the cylinder.
    height : float
        The height of the cylinder.
    center : float
        The center of the cylinder along the Z-axis.
    nt : int
        Number of points for angular discretization.
    nv : int
        Number of points for height discretization.
    color : str
        The color of the cylinder.
    """
    theta = np.linspace(0, 2 * np.pi, nt)
    v = np.linspace(center - height / 2, center + height / 2, nv)
    theta, v = np.meshgrid(theta, v)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = v
    
    local_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    traces = []
    if mid_color is None:
        mid_color = 'rgba(242, 5, 215, 0.1)'
    if init_color is None:
        init_color = 'rgba(190, 0, 247, 0.3)'
    if final_color is None:
        final_color = 'rgba(255, 132, 0, 0.4)'

    for i, (p, R) in enumerate(zip(p_hist, R_hist)):
        global_points = np.dot(R, local_points) + p[:, None]
        x_global = global_points[0, :].reshape(x.shape)
        y_global = global_points[1, :].reshape(y.shape)
        z_global = global_points[2, :].reshape(z.shape)
        if i == 0:
            color = init_color
        elif i == len(p_hist) - 1:
            color = final_color
        else:
            color = mid_color
        traces.append(go.Surface(
            x=x_global,
            y=y_global,
            z=z_global,
            colorscale=[[0, color], [1, color]],
            opacity=0.8,
            showscale=False
            ))

    return traces

import plotly.colors as pc
import plotly.graph_objects as go

def vector_field_plot(
    coordinates,
    field_values,
    orientations,
    curve,
    num_arrows=10,
    init_ball=0,
    final_ball=None,
    num_balls=10,
    add_lineplot=False,
    colorscale=None,
    show_curve=True,
    plot_balls=True,
    plot_vectorfield=True,
    ball_size=5,
    curve_width=2,
    path_width=5,
    frame_scale=0.05,
    frame_width=2,
    curr_path_style="solid",
    prev_path_style="dash",
    **kwargs
):
    """Plot a vector field in 3D. The vectors are represented as cones and the
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function. Also plots the target curve, and the path of the
    object. The object is represented as a sphere. The orientations are represented
    as frames with the x, y and z axis of the frame.

    Parameters
    ----------
    coordinates : list or np.array
        Mx3 array of coordinates of the vectors. Each row corresponds to x,y,z
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        Mx3 array of field values of the vectors. Each row corresponds to u,v,w
        respectively, i.e. the LINEAR velocity of the field in each direction.
        The column entries are the respective values.
    orientations : list or np.array
        Mx3x3 array of orientations of the object. Each row corresponds to the
        orientation of the object at that point. The 'column' entries are the
        respective 3x3 rotation matrices.
    curve : np.array
        Nx3 array of the curve points. Each row corresponds to x,y,z respectively.
    num_arrows : int, optional
        Number of vector field arrows (cones) to plot. The default is 10.
    init_ball : int, optional
        Initial ball index to plot. The default is 0.
    final_ball : int, optional
        Final ball index to plot. The default is None, which plots until the end.
    num_balls : int, optional
        Number of balls to plot. The default is 10.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
        This is used to connect the vector field arrows.
    colorscale : list, optional
        List of colors to use in the plot. The default is None, which uses the
        Plotly default colors. The list must have at least 6 colors, which are
        used for the curve, previous path, current path, initial ball, final ball
        and the object, respectively.
    show_curve : bool, optional
        Whether to show the target curve. The default is True.
    ball_size : int, optional
        Size of the object balls. The default is 5.
    curve_width : int, optional
        Width of the curve line. The default is 2.
    path_width : int, optional
        Width of the path line. The default is 5.
    frame_scale : float or list, optional
        Scale factor for the orientation frames. The default is 0.05. If a list
        is given, the scale factor is applied to each axis of the frame.
    frame_width : int, optional
        Width of the orientation frame lines. The default is 2.
    curr_path_style : str, optional
        Style of the current path line. The default is "solid".
    prev_path_style : str, optional
        Style of the previous path line. The default is "dash".
    **kwargs
        Additional keyword arguments to pass to the go.Cone function.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Resulting plotly figure.
    """
    if final_ball is None:
        final_ball = len(coordinates) - 1

    if isinstance(frame_scale, (int, float)):
        frame_scale = [frame_scale] * 3

    coordinates = np.array(coordinates).reshape(-1, 3)
    arrows_idx = np.round(np.linspace(0, len(coordinates) - 1, num_arrows)).astype(int)
    coord_field = coordinates[arrows_idx].T
    field_values = np.array(field_values).reshape(-1, 3)[arrows_idx].T
    ball_idx = np.round(np.linspace(init_ball, final_ball, num_balls)).astype(int)
    coord_balls = coordinates[ball_idx]
    ori_balls = np.array(orientations)[ball_idx]
    coordinates = coordinates.T

    if colorscale is None:
        colorscale = pc.qualitative.Plotly

    if isinstance(curve, tuple):
        curve = curve[0]

    fig = go.Figure()

    # Curve
    if show_curve:
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines",
                line=dict(width=curve_width, color=colorscale[1]),
            )
        )
    # Previous path
    if init_ball > 0:
        fig.add_trace(
            (
                go.Scatter3d(
                    x=coordinates[0, 0:init_ball],
                    y=coordinates[1, 0:init_ball],
                    z=coordinates[2, 0:init_ball],
                    mode="lines",
                    line=dict(width=path_width, dash=prev_path_style, color=colorscale[5]),
                )
            )
        )

    # Current path
    fig.add_trace(
        go.Scatter3d(
            x=coordinates[0, init_ball:final_ball],
            y=coordinates[1, init_ball:final_ball],
            z=coordinates[2, init_ball:final_ball],
            mode="lines",
            line=dict(width=path_width, dash=curr_path_style, color=colorscale[0]),
        )
    )

    # Vector field arrows
    if plot_vectorfield:
        fig.add_trace(
            go.Cone(
                x=coord_field[0, :],
                y=coord_field[1, :],
                z=coord_field[2, :],
                u=field_values[0, :],
                v=field_values[1, :],
                w=field_values[2, :],
                colorscale=[[0, colorscale[5]], [1, colorscale[5]]],  # Set the colorscale
                showscale=False,
                **kwargs,
            )
        )

    # Orientation frames
    if orientations is not None:
        for i, ori in enumerate(ori_balls):
            px, py, pz = coord_balls[i, :]
            ux, uy, uz =  ori[:, 0] / (np.linalg.norm(ori[:, 0] + 1e-6)) * frame_scale
            vx, vy, vz =  ori[:, 1] / (np.linalg.norm(ori[:, 1] + 1e-6)) * frame_scale
            wx, wy, wz =  ori[:, 2] / (np.linalg.norm(ori[:, 2] + 1e-6)) * frame_scale
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + ux],
                    y=[py, py + uy],
                    z=[pz, pz + uz],
                    mode="lines",
                    line=dict(color="red", width=frame_width),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + vx],
                    y=[py, py + vy],
                    z=[pz, pz + vz],
                    mode="lines",
                    line=dict(color="lime", width=frame_width),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + wx],
                    y=[py, py + wy],
                    z=[pz, pz + wz],
                    mode="lines",
                    line=dict(color="blue", width=frame_width),
                    showlegend=False,
                )
            )

    # Object
    if plot_balls:
        for i, coord in enumerate(coord_balls):
            if i == 0:
                color = colorscale[3]
            elif i == len(coord_balls) - 1:
                color = colorscale[4]
            else:
                color = "rgba(172, 99, 250, 0.6)"
            fig.add_trace(
                go.Scatter3d(
                    x=[coord[0]],
                    y=[coord[1]],
                    z=[coord[2]],
                    mode="markers",
                    marker=dict(size=ball_size, color=color),
                    showlegend=False,
                )
            )

    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )

    return fig

# PLOT 1
final_ball = 13000
init_ball = 0
# PLOT 2
final_ball = 26000
init_ball = 13000
# PLOT 3
final_ball = 39000
init_ball = 26000

num_balls = 8
idxs = np.linspace(init_ball, final_ball, num_balls, dtype=int)
p_plot, R_plot = np.array(p)[idxs], np.array(R_cols)[idxs]
p_curve = np.array([h[3:6, 6] for h in curve]).reshape(-1, 3)
cylinders = plot_cylinder(p_plot, R_plot, 0.25, 1, 0.5)
# fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='lines', line=dict(color='red', width=4)))
fig = vector_field_plot(p, xi_t, R_cols, p_curve, num_arrows=0, init_ball=init_ball, 
                        final_ball=final_ball, num_balls=num_balls, add_lineplot=False,
                        frame_scale=0.5, ball_size=5, path_width=5, curve_width=3,
                        curr_path_style='solid', plot_balls=False, plot_vectorfield=False)
fig.add_traces(cylinders)
# Remove legend
fig.update_layout(width=600, height=600, margin=dict(l=0, r=0, t=0, b=0), 
                  showlegend=False, plot_bgcolor='white', paper_bgcolor='white')
# PLOT 1
eye = {'x': -0.5188374604151841, 'y': 1.7775818819597617, 'z': 1.9763689762474717}
center = {'x': 0.10410037017924523, 'y': 0.08148809126474614, 'z': -0.15547200455041466}
# PLOT 2
eye = {'x': -1.687735973547643, 'y': -1.7443866178498613, 'z': 1.5321718288279524}
center = {'x': 0.04431795503188255, 'y': 0.07069721419689137, 'z': -0.2285652975541114}
# PLOT3
eye={'x': 0.5837593159603366, 'y': 1.4488594523919796, 'z': 2.397553293755867}
center={'x': -0.04772314482037555, 'y': 0.02957920444399699, 'z': -0.19415339574852067}

xticks = [-2.5, 0, 2.5]
yticks = [-2.5, 0, 2.5]
zticks = [-1.7, 0, 1.7]
fig.update_layout(scene=dict(
    camera=dict(eye=eye, center=center),
    xaxis=dict(tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
                 zerolinecolor='rgba(0.0, 0, 0, 0.5)', backgroundcolor='white',
                 showticklabels=False, tickvals=xticks, range=[xticks[0], xticks[-1]],
                 ticks="inside"),
    yaxis=dict(tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
                 zerolinecolor='rgba(0.0, 0, 0, 0.5)', backgroundcolor='white',
                 showticklabels=False, tickvals=yticks, range=[yticks[0], yticks[-1]],
                 ticks="inside"),
    zaxis=dict(tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
                 zerolinecolor='rgba(0.0, 0, 0, 0.5)', backgroundcolor='white',
                 showticklabels=False, tickvals=zticks, range=[zticks[0], zticks[-1]],
                 ticks="inside"),
    aspectmode='manual', aspectratio=dict(x=1.47, y=1.47, z=1),
    ))
# fig.update_layout(scene_aspectmode='data')
# fig.update_xaxes(tickprefix="£", ticksuffix="£",gridcolor='rgba(0.0, 0, 0, 0.5)', 
#                  zerolinecolor='rgba(0.0, 0, 0, 0.5)', title_standoff=0)
fig.show()

# %%
""" PLOT WRENCHES"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

path_ = '/home/fbartelt/Documents/Projetos/dissertation/scripts'
file_name = 'adaptive_wrenches.csv'
file_path = os.path.join(path_, file_name)

df = pd.read_csv(file_path, header=None)
df = df.dropna(axis=1)

force_1 = np.array(df.iloc[:, :3]).reshape(-1, 3)
torque_1 = np.array(df.iloc[:, 3:6]).reshape(-1, 3)
force_2 = np.array(df.iloc[:, 6:9]).reshape(-1, 3)
torque_2 = np.array(df.iloc[:, 9:12]).reshape(-1, 3)
force_3 = np.array(df.iloc[:, 12:15]).reshape(-1, 3)
torque_3 = np.array(df.iloc[:, 15:18]).reshape(-1, 3)
force_4 = np.array(df.iloc[:, 18:21]).reshape(-1, 3)
torque_4 = np.array(df.iloc[:, 21:24]).reshape(-1, 3)
force_5 = np.array(df.iloc[:, 24:27]).reshape(-1, 3)
torque_5 = np.array(df.iloc[:, 27:30]).reshape(-1, 3)
force_6 = np.array(df.iloc[:, 30:33]).reshape(-1, 3)
torque_6 = np.array(df.iloc[:, 33:36]).reshape(-1, 3)

forces_norm = np.array([np.linalg.norm(f, axis=1) for f in [force_1, force_2, force_3, force_4, force_5, force_6]]).T
torques_norm = np.array([np.linalg.norm(t, axis=1) for t in [torque_1, torque_2, torque_3, torque_4, torque_5, torque_6]]).T

for i in range(6):
    print(i+1)
    print("Faverage: ", np.mean(forces_norm[:, i]), " pm ", np.std(forces_norm[:, i]))
    print("Fmin: ", np.min(forces_norm[:, i]))
    print("Fmax: ", np.max(forces_norm[:, i]))
    print("Taverage: ", np.mean(torques_norm[:, i]), " pm ", np.std(torques_norm[:, i]))
    print("Tmin: ", np.min(torques_norm[:, i]))
    print("Tmax: ", np.max(torques_norm[:, i]))
# %%
