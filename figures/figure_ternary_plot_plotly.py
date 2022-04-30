# This code was taken and modified from:
# https://nbviewer.org/github/empet/Ternary-contour-plot/blob/master/Plotly-ternary-contour-plot.ipynb

import plotly.graph_objs as go
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import pickle

#%%
CMIN_VMIN = 1.03
CMAX_VMAX = 1.061
LINE_WIDTH_LAYOUT = 0.3
LINE_WIDTH_CONTOUR = 0.3

def load_data(load: float,
              pv: float,
              quant_value: str,
              file_path: Path=None,):

    assert (0 <= pv <= 1.0) and (0 <= load <= 1.0), "Load and PV must be between [0,1]"

    if file_path is None:
        path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
    else:
        path_file_parent = file_path

    file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quant_value}.pkl"
    with open(path_file_parent / file_name_solutions_dictionary, "rb") as pickle_file:
        solutions_dict = pickle.load(pickle_file)

    mixture_count = 0
    solutions_ternary = {}
    for solution_key in solutions_dict:
        mixture_, load_, pv_ = solution_key
        if (load == load_) and (pv == pv_):
            mixture_count += 1
            solutions_ternary[mixture_] = solutions_dict[solution_key]["max_q_" + quant_value].max()

    print(f"Total mixture in load-PV combination: {mixture_count}")
    print(f"Min. voltage: {min(solutions_ternary.values())}")
    print(f"Max. voltage: {max(solutions_ternary.values())}")

    mixture_values = np.array(list(solutions_ternary.keys()))
    voltage_values = np.array(list(solutions_ternary.values()))

    mixture_frame = pd.DataFrame(mixture_values, columns=["cloudy", "sunny", "overcast"])
    max_voltage_frame = pd.DataFrame(voltage_values, columns=["voltage"])

    ternary_data = pd.concat([mixture_frame, max_voltage_frame], axis=1)

    return ternary_data


def tr_b2c2b():
    # returns the transformation matrix from barycentric to cartesian coordinates and conversely
    tri_verts = np.array([[0.5, np.sqrt(3)/2], [0, 0], [1, 0]])# reference triangle
    M = np.array([tri_verts[:,0], tri_verts[:, 1], np.ones(3)])
    return M, np.linalg.inv(M)


def contour_trace(x, y, z, tooltip,
                  colorscale='Viridis', reversescale=False,
                  linewidth=0.5, linecolor='rgb(150,150,150)'):
    return dict(type='contour',
                x=x,
                y=y,
                z=z,
                text=tooltip,
                hoverinfo='text',
                colorscale=colorscale,
                reversescale=reversescale,
                line=dict(width=linewidth, color=linecolor),
                colorbar=dict(
                              dtick=0.01,  # How often you show the ticks (e.g., every 0.01 volts)
                              thickness=10,
                              tickcolor="#FFFFFF",
                              # nticks=1,
                              # outlinewidth=0.5,
                              # bordercolor="#FFFFFF",
                              ticklen=0.5,
                              title="Voltage",
                              titleside='right',
                              # showticklabels=False,
                              ),
                contours=dict(showlabels=True,
                              labelfont=dict(  # label font properties
                                             size=12,
                                             color='black',
                                             ),
                                # range_color = (0.5, 1.5),
                              start=CMIN_VMIN,  # overrides zmin
                              end=CMAX_VMAX,   # overrides zmax
                              size=0.00105  # Controls the resolution of the levels
                              ),
                # zmin = 0.5,
                # zmax = 2.5,
                # line_width=1.0,
                showscale=True,  # Hide the colorbar

                )


def barycentric_ticks(side):
    # side 0, 1 or  2; side j has 0 in the  j^th position of barycentric coords of tick origin
    # returns the list of tick origin barycentric coords
    p = 10
    if side == 0:  # where a=0
        return np.array([(0, j / p, 1 - j / p) for j in range(p - 2, 0, -2)])
    elif side == 1:  # b=0
        return np.array([(i / p, 0, 1 - i / p) for i in range(2, p, 2)])
    elif side == 2:  # c=0
        return np.array([(i / p, j / p, 0) for i in range(p - 2, 0, -2) for j in range(p - i, -1, -1) if i + j == p])
    else:
        raise ValueError('The side can be only 0, 1, 2')


def cart_coord_ticks(side, M, xt, yt, posx, posy,  t=0.01):
    # side 0, 1 or 2
    # each tick segment is parameterized as (x(s), y(s)), s in [0, t]

    # global M, xt, yt, posx, posy

    # M is the transformation matrix from barycentric to cartesian coords
    # xt, yt are the lists of x, resp y-coords of tick segments
    # posx, posy are the lists of ticklabel positions for side 0, 1, 2 (concatenated)

    baryc = barycentric_ticks(side)
    xy1 = np.dot(M, baryc.T)
    xs, ys = xy1[:2]

    if side == 0:
        for i in range(4):
            xt.extend([xs[i], xs[i] + t, None])
            yt.extend([ys[i], ys[i] - np.sqrt(3) * t, None])
        posx.extend([xs[i] + t for i in range(4)])
        posy.extend([ys[i] - np.sqrt(3) * t for i in range(4)])

    elif side == 1:
        for i in range(4):
            xt.extend([xs[i], xs[i] + t, None])
            yt.extend([ys[i], ys[i] + np.sqrt(3) * t, None])
        posx.extend([xs[i] + t for i in range(4)])
        posy.extend([ys[i] + np.sqrt(3) * t for i in range(4)])

    elif side == 2:
        for i in range(4):
            xt.extend([xs[i], xs[i] - 2 * t, None])
            yt.extend([ys[i], ys[i], None])
        posx.extend([xs[i] - 2 * t for i in range(4)])
        posy.extend([ys[i] for i in range(4)])
    else:
        raise ValueError('side can be only 0,1,2')


def ternary_layout(title='Ternary contour plot', width=550, height=525,
                   fontfamily='Balto, sans-serif', lfontsize=14,
                   # plot_bgcolor='rgb(240,240,240)',
                   plot_bgcolor='rgb(255,255,255)',
                   vertex_text=['a', 'b', 'c'], v_fontsize=14):
    return dict(title=title,
                title_x=0.5,
                font=dict(family=fontfamily,
                          size=lfontsize),
                width=width,
                height=height,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor=plot_bgcolor,
                showlegend=False,
                # annotations for strings  placed at the triangle vertices
                annotations=[dict(showarrow=False,
                                  text=vertex_text[0],
                                  x=0.5,
                                  y=np.sqrt(3) / 2,
                                  align='center',
                                  xanchor='center',
                                  yanchor='bottom',
                                  font=dict(size=v_fontsize)),
                             dict(showarrow=False,
                                  text=vertex_text[1],
                                  x=0,
                                  y=0,
                                  align='left',
                                  xanchor='right',
                                  yanchor='top',
                                  font=dict(size=v_fontsize)),
                             dict(showarrow=False,
                                  text=vertex_text[2],
                                  x=1,
                                  y=0,
                                  align='right',
                                  xanchor='left',
                                  yanchor='top',
                                  font=dict(size=v_fontsize))
                             ])


def set_ticklabels(annotations, posx, posy, proportion=True):
    # annotations: list of annotations previously defined in layout definition as a dict,
    #     not as an instance of go.Layout
    # posx, posy:  lists containing ticklabel position coordinates
    # proportion - boolean; True when ticklabels are 0.2, 0.4, ... False when they are 20%, 40%...

    if not isinstance(annotations, list):
        raise ValueError('annotations should be a list')

    ticklabel = [0.8, 0.6, 0.4, 0.2] if proportion else ['80%', '60%', '40%', '20%']

    annotations.extend([dict(showarrow=False,  # annotations for ticklabels on side 0
                             text=f'{ticklabel[j]}',
                             x=posx[j],
                             y=posy[j],
                             align='center',
                             xanchor='center',
                             yanchor='top',
                             font=dict(size=12)) for j in range(4)])

    annotations.extend([dict(showarrow=False,  # annotations for ticklabels on  side 1
                             text=f'{ticklabel[j]}',
                             x=posx[j + 4],
                             y=posy[j + 4],
                             align='center',
                             xanchor='left',
                             yanchor='middle',
                             font=dict(size=12)) for j in range(4)])

    annotations.extend([dict(showarrow=False,  # annotations for ticklabels on side 2
                             text=f'{ticklabel[j]}',
                             x=posx[j + 8],
                             y=posy[j + 8],
                             align='center',
                             xanchor='right',
                             yanchor='middle',
                             font=dict(size=12)) for j in range(4)])
    return annotations


def styling_traces(xt, yt):
    # global xt, yt
    side_trace = dict(type='scatter',
                      x=[0.5, 0, 1, 0.5],
                      y=[np.sqrt(3) / 2, 0, 0, np.sqrt(3) / 2],
                      mode='lines',
                      line=dict(width=2, color='#444444'),
                      hoverinfo='none')

    tick_trace = dict(type='scatter',
                      x=xt,
                      y=yt,
                      mode='lines',
                      line=dict(width=1, color='#444444'),
                      hoverinfo='none')

    return side_trace, tick_trace

LOAD = 0.5
PV = 0.5

data_voltage = load_data(load=LOAD, pv=PV, quant_value="90")

A = data_voltage["cloudy"].values
C = data_voltage["sunny"].values
B = data_voltage["overcast"].values
z = data_voltage["voltage"].values

pl_ternary = dict(type='scatterternary',
                  a=A,
                  b=B,
                  c=C,
                  mode='markers',
                  marker=dict(size=10, color='red'))

layout = dict(width=500, height=400,
              ternary={'sum': 1,
                       'aaxis': {'title': 'a',  'min': 0.001, 'linewidth': LINE_WIDTH_LAYOUT, 'ticks':'outside'},
                       'baxis': {'title': 'b',  'min': 0.001, 'linewidth': LINE_WIDTH_LAYOUT, 'ticks':'outside'},
                       'caxis': {'title': 'c',  'min': 0.001, 'linewidth': LINE_WIDTH_LAYOUT, 'ticks':'outside'}},
              showlegend=False,
              paper_bgcolor='#EBF0F8')

# fw = go.FigureWidget(data=[pl_ternary], layout=layout)
# fw.write_html('outliers_highlight.html', auto_open=True)

M, invM =  tr_b2c2b()
cartes_coord_points = np.einsum('ik, kj -> ij', M, np.stack((A, B, C)))
xx, yy = cartes_coord_points[:2]
a, b = xx.min(), xx.max()
c, d = yy.min(), yy.max()

N=150
gr_x = np.linspace(a,b, N)
gr_y = np.linspace(c,d, N)
grid_x, grid_y = np.meshgrid(gr_x, gr_y)

#interpolate data (cartes_coords[:2].T; z) and evaluate the  interpolatory function at the meshgrid points to get grid_z
grid_z = griddata(cartes_coord_points[:2].T, z, (grid_x, grid_y), method='cubic')

# Go back to barycentric coordinates
bar_coords = np.einsum('ik, kmn -> imn', invM, np.stack((grid_x, grid_y, np.ones(grid_x.shape))))
bar_coords[np.where(bar_coords<0)] = None # invalidate the points outside of the reference triangle
xy1 = np.einsum('ik, kmn -> imn', M, bar_coords) # recompute back the cartesian coordinates of bar_coords with invalid
                                                 # positions and extract indices where x are nan

I = np.where(np.isnan(xy1[0]))
grid_z[I] = None

# tooltips for  proportions, i.e. a+b+c=1

t_names = {"a": "Cloudy",
           "b": "Overcast",
           "c": "Sunny",
           "z": "Voltage"}

t_proportions = [[f'{t_names["a"]}: {round(bar_coords[0][i,j], 2)}<br>{t_names["b"]}: {round(bar_coords[1][i,j], 2)}'+\
                  f'<br>{t_names["c"]}: {round(1-round(bar_coords[0][i,j], 2)-round(bar_coords[1][i,j], 2), 2)}'+\
                  f'<br>{t_names["z"]}: {round(grid_z[i,j],4)}'  if ~np.isnan(xy1[0][i,j]) else '' for j in range(N)]
                                       for i in range(N)]

# tooltips for  percents, i.e. a+b+c=100
t_percents=[[f'a: {int(100*bar_coords[0][i,j]+0.5)}<br>b: {int(100*bar_coords[1][i,j]+0.5)}'+\
             f'<br>c: {100-int(100*bar_coords[0][i,j]+0.5)-int(100*bar_coords[1][i,j]+0.5)}'+\
             f'<br>z: {round(grid_z[i,j],2)}'  if ~np.isnan(xy1[0][i,j]) else '' for j in range(N)]
                                         for i in range(N)]

pl_deep = [[0.0, 'rgb(253, 253, 204)'],
           [0.1, 'rgb(201, 235, 177)'],
           [0.2, 'rgb(145, 216, 163)'],
           [0.3, 'rgb(102, 194, 163)'],
           [0.4, 'rgb(81, 168, 162)'],
           [0.5, 'rgb(72, 141, 157)'],
           [0.6, 'rgb(64, 117, 152)'],
           [0.7, 'rgb(61, 90, 146)'],
           [0.8, 'rgb(65, 64, 123)'],
           [0.9, 'rgb(55, 44, 80)'],
           [1.0, 'rgb(39, 26, 44)']]

# Ternary contour plot that displays proportions
xt = []
yt = []
posx = []
posy = []
for side in [0, 1, 2]:
    cart_coord_ticks(side, M, xt, yt, posx, posy,  t=0.01)

tooltip = t_proportions
# layout = ternary_layout(width=600, height=525, vertex_text=[r"$\text{Cloudy}$", r"$\text{Dark}$", r"$\text{Sunny}$"])
layout = ternary_layout(title="Ternary plot", width=600, height=510, vertex_text=["Cloudy", "Overcast", "Sunny"])
# layout = ternary_layout(title="Ternary plot", width=int(600 * (2/3)), height=int(255 * (2/3)), vertex_text=["Cloudy", "Overcast", "Sunny"])
annotations = set_ticklabels(layout['annotations'], posx, posy, proportion=True)

norm_individual = mpl.colors.Normalize(vmin=CMIN_VMIN, vmax=CMAX_VMAX)
v_safe = norm_individual(1.045)
v_caution = norm_individual(1.05)

alpha = 0.9
green_color = f"rgba(127,191,127, {alpha})"
orange_color = f"rgba(255,192,76, {alpha})"
red_color = f"rgba(219,76,76, {alpha})"
colorscale = [(0.00,  green_color),   (v_safe, green_color),
              (v_safe, orange_color), (v_caution, orange_color),
              (v_caution, red_color),  (1.00, red_color)]

c_trace = contour_trace(gr_x, gr_y, grid_z, tooltip, linewidth=LINE_WIDTH_CONTOUR, colorscale=colorscale, reversescale=False, linecolor="rgb(0,0,0)")

side_trace, tick_trace = styling_traces(xt, yt)
fw1 = go.Figure(data=[c_trace, tick_trace, side_trace], layout=layout)
fw1.layout.annotations = annotations

fw1.update_coloraxes(showscale=False)
fw1.write_html('ternary_plot/outliers_highlight.html', auto_open=True)
fw1.write_image(f"ternary_load{int(LOAD*100)}_pv{int(PV*100)}.pdf")
fw1.write_image(f"ternary_load{int(LOAD*100)}_pv{int(PV*100)}.svg")
fw1.write_image(f"ternary_load{int(LOAD*100)}_pv{int(PV*100)}.png", width=300*7, height=300*7, scale=1)
