import ternary


# Load some data, tuples (x,y,z)
points = []
with open("sample_data/curve.txt") as handle:
    for line in handle:
        points.append(list(map(float, line.split(' '))))

## Sample trajectory plot
figure, tax = ternary.figure(scale=1.0)
figure.set_size_inches(5, 5)

tax.boundary()
tax.gridlines(multiple=0.2, color="black")
tax.set_title("Plotting of sample trajectory data\n", fontsize=10)

# Plot the data
tax.plot(points, linewidth=2.0, label="Curve")
tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.legend()
tax.show()


#%%

import matplotlib.pyplot as plt
import math

def color_point(x, y, z, scale):
    w = 255
    x_color = x * w / float(scale)
    y_color = y * w / float(scale)
    z_color = z * w / float(scale)
    r = math.fabs(w - y_color) / w
    g = math.fabs(w - x_color) / w
    b = math.fabs(w - z_color) / w
    return (r, g, b, 1.)


def generate_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, scale)
    return d

scale = 90
data = generate_heatmap_data(scale)
figure, tax = ternary.figure(scale=scale)
tax.heatmap(data, style="hexagonal", use_rgba=True, colorbar=False)
# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.boundary()
tax.set_title("RGBA Heatmap")
plt.show()


#%%

## Generate Data
import random

def random_points(num_points=25, scale=40):
    points = []
    for i in range(num_points):
        x = random.randint(1, scale)
        y = random.randint(0, scale - x)
        z = scale - x - y
        points.append((x,y,z))
    return points

# Scatter Plot
scale = 40
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 10)
# Plot a few different styles with a legend
points = random_points(30, scale=scale)
tax.scatter(points, marker='s', color='red', label="Red Squares")
points = random_points(30, scale=scale)
tax.scatter(points, marker='D', color='green', label="Green Diamonds")
tax.legend()

tax.set_title("Scatter Plot", fontsize=20)
tax.boundary(linewidth=2.0)
tax.gridlines(multiple=5, color="blue")
tax.ticks(axis='lbr', linewidth=1, multiple=5)
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')

tax.show()


#%%
import math

def shannon_entropy(p):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1. * s

scale = 60
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 8)
tax.heatmapf(shannon_entropy, boundary=True, style="triangular")
tax.boundary(linewidth=2.0)
tax.set_title("Shannon Entropy Heatmap")
tax.ticks(axis='lbr', linewidth=1, multiple=5)
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.show()

#%%
import random

def generate_random_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i,j,k) in simplex_iterator(scale):
        d[(i,j)] = random.random()
    return d

scale = 20
d = generate_random_heatmap_data(scale)
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 8)
tax.heatmap(d, style="h")
tax.boundary()
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.set_title("Heatmap Test: Hexagonal")





#%%

import ternary
import matplotlib.pyplot as plt


# Function to visualize for heat map
def f(x):
    return 1.0 * x[0] / (1.0 * x[0] + 0.2 * x[1] + 0.05 * x[2])

# Dictionary of axes colors for bottom (b), left (l), right (r).
axes_colors = {'b': 'g', 'l': 'r', 'r': 'b'}

scale = 10

fig, ax = plt.subplots()
ax.axis("off")
figure, tax = ternary.figure(ax=ax, scale=scale)

tax.heatmapf(f, boundary=False,
             style="hexagonal", cmap=plt.cm.get_cmap('Blues'),
             cbarlabel='Component 0 uptake',
             vmax=1.0, vmin=0.0)

tax.boundary(linewidth=2.0, axes_colors=axes_colors)

tax.left_axis_label("$x_1$", offset=0.16, color=axes_colors['l'])
tax.right_axis_label("$x_0$", offset=0.16, color=axes_colors['r'])
tax.bottom_axis_label("$x_2$", offset=0.06, color=axes_colors['b'])

tax.gridlines(multiple=1, linewidth=2,
              horizontal_kwargs={'color': axes_colors['b']},
              left_kwargs={'color': axes_colors['l']},
              right_kwargs={'color': axes_colors['r']},
              alpha=0.7)

# Set and format axes ticks.
ticks = [i / float(scale) for i in range(scale+1)]
tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True,
          axes_colors=axes_colors, offset=0.03, tick_formats="%0.1f")

tax.clear_matplotlib_ticks()
tax._redraw_labels()
plt.tight_layout()
tax.show()


#%%

## Boundary and Gridlines
scale = 9
figure, tax = ternary.figure(scale=scale)
tax.ax.axis("off")
figure.set_facecolor('w')

# Draw Boundary and Gridlines
tax.boundary(linewidth=1.0)
tax.gridlines(color="black", multiple=1, linewidth=0.5,ls='-')

# Set Axis labels and Title
fontsize = 15
tax.left_axis_label("Barleygrow", fontsize=fontsize, offset=0.12)
tax.right_axis_label("Beans", fontsize=fontsize, offset=0.12)
tax.bottom_axis_label("Oats", fontsize=fontsize, offset=0.025)

# Set ticks
tax.ticks(axis='blr', linewidth=1,multiple=1)



# Scatter some points
points = [(2,3,5),(3,6,1),(5,4,1),(3,4,3),(2,2,6)]
c = [200,20,30,10,64]

cb_kwargs = {"shrink" : 0.6,
             "orientation" : "horizontal",
             "fraction" : 0.1,
             "pad" : 0.05,
             "aspect" : 30}

tax.scatter(points,marker='s',c=c,edgecolor='k',s=40,linewidths=0.5,
            vmin=0,vmax=100,colorbar=True,colormap='jet',cbarlabel='Farmers',
            cb_kwargs=cb_kwargs,zorder=3)


tax._redraw_labels()

# Color coded heatmap example with colorbar kwargs
# Slight modification so that we don't have to re-import pyplot
# but make use of ternary.plt


#%%

# Function to visualize for heat map
def f(x):
    return 1.0 * x[0] / (1.0 * x[0] + 0.2 * x[1] + 0.05 * x[2])

# dictionary of axes colors for bottom (b), left (l), right (r)
axes_colors = {'b': 'g', 'l': 'r', 'r':'b'}

scale = 10

figure, tax = ternary.figure(scale=scale)
tax.ax.axis("off")
cb_kwargs = {"shrink" : 0.6,
             "pad" : 0.05,
             "aspect" : 30,
             "orientation" : "horizontal"}

tax.heatmapf(f, boundary=False,
            style="hexagonal", cmap=ternary.plt.cm.get_cmap('Blues'),
            cbarlabel='Component 0 uptake',
            vmax=1.0, vmin=0.0, cb_kwargs=cb_kwargs)

tax.boundary(linewidth=2.0, axes_colors=axes_colors)

tax.left_axis_label("$x_1$", offset=0.16, color=axes_colors['l'])
tax.right_axis_label("$x_0$", offset=0.16, color=axes_colors['r'])
tax.bottom_axis_label("$x_2$", offset=-0.06, color=axes_colors['b'])

tax.gridlines(multiple=1, linewidth=2,
              horizontal_kwargs={'color':axes_colors['b']},
              left_kwargs={'color':axes_colors['l']},
              right_kwargs={'color':axes_colors['r']},
              alpha=0.7)

ticks = [round(i / float(scale), 1) for i in range(scale+1)]
tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True,
          axes_colors=axes_colors, offset=0.03)

tax.clear_matplotlib_ticks()
tax._redraw_labels()
ternary.plt.tight_layout()
ternary.plt.show()


























#









