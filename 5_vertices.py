import numpy as np
import sys
from pathlib import Path
sys.path.append((str(Path.cwd() / "src")))
from polytope import Polytope
from plotting_utils import save_plot_and_data_simple, plot_smooth

# set up the polytope, columns are vertices
vertices = np.array([[0, 2, 1, 0, 2.5],
                    [0, 0, 2, 2, 0.9]])
vertices = vertices.astype(np.float32)
poly = Polytope(vertices)

# define the point y for the vertex distance
y = np.array([2.5, 0.9])

# run the vertex distance computation
grid, dists = poly.vertex_distance_grid(y, num_steps=100, epsilon=1e-5, tol=1e-7, parallel=True, num_cores=10)
fig = plot_smooth(grid, dists, y, title="Vertex distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="5_vertices")


# y relint
y = np.array([2.3, 0.7])

# run the vertex distance computation
grid, dists = poly.vertex_distance_grid(y, num_steps=10, epsilon=1e-4, tol=1e-6, parallel=True, num_cores=10)
fig = plot_smooth(grid, dists, y, title="Vertex distance", xlable="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="5_vertices_y_relint")
