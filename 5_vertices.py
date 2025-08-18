import numpy as np
import sys
from pathlib import Path
sys.path.append((str(Path.cwd() / "src")))
from polytope import Polytope
from plotting_utils import save_plot_and_data_simple, plot_facial_distances_delaunay

# set up the polytope, columns are vertices
vertices = np.array([[0, 2, 1, 0, 2.5],
                    [0, 0, 2, 2, 0.9]])
vertices = vertices.astype(np.float32)
poly = Polytope(vertices)

# define the point y for the vertex distance
y = np.array([2.5, 0.9])

# run the vertex distance computation
grid, dists = poly.vertex_distance_grid(y, num_steps=150, epsilon=1e-5, tol=1e-7, parallel=True, num_cores=10)
fig = plot_facial_distances_delaunay(grid, dists, y)
save_plot_and_data_simple(fig, vertices, y, name="5_vertices")


# y relint
y = np.array([2.3, 0.7])

# run the vertex distance computation
grid, dists = poly.vertex_distance_grid(y, num_steps=150, epsilon=1e-5, tol=1e-7, parallel=True, num_cores=10)
fig = plot_facial_distances_delaunay(grid, dists, y)
save_plot_and_data_simple(fig, vertices, y, name="5_vertices_y_relint")