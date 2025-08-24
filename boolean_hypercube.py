import numpy as np
import sys
from pathlib import Path
sys.path.append((str(Path.cwd() / "src")))
from polytope import Polytope
from plotting_utils import save_plot_and_data_simple, plot_delaunay, plot_smooth

# set up the polytope, columns are vertices
vertices = np.array([[0, 1, 1, 0],
                    [0, 0, 1, 1]])
vertices = vertices.astype(np.float32)
poly = Polytope(vertices)

# define the point y for the vertex distance
y = np.array([0, 0])

# run the vertex distance computation
grid, dists = poly.vertex_distance_grid(y, num_steps=50, epsilon=1e-4, tol=1e-6, parallel=True, num_cores=10)
fig = plot_smooth(grid, dists, y, title="Vertex distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="boolean_hypercube")