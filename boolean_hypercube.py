import numpy as np
import sys
from pathlib import Path

# access the code
sys.path.append((str(Path.cwd() / "src")))
from polytope import Polytope
from plotting_utils import save_plot_and_data_simple, plot_facial_distances_smooth


vertices = np.array([[0, 1, 1, 0],
                    [0, 0, 1, 1]])
vertices = vertices.astype(np.float32)
poly = Polytope(vertices)

y = np.array([0, 0])
grid, dists = poly.vertex_distance_grid(y, step=0.1, epsilon=1e-5, tol=1e-7, parallel=True, num_cores=10)
fig = plot_facial_distances_smooth(grid, dists, y)
save_plot_and_data_simple(fig, vertices, y, name="boolean_hypercube")