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

num_steps = 50
num_steps_line = 100
num_cores = 10
epsilon = 1e-5
tol = 1e-7


y = np.array([0, 0], dtype=np.float32)
grid, frag_dists, vert_dists = poly.fragmentation_grid(y, num_steps=num_steps, num_steps_line=num_steps_line, epsilon=epsilon, tol=tol, parallel=True, num_cores=num_cores)
fig = plot_smooth(grid, frag_dists, y, title="Fragmentation distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="fragmentation_distance_boolean_hypercube_vertex_1")
fig = plot_smooth(grid, vert_dists, y, title="Vertex distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="vertex_distance_boolean_hypercube_vertex_1")


y = np.array([0.2, 0.4], dtype=np.float32)
grid, frag_dists, vert_dists = poly.fragmentation_grid(y, num_steps=num_steps, num_steps_line=num_steps_line, epsilon=epsilon, tol=tol, parallel=True, num_cores=num_cores)
fig = plot_smooth(grid, frag_dists, y, title="Fragmentation distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="fragmentation_distance_boolean_hypercube_relint_1")
fig = plot_smooth(grid, vert_dists, y, title="Vertex distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="vertex_distance_boolean_hypercube_relint_1")

y = np.array([0.5, 0.5], dtype=np.float32)
grid, frag_dists, vert_dists = poly.fragmentation_grid(y, num_steps=num_steps, num_steps_line=num_steps_line, epsilon=epsilon, tol=tol, parallel=True, num_cores=num_cores)
fig = plot_smooth(grid, frag_dists, y, title="Fragmentation distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="fragmentation_distance_boolean_hypercube_relint_2")
fig = plot_smooth(grid, vert_dists, y, title="Vertex distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="vertex_distance_boolean_hypercube_relint_2")

y = np.array([0.2, 0.0], dtype=np.float32)
grid, frag_dists, vert_dists = poly.fragmentation_grid(y, num_steps=num_steps, num_steps_line=num_steps_line, epsilon=epsilon, tol=tol, parallel=True, num_cores=num_cores)
fig = plot_smooth(grid, frag_dists, y, title="Fragmentation distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="fragmentation_distance_boolean_hypercube_boundary_1")
fig = plot_smooth(grid, vert_dists, y, title="Vertex distance", xlabel="x1", ylabel="x2")
save_plot_and_data_simple(fig, vertices, y, name="vertex_distance_boolean_hypercube_boundary_1")