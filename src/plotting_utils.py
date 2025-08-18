
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from itertools import product
from scipy.interpolate import griddata
import os


def plot_facial_distances_smooth(grid_points: np.ndarray, distances: np.ndarray, y: np.ndarray, resolution=100, offset: float = 0.1) -> plt.Figure:
    if grid_points.shape[0] != 2:
        raise ValueError("Plotting only supported for 2D polytopes.")
    
    xi = np.linspace(-offset + grid_points[0].min(), offset + grid_points[0].max(), resolution)
    yi = np.linspace(-offset + grid_points[1].min(), offset + grid_points[1].max(), resolution)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata(grid_points.T, distances, (XI, YI), method='cubic')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(XI, YI, ZI, levels=250, cmap='gray_r')
    cbar = plt.colorbar(cf, ax=ax, label='Vertex distance to y', ticks=np.arange(0, 1.1, 0.2))
    cbar.ax.set_ylim(0, 1)
    ax.scatter(y[0], y[1], color='red', marker='x', s=100, label='y')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Vertex distances over polytope')
    ax.legend()
    return fig


def plot_facial_distances_delaunay(grid_points: np.ndarray, distances: np.ndarray, y: np.ndarray, offset: float = 0.1) -> plt.Figure:
    if grid_points.shape[0] != 2:
        raise ValueError("Delaunay plotting only supported for 2D polytopes.")

    # Create a Delaunay triangulation
    triang = mtri.Triangulation(grid_points[0], grid_points[1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Use tricontourf to plot the filled contours
    cf = ax.tricontourf(triang, distances, levels=250, cmap='gray_r')
    cbar = plt.colorbar(cf, ax=ax, label='Vertex distance to y', ticks=np.arange(0, 1.1, 0.2))
    cbar.ax.set_ylim(0, 1)

    # Add the offset to the plot limits
    x_min, x_max = grid_points[0].min(), grid_points[0].max()
    y_min, y_max = grid_points[1].min(), grid_points[1].max()
    ax.set_xlim(x_min - offset, x_max + offset)
    ax.set_ylim(y_min - offset, y_max + offset)
    
    # Plot the original grid points and the target point y
    ax.scatter(grid_points[0], grid_points[1], c=distances, s=5, zorder=2, cmap='gray_r')
    ax.scatter(y[0], y[1], color='red', marker='x', s=100, label='y', zorder=3)
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Vertex distances over polytope')
    ax.legend()
    
    return fig

def save_plot_and_data_simple(fig: plt.Figure, vertices: np.ndarray, y: np.ndarray, name: str):
    """
    Save figure and related data (vertices and y only).

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to save.
    vertices : np.ndarray
        Array of shape (dim, n_vertices) with polytope vertices.
    y : np.ndarray
        Target point y.
    name : str
        Base name for saving files.
    """
    os.makedirs("results", exist_ok=True)
    
    # Save figure
    fig_path = os.path.join("results", f"{name}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    
    # Round vertices and y
    vertices_rounded = np.round(vertices, 3)
    y_rounded = np.round(y, 3).reshape(-1, 1)  # keep y as column
    
    # Save vertices and y
    txt_path = os.path.join("results", f"{name}.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Vertices (shape {vertices_rounded.shape}):\n")
        for row in vertices_rounded:
            f.write(" ".join(f"{val:.3f}" for val in row) + "\n")
        
        f.write(f"\n# Target y (shape {y_rounded.shape}):\n")
        for row in y_rounded:
            f.write(" ".join(f"{val:.3f}" for val in row) + "\n")
    
    print(f"Saved plot to {fig_path} and data to {txt_path}")

