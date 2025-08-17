
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from scipy.interpolate import griddata
import os

def plot_facial_distances(grid_points: np.ndarray, distances: np.ndarray, y: np.ndarray) -> plt.Figure:
    if grid_points.shape[0] != 2:
        raise ValueError("Plotting only supported for 2D polytopes.")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(grid_points[0], grid_points[1], c=distances, cmap='gray_r')
    plt.colorbar(sc, ax=ax, label='Vertex distance to y')
    ax.scatter(y[0], y[1], color='red', marker='x', s=100, label='y')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Facial distances over polytope')
    ax.legend()
    return fig


def plot_facial_distances_smooth(grid_points: np.ndarray, distances: np.ndarray, y: np.ndarray, resolution=100) -> plt.Figure:
    if grid_points.shape[0] != 2:
        raise ValueError("Plotting only supported for 2D polytopes.")
    
    xi = np.linspace(-0.05 + grid_points[0].min(), 0.05 + grid_points[0].max(), resolution)
    yi = np.linspace(-0.05 + grid_points[1].min(), 0.05 + grid_points[1].max(), resolution)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata(grid_points.T, distances, (XI, YI), method='cubic')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(XI, YI, ZI, levels=50, cmap='gray_r')
    plt.colorbar(cf, ax=ax, label='Vertex distance to y')
    ax.scatter(y[0], y[1], color='red', marker='x', s=100, label='y')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Smoothed facial distances over polytope')
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