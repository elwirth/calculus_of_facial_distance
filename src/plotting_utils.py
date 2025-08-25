
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import griddata
import os


def plot_smooth(grid_points: np.ndarray, distances: np.ndarray, y: np.ndarray, resolution=1250, offset: float = 0.1,
                title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> plt.Figure:
    if grid_points.shape[0] != 2:
        raise ValueError("Plotting only supported for 2D polytopes.")
    
    xi = np.linspace(grid_points[0].min() - offset, grid_points[0].max() + offset, resolution)
    yi = np.linspace(grid_points[1].min() - offset, grid_points[1].max() + offset, resolution)
    XI, YI = np.meshgrid(xi, yi)
    
    ZI = griddata(grid_points.T, distances, (XI, YI), method='linear')
    fig, ax = plt.subplots(figsize=(6, 5))
    
    finite_vals = ZI[np.isfinite(ZI)]
    min_val = finite_vals.min()
    max_val = finite_vals.max()
    
    cf = ax.contourf(XI, YI, ZI, levels=np.linspace(min_val, max_val, 500), cmap='viridis')
    
    legend_title = f"{title} to y" if title else "Vertex distance to y"
    cbar = plt.colorbar(cf, ax=ax, label=legend_title, ticks=np.linspace(min_val, max_val, 5))
    
    ax.scatter(y[0], y[1], color='red', marker='x', s=100, label='y')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    return fig


def plot_delaunay(grid_points: np.ndarray, distances: np.ndarray, y: np.ndarray, offset: float = 0.1,
                  title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> plt.Figure:
    if grid_points.shape[0] != 2:
        raise ValueError("Delaunay plotting only supported for 2D polytopes.")

    triang = mtri.Triangulation(grid_points[0], grid_points[1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    finite_vals = distances[np.isfinite(distances)]
    min_val = finite_vals.min()
    max_val = finite_vals.max()
    
    cf = ax.tricontourf(triang, distances, levels=np.linspace(min_val, max_val, 500), cmap='viridis')
    
    legend_title = f"{title} to y" if title else "Vertex distance to y"
    cbar = plt.colorbar(cf, ax=ax, label=legend_title, ticks=np.linspace(min_val, max_val, 5))
    
    x_min, x_max = grid_points[0].min(), grid_points[0].max()
    y_min, y_max = grid_points[1].min(), grid_points[1].max()
    ax.set_xlim(x_min - offset, x_max + offset)
    ax.set_ylim(y_min - offset, y_max + offset)
    
    ax.scatter(grid_points[0], grid_points[1], c=distances, s=5, zorder=2, cmap='viridis')
    ax.scatter(y[0], y[1], color='red', marker='x', s=100, label='y', zorder=3)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
    print(f"Plotting {name}.")
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

