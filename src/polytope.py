
import numpy as np
from scipy.optimize import linprog
from itertools import product, combinations
from joblib import Parallel, delayed
import multiprocessing

class Polytope:
    def __init__(self, vertices: np.ndarray):
        """
        vertices: (dim, n_vertices) array, each column is a vertex
        """
        self.vertices = vertices
        self.dim, self.n_vertices = vertices.shape
    
    def membership_oracle(self, point: np.ndarray, epsilon = 0) -> tuple[bool, object]:
        """
        Check if `point` is inside the polytope.

        Parameters
        ----------
        point : (dim,) array
        epsilon : float
            Minimum lambda bound; default 0 (no positivity constraint)

        Returns
        -------
        tuple
            (inside: bool, lp_result: LPResult)
        """

        # We only care about feasibility, objective of the LP is irrelevant.
        c = np.zeros(self.n_vertices)

        # Equality constraints: sum_i lambda_i * v_i = point, last row is sum lambda_i = 1 constraint
        A_eq = np.vstack([self.vertices, np.ones(self.n_vertices)])
        b_eq = np.append(point, 1)

        # Bounds for lambda_i >= 0 + epsilon
        bounds = [(epsilon, None) for _ in range(self.n_vertices)]

        # Solve LP
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        # Feasible solution means point is inside
        return (res.success, res)

    def support_set_oracle(self, point: np.ndarray, epsilon = 1e-6) -> list[list[int]]:
        """
        Return all subsets of vertices S such that `point` has a convex representation
        over S with strictly positive coefficients (> epsilon).
        """
        support_sets = []

        for k in range(1, self.n_vertices + 1):
            for indices in combinations(range(self.n_vertices), k):
                subset_vertices = self.vertices[:, indices]
                contained, _ = Polytope(subset_vertices).membership_oracle(point, epsilon=epsilon)
                if contained:
                    support_sets.append(list(indices))

        return support_sets
    
    def gamma_feasible(self, x: np.ndarray, y: np.ndarray, support_indices: list[int], gamma: float) -> tuple[bool, object]:
        """
        Check if there exist u ∈ conv(S) and v ∈ conv(V) such that y - x = gamma * (v - u)
        """
        V = self.vertices                      # v ∈ conv(V)
        S = self.vertices[:, support_indices]  # u ∈ conv(S)
        
        n_V = V.shape[1]
        n_S = S.shape[1]

        # Variables: first lambda_i for V, then mu_j for S
        c = np.zeros(n_V + n_S)

        # Equality: gamma * v - gamma * u = y - x
        A_eq = np.hstack([gamma * V, -gamma * S])
        b_eq = y - x

        # Add sum(lambda_i) = 1 and sum(mu_j) = 1
        sum_constraints = np.zeros((2, n_V + n_S))
        sum_constraints[0, :n_V] = 1
        sum_constraints[1, n_V:] = 1
        A_eq = np.vstack([A_eq, sum_constraints])
        b_eq = np.append(b_eq, [1, 1])

        # Bounds: lambda_i >= 0, mu_j >= 0
        bounds = [(0, None)] * n_V + [(0, None)] * n_S

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        return (res.success, res)

    def maximal_support_sets(support_sets: list[list[int]]) -> list[list[int]]:
        """
        Return only support sets that are not proper subsets of any other support set.
        """
        maximal_sets = []
        for S in support_sets:
            if not any(set(S) < set(T) for T in support_sets if S != T):
                maximal_sets.append(S)
        return maximal_sets

    def vertex_distance(self, x: np.ndarray, y: np.ndarray, epsilon=1e-6, tol=1e-8, gamma_max=1, max_iter=50):
        """
        Compute the vertex distance ν(y, x) = max_{S in S(x)} min { γ ≥ 0 : γ feasible for S }.
        """
        if np.allclose(x, y):
            return 0.0
        # print("computing support sets S(x)...")
        support_sets = self.support_set_oracle(x, epsilon=epsilon)
        # print(f"Found {len(support_sets)} support sets: {support_sets}")
        max_gamma = 0.0

        for S in support_sets:
            # Bisection search for minimal feasible gamma
            low, high = 0.0, gamma_max
            feasible_gamma = gamma_max

            for _ in range(max_iter):
                mid = (low + high) / 2
                feasible, _ = self.gamma_feasible(x, y, S, mid)
                if feasible:
                    feasible_gamma = mid
                    high = mid
                else:
                    low = mid

                if high - low < tol:
                    break

            max_gamma = max(max_gamma, feasible_gamma)

        return max_gamma
    
    def convex_grid(self, step=0.1, dtype=np.float32) -> np.ndarray:
        """
        Generate a grid of points in conv(V) by sampling convex combinations.
        NOTE: It is important to sample like this. We really want to explore convex combinations of vertices with other vertices set to
        0 explicitly to get correct results.
        Returns: (dim, n_points)
        """
        steps = np.arange(0, 1 + step, step)
        grid_points = []

        for weights in product(steps, repeat=self.n_vertices):
            weights = np.array(weights)
            if np.isclose(weights.sum(), 1):
                grid_points.append(self.vertices @ weights)

        return np.array(grid_points, dtype=dtype).T

    def vertex_distance_grid(self, y: np.ndarray, step=0.1, epsilon=1e-6, tol=1e-8, gamma_max=1,
                            parallel=False, num_cores=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute vertex distances for all grid points in the polytope to a fixed y.
        If `parallel=True`, computations are done in parallel using `num_cores` (defaults to all cores).
        Returns: (grid_points (dim, n_points), distances (n_points,))
        """
        grid_points = self.convex_grid(step=step)

        if parallel:
            if num_cores is None:
                num_cores = multiprocessing.cpu_count()
            distances = Parallel(n_jobs=num_cores)(
                delayed(self.vertex_distance)(x, y, epsilon=epsilon, tol=tol, gamma_max=gamma_max)
                for x in grid_points.T
            )
            distances = np.array(distances)
        else:
            distances = np.array([self.vertex_distance(x, y, epsilon=epsilon, tol=tol, gamma_max=gamma_max)
                                for x in grid_points.T])

        return grid_points, distances