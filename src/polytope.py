
import numpy as np
from scipy.optimize import linprog


class Polytope:
    def __init__(self, vertices: np.ndarray):
        """
        vertices: (dim, n_vertices) array, each column is a vertex
        """
        self.vertices = vertices
        self.dim, self.n_vertices = vertices.shape
    
    def membership_oracle(self, point: np.ndarray, epsilon = 0, tol=1e-8) -> tuple[bool, object]:
        """
        Check if `point` is inside the polytope.

        Parameters
        ----------
        point : (dim,) array
        epsilon : float
            Minimum lambda bound; default 0 (no positivity constraint)
        tol : float
            Tolerance for feasibility (ignored by default LP solver)

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

    def support_set_oracle(self, point: np.ndarray, epsilon = 1e-6, tol=1e-8) -> list[list[int]]:
        """
        Return all subsets of vertices S such that `point` has a convex representation
        over S with strictly positive coefficients (> epsilon).
        """
        from itertools import combinations
        support_sets = []

        for k in range(1, self.n_vertices + 1):
            for indices in combinations(range(self.n_vertices), k):
                subset_vertices = self.vertices[:, indices]
                contained, _ = Polytope(subset_vertices).membership_oracle(point, epsilon=epsilon, tol=tol)
                if contained:
                    support_sets.append(list(indices))

        return support_sets
    
    def max_step_oracle(self, point: np.ndarray, direction: np.ndarray, tol=1e-8, max_step=1e3) -> float:
        """
        Compute the largest step size `gamma` such that 
        `start + gamma * direction` remains inside the polytope.

        Parameters
        ----------
        point : (dim,) array
            Starting point inside the polytope.
        direction : (dim,) array
            Direction vector along which to move.
        tol : float
            Tolerance for bisection convergence.
        max_step : float
            Initial upper bound for step size.

        Returns
        -------
        float
            Maximum gamma >= 0 such that start + gamma * direction is in the polytope.
        """
        low, high = 0.0, max_step

        while high - low > tol:
            mid = (low + high) / 2
            candidate_point = point + mid * direction
            inside, _ = self.membership_oracle(candidate_point, tol=tol)
            if inside:
                low = mid  # Can go at least this far
            else:
                high = mid  # Too far, reduce
        return low