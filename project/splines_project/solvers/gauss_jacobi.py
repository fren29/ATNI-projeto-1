import numpy as np
from typing import Tuple, Optional

from project.splines_project import utils
from project.splines_project.utils import verify_diagonal


def jacobi(A: np.ndarray,  b: np.ndarray,  x0: np.ndarray, tol: float = 1e-10, max_iter: int = 1000) -> Tuple[np.ndarray, int, list]:
     n = len(b)
     x_old = x0.copy()
     x_new = np.zeros(n)
     errors = []

     utils.verify_diagonal(A)

     for k in range(max_iter):
         for i in range(n):
             sum = 0.0
             for j in range(n):
                 if  j != i:
                     sum += A[i, j] * x_old[j]
             x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

         error = np.linalg.norm(x_new - x_old, ord=np.inf)
         errors.append(error)

         if error < tol:
             return x_new, k + 1, errors

         x_old = x_new.copy()

     return x_new, max_iter, errors
