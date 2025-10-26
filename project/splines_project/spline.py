def build_tridiagonal_system(x, y, bc="natural", ypp0=0.0, yppn=0.0):
    """Monta T*M = d do spline cúbico."""
    # TODO
    pass

def compute_M(T, d, solver):
    """Usa solver(A,b)->x para obter M."""
    # TODO
    pass

def compute_AB(x, y, M):
    """Calcula coeficientes A_i e B_i para cada intervalo."""
    # TODO
    pass

def spline_eval(x, y, M, A, B, x_star, locator):
    """Avalia S(x_star) usando a busca binária."""
    # TODO
    pass

def spline_function(x, y, bc="natural", ypp0=0.0, yppn=0.0, solver=None, locator=None):
    """Retorna um callable S(x_star)."""
    # TODO
    pass
