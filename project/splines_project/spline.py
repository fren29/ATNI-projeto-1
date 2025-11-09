import numpy as np
from gauss import solve_by_gaussian_elimination


def build_tridiagonal_system(x, y, bc="natural", ypp0=0.0, yppn=0.0):
    """
    Monta o sistema tridiagonal T·M = d para o spline cúbico interpolador.

    Parâmetros
    ----------
    x, y : sequências de floats com o mesmo tamanho (n+1)
    bc   : "natural"  -> impõe M0 = 0 e Mn = 0
           "complete" -> impõe derivadas primeiras nas bordas:
                         S'(a)=ypp0, S'(b)=yppn  (ypp0, yppn = f'(a), f'(b))
    ypp0, yppn : floats
        Se bc="complete", são f'(a) e f'(b). Ignorados em "natural".

    Retorna
    -------
    T : list[list[float]]  (matriz (n+1)×(n+1))
    d : list[float]        (vetor (n+1))
    """
    n = len(x) - 1
    if n < 1 or len(y) != len(x):
        raise ValueError("x e y devem ter o mesmo tamanho e conter ao menos dois pontos.")

    # passos h_i
    h = [x[i+1] - x[i] for i in range(n)]
    if any(hi <= 0 for hi in h):
        raise ValueError("x deve ser estritamente crescente.")

    T = [[0.0]*(n+1) for _ in range(n+1)]
    d = [0.0]*(n+1)

    # --- Condições de contorno ---
    if bc == "natural":
        # M0 = 0, Mn = 0
        T[0][0] = 1.0
        T[n][n] = 1.0
        d[0] = 0.0
        d[n] = 0.0

    elif bc == "complete":
        # Bordas com derivadas S'(a)=ypp0 e S'(b)=yppn
        # 2*h0*M0 + h0*M1 = 6[(y1-y0)/h0 - S'(a)]
        T[0][0] = 2.0 * h[0]
        T[0][1] = 1.0 * h[0]
        d[0]    = 6.0 * ((y[1] - y[0]) / h[0] - float(ypp0))

        # hn-1*Mn-1 + 2*hn-1*Mn = 6[S'(b) - (yn - y{n-1})/hn-1]
        T[n][n-1] = 1.0 * h[-1]
        T[n][n]   = 2.0 * h[-1]
        d[n]      = 6.0 * (float(yppn) - (y[-1] - y[-2]) / h[-1])

    else:
        raise ValueError("bc deve ser 'natural' ou 'complete'.")

    # --- Equações internas (i = 1..n-1) ---
    for i in range(1, n):
        him1 = h[i-1]
        hi   = h[i]
        denom = him1 + hi

        mu  = hi   / denom   # coeficiente de M_{i-1}
        lam = him1 / denom   # coeficiente de M_{i+1}
        rhs = 6.0 * ( (y[i+1] - y[i]) / hi - (y[i] - y[i-1]) / him1 ) / denom

        T[i][i-1] = mu
        T[i][i]   = 2.0
        T[i][i+1] = lam
        d[i] = rhs

    return T, d



def compute_M(T, d, solver=solve_by_gaussian_elimination):
    """
    Resolve o sistema linear T·M = d para obter o vetor de segundas derivadas M.

    Origem: Algoritmo 1 — Eliminação de Gauss (PDF, p. 4).

    Parâmetros
    ----------
    T : list[list[float]]
        Matriz quadrada (n+1)x(n+1) tridiagonal.
    d : list[float]
        Vetor (n+1) do termo independente.
    solver : callable
        Função para resolver sistema linear Ax = b.
        Padrão: solve_by_gaussian_elimination do módulo gauss.

    Retorno
    -------
    M : list[float]
        Vetor solução (segundas derivadas nos nós x_i).
    """
    # Copia profunda para não alterar T nem d originais
    A = [row[:] for row in T]
    b = d[:]
    n = len(A)

    # Sanidade básica
    for i in range(n):
        if len(A[i]) != n:
            raise ValueError("Matriz T deve ser quadrada (n x n).")
    if len(b) != n:
        raise ValueError("Dimensão inconsistente entre T e d.")

    # Chama o solver definido
    M = solver(A, b)

    # Garantir formato lista de floats
    M = [float(val) for val in M]
    return M

def compute_AB(x, y, M):
    """
    Calcula os coeficientes A_i e B_i do spline cúbico interpolador.

    Origem: Eq. (3) — PDF (p. 4).

    Para cada i = 0..n-1, com h_i = x_{i+1} - x_i:
        A_i = (M_{i+1} - M_i) / (6*h_i)
        B_i = M_i / 2

    Parâmetros
    ----------
    x, y : listas (n+1) de floats
    M : lista (n+1) de floats, segundas derivadas

    Retorno
    -------
    A, B : listas (n) de floats
    """
    n = len(x) - 1
    if not (len(y) == len(x) == len(M)):
        raise ValueError("x, y e M devem ter o mesmo comprimento.")

    h = [x[i+1] - x[i] for i in range(n)]
    if any(hi <= 0 for hi in h):
        raise ValueError("x deve ser estritamente crescente.")

    A, B = [], []
    for i in range(n):
        hi = h[i]
        Ai = (M[i+1] - M[i]) / (6.0 * hi)
        Bi = M[i] / 2.0
        A.append(Ai)
        B.append(Bi)

    return A, B


from bsearch import find_interval

def spline_eval(x, y, M, A, B, x_star, locator=find_interval):
    n = len(x) - 1

    if not (len(y) == len(M) == len(x)):
        raise ValueError("x, y e M devem ter o mesmo comprimento.")
    if len(A) != n or len(B) != n:
        raise ValueError("A e B devem ter n elementos (n = len(x)-1).")

    # Vetorização: aceita escalar ou array
    x_star = np.atleast_1d(x_star)
    results = []

    for xx in x_star:
        if not (x[0] <= xx <= x[-1]):
            raise ValueError(f"x_star={xx} fora do domínio [{x[0]}, {x[-1]}].")

        i = locator(x, xx)
        xi, xi1 = x[i], x[i+1]
        yi, yi1 = y[i], y[i+1]
        Mi, Mi1 = M[i], M[i+1]
        hi = xi1 - xi

        term1 = Mi * (xi1 - xx)**3 / (6.0 * hi)
        term2 = Mi1 * (xx - xi)**3 / (6.0 * hi)
        term3 = (yi / hi - Mi * hi / 6.0) * (xi1 - xx)
        term4 = (yi1 / hi - Mi1 * hi / 6.0) * (xx - xi)

        results.append(term1 + term2 + term3 + term4)

    results = np.array(results)
    # Retorna escalar se entrada era escalar
    return results[0] if results.size == 1 else results


def spline_function(x, y, bc="natural", ypp0=0.0, yppn=0.0, solver=solve_by_gaussian_elimination):
    """
    Constrói e retorna uma função callable S(x_star) que avalia o spline cúbico interpolador.

    Junta todos os passos:
        1. Monta o sistema T·M = d (Eq. 2 – PDF, p. 3)
        2. Resolve via eliminação de Gauss (Alg. 1 – PDF, p. 4)
        3. Calcula A_i e B_i (Eq. 3 – PDF, p. 4)
        4. Avalia S(x*) conforme Eq. 1 (PDF, p. 2)

    Parâmetros
    ----------
    x, y : listas de floats
        Pontos de interpolação (mesmo tamanho)
    bc : str
        Condição de contorno ("natural" ou "complete")
    ypp0, yppn : floats
        Segundas derivadas nos extremos (usadas se bc="complete")
    solver : callable
        Função de resolução de sistemas lineares (padrão: gauss.solve_by_gaussian_elimination)

    Retorno
    -------
    S : callable
        Função S(x_star) que avalia o spline cúbico interpolador.
    """
    # Passos fundamentais
    T, d = build_tridiagonal_system(x, y, bc, ypp0, yppn)
    M = compute_M(T, d, solver)
    A, B = compute_AB(x, y, M)

    def S(x_star):
        return spline_eval(x, y, M, A, B, x_star)

    return S



# Exemplo pequeno
from spline import spline_function

x = [0.0, 1.0, 2.0]
y = [0.0, 1.0, 0.0]

S = spline_function(x, y)

for x_star in [0.0, 0.5, 1.0, 1.5, 2.0]:
    print(f"S({x_star}) = {S(x_star):.6f}")
