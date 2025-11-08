from gauss import solve_by_gaussian_elimination


def build_tridiagonal_system(x, y, bc="natural", ypp0=0.0, yppn=0.0):
    """
    Monta o sistema linear tridiagonal T·M = d do spline cúbico interpolador.

    Origem: Eq. (2) — PDF (λ_i, μ_i, d_i). Para i = 1..n-1 (n = len(x)-1):
        μ_i M_{i-1} + 2 M_i + λ_i M_{i+1} = d_i
        μ_i = h_i / (h_{i-1} + h_i),  λ_i = h_{i-1} / (h_{i-1} + h_i)
        d_i = 6 * [ ( (y_{i+1}-y_i)/h_i ) - ( (y_i - y_{i-1})/h_{i-1} ) ] / (h_{i-1}+h_i)

    Condições de contorno:
      - bc == "natural": M_0 = 0, M_n = 0 (ou valores dados por ypp0,yppn).
      - bc == "complete": usa M_0 = ypp0 e M_n = yppn (segunda derivada conhecida).
    Em ambos os casos, impomos linhas identidade nas bordas.

    Parâmetros
    ----------
    x, y : sequências de floats com mesmo tamanho (n+1)
    bc : "natural" ou "complete"
    ypp0, yppn : valores de M_0 e M_n quando especificados

    Retorno
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

    size = n + 1
    T = [[0.0]*(size) for _ in range(size)]
    d = [0.0]*(size)

    # bordas: identidade + valores conforme bc
    T[0][0] = 1.0
    T[n][n] = 1.0
    d[0] = 0.0 if bc == "natural" else float(ypp0)
    d[n] = 0.0 if bc == "natural" else float(yppn)
    if bc == "natural" and (ypp0 != 0.0 or yppn != 0.0):
        # permite natural com valores explícitos nas bordas
        d[0] = float(ypp0)
        d[n] = float(yppn)

    # equações internas i=1..n-1
    for i in range(1, n):
        him1 = h[i-1]
        hi = h[i]
        denom = him1 + hi
        mu = hi / denom
        lam = him1 / denom
        rhs = 6.0 * ( (y[i+1]-y[i])/hi - (y[i]-y[i-1])/him1 ) / denom

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
    """
    Avalia o spline cúbico interpolador em um ponto x_star.

    Origem: Eq. (1) — PDF (p. 2)

    Para o subintervalo [x_i, x_{i+1}] tal que x_i ≤ x_star ≤ x_{i+1}:
        S_i(x) = (M_i*(x_{i+1}-x)^3)/(6*h_i)
                + (M_{i+1}*(x-x_i)^3)/(6*h_i)
                + ((y_i/h_i) - (M_i*h_i)/6)*(x_{i+1}-x)
                + ((y_{i+1}/h_i) - (M_{i+1}*h_i)/6)*(x-x_i)

    Parâmetros
    ----------
    x, y, M, A, B : listas do spline
    x_star : float
        Ponto a avaliar
    locator : callable
        Função para localizar o índice i tal que x ∈ [x_i, x_{i+1}]
        Padrão: bsearch.find_interval

    Retorno
    -------
    Sx : float
        Valor do spline em x_star
    """
    n = len(x) - 1
    if not (len(y) == len(M) == len(x)):
        raise ValueError("x, y e M devem ter o mesmo comprimento.")
    if len(A) != n or len(B) != n:
        raise ValueError("A e B devem ter n elementos (n = len(x)-1).")
    if not (x[0] <= x_star <= x[-1]):
        raise ValueError(f"x_star={x_star} fora do domínio [{x[0]}, {x[-1]}].")

    i = locator(x, x_star)  # índice tal que x[i] ≤ x_star ≤ x[i+1]
    hi = x[i+1] - x[i]
    xi, xi1 = x[i], x[i+1]
    yi, yi1 = y[i], y[i+1]
    Mi, Mi1 = M[i], M[i+1]

    term1 = Mi * (xi1 - x_star)**3 / (6.0 * hi)
    term2 = Mi1 * (x_star - xi)**3 / (6.0 * hi)
    term3 = (yi / hi - Mi * hi / 6.0) * (xi1 - x_star)
    term4 = (yi1 / hi - Mi1 * hi / 6.0) * (x_star - xi)

    Sx = term1 + term2 + term3 + term4
    return Sx


def spline_function(x, y, bc="natural", ypp0=0.0, yppn=0.0, solver=None, locator=None):
    """Retorna um callable S(x_star)."""
    # TODO
    pass


# Exemplo pequeno
x = [0.0, 1.0, 2.0]
y = [0.0, 1.0, 0.0]
T, d = build_tridiagonal_system(x, y)
M = compute_M(T, d)
A, B = compute_AB(x, y, M)

for x_star in [0.0, 0.5, 1.0, 1.5, 2.0]:
    print(f"S({x_star}) = {spline_eval(x, y, M, A, B, x_star):.6f}")