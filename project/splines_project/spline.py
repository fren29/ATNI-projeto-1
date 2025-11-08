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

x = [0.0, 1.0, 2.0]
y = [0.0, 1.0, 0.0]
T, d = build_tridiagonal_system(x, y, bc="natural")

print("T =")
for row in T: print(row)
print("d =", d)
