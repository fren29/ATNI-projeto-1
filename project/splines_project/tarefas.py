import numpy as np
from spline import (
    build_tridiagonal_system,
    compute_M,
    compute_AB,
    spline_eval,
    spline_function
)
from utils import assert_strictly_increasing, make_uniform_mesh, sup_error
import numpy as np
from spline import spline_function

def tarefa_validar_pontos_exemplo():
    """
    Reproduz as Tabelas 1 e 2 do PDF (Prof. André Pierro, UFABC 2025).
    Valida a montagem do sistema tridiagonal, solução via Gauss e
    avaliação do spline cúbico natural nos pontos solicitados.
    """
    # --- Dados de exemplo (extraídos do PDF) -------------------------------
    x = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y = [-0.71736, -0.47943, 0.0, 0.47943, 0.71736]
    assert_strictly_increasing(x)

    # --- Montagem do sistema ----------------------------------------------
    T, d = build_tridiagonal_system(x, y, bc="natural")

    print("\nMatriz T:")
    for row in T:
        print(" ".join(f"{v:12.8f}" for v in row))

    print("\nVetor d:")
    print(" ".join(f"{v:12.8f}" for v in d))

    # --- Solução ----------------------------------------------------------
    M = compute_M(T, d)

    print("\nVetor M (segundas derivadas):")
    print(" ".join(f"{v:12.8f}" for v in M))

    # --- Coeficientes -----------------------------------------------------
    A, B = compute_AB(x, y, M)

    print("\nCoeficientes A e B:")
    for i, (Ai, Bi) in enumerate(zip(A, B)):
        print(f"i={i:2d}  A={Ai:12.8f}  B={Bi:12.8f}")

    # --- Avaliação do spline ---------------------------------------------
    xs_test = [-0.6, 0.25, 0.5]
    print("\nAvaliação do spline cúbico:")
    for x_star in xs_test:
        Sx = spline_eval(x, y, M, A, B, x_star)
        print(f"S({x_star:6.2f}) = {Sx:12.8f}")

#tarefa_validar_pontos_exemplo()

def tarefa_convergencia(f, a, b, ns, bc="natural"):
    """
    Estuda empiricamente a convergência do spline cúbico interpolador.

    Para cada n em ns:
      1. Gera malha uniforme [a,b] com n subintervalos.
      2. Constrói spline cúbico S_n(x) com condição bc.
      3. Calcula erro máximo E_n = max |f(x) - S_n(x)| em malha densa.
      4. Exibe tabela com (n, h, E_n).

    Parâmetros
    ----------
    f : callable
        Função original a interpolar.
    a, b : floats
        Intervalo de definição.
    ns : list[int]
        Tamanhos de malha (ex.: [4, 8, 16, 32, 64]).
    bc : str
        Condição de contorno ("natural" ou "complete").
    """
    print(f"\n=== Estudo de Convergência do Spline Cúbico ({bc}) ===")
    print(f"{'n':>6} {'h':>12} {'E_n':>16}")

    results = []
    for n in ns:
        xs = make_uniform_mesh(a, b, n)
        ys = f(xs)
        S = spline_function(xs.tolist(), ys.tolist(), bc=bc)

        # Malha densa para medir erro
        xs_dense = np.linspace(a, b, 2000)
        E_n = sup_error(f, S, xs_dense)

        h = (b - a) / n
        results.append((n, h, E_n))
        print(f"{n:6d} {h:12.6e} {E_n:16.8e}")

    return results

import numpy as np
from utils import make_uniform_mesh, sup_error
from spline import spline_function

def tarefa_convergencia_completa(f, df, a, b, ns):
    """
    Estuda empiricamente a convergência do spline cúbico completo,
    com derivadas exatas nas extremidades.

    Parâmetros
    ----------
    f : callable
        Função original.
    df : callable
        Derivada primeira exata de f.
    a, b : floats
        Intervalo de definição.
    ns : list[int]
        Números de subintervalos.
    """
    print(f"\n=== Estudo de Convergência do Spline Cúbico (completo) ===")
    print(f"{'n':>6} {'h':>12} {'E_n':>16}")

    results = []
    for n in ns:
        xs = make_uniform_mesh(a, b, n)
        ys = f(xs)
        S = spline_function(xs.tolist(), ys.tolist(),
                            bc="complete",
                            ypp0=df(a),
                            yppn=df(b))

        xs_dense = np.linspace(a, b, 2000)
        E_n = sup_error(f, S, xs_dense)

        h = (b - a) / n
        results.append((n, h, E_n))
        print(f"{n:6d} {h:12.6e} {E_n:16.8e}")

    return results

import numpy as np
import matplotlib.pyplot as plt

def ajuste_ordem_convergencia(results, titulo="Spline Cúbico"):
    """
    Estima numericamente a ordem de convergência ρ a partir de (h, E_n).

    Parâmetros
    ----------
    results : list of tuples (n, h, E_n)
        Saída das funções tarefa_convergencia ou tarefa_convergencia_completa.
    titulo : str
        Título do gráfico (opcional).

    Retorno
    -------
    rho : float
        Estimativa da ordem de convergência.
    """
    # Extrai vetores
    hs  = np.array([h for _, h, _ in results])
    Es  = np.array([E for _, _, E in results])

    logh = np.log(hs)
    logE = np.log(Es)

    # Ajuste linear: logE = α + ρ·logh
    A = np.vstack([logh, np.ones_like(logh)]).T
    rho, alpha = np.linalg.lstsq(A, logE, rcond=None)[0]

    # Exibe resultados
    print("\n=== Ajuste log–log de Convergência ===")
    print(f"ρ (ordem estimada) = {rho:8.4f}")
    print(f"Coeficiente linear  = {alpha:8.4f}")
    print(f"Relação estimada: log(E) ≈ {alpha:.4f} + {rho:.4f}·log(h)")

    # Plot log–log
    plt.figure(figsize=(6,4))
    plt.plot(logh, logE, "o", label="dados numéricos")
    plt.plot(logh, alpha + rho*logh, "-", label=f"ajuste linear (ρ={rho:.2f})")
    plt.xlabel("log(h)")
    plt.ylabel("log(E_n)")
    plt.title(f"Convergência {titulo}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rho

from tarefas import tarefa_convergencia_completa, ajuste_ordem_convergencia
import numpy as np

f  = np.cos
df = lambda x: -np.sin(x)
a, b = 0.0, np.pi / 2
ns = [4, 8, 16, 32, 64]

results = tarefa_convergencia_completa(f, df, a, b, ns)
rho = ajuste_ordem_convergencia(results, titulo="Spline Completo")
print(f"Ordem estimada: ρ ≈ {rho:.3f}")