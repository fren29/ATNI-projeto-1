import numpy as np
from spline import (
    build_tridiagonal_system,
    compute_M,
    compute_AB,
    spline_eval
)
from utils import assert_strictly_increasing


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
tarefa_validar_pontos_exemplo()

def tarefa_convergencia(f, a, b, ns):
    """Estudo empírico de convergência."""
    # TODO
    pass
