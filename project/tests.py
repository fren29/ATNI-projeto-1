# tests.py
from math import isclose

# ajuste os imports conforme a sua estrutura de pastas
from splines_project.bsearch import find_interval
from splines_project.gauss import solve_by_gaussian_elimination as solve

def almost_equal_list(a, b, tol=1e-10):
    return len(a) == len(b) and all(isclose(ai, bi, rel_tol=0, abs_tol=tol) for ai, bi in zip(a, b))

def test_bsearch_basic():
    nodes = [-1.0, -0.5, 0.25, 0.5]
    # bordas e meio
    assert find_interval(nodes, -1.0) == 0
    assert find_interval(nodes, -0.75) == 0
    assert find_interval(nodes, -0.5) == 1
    assert find_interval(nodes, 0.25) == 2
    assert find_interval(nodes, 0.3) == 2
    assert find_interval(nodes, 0.5) == 2  # último intervalo é [0.25, 0.5]

def test_bsearch_out_of_domain():
    nodes = [0.0, 1.0]
    try:
        find_interval(nodes, -0.1)
        assert False, "deveria ter lançado ValueError"
    except ValueError:
        pass
    try:
        find_interval(nodes, 1.1)
        assert False, "deveria ter lançado ValueError"
    except ValueError:
        pass

def test_gauss_2x2_easy():
    A = [[3, 2], [1, -1]]
    b = [5, -1]
    expected = [3/5, 8/5]  # [0.6, 1.6]
    x = solve(A, b)
    assert almost_equal_list(x, expected)

def test_gauss_2x2_pivot():
    A = [[0, 1], [2, 3]]
    b = [4, 5]
    expected = [-7/2, 4]
    x = solve(A, b)
    assert almost_equal_list(x, expected)

def test_gauss_3x3_classic():
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    expected = [2, 3, -1]
    x = solve(A, b)
    assert almost_equal_list(x, expected)

def test_gauss_3x3_pivot():
    A = [[0, 2, 9], [2, 1, -1], [-3, -1, 2]]
    b = [7, 8, -11]
    expected = [24/7, 11/7, 3/7]
    x = solve(A, b)
    assert almost_equal_list(x, expected, tol=1e-12)

def test_gauss_singular():
    A = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
    b = [6, 12, 3]
    try:
        _ = solve(A, b)
        assert False, "deveria ter lançado ValueError (matriz singular)"
    except ValueError:
        pass

def test_gauss_tridiagonal_4x4():
    A = [
        [4, -1, 0, 0],
        [-1, 4, -1, 0],
        [0, -1, 4, -1],
        [0, 0, -1, 3],
    ]
    b = [5, 5, 5, 5]
    expected = [280/153, 355/153, 125/51, 380/153]  # ≈ [1.83007, 2.32026, 2.45098, 2.48366]
    x = solve(A, b)
    assert almost_equal_list(x, expected, tol=1e-10)

def test_compute_M_basic():
    from splines_project.spline import build_tridiagonal_system, compute_M
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    T, d = build_tridiagonal_system(x, y)
    M = compute_M(T, d)
    assert all(abs(a - b) < 1e-12 for a, b in zip(M, [0.0, -3.0, 0.0]))


def test_compute_AB_basic():
    from splines_project.spline import build_tridiagonal_system, compute_M, compute_AB
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    T, d = build_tridiagonal_system(x, y)
    M = compute_M(T, d)
    A, B = compute_AB(x, y, M)
    assert all(abs(a - b) < 1e-12 for a, b in zip(A, [-0.5, 0.5]))
    assert all(abs(a - b) < 1e-12 for a, b in zip(B, [0.0, -1.5]))

def test_spline_eval_basic():
    from splines_project.spline import build_tridiagonal_system, compute_M, compute_AB, spline_eval
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    T, d = build_tridiagonal_system(x, y)
    M = compute_M(T, d)
    A, B = compute_AB(x, y, M)
    Sx = spline_eval(x, y, M, A, B, 0.5)
    assert abs(Sx - 0.6875) < 1e-6

def test_spline_function_basic():
    from splines_project.spline import spline_function
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 0.0]
    S = spline_function(x, y)
    # interpolação exata nos nós
    for xi, yi in zip(x, y):
        assert abs(S(xi) - yi) < 1e-12
    # ponto intermediário
    assert abs(S(0.5) - 0.6875) < 1e-6

def test_assert_strictly_increasing_valid():
    from splines_project.utils import assert_strictly_increasing
    assert assert_strictly_increasing([0, 1, 2, 3])

def test_assert_strictly_increasing_invalid():
    from splines_project.utils import assert_strictly_increasing
    import pytest
    with pytest.raises(ValueError):
        assert_strictly_increasing([0, 1, 1])

def test_make_uniform_mesh_basic():
    from splines_project.utils import make_uniform_mesh
    import numpy as np
    xs = make_uniform_mesh(0.0, 1.0, 4)
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(xs, expected)

def test_sup_error_cosine():
    import numpy as np
    from splines_project.utils import sup_error
    f = np.cos
    g = lambda x: np.cos(x) + 0.001
    xs = np.linspace(0, np.pi, 100)
    E = sup_error(f, g, xs)
    assert abs(E - 0.001) < 1e-6

def test_tarefa_validar_runs():
    from splines_project.tarefas import tarefa_validar_pontos_exemplo
    # A função deve rodar sem erros e imprimir resultados
    tarefa_validar_pontos_exemplo()


def run_all():
    # chama todas as funções que começam com test_
    import inspect, sys
    current_module = sys.modules[__name__]
    tests = [obj for name, obj in inspect.getmembers(current_module) if name.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"💥 {t.__name__}: exceção inesperada: {e}")
            failures += 1
    print(f"\nTotal: {len(tests)} testes, {failures} falhas.")
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    run_all()
