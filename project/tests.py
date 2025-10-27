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
