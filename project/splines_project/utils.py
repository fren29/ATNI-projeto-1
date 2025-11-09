import numpy as np


def assert_strictly_increasing(x):
    """
    Verifica se a sequência x é estritamente crescente.
    Lança ValueError se algum valor não obedecer a x[i] < x[i+1].
    """
    if any(x[i+1] <= x[i] for i in range(len(x) - 1)):
        raise ValueError("Os valores de x devem ser estritamente crescentes.")
    return True


def make_uniform_mesh(a, b, n):
    """
    Gera uma malha uniforme de n+1 pontos no intervalo [a, b].

    Parâmetros
    ----------
    a, b : floats
        Limites do intervalo.
    n : int
        Número de subintervalos (retorna n+1 nós).

    Retorno
    -------
    xs : numpy.ndarray
        Vetor de nós uniformemente espaçados.
    """
    if n <= 0:
        raise ValueError("O número de subintervalos n deve ser positivo.")
    if b <= a:
        raise ValueError("Requer a < b para gerar malha uniforme.")

    return np.linspace(a, b, n + 1)

def sup_error(f, g, xs):
    """||f - g||_∞ discreto nos pontos xs."""
    # TODO
    pass

# Teste 1: validação de ordem
x = [0, 1, 2, 3]
assert assert_strictly_increasing(x)

try:
    assert_strictly_increasing([0, 2, 2])
except ValueError:
    print("Erro detectado corretamente.")

# Teste 2: geração de malha uniforme
xs = make_uniform_mesh(0, 1, 4)
print(xs)
# Esperado: [0.   0.25 0.5  0.75 1.  ]