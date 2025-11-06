def solve_by_gaussian_elimination(coeff_matrix, rhs_vector):
    """
    Resolve o sistema linear A·x = b usando eliminação de Gauss
    com pivoteamento parcial (troca de linhas quando o pivô é nulo ou pequeno).

    Parâmetros
    ----------
    coeff_matrix : list[list[float]]
        Matriz quadrada de coeficientes A.
    rhs_vector : list[float]
        Vetor de termos independentes b.

    Retorna
    -------
    list[float]
        Vetor solução x.

    Observações
    ------------
    O método segue dois estágios:
      1. Eliminação direta (triangularização com pivoteamento parcial).
      2. Retro-substituição (resolução do sistema triangular resultante).

    Este algoritmo é estável para a maioria das matrizes não singulares,
    e lança ValueError caso a matriz seja singular (pivô nulo em toda a coluna).
    """
    n = len(coeff_matrix)
    A = [row[:] for row in coeff_matrix]  # cópia profunda da matriz
    b = rhs_vector[:]                     # cópia do vetor de termos independentes

    EPSILON = 1e-12  # tolerância para tratar pivôs "quase nulos"

    # --- Etapa 1: Eliminação direta com pivoteamento parcial ---
    for pivot_index in range(n - 1):
        # 1) Escolhe pivô — se for zero ou muito pequeno, faz troca de linhas
        if abs(A[pivot_index][pivot_index]) < EPSILON:
            for k in range(pivot_index + 1, n):
                if abs(A[k][pivot_index]) > EPSILON:
                    # troca linhas pivot_index <-> k
                    A[pivot_index], A[k] = A[k], A[pivot_index]
                    b[pivot_index], b[k] = b[k], b[pivot_index]
                    break
            else:
                raise ValueError("Matriz singular — pivô nulo em toda a coluna")

        pivot_value = A[pivot_index][pivot_index]

        # 2) Elimina elementos abaixo do pivô
        for target_row in range(pivot_index + 1, n):
            elimination_factor = A[target_row][pivot_index] / pivot_value

            # Atualiza linha da matriz
            for col in range(pivot_index, n):
                A[target_row][col] -= elimination_factor * A[pivot_index][col]

            # Atualiza o termo independente correspondente
            b[target_row] -= elimination_factor * b[pivot_index]

    # --- Etapa 2: Retro-substituição ---
    solution = [0.0] * n

    for row in range(n - 1, -1, -1):
        # soma parcial dos termos já resolvidos
        accumulated_sum = sum(A[row][col] * solution[col] for col in range(row + 1, n))
        diagonal_value = A[row][row]

        if abs(diagonal_value) < EPSILON:
            raise ValueError(f"Pivô nulo na retro-substituição (linha {row})")

        solution[row] = (b[row] - accumulated_sum) / diagonal_value

    return solution

if __name__ == "__main__":
    A = [
        [0.0, 2.0, 9.0],   # note que o pivô inicial (A[0][0]) é 0 -> força pivoteamento
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
    ]
    b = [7.0, 8.0, -11.0]

    x = solve_by_gaussian_elimination(A, b)
    print("Solução:", x)