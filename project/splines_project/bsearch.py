def find_interval(nodes, target):
    """
    Retorna o índice i tal que nodes[i] <= target <= nodes[i + 1].

    Este método é usado em interpolação (por exemplo, splines cúbicos)
    para localizar o subintervalo [x_i, x_{i+1}] que contém o ponto `target`.
    Lança ValueError se o valor estiver fora do domínio [nodes[0], nodes[-1]].
    """
    total_nodes = len(nodes)
    if total_nodes < 2:
        raise ValueError("A sequência de nós deve ter pelo menos dois pontos.")

    first_node = nodes[0]
    last_node = nodes[-1]

    # Validação de domínio
    if target < first_node or target > last_node:
        raise ValueError("O valor está fora do domínio de interpolação.")

    # Caso especial: target exatamente no último nó
    if target == last_node:
        return total_nodes - 2  # intervalo [n-2, n-1]

    left_index = 0
    right_index = total_nodes - 1

    while right_index - left_index > 1:
        middle_index = (left_index + right_index) // 2

        if target >= nodes[middle_index]:
            left_index = middle_index
        else:
            right_index = middle_index

    return left_index

if __name__ == "__main__":
    nodes = [-1.0, -0.5, 0.25, 0.5]
    query_points = [-1.0, -0.75, -0.5, -0.1, 0.25, 0.3, 0.5]

    for point in query_points:
        interval_index = find_interval(nodes, point)
        left, right = nodes[interval_index], nodes[interval_index + 1]
        print(f"{point:>5.2f} → intervalo [{left}, {right}]")
