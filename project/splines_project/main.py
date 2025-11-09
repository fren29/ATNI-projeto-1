from project.splines_project.solvers import gauss_jacobi


def main():
    print("Selecione a tarefa:")
    print("1 - Validar pontos do exemplo")
    print("2 - Estudo de convergência")
    choice = input("> ")
    if choice == "1":
        from tarefas import tarefa_validar_pontos_exemplo
        tarefa_validar_pontos_exemplo()
    elif choice == "2":
        from tarefas import tarefa_convergencia
        import math
        tarefa_convergencia(math.sin, 0.0, math.pi, [4, 8, 16, 32])
    else:
        print("Opção inválida.")

if __name__ == "__main__":
    main()
    print(gauss_jacobi.jacobi())
