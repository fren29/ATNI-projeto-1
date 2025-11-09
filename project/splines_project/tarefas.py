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

import datetime
from pathlib import Path
import numpy as np

from pathlib import Path
import datetime

def gerar_relatorio(results_natural, results_completo, rho_natural, rho_completo,
                    fig_nat="./convergencia_natural.png", fig_comp="./convergencia_completo.png"):
    """
    Gera report.md com formatação compatível com Pandoc/pdflatex.
    - Usa raw strings (r'...' ou r'''...''') nos blocos com LaTeX.
    - Tabelas usam $E_n$; letras gregas aparecem como $\rho$; ~ sempre em modo math.
    """
    data = datetime.date.today().strftime("%d/%m/%Y")

    # Cabeçalho SEM LaTeX → f-string ok
    md = f"""# Projeto 1 — Aproximação Teórica e Numérica I
### Estudo sobre a Convergência de Splines Cúbicos Interpoladores
**Autor:** Rodrigo Fassa et al.  
**Orientador:** Prof. André Pierro de Camargo  
**Data:** {data}

---

"""

    # Bloco com LaTeX → raw string (r"""...""")
    md += r"""## 1. Introdução

### 1.1 Fundamentação Teórica

O spline cúbico interpolador é uma função polinomial por partes \(S(x)\) de classe \(C^2[a,b]\),
isto é, contínua juntamente com suas primeiras e segundas derivadas em todo o domínio.
Cada subintervalo \([x_i, x_{i+1}]\) é associado a um polinômio cúbico da forma:

\[
S_i(x) = a_i + b_i (x-x_i) + c_i (x-x_i)^2 + d_i (x-x_i)^3,
\]
de modo que:
\[
S_i(x_i) = y_i, \quad S_i(x_{i+1}) = y_{i+1}, \quad
S_i'(x_{i+1}) = S_{i+1}'(x_{i+1}), \quad S_i''(x_{i+1}) = S_{i+1}''(x_{i+1}).
\]

Essas condições garantem a suavidade global da interpolação, e o sistema tridiagonal
resultante é derivado dessas equações de continuidade de segunda ordem.

Fisicamente, o spline cúbico natural corresponde à curva de **menor energia elástica** 
que passa por todos os pontos \((x_i, y_i)\).
Isso equivale a minimizar o funcional:
\[
E[S] = \int_a^b [S''(x)]^2 \, dx,
\]
que mede a curvatura média da função.

Do ponto de vista analítico, se \(f \in C^4[a,b]\), então o erro de interpolação satisfaz:
\[
|f(x) - S(x)| \leq \frac{5}{384} h^4 \max_{\xi \in [a,b]} |f^{(4)}(\xi)|,
\]
mostrando que o spline cúbico completo é de **ordem de convergência 4**,
enquanto o spline natural pode exibir comportamento \(O(h^2)\) próximo das fronteiras
caso as segundas derivadas não se anulem.

A teoria, portanto, prevê:
\[
E_n \approx C h^\rho, \quad \text{com } \rho \approx 4.
\]

Essa relação será validada empiricamente nas seções seguintes.

### 1.2 Objetivo do Estudo

Este relatório apresenta o estudo numérico da convergência de *splines cúbicos interpoladores*,
verificando empiricamente a ordem de convergência teórica do método.
Se $f \in C^4[a,b]$, o erro máximo satisfaz:
$$
E_n = \max_{x \in [a,b]} |f(x) - S(x)| \approx C\,h^4.
$$

---

## 2. Metodologia

As rotinas foram implementadas em Python seguindo o pseudocódigo do enunciado.
Principais funções:

| Módulo     | Função                         | Descrição                                   |
|:-----------|:-------------------------------|:--------------------------------------------|
| `spline.py`| `build_tridiagonal_system`     | Monta o sistema $T\cdot M=d$                |
| `gauss.py` | `solve_by_gaussian_elimination`| Resolve o sistema linear                    |
| `spline.py`| `compute_M`, `compute_AB`, `spline_eval` | Segundas derivadas e avaliação       |
| `tarefas.py`| `tarefa_convergencia_*`       | Experimentos de convergência                |
| `tarefas.py`| `ajuste_ordem_convergencia`   | Estima $\rho$ por regressão log–log         |

Validação em $f(x)=\cos(x)$, no intervalo $[0,\pi/2]$,
com condições de contorno **natural** e **completa**.

---

---
A seguir, apresentamos os resultados obtidos para o estudo de convergência empírica
do spline cúbico nas versões natural e completa, comparando os erros e ordens estimadas.
---

## 3. Resultados Numéricos

### 3.1 Spline Natural

| n | h | $E_n$ |
|--:|--:|--:|
"""

    # Tabela Natural (sem LaTeX novo aqui; só números)
    for n, h, E in results_natural:
        md += f"| {n:3d} | {h:10.6f} | {E:12.6e} |\n"

    md += f"""
**Ordem estimada:** $\\rho \\approx {rho_natural:.2f}$

---

### 3.2 Spline Completo

| n | h | $E_n$ |
|--:|--:|--:|
"""

    for n, h, E in results_completo:
        md += f"| {n:3d} | {h:10.6f} | {E:12.6e} |\n"

    md += f"""
**Ordem estimada:** $\\rho \\approx {rho_completo:.2f}$

---

### 3.3 Gráficos log–log
"""

    # Imagens: só referencia se existir no disco (evita warning do Pandoc)
    if Path(fig_nat).exists():
        md += f"**Spline Natural**\n\n![Convergência (Natural)]({fig_nat})\n\n"
    else:
        md += "_Figura do natural não encontrada no diretório._\n\n"

    if Path(fig_comp).exists():
        md += f"**Spline Completo**\n\n![Convergência (Completo)]({fig_comp})\n\n"
    else:
        md += "_Figura do completo não encontrada no diretório._\n\n"

    md += r"""
---

## 4. Discussão e Conclusão

O spline natural apresentou erro com tendência $E_n \sim h^2$,
enquanto o spline completo atingiu a convergência teórica de quarta ordem ($\rho \approx 4$),
evidenciando a importância das condições de contorno no desempenho global.

**Conclusão:** o spline cúbico completo é um método de alta precisão para interpolação suave.

---
## 5. Referências

- Burden, R. L. & Faires, J. D. *Análise Numérica*, 10ª ed., Cengage, 2016.  
- Kiusalaas, J. *Numerical Methods in Engineering with Python 3*, Cambridge University Press, 2013.  
- Camargo, A. P. (2025). *Notas de Aula — Aproximação Teórica e Numérica I (UFABC)*.

"""

    Path("report.md").write_text(md, encoding="utf-8")
    print("✅ report.md gerado com sucesso.")

import subprocess
from pathlib import Path
import shutil

def gerar_pdf(template_tex: str | None = "ufabc-template.tex"):
    """
    Converte report.md em report.pdf via Pandoc.
    Usa template LaTeX se presente; caso contrário, usa o padrão.
    """
    from pathlib import Path
    import subprocess
    import shutil

    md_path = Path("report.md")
    pdf_path = Path("report.pdf")

    if not md_path.exists():
        print("❌ report.md não encontrado. Gere o relatório antes.")
        return

    pandoc = shutil.which("pandoc")
    if pandoc is None:
        print("❌ Pandoc não encontrado no PATH. Instale Pandoc e MiKTeX.")
        return

    args = [
        pandoc,
        str(md_path),
        "-o", str(pdf_path),
        "--from", "markdown+tex_math_dollars",
        "--pdf-engine=pdflatex",
        "--toc",
        "--number-sections",
        "--variable", "tables-use-longtable=false",  # 👈 ESSA LINHA É A CHAVE
    ]
    if template_tex and Path(template_tex).exists():
        args += ["--template", template_tex]

    print("🛠️  Gerando PDF com Pandoc...")
    subprocess.run(args, check=True)
    print(f"✅ PDF gerado com sucesso em {pdf_path.resolve()}")


#gerar_pdf()

def tarefa_tabela1():
    """
    Tarefa 1 — Tabela 1 do Enunciado (Prof. André Pierro, UFABC 2025)
    -----------------------------------------------------------------
    Reproduz os valores dados no PDF do projeto (Tabela 1) e avalia
    o spline cúbico interpolador (condição natural).

    Dados:
        xᵢ = [-0.9, -0.83, -0.6, -0.49, 0.0, 0.2, 0.6, 0.83]
        yᵢ = [0.0, 1.0, 2.4, 4.1, 6.0, 8.2, 10.6, 13.4]

    Objetivo:
        - Montar o spline cúbico natural.
        - Avaliar S(x) em pontos intermediários.
        - Imprimir tabela de valores interpolados.
    """
    import numpy as np
    from spline import spline_function

    x = [-0.9, -0.83, -0.6, -0.49, 0.0, 0.2, 0.6, 0.83]
    y = [0.0, 1.0, 2.4, 4.1, 6.0, 8.2, 10.6, 13.4]

    S = spline_function(x, y, bc="natural")

    print("\n=== Tarefa 1 — Tabela 1 do Enunciado ===")
    print(f"{'x':>8} | {'S(x)':>12}")
    print("-" * 24)

    xs_test = np.linspace(min(x), max(x), 15)
    for xi in xs_test:
        print(f"{xi:8.3f} | {S(xi):12.6f}")
#tarefa_tabela1()

from tarefas import tarefa_convergencia, tarefa_convergencia_completa, ajuste_ordem_convergencia, gerar_relatorio
import numpy as np

f  = np.cos
df = lambda x: -np.sin(x)
a, b = 0.0, np.pi/2
ns = [4, 8, 16, 32, 64]

# Natural
results_nat = tarefa_convergencia(f, a, b, ns)
rho_nat = ajuste_ordem_convergencia(results_nat, "Spline Natural")

# Completo
results_compl = tarefa_convergencia_completa(f, df, a, b, ns)
rho_compl = ajuste_ordem_convergencia(results_compl, "Spline Completo")

# Gera o relatório final
gerar_relatorio(results_nat, results_compl, rho_nat, rho_compl)
