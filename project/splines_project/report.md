# Projeto 1 — Aproximação Teórica e Numérica I

### Estudo sobre a Convergência de Splines Cúbicos Interpoladores

**Autor:** Rodrigo Fassa et al.  
**Orientador:** Prof. André Pierro de Camargo  
**Data:** 09/11/2025

---

## 1. Introdução

### 1.1 Fundamentação Teórica

O spline cúbico interpolador é uma função polinomial por partes $S(x)$ de classe $C^2[a,b]$,
isto é, contínua juntamente com suas primeiras e segundas derivadas em todo o domínio.
Cada subintervalo $[x_i, x_{i+1}]$ é associado a um polinômio cúbico da forma:

$$
S_i(x) = a_i + b_i (x-x_i) + c_i (x-x_i)^2 + d_i (x-x_i)^3,
$$
de modo que:
$$
S_i(x_i) = y_i, \quad S_i(x_{i+1}) = y_{i+1}, \quad
S_i'(x_{i+1}) = S_{i+1}'(x_{i+1}), \quad S_i''(x_{i+1}) = S_{i+1}''(x_{i+1}).
$$

Essas condições garantem a suavidade global da interpolação, e o sistema tridiagonal
resultante é derivado dessas equações de continuidade de segunda ordem.

Fisicamente, o spline cúbico natural corresponde à curva de **menor energia elástica**
que passa por todos os pontos $(x_i, y_i)$.
Isso equivale a minimizar o funcional:
$$
E[S] = \int_a^b [S''(x)]^2 \, dx,
$$
que mede a curvatura média da função.

Do ponto de vista analítico, se $f \in C^4[a,b]$, então o erro de interpolação satisfaz:
$$
|f(x) - S(x)| \leq \frac{5}{384} h^4 \max_{\xi \in [a,b]} |f^{(4)}(\xi)|,
$$
mostrando que o spline cúbico completo é de **ordem de convergência 4**,
enquanto o spline natural pode exibir comportamento $O(h^2)$ próximo das fronteiras
caso as segundas derivadas não se anulem.

A teoria, portanto, prevê:
$$
E_n \approx C h^\rho, \quad \text{com } \rho \approx 4.
$$

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
do spline cúbico nas versões natural e completa, comparando os erros e ordens estimadas
---

## 3. Resultados Numéricos

### 3.1 Spline Natural

| n | h | $E_n$ |
|--:|--:|--:|
|   4 |   0.392699 | 7.725073e-03 |
|   8 |   0.196350 | 1.902205e-03 |
|  16 |   0.098175 | 4.737284e-04 |
|  32 |   0.049087 | 1.183202e-04 |
|  64 |   0.024544 | 2.957305e-05 |

**Ordem estimada:** $\rho \approx 2.01$

---

### 3.2 Spline Completo

| n | h | $E_n$ |
|--:|--:|--:|
|   4 |   0.392699 | 6.324039e-05 |
|   8 |   0.196350 | 3.889330e-06 |
|  16 |   0.098175 | 2.421787e-07 |
|  32 |   0.049087 | 1.512267e-08 |
|  64 |   0.024544 | 9.443273e-10 |

**Ordem estimada:** $\rho \approx 4.01$

---

### 3.3 Gráficos log–log

**Spline Natural**

![Convergência Natural](./convergencia_natural.png)

**Spline Completo**

![Convergência Completo](./convergencia_completo.png)

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
