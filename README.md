# Curso de Matemática com Python

Curso **gratuito** que ensina matemática junto com programação, do básico até conteúdo quase universitário. O material é entregue como uma coleção de notebooks Jupyter, pensados para jovens estudarem de forma autônoma e progressiva — cada notebook ensina sozinho (com o auxílio de conteúdos da Wikipedia), com teoria, exemplos, visualizações, widgets interativos e exercícios com autocorreção.

## Estrutura do curso

A ordem das pastas é a ordem sugerida de estudo:

| Módulo | Status |
|---|---|
| `00_fundamentos_e_python` | ✅ em andamento |
| `01_conjuntos_e_funcoes` | ✅ em andamento |
| `02_logaritmos` | ⏳ não iniciado |
| `03_progressoes_matrizes_e_sistemas_lineares` | ⏳ não iniciado |
| `04_geometria_plana` | ⏳ não iniciado |
| `05_trigonometria` | ⏳ não iniciado |
| `06_geometria_espacial` | ⏳ não iniciado |
| `07_geometria_analitica_plana_e_espacial` | ⏳ não iniciado |
| `08_analise_combinatoria_e_probabilidade` | ⏳ não iniciado |
| `09_complexos_polinomios_e_equacoes` | ⏳ não iniciado |
| `10_matematica_financeira_comercial_e_estatistica_descritiva` | ⏳ não iniciado |
| `11_limite_integral_e_derivada` | ⏳ não iniciado |
| `12_matematica_discreta` (opcional) | ⏳ não iniciado |
| `13_algebra_linear` (opcional) | ⏳ não iniciado |
| `14_equacoes_diferenciais` (opcional) | ⏳ não iniciado |

Os módulos 12 a 14 são opcionais, voltados a quem quer seguir para Engenharia ou Computação.

## Progresso atual

### 00 — Fundamentos e Python

- **[00_introducao_python.ipynb](00_fundamentos_e_python/00_introducao_python.ipynb)** — primeiro contato com programação: o que é um programa, variáveis, tipos de dados básicos (`int`, `float`, `str`, `bool`), operadores aritméticos (incluindo divisão inteira e resto), `print`/f-strings, comentários e um primeiro widget interativo com `ipywidgets`.
- **[00_introducao_python_gabarito.ipynb](00_fundamentos_e_python/00_introducao_python_gabarito.ipynb)** — gabarito comentado dos exercícios.

### 01 — Conjuntos e Funções

- **[011a_nocoes_de_logica.ipynb](01_conjuntos_e_funcoes/011a_nocoes_de_logica.ipynb)** — noções de lógica proposicional: proposições, sentenças abertas e quantificadores (∀, ∃), negação, conectivos (conjunção e disjunção), condicionais e bicondicionais, construção automática de tabelas-verdade, classificação em tautologia/contradição/contingência, implicação e equivalência lógica, recíproca/contrária/contrapositiva e as Leis de De Morgan. Traz representações algébricas, gráficas (paleta Nord) e numéricas, além de widgets interativos para testar quantificadores e montar proposições compostas.
- **[011b_nocoes_de_logica_gabarito.ipynb](01_conjuntos_e_funcoes/011b_nocoes_de_logica_gabarito.ipynb)** — gabarito comentado dos exercícios.
- **[012a_conjuntos.ipynb](01_conjuntos_e_funcoes/012a_conjuntos.ipynb)** — teoria de conjuntos: pertinência, descrição por extensão/compreensão, conjunto unitário/vazio/universo, igualdade, subconjuntos, conjunto das partes, união, interseção, diferença, complementar, propriedades (comutativa, associativa, distributiva) e Leis de De Morgan aplicadas a conjuntos. Traz um diagrama de Venn interativo reutilizado em vários blocos (paleta Nord) e um problema de contagem com união (n(A∪B) = n(A) + n(B) − n(A∩B)).
- **[012b_conjuntos_gabarito.ipynb](01_conjuntos_e_funcoes/012b_conjuntos_gabarito.ipynb)** — gabarito comentado dos exercícios.
- **[013a_conjuntos_numericos.ipynb](01_conjuntos_e_funcoes/013a_conjuntos_numericos.ipynb)** — conjuntos numéricos N, Z, Q, R e C, motivados por uma equação que trava em cada conjunto sucessivo; princípio da indução finita (base e passo indutivo); dízimas periódicas e expansão decimal por divisão longa; prova por contradição de que √2 é irracional; módulo (valor absoluto) como distância na reta; e notação de intervalos (aberto, fechado, semiaberto, infinito). Traz uma reta numérica interativa reutilizada em vários blocos (paleta Nord) e widgets para explorar frações, módulo e intervalos.
- **[013b_conjuntos_numericos_gabarito.ipynb](01_conjuntos_e_funcoes/013b_conjuntos_numericos_gabarito.ipynb)** — gabarito comentado dos exercícios.
- **[014a_relacoes.ipynb](01_conjuntos_e_funcoes/014a_relacoes.ipynb)** — par ordenado e sua propriedade de igualdade; sistema cartesiano ortogonal e quadrantes; produto cartesiano (A × B) e cardinalidade; relação binária como subconjunto de A × B, com suas cinco formas de representação (sentença, pares, tabela, gráfico, diagrama de flechas); domínio, contradomínio e imagem; relação inversa; e relação em A com as propriedades reflexiva, simétrica, transitiva e antissimétrica. Traz diagrama de flechas (sagital) e matriz de relação interativos (paleta Nord), além de widgets para montar relações escolhendo os pares.
- **[014b_relacoes_gabarito.ipynb](01_conjuntos_e_funcoes/014b_relacoes_gabarito.ipynb)** — gabarito comentado dos exercícios.

## Como rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
