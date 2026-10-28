"""Funções de desenho do Módulo 06 — Geometria Espacial.

Propósito
---------
Este arquivo reúne todas as funções que desenham pontos, retas, planos e sólidos
nos notebooks do módulo 06. Assim os notebooks ficam curtos e todas as figuras
ficam com a mesma cara.

Você não precisa entender este arquivo para estudar a matemática. Mas, se gosta de
programar, fique à vontade para ler, testar e modificar: o código foi escrito para
ser lido por quem está começando.

Como usar
---------
    import utils_3d as u3d

    fig = u3d.criar_figura(titulo="Meu primeiro desenho")
    fig = u3d.desenhar_ponto(fig, (0, 0, 0), rotulo="A")
    fig = u3d.desenhar_plano(fig, ponto=(0, 0, 0), normal=(0, 0, 1), rotulo="α")
    u3d.mostrar(fig)

Toda função `desenhar_...` recebe a figura, acrescenta algo a ela e devolve a mesma
figura. Por isso escrevemos sempre `fig = u3d.desenhar_...(fig, ...)`.

As coordenadas (x, y, z) que aparecem nos exemplos servem apenas para posicionar os
desenhos. O estudo do espaço com coordenadas é assunto do módulo 07 (Geometria
Analítica).

Histórico de versões
--------------------
v1 — notebook 01: núcleo (figura, ponto, segmento, reta, plano, interseção de planos,
     cubo, conferência de posição de retas)
v2 — notebook 02: construtores (plano por três pontos, plano paralelo a uma reta,
     interseção reta-plano), paralela por um ponto, interseção reta-plano desenhada,
     ângulos (arco, ângulo reto, ângulo entre retas), conferência de reta-plano, de
     planos, distância e ângulo entre retas
v3 — notebook 03: construtores de perpendicularidade (pé da perpendicular ao plano e
     à reta, plano perpendicular a uma reta, plano perpendicular a outro por uma
     reta), desenhos correspondentes e conferências de reta perpendicular a plano e
     de planos perpendiculares
v4 — notebook 04: polígono, circunferência, esfera, nuvem de pontos, projeção ortogonal
     de figuras, perpendicular comum, plano mediador, planos bissetores, distâncias de
     ponto a plano e entre retas, ângulo entre reta e plano; medir_distancia passa a
     aceitar vários pontos de uma vez
v5 — notebook 05: semiplanos e diedros (desenho, diedro padrão, seção por um plano,
     seção igualmente inclinada, semiplano bissetor), medidas de ângulo entre
     semirretas, de diedro e entre planos

Semiplanos e diedros
--------------------
Um semiplano é dado por dois pontos a e b da sua reta de origem e por um ponto p
fora dessa reta, que indica DE QUE LADO da reta está o semiplano. Um diedro é dado
pela aresta (os pontos a e b) e por um ponto de cada face (p1 e p2).
"""

# ==================================================================================
# Importações
# ==================================================================================
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import plotly.graph_objects as go

# Um ponto ou um vetor pode ser escrito como tupla, lista ou array do NumPy,
# sempre com 3 números: (x, y, z).
Vetor3D: TypeAlias = Sequence[float] | np.ndarray

# Um plano é descrito por um ponto dele e um vetor perpendicular a ele (o "normal").
Plano: TypeAlias = tuple[Vetor3D, Vetor3D]

# Tolerância numérica: o computador faz contas com pequenos erros de arredondamento,
# então "zero" na prática quer dizer "menor que este número".
TOLERANCIA = 1e-9


# ==================================================================================
# Cores (paleta Nord, tema escuro)
# ==================================================================================
NORD = {
    "nord0": "#2E3440", "nord1": "#3B4252", "nord2": "#434C5E", "nord3": "#4C566A",
    "nord4": "#D8DEE9", "nord5": "#E5E9F0", "nord6": "#ECEFF4",
    "nord7": "#8FBCBB", "nord8": "#88C0D0", "nord9": "#81A1C1", "nord10": "#5E81AC",
    "nord11": "#BF616A", "nord12": "#D08770", "nord13": "#EBCB8B",
    "nord14": "#A3BE8C", "nord15": "#B48EAD",
}

# Cores por papel: assim o "plano principal" tem a mesma cor em todos os notebooks.
# O fundo é escuro (padrão do curso), por isso textos, pontos e linhas de apoio usam
# os tons claros da paleta.
CORES = {
    "fundo": NORD["nord0"],
    "texto": NORD["nord6"],
    "apoio": NORD["nord4"],
    "ponto": NORD["nord6"],
    "destaque": NORD["nord11"],
    "reta": NORD["nord10"],
    "reta_secundaria": NORD["nord12"],
    "plano": NORD["nord8"],
    "plano_secundario": NORD["nord14"],
    "plano_terciario": NORD["nord15"],
    "solido": NORD["nord9"],
}


# ==================================================================================
# Funções auxiliares internas (não são usadas diretamente nos notebooks)
# ==================================================================================
def _como_vetor(valor: Vetor3D) -> np.ndarray:
    """Converte uma tupla, lista ou array de 3 números num array do NumPy."""
    vetor = np.asarray(valor, dtype=float)
    if vetor.shape != (3,):
        raise ValueError(
            f"Esperava 3 números (x, y, z), mas recebi {valor!r}. "
            "Escreva pontos e vetores como (x, y, z)."
        )
    return vetor


def _normalizar(vetor: Vetor3D) -> np.ndarray:
    """Devolve um vetor com a mesma direção e sentido, mas de comprimento 1."""
    vetor = _como_vetor(vetor)
    comprimento = np.linalg.norm(vetor)
    if comprimento < TOLERANCIA:
        raise ValueError("O vetor (0, 0, 0) não tem direção: use um vetor não nulo.")
    return vetor / comprimento


def _base_do_plano(normal: Vetor3D) -> tuple[np.ndarray, np.ndarray]:
    """Encontra dois vetores u e v, de comprimento 1, perpendiculares entre si e ao
    vetor normal. Andando nas direções de u e v, ficamos sempre dentro do plano."""
    n = _normalizar(normal)

    # Escolhemos um vetor "de apoio" que não seja paralelo ao normal. Preferimos o
    # eixo x, mas se o normal estiver quase na direção x usamos o eixo z.
    apoio = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(apoio, n)) > 0.9:
        apoio = np.array([0.0, 0.0, 1.0])

    # O produto vetorial de dois vetores dá um vetor perpendicular aos dois.
    u = _normalizar(np.cross(apoio, n))
    v = np.cross(n, u)
    return u, v


def _intersecao_de_planos(
    plano1: Plano, plano2: Plano
) -> tuple[np.ndarray, np.ndarray] | None:
    """Calcula a reta comum a dois planos.

    Devolve (ponto, direcao) da reta, escolhendo o ponto da reta mais próximo da
    média dos pontos dos dois planos (para o desenho ficar centrado). Devolve None
    se os planos forem paralelos ou coincidentes.
    """
    p1, n1 = _como_vetor(plano1[0]), _como_vetor(plano1[1])
    p2, n2 = _como_vetor(plano2[0]), _como_vetor(plano2[1])

    # A reta comum é perpendicular aos dois normais.
    direcao = np.cross(n1, n2)
    if np.linalg.norm(direcao) <= TOLERANCIA * np.linalg.norm(n1) * np.linalg.norm(n2):
        return None

    # O ponto procurado está nos dois planos e no plano perpendicular à reta que
    # passa pela média dos pontos: são três equações com três incógnitas.
    media = (p1 + p2) / 2
    coeficientes = np.array([n1, n2, direcao])
    termos = np.array([np.dot(n1, p1), np.dot(n2, p2), np.dot(direcao, media)])
    ponto = np.linalg.solve(coeficientes, termos)
    return ponto, _normalizar(direcao)


def _encontro_de_retas(
    a1: Vetor3D, a2: Vetor3D, b1: Vetor3D, b2: Vetor3D
) -> np.ndarray:
    """Calcula o ponto de encontro de duas retas concorrentes (a1a2 e b1b2).

    Procuramos números t e k tais que a1 + t·(a2 - a1) = b1 + k·(b2 - b1). São três
    equações (uma para x, uma para y, uma para z) e duas incógnitas; o método dos
    mínimos quadrados do NumPy resolve esse sistema "sobrando equação".
    """
    pa1, pa2 = _como_vetor(a1), _como_vetor(a2)
    pb1, pb2 = _como_vetor(b1), _como_vetor(b2)
    coeficientes = np.column_stack([pa2 - pa1, -(pb2 - pb1)])
    (t, _), *_ = np.linalg.lstsq(coeficientes, pb1 - pa1, rcond=None)
    return pa1 + t * (pa2 - pa1)


def _direcao_do_semiplano(
    a: Vetor3D, b: Vetor3D, ponto: Vetor3D
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve as duas direções que descrevem o semiplano de origem ab que passa
    por `ponto`.

    A primeira direção (d) corre ao longo da reta de origem, de a para b. A segunda
    (w) fica dentro do semiplano, forma 90° com a reta de origem e aponta para o lado
    de `ponto`. Ambas têm comprimento 1.
    """
    pa, pb, p = _como_vetor(a), _como_vetor(b), _como_vetor(ponto)
    if np.linalg.norm(pb - pa) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar a reta de origem."
        )
    d = _normalizar(pb - pa)

    # Tiramos de a→ponto a parte que corre ao longo da reta de origem; o que sobra é
    # perpendicular a ela e aponta para o lado do semiplano.
    ap = p - pa
    lateral = ap - np.dot(ap, d) * d
    if np.linalg.norm(lateral) <= TOLERANCIA * max(np.linalg.norm(ap), 1.0):
        raise ValueError(
            "O ponto está na reta de origem: ele não indica um lado do semiplano."
        )
    return d, _normalizar(lateral)


# ==================================================================================
# Figura e exibição
# ==================================================================================
def criar_figura(
    titulo: str | None = None, mostrar_eixos: bool = False, altura: int = 500
) -> go.Figure:
    """Cria uma figura 3D vazia, com o visual padrão do módulo.

    Parâmetros:
        titulo: texto que aparece no alto da figura (opcional).
        mostrar_eixos: se True, mostra os eixos x, y e z. Por padrão eles ficam
            escondidos, porque as coordenadas não são o assunto deste módulo.
        altura: altura da figura, em pixels.

    Retorna:
        Uma figura do Plotly (`go.Figure`), pronta para receber desenhos.

    Exemplo:
        >>> fig = criar_figura(titulo="Ponto, reta e plano")
    """
    if mostrar_eixos:
        eixo = dict(
            showbackground=True,
            backgroundcolor=NORD["nord1"],
            gridcolor=NORD["nord3"],
            zerolinecolor=NORD["nord3"],
            color=NORD["nord4"],
        )
    else:
        # Sem eixos: nada de grade, fundo, números ou nomes dos eixos.
        eixo = dict(visible=False)

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, x=0.5) if titulo else None,
        height=altura,
        paper_bgcolor=CORES["fundo"],
        plot_bgcolor=CORES["fundo"],
        font=dict(color=CORES["texto"], size=13),
        margin=dict(l=0, r=0, b=0, t=50 if titulo else 10),
        showlegend=False,
        scene=dict(
            aspectmode="data",  # mantém as proporções reais (um cubo parece cubo)
            bgcolor=CORES["fundo"],
            xaxis=eixo,
            yaxis=eixo,
            zaxis=eixo,
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.0)),
        ),
    )
    return fig


def definir_camera(fig: go.Figure, olho: Vetor3D = (1.6, 1.4, 1.0)) -> go.Figure:
    """Muda o ponto de vista inicial da figura.

    Parâmetros:
        fig: a figura.
        olho: posição de onde a "câmera" olha para o centro da figura. Números
            maiores afastam a câmera.

    Retorna:
        A mesma figura, com a nova câmera.

    Exemplo:
        >>> fig = definir_camera(fig, olho=(2, 0.5, 0.8))
    """
    x, y, z = _como_vetor(olho)
    fig.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=z)))
    return fig


def mostrar(fig: go.Figure) -> None:
    """Exibe a figura no notebook (sem o logotipo do Plotly na barra de botões).

    Parâmetros:
        fig: a figura a exibir.

    Exemplo:
        >>> mostrar(fig)
    """
    fig.show(config={"displaylogo": False})


# ==================================================================================
# Elementos básicos: ponto, segmento, reta, plano e rótulo
# ==================================================================================

# --- Construtores (devolvem dados, não desenham) ---
def plano_por_tres_pontos(a: Vetor3D, b: Vetor3D, c: Vetor3D) -> Plano:
    """Devolve o plano determinado por três pontos não colineares.

    Esta função não desenha nada: ela devolve o plano no formato (ponto, normal),
    o mesmo usado por `desenhar_plano`, `desenhar_intersecao_de_planos` e pelas
    funções de conferência. As contas internas usam vetores, assunto do módulo 07
    (Geometria Analítica).

    Parâmetros:
        a, b, c: os três pontos, cada um como (x, y, z).

    Retorna:
        Uma tupla (ponto, normal). O ponto devolvido é o baricentro de a, b e c (o
        "ponto de equilíbrio" do triângulo abc).

    Levanta:
        ValueError: se os três pontos forem colineares.

    Exemplo:
        >>> base = plano_por_tres_pontos((0, 0, 0), (2, 0, 0), (2, 2, 0))
        >>> fig = desenhar_plano(fig, *base, rotulo="α")
    """
    pa, pb, pc = _como_vetor(a), _como_vetor(b), _como_vetor(c)

    # Se AB e AC apontam na mesma direção, o produto vetorial dá (quase) zero:
    # os três pontos estão numa mesma reta.
    normal = np.cross(pb - pa, pc - pa)
    escala = np.linalg.norm(pb - pa) * np.linalg.norm(pc - pa)
    if np.linalg.norm(normal) <= TOLERANCIA * max(escala, 1.0):
        raise ValueError(
            "Os três pontos são colineares: eles não determinam um único plano."
        )

    baricentro = (pa + pb + pc) / 3
    return baricentro, normal


def plano_paralelo_a_reta(
    a1: Vetor3D, a2: Vetor3D, b1: Vetor3D, b2: Vetor3D
) -> Plano:
    """Devolve o plano que contém a reta a1a2 e é paralelo à reta b1b2.

    É a construção do notebook 02: por um ponto da primeira reta traçamos a
    paralela à segunda; as duas retas concorrentes determinam o plano. Esta função
    não desenha nada, e as contas internas usam vetores, assunto do módulo 07
    (Geometria Analítica).

    Parâmetros:
        a1, a2: dois pontos distintos da reta que fica contida no plano.
        b1, b2: dois pontos distintos da reta que fica paralela ao plano.

    Retorna:
        Uma tupla (ponto, normal). O ponto devolvido é o ponto médio de a1a2.

    Levanta:
        ValueError: se as duas retas forem paralelas (ou coincidentes).

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> alfa = plano_paralelo_a_reta(v["A"], v["B"], v["C"], v["G"])
    """
    pa1, pa2 = _como_vetor(a1), _como_vetor(a2)
    pb1, pb2 = _como_vetor(b1), _como_vetor(b2)
    u = pa2 - pa1  # direção da reta contida no plano
    v = pb2 - pb1  # direção da reta paralela ao plano
    if np.linalg.norm(u) < TOLERANCIA or np.linalg.norm(v) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )

    # O plano procurado tem as duas direções, u e v; o normal é perpendicular a
    # ambas. Se u e v forem paralelos, o produto vetorial dá zero e o plano não é
    # único (qualquer plano do feixe que passa pela primeira reta serviria).
    normal = np.cross(u, v)
    if np.linalg.norm(normal) <= TOLERANCIA * np.linalg.norm(u) * np.linalg.norm(v):
        raise ValueError(
            "As retas são paralelas: existem infinitos planos que contêm a primeira "
            "e são paralelos à segunda."
        )
    return (pa1 + pa2) / 2, normal


def intersecao_reta_plano(
    a: Vetor3D, b: Vetor3D, plano: Plano
) -> tuple[float, float, float] | None:
    """Devolve o ponto onde a reta ab "fura" o plano.

    Esta função não desenha nada. As contas internas usam vetores, assunto do
    módulo 07 (Geometria Analítica).

    Parâmetros:
        a, b: dois pontos distintos da reta.
        plano: o plano, como (ponto, normal).

    Retorna:
        O ponto de interseção, como (x, y, z), ou None se a reta for paralela ao
        plano ou estiver contida nele (nesses casos não há um único ponto comum).

    Exemplo:
        >>> intersecao_reta_plano((0, 0, 0), (2, 2, 2), ((0, 0, 1), (0, 0, 1)))
        (1.0, 1.0, 1.0)
    """
    pa, pb = _como_vetor(a), _como_vetor(b)
    ponto, normal = _como_vetor(plano[0]), _como_vetor(plano[1])
    direcao = pb - pa
    if np.linalg.norm(direcao) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )

    # Andando sobre a reta (a + t·direcao), a "altura" em relação ao plano muda
    # na razão normal·direcao. Se essa razão é zero, a reta corre paralela ao plano.
    velocidade = np.dot(normal, direcao)
    if abs(velocidade) <= TOLERANCIA * np.linalg.norm(normal) * np.linalg.norm(direcao):
        return None

    # Escolhemos t para que a "altura" de a + t·direcao em relação ao plano seja 0.
    t = np.dot(normal, ponto - pa) / velocidade
    x, y, z = pa + t * direcao
    return float(x), float(y), float(z)


def pe_da_perpendicular_ao_plano(
    ponto: Vetor3D, plano: Plano
) -> tuple[float, float, float]:
    """Devolve o pé da perpendicular traçada de `ponto` até o plano.

    O pé é o ponto onde a reta que passa por `ponto` e é perpendicular ao plano
    encontra o plano. Se o ponto já estiver no plano, a função devolve o próprio
    ponto. Esta função não desenha nada, e as contas internas usam vetores, assunto
    do módulo 07 (Geometria Analítica).

    O pé da perpendicular é também a projeção ortogonal do ponto sobre o plano
    (notebook 04).

    Parâmetros:
        ponto: o ponto, como (x, y, z).
        plano: o plano, como (ponto, normal).

    Retorna:
        O pé da perpendicular, como (x, y, z).

    Exemplo:
        >>> pe_da_perpendicular_ao_plano((1, 2, 5), ((0, 0, 0), (0, 0, 1)))
        (1.0, 2.0, 0.0)
    """
    p = _como_vetor(ponto)
    ponto_do_plano, normal = _como_vetor(plano[0]), _como_vetor(plano[1])
    n = _normalizar(normal)

    # A "altura" do ponto em relação ao plano, medida na direção do normal.
    # Descendo essa altura ao longo do normal, chegamos ao plano.
    altura = np.dot(p - ponto_do_plano, n)
    x, y, z = p - altura * n
    return float(x), float(y), float(z)


def pe_da_perpendicular_a_reta(
    ponto: Vetor3D, a: Vetor3D, b: Vetor3D
) -> tuple[float, float, float]:
    """Devolve o pé da perpendicular traçada de `ponto` até a reta ab.

    É o ponto da reta ab mais próximo de `ponto`. Se o ponto estiver na reta, a
    função devolve o próprio ponto. Esta função não desenha nada, e as contas
    internas usam vetores, assunto do módulo 07 (Geometria Analítica).

    Parâmetros:
        ponto: o ponto, como (x, y, z).
        a, b: dois pontos distintos da reta.

    Retorna:
        O pé da perpendicular, como (x, y, z).

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> pe_da_perpendicular_a_reta((1, 3, 0), (0, 0, 0), (2, 0, 0))
        (1.0, 0.0, 0.0)
    """
    p, pa, pb = _como_vetor(ponto), _como_vetor(a), _como_vetor(b)
    direcao = pb - pa
    if np.linalg.norm(direcao) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )

    # Andamos sobre a reta (a + t·direcao) até o ponto em que o segmento que vai
    # até `ponto` forma 90° com a reta. Esse t é a "sombra" de a→ponto sobre a reta.
    t = np.dot(p - pa, direcao) / np.dot(direcao, direcao)
    x, y, z = pa + t * direcao
    return float(x), float(y), float(z)


def plano_perpendicular_a_reta(a: Vetor3D, b: Vetor3D, ponto: Vetor3D) -> Plano:
    """Devolve o plano que passa por `ponto` e é perpendicular à reta ab.

    Pela propriedade 1 do notebook 03, esse plano é único. Esta função não desenha
    nada, e as contas internas usam vetores, assunto do módulo 07 (Geometria
    Analítica).

    Parâmetros:
        a, b: dois pontos distintos da reta.
        ponto: o ponto por onde o plano passa, como (x, y, z).

    Retorna:
        Uma tupla (ponto, normal). O ponto devolvido é o próprio `ponto`.

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> pi = plano_perpendicular_a_reta(v["A"], v["B"], v["A"])
    """
    pa, pb = _como_vetor(a), _como_vetor(b)
    direcao = pb - pa
    if np.linalg.norm(direcao) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    # Um plano perpendicular à reta tem a própria direção da reta como normal
    # (como um mastro fincado no chão: o mastro é o normal do chão).
    return _como_vetor(ponto), direcao


def plano_perpendicular_por_reta(a: Vetor3D, b: Vetor3D, plano: Plano) -> Plano:
    """Devolve o plano que contém a reta ab e é perpendicular a `plano`.

    Pela propriedade 3 da seção 5 do notebook 03, esse plano é único quando a reta
    não é perpendicular ao plano dado. Esta função não desenha nada, e as contas
    internas usam vetores, assunto do módulo 07 (Geometria Analítica).

    Parâmetros:
        a, b: dois pontos distintos da reta que fica contida no plano procurado.
        plano: o plano dado, como (ponto, normal).

    Retorna:
        Uma tupla (ponto, normal). O ponto devolvido é o ponto médio de ab.

    Levanta:
        ValueError: se a e b forem iguais, ou se a reta ab for perpendicular ao
            plano dado.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> base = plano_por_tres_pontos(v["A"], v["B"], v["C"])
        >>> pi = plano_perpendicular_por_reta(v["A"], v["G"], base)
    """
    pa, pb = _como_vetor(a), _como_vetor(b)
    normal_dado = _como_vetor(plano[1])
    direcao = pb - pa
    if np.linalg.norm(direcao) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )

    # O plano procurado contém a direção da reta e a direção do normal do plano
    # dado (é ela que o deixa "em pé" sobre o plano dado). O seu normal é
    # perpendicular a essas duas direções. Se a reta já tem a direção do normal,
    # o produto vetorial dá zero: qualquer plano do feixe por ab serviria.
    normal = np.cross(direcao, normal_dado)
    escala = np.linalg.norm(direcao) * np.linalg.norm(normal_dado)
    if np.linalg.norm(normal) <= TOLERANCIA * escala:
        raise ValueError(
            "A reta é perpendicular ao plano: existem infinitos planos que a contêm "
            "e são perpendiculares a ele."
        )
    return (pa + pb) / 2, normal


def perpendicular_comum(
    a1: Vetor3D, a2: Vetor3D, b1: Vetor3D, b2: Vetor3D
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Devolve os pés da perpendicular comum às retas reversas a1a2 e b1b2.

    A perpendicular comum é a única reta que corta as duas retas reversas e forma
    90° com ambas (notebook 04). O segmento entre os dois pés mede a distância entre
    as retas. Esta função não desenha nada, e as contas internas usam vetores,
    assunto do módulo 07 (Geometria Analítica).

    Parâmetros:
        a1, a2: dois pontos distintos da reta r.
        b1, b2: dois pontos distintos da reta s.

    Retorna:
        Uma tupla (p, q): p é o pé na reta r e q é o pé na reta s, cada um como
        (x, y, z).

    Levanta:
        ValueError: se as retas não forem reversas.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> p, q = perpendicular_comum(v["A"], v["B"], v["C"], v["G"])
        >>> p, q
        ((2.0, 0.0, 0.0), (2.0, 2.0, 0.0))
    """
    if classificar_posicao_retas(a1, a2, b1, b2) != "reversas":
        raise ValueError(
            "As retas não são reversas: a perpendicular comum só é definida (e única) "
            "para retas reversas."
        )
    pa1, pa2 = _como_vetor(a1), _como_vetor(a2)
    pb1, pb2 = _como_vetor(b1), _como_vetor(b2)
    u = pa2 - pa1  # direção de r
    v = pb2 - pb1  # direção de s
    w = pa1 - pb1  # vai de um ponto de s até um ponto de r

    # Procuramos p = a1 + t·u em r e q = b1 + k·v em s tais que o segmento pq forme
    # 90° com u e com v. Isso dá duas equações (produtos escalares iguais a zero)
    # com duas incógnitas, t e k, resolvidas pela regra de Cramer.
    uu, uv, vv = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    uw, vw = np.dot(u, w), np.dot(v, w)
    determinante = uu * vv - uv**2  # não é zero, porque as retas não são paralelas
    t = (uv * vw - vv * uw) / determinante
    k = (uu * vw - uv * uw) / determinante

    px, py, pz = pa1 + t * u
    qx, qy, qz = pb1 + k * v
    return (float(px), float(py), float(pz)), (float(qx), float(qy), float(qz))


def plano_mediador(a: Vetor3D, b: Vetor3D) -> Plano:
    """Devolve o plano mediador do segmento ab.

    O plano mediador é o plano perpendicular a ab que passa pelo seu ponto médio.
    É o lugar geométrico dos pontos equidistantes de a e de b (notebook 04), a
    versão espacial da mediatriz. Esta função não desenha nada, e as contas
    internas usam vetores, assunto do módulo 07 (Geometria Analítica).

    Parâmetros:
        a, b: as extremidades do segmento, cada uma como (x, y, z).

    Retorna:
        Uma tupla (ponto, normal). O ponto devolvido é o ponto médio de ab.

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> plano_mediador((0, 0, 0), (2, 0, 0))[0]
        array([1., 0., 0.])
    """
    pa, pb = _como_vetor(a), _como_vetor(b)
    if np.linalg.norm(pb - pa) < TOLERANCIA:
        raise ValueError("Os dois pontos são iguais: não existe plano mediador.")
    # O próprio segmento ab dá a direção do normal (o plano é perpendicular a ele).
    return (pa + pb) / 2, pb - pa


def planos_bissetores(plano1: Plano, plano2: Plano) -> tuple[Plano, Plano]:
    """Devolve os dois planos bissetores de dois planos secantes.

    Os planos bissetores são formados pelos pontos equidistantes dos dois planos
    dados. Eles contêm a reta de interseção dos planos dados e são perpendiculares
    entre si (notebook 04). Esta função não desenha nada, e as contas internas usam
    vetores, assunto do módulo 07 (Geometria Analítica).

    Parâmetros:
        plano1: o primeiro plano, como (ponto, normal).
        plano2: o segundo plano, como (ponto, normal).

    Retorna:
        Uma tupla (bissetor1, bissetor2), cada um como (ponto, normal). O ponto
        devolvido é um ponto da reta de interseção dos planos dados.

    Levanta:
        ValueError: se os planos não forem secantes.

    Exemplo:
        >>> chao = ((0, 0, 0), (0, 0, 1))
        >>> parede = ((0, 0, 0), (0, 1, 0))
        >>> bissetor1, bissetor2 = planos_bissetores(chao, parede)
    """
    reta = _intersecao_de_planos(plano1, plano2)
    if reta is None:
        raise ValueError(
            "Os planos não são secantes: os planos bissetores só existem para planos "
            "que se cortam."
        )
    ponto, _ = reta

    # Com os normais de comprimento 1, a soma e a diferença deles apontam nas
    # direções que "dividem ao meio" os ângulos entre os planos (como as diagonais
    # de um losango). São os normais dos dois bissetores.
    n1 = _normalizar(plano1[1])
    n2 = _normalizar(plano2[1])
    return (ponto, n1 + n2), (ponto.copy(), n1 - n2)


def vertices_do_diedro(
    abertura: float, comprimento: float = 6, largura: float = 4
) -> dict[str, tuple[float, float, float]]:
    """Devolve os pontos de um diedro "padrão", pronto para os desenhos.

    O diedro padrão tem a aresta deitada, centrada na origem, com tamanho
    `comprimento`. A primeira face sai da aresta "para a frente" e a segunda face
    forma com a primeira o ângulo `abertura`. Assim, nos widgets, basta mexer num
    número para abrir ou fechar o diedro.

    Esta função não desenha nada. As contas internas usam coordenadas, assunto do
    módulo 07 (Geometria Analítica).

    Parâmetros:
        abertura: a medida do diedro, em graus (entre 0° e 180°, sem incluir os
            extremos).
        comprimento: o tamanho do trecho da aresta.
        largura: a distância dos pontos das faces até a aresta.

    Retorna:
        Um dicionário com as chaves "aresta_inicio", "aresta_fim" (as pontas da
        aresta), "ponto_face1" e "ponto_face2" (um ponto de cada face).

    Levanta:
        ValueError: se a abertura não estiver entre 0° e 180°.

    Exemplo:
        >>> d = vertices_do_diedro(90)
        >>> d["ponto_face2"]
        (0.0, 0.0, 4.0)
    """
    if not 0 < abertura < 180:
        raise ValueError("A abertura de um diedro deve estar entre 0° e 180°.")
    c, largura_ = float(comprimento), float(largura)

    # Seno e cosseno servem só para "abrir" a segunda face: ela gira em volta da
    # aresta, saindo da primeira face, até formar o ângulo pedido.
    theta = np.radians(abertura)
    y2 = largura_ * np.cos(theta)
    z2 = largura_ * np.sin(theta)
    # Arredondamos para que aberturas como 90° deem 0.0, e não 2.4e-16.
    y2, z2 = round(float(y2), 12) + 0.0, round(float(z2), 12) + 0.0
    return {
        "aresta_inicio": (-c / 2, 0.0, 0.0),
        "aresta_fim": (c / 2, 0.0, 0.0),
        "ponto_face1": (0.0, largura_, 0.0),
        "ponto_face2": (0.0, y2, z2),
    }


def secao_do_diedro(
    a: Vetor3D, b: Vetor3D, p1: Vetor3D, p2: Vetor3D, plano: Plano
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
] | None:
    """Devolve a seção do diedro pelo plano dado: um ângulo.

    O diedro é dado pela aresta (os pontos a e b) e por um ponto de cada face (p1 e
    p2). O plano corta a aresta num ponto, o vértice da seção, e corta cada face numa
    semirreta que sai desse vértice: as duas semirretas formam a seção.

    Esta função não desenha nada. As contas internas usam vetores, assunto do módulo
    07 (Geometria Analítica).

    Parâmetros:
        a, b: dois pontos distintos da aresta.
        p1: um ponto da primeira face (fora da aresta).
        p2: um ponto da segunda face (fora da aresta).
        plano: o plano de corte, como (ponto, normal).

    Retorna:
        Uma tupla (vertice, lado1, lado2): o vértice da seção e um ponto de cada
        semirreta da seção, a distância 1 do vértice. Devolve None se o plano for
        paralelo à aresta ou a contiver (nesses casos a seção não é um ângulo).

    Exemplo:
        >>> d = vertices_do_diedro(60)
        >>> vertice, lado1, lado2 = secao_do_diedro(
        ...     d["aresta_inicio"], d["aresta_fim"], d["ponto_face1"], d["ponto_face2"],
        ...     ((0, 0, 0), (1, 0, 0)),
        ... )
    """
    vertice = intersecao_reta_plano(a, b, plano)
    if vertice is None:
        return None
    v = np.array(vertice)
    normal_corte = _como_vetor(plano[1])

    lados = []
    for ponto_da_face in (p1, p2):
        d, w = _direcao_do_semiplano(a, b, ponto_da_face)
        # A semirreta da seção está ao mesmo tempo no plano da face e no plano de
        # corte, então é perpendicular aos dois normais (produto vetorial).
        normal_face = np.cross(d, w)
        direcao = _normalizar(np.cross(normal_face, normal_corte))
        # Das duas semirretas dessa reta, ficamos com a que entra no semiplano
        # (a que aponta para o lado do ponto da face).
        if np.dot(direcao, w) < 0:
            direcao = -direcao
        x, y, z = v + direcao
        lados.append((float(x), float(y), float(z)))
    return vertice, lados[0], lados[1]


def secao_igualmente_inclinada(
    a: Vetor3D,
    b: Vetor3D,
    p1: Vetor3D,
    p2: Vetor3D,
    vertice: Vetor3D,
    inclinacao: float,
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    """Devolve uma seção igualmente inclinada do diedro.

    A partir de `vertice` (um ponto da aresta), traçamos uma semirreta em cada face,
    as duas formando o mesmo ângulo `inclinacao` com a aresta, do mesmo lado (no
    sentido que vai de a para b). Com inclinacao=90, a seção obtida é a seção reta.

    O diedro é dado pela aresta (os pontos a e b) e por um ponto de cada face (p1 e
    p2). Esta função não desenha nada. As contas internas usam vetores, assunto do
    módulo 07 (Geometria Analítica).

    Parâmetros:
        a, b: dois pontos distintos da aresta.
        p1: um ponto da primeira face (fora da aresta).
        p2: um ponto da segunda face (fora da aresta).
        vertice: o vértice da seção, um ponto da aresta.
        inclinacao: o ângulo, em graus, entre cada semirreta e a aresta (maior que
            0° e no máximo 90°).

    Retorna:
        Uma tupla (vertice, lado1, lado2): o vértice e um ponto de cada semirreta,
        a distância 1 do vértice.

    Levanta:
        ValueError: se o vértice não estiver na aresta ou se a inclinação não
            estiver entre 0° (exclusive) e 90°.

    Exemplo:
        >>> d = vertices_do_diedro(90)
        >>> vertice, lado1, lado2 = secao_igualmente_inclinada(
        ...     d["aresta_inicio"], d["aresta_fim"], d["ponto_face1"], d["ponto_face2"],
        ...     (0, 0, 0), 45,
        ... )
    """
    if not 0 < inclinacao <= 90:
        raise ValueError("A inclinação deve estar entre 0° (exclusive) e 90°.")
    pa, v = _como_vetor(a), _como_vetor(vertice)
    d, w1 = _direcao_do_semiplano(a, b, p1)
    _, w2 = _direcao_do_semiplano(a, b, p2)

    # O vértice está na aresta quando a→vértice não tem nenhuma parte "lateral".
    av = v - pa
    lateral = av - np.dot(av, d) * d
    if np.linalg.norm(lateral) > 1e-6 * max(np.linalg.norm(av), 1.0):
        raise ValueError("O vértice da seção precisa estar na aresta do diedro.")

    # Em cada face, andamos cos(φ) ao longo da aresta e sen(φ) para dentro da face:
    # a semirreta forma o ângulo φ com a aresta.
    phi = np.radians(inclinacao)
    lados = []
    for w in (w1, w2):
        x, y, z = v + np.cos(phi) * d + np.sin(phi) * w
        lados.append((float(x), float(y), float(z)))
    x, y, z = v
    return (float(x), float(y), float(z)), lados[0], lados[1]


def ponto_do_semiplano_bissetor(
    a: Vetor3D, b: Vetor3D, p1: Vetor3D, p2: Vetor3D
) -> tuple[float, float, float]:
    """Devolve um ponto do semiplano bissetor do diedro.

    O semiplano bissetor tem origem na aresta e divide o diedro em dois diedros
    adjacentes congruentes. O semiplano de origem ab que passa pelo ponto devolvido
    é o bissetor: use-o, por exemplo, em `desenhar_semiplano(fig, a, b, ponto)`.

    O diedro é dado pela aresta (os pontos a e b) e por um ponto de cada face (p1 e
    p2). Esta função não desenha nada. As contas internas usam vetores, assunto do
    módulo 07 (Geometria Analítica).

    Parâmetros:
        a, b: dois pontos distintos da aresta.
        p1: um ponto da primeira face (fora da aresta).
        p2: um ponto da segunda face (fora da aresta).

    Retorna:
        Um ponto do semiplano bissetor, a distância 1 do ponto a, como (x, y, z).

    Levanta:
        ValueError: se as duas faces formarem um único plano (não há diedro).

    Exemplo:
        >>> ponto_do_semiplano_bissetor((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        (0.0, 0.7071067811865475, 0.7071067811865475)
    """
    _, w1 = _direcao_do_semiplano(a, b, p1)
    _, w2 = _direcao_do_semiplano(a, b, p2)

    # Como w1 e w2 têm o mesmo comprimento, a soma deles aponta no meio das duas
    # faces (é a diagonal de um losango, que divide o ângulo ao meio).
    meio = w1 + w2
    if np.linalg.norm(meio) < 1e-6:
        raise ValueError(
            "As duas faces formam um único plano: elas não determinam um diedro."
        )
    x, y, z = _como_vetor(a) + _normalizar(meio)
    return float(x), float(y), float(z)


# --- Desenhos ---
def desenhar_ponto(
    fig: go.Figure,
    ponto: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    tamanho: float = 6,
) -> go.Figure:
    """Desenha um ponto, com um rótulo opcional ao lado.

    Parâmetros:
        fig: a figura.
        ponto: posição do ponto, como (x, y, z).
        rotulo: nome do ponto, por exemplo "A".
        cor: cor do ponto. Se não for informada, usa CORES["ponto"].
        tamanho: tamanho do marcador.

    Retorna:
        A mesma figura, com o ponto desenhado.

    Exemplo:
        >>> fig = desenhar_ponto(fig, (1, 1, 0), rotulo="A")
    """
    x, y, z = _como_vetor(ponto)
    cor = cor or CORES["ponto"]
    fig.add_trace(
        go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers+text" if rotulo else "markers",
            marker=dict(size=tamanho, color=cor),
            text=[rotulo] if rotulo else None,
            # O texto fica um pouco acima do ponto, para não cobrir o marcador.
            textposition="top center",
            textfont=dict(color=cor, size=15),
            hoverinfo="text" if rotulo else "skip",
        )
    )
    return fig


def desenhar_pontos(
    fig: go.Figure,
    pontos: Sequence[Vetor3D],
    rotulos: Sequence[str] | None = None,
    cor: str | None = None,
    tamanho: float = 6,
) -> go.Figure:
    """Desenha vários pontos de uma vez.

    Parâmetros:
        fig: a figura.
        pontos: lista de pontos, cada um como (x, y, z).
        rotulos: lista de nomes, na mesma ordem dos pontos (opcional).
        cor: cor dos pontos. Se não for informada, usa CORES["ponto"].
        tamanho: tamanho dos marcadores.

    Retorna:
        A mesma figura, com os pontos desenhados.

    Exemplo:
        >>> fig = desenhar_pontos(fig, [(0, 0, 0), (1, 0, 0)], rotulos=["A", "B"])
    """
    if rotulos is not None and len(rotulos) != len(pontos):
        raise ValueError("A lista de rótulos precisa ter um nome para cada ponto.")
    for indice, ponto in enumerate(pontos):
        rotulo = rotulos[indice] if rotulos is not None else None
        fig = desenhar_ponto(fig, ponto, rotulo=rotulo, cor=cor, tamanho=tamanho)
    return fig


def desenhar_segmento(
    fig: go.Figure,
    inicio: Vetor3D,
    fim: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    espessura: float = 4,
    tracejado: bool = False,
) -> go.Figure:
    """Desenha o segmento de reta que liga dois pontos.

    Parâmetros:
        fig: a figura.
        inicio: uma das pontas do segmento, como (x, y, z).
        fim: a outra ponta.
        rotulo: nome do segmento, escrito no ponto médio (opcional).
        cor: cor do segmento. Se não for informada, usa CORES["apoio"].
        espessura: espessura da linha.
        tracejado: se True, desenha a linha tracejada.

    Retorna:
        A mesma figura, com o segmento desenhado.

    Exemplo:
        >>> fig = desenhar_segmento(fig, (0, 0, 0), (2, 0, 0), rotulo="AB")
    """
    a = _como_vetor(inicio)
    b = _como_vetor(fim)
    cor = cor or CORES["apoio"]
    fig.add_trace(
        go.Scatter3d(
            x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
            mode="lines",
            line=dict(
                color=cor, width=espessura, dash="dash" if tracejado else "solid"
            ),
            hoverinfo="skip",
        )
    )
    if rotulo:
        fig = adicionar_rotulo(fig, (a + b) / 2, rotulo, cor=cor)
    return fig


def desenhar_reta(
    fig: go.Figure,
    ponto: Vetor3D,
    direcao: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    comprimento: float = 10,
    espessura: float = 4,
    tracejado: bool = False,
) -> go.Figure:
    """Desenha (um trecho de) uma reta.

    Uma reta é infinita, e não dá para desenhar algo infinito. Por isso a função
    desenha apenas um trecho de tamanho `comprimento`, centrado em `ponto` e
    apontando na `direcao` dada. Lembre-se: a reta continua além das pontas.

    Parâmetros:
        fig: a figura.
        ponto: um ponto da reta, como (x, y, z). O trecho fica centrado nele.
        direcao: um vetor que indica para onde a reta "aponta".
        rotulo: nome da reta, escrito perto de uma das pontas (opcional).
        cor: cor da reta. Se não for informada, usa CORES["reta"].
        comprimento: tamanho do trecho desenhado.
        espessura: espessura da linha.
        tracejado: se True, desenha a linha tracejada.

    Retorna:
        A mesma figura, com a reta desenhada.

    Exemplo:
        >>> fig = desenhar_reta(fig, ponto=(0, 0, 0), direcao=(1, 1, 0), rotulo="r")
    """
    centro = _como_vetor(ponto)
    d = _normalizar(direcao)
    cor = cor or CORES["reta"]

    # As duas pontas ficam a meio comprimento do centro, uma para cada lado.
    ponta1 = centro - d * comprimento / 2
    ponta2 = centro + d * comprimento / 2
    fig = desenhar_segmento(
        fig, ponta1, ponta2, cor=cor, espessura=espessura, tracejado=tracejado
    )
    if rotulo:
        # O nome fica perto da ponta 2, um pouco para dentro do trecho.
        fig = adicionar_rotulo(fig, centro + d * comprimento * 0.45, rotulo, cor=cor)
    return fig


def desenhar_reta_por_dois_pontos(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    comprimento: float = 10,
    espessura: float = 4,
    tracejado: bool = False,
) -> go.Figure:
    """Desenha a reta determinada por dois pontos distintos.

    O trecho desenhado fica centrado no ponto médio de AB. Como em `desenhar_reta`,
    lembre-se de que a reta continua além das pontas.

    Parâmetros:
        fig: a figura.
        a: primeiro ponto, como (x, y, z).
        b: segundo ponto (diferente de a).
        rotulo: nome da reta (opcional).
        cor: cor da reta. Se não for informada, usa CORES["reta"].
        comprimento: tamanho do trecho desenhado.
        espessura: espessura da linha.
        tracejado: se True, desenha a linha tracejada.

    Retorna:
        A mesma figura, com a reta desenhada.

    Levanta:
        ValueError: se os dois pontos forem iguais.

    Exemplo:
        >>> fig = desenhar_reta_por_dois_pontos(fig, (0, 0, 0), (1, 2, 0), rotulo="r")
    """
    pa = _como_vetor(a)
    pb = _como_vetor(b)
    if np.linalg.norm(pb - pa) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    return desenhar_reta(
        fig, (pa + pb) / 2, pb - pa, rotulo=rotulo, cor=cor,
        comprimento=comprimento, espessura=espessura, tracejado=tracejado,
    )


def desenhar_paralela_por_ponto(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    ponto: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    comprimento: float = 10,
    espessura: float = 4,
    tracejado: bool = False,
) -> go.Figure:
    """Desenha a reta que passa por `ponto` e é paralela à reta ab.

    Pelo postulado das paralelas, essa reta é única. O trecho desenhado fica
    centrado em `ponto`.

    Parâmetros:
        fig: a figura.
        a, b: dois pontos distintos da reta de referência.
        ponto: o ponto por onde passa a paralela, como (x, y, z).
        rotulo: nome da paralela (opcional).
        cor: cor da reta. Se não for informada, usa CORES["reta"].
        comprimento: tamanho do trecho desenhado.
        espessura: espessura da linha.
        tracejado: se True, desenha a linha tracejada.

    Retorna:
        A mesma figura, com a paralela desenhada.

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> fig = desenhar_paralela_por_ponto(fig, (0, 0, 0), (1, 0, 0), (0, 2, 1))
    """
    pa, pb = _como_vetor(a), _como_vetor(b)
    if np.linalg.norm(pb - pa) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    # A paralela tem a mesma direção de ab; só muda o ponto por onde passa.
    return desenhar_reta(
        fig, ponto, pb - pa, rotulo=rotulo, cor=cor, comprimento=comprimento,
        espessura=espessura, tracejado=tracejado,
    )


def desenhar_perpendicular_ao_plano(
    fig: go.Figure,
    plano: Plano,
    ponto: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    comprimento: float = 6,
    espessura: float = 4,
    tracejado: bool = False,
    marcar_angulo_reto: bool = True,
) -> go.Figure:
    """Desenha a reta que passa por `ponto` e é perpendicular ao plano.

    Pela propriedade 2 do notebook 03, essa reta é única. O trecho desenhado fica
    centrado no pé da perpendicular (o ponto onde a reta encontra o plano).

    Parâmetros:
        fig: a figura.
        plano: o plano, como (ponto, normal).
        ponto: o ponto por onde a perpendicular passa, como (x, y, z). Pode estar
            dentro ou fora do plano.
        rotulo: nome da reta (opcional).
        cor: cor da reta. Se não for informada, usa CORES["reta"].
        comprimento: tamanho do trecho desenhado.
        espessura: espessura da linha.
        tracejado: se True, desenha a linha tracejada.
        marcar_angulo_reto: se True, desenha o quadradinho de ângulo reto no pé,
            entre a reta e uma direção qualquer do plano.

    Retorna:
        A mesma figura, com a perpendicular desenhada.

    Exemplo:
        >>> chao = ((0, 0, 0), (0, 0, 1))
        >>> fig = desenhar_perpendicular_ao_plano(fig, chao, (1, 1, 3), rotulo="r")
    """
    p = _como_vetor(ponto)
    pe = np.array(pe_da_perpendicular_ao_plano(p, plano))

    # A perpendicular tem a direção do normal. Escolhemos o sentido que vai do pé
    # até o ponto, para que o rótulo e o quadradinho fiquem do lado do ponto.
    direcao = _normalizar(plano[1])
    if np.dot(p - pe, direcao) < 0:
        direcao = -direcao

    fig = desenhar_reta(
        fig, pe, direcao, rotulo=rotulo, cor=cor, comprimento=comprimento,
        espessura=espessura, tracejado=tracejado,
    )
    if marcar_angulo_reto:
        # Qualquer direção do plano serve: a perpendicular forma 90° com todas.
        u, _ = _base_do_plano(direcao)
        fig = desenhar_angulo_reto(fig, pe, pe + u, pe + direcao, cor=cor)
    return fig


def desenhar_plano(
    fig: go.Figure,
    ponto: Vetor3D,
    normal: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.35,
    tamanho: float = 6,
    mostrar_borda: bool = True,
) -> go.Figure:
    """Desenha (um pedaço de) um plano.

    Um plano é infinito em todas as direções. Como não dá para desenhar isso, a
    função desenha um "pedaço" quadrado do plano, de lado `tamanho`, centrado em
    `ponto`. É a mesma ideia dos livros, que representam um plano por um
    paralelogramo. Lembre-se: o plano continua além da borda.

    A posição do plano é dada por um vetor `normal`, que é perpendicular a ele (como
    um mastro fincado no chão é perpendicular ao chão).

    Parâmetros:
        fig: a figura.
        ponto: um ponto do plano, como (x, y, z). O quadrado fica centrado nele.
        normal: vetor perpendicular ao plano. (0, 0, 1) dá um plano horizontal.
        rotulo: nome do plano, normalmente uma letra grega como "α" (opcional).
        cor: cor do plano. Se não for informada, usa CORES["plano"].
        opacidade: de 0 (invisível) a 1 (totalmente opaco).
        tamanho: lado do quadrado desenhado.
        mostrar_borda: se True, desenha o contorno do quadrado.

    Retorna:
        A mesma figura, com o plano desenhado.

    Exemplo:
        >>> fig = desenhar_plano(fig, ponto=(0, 0, 0), normal=(0, 0, 1), rotulo="α")
    """
    centro = _como_vetor(ponto)
    u, v = _base_do_plano(normal)
    cor = cor or CORES["plano"]
    metade = tamanho / 2

    # Os quatro cantos do quadrado: andamos meio lado na direção de u e de v,
    # para a frente ou para trás.
    cantos = np.array([
        centro - metade * u - metade * v,
        centro + metade * u - metade * v,
        centro + metade * u + metade * v,
        centro - metade * u + metade * v,
    ])

    # O Plotly desenha superfícies com triângulos: o quadrado vira dois triângulos,
    # (canto 0, canto 1, canto 2) e (canto 0, canto 2, canto 3).
    fig.add_trace(
        go.Mesh3d(
            x=cantos[:, 0], y=cantos[:, 1], z=cantos[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=cor,
            opacity=opacidade,
            # Iluminação "chapada": a cor do plano não muda com o ângulo de visão.
            lighting=dict(ambient=1, diffuse=0, specular=0, fresnel=0),
            hoverinfo="skip",
        )
    )

    if mostrar_borda:
        # Repetimos o primeiro canto no fim para fechar o contorno.
        contorno = np.vstack([cantos, cantos[:1]])
        fig.add_trace(
            go.Scatter3d(
                x=contorno[:, 0], y=contorno[:, 1], z=contorno[:, 2],
                mode="lines",
                line=dict(color=cor, width=2),
                hoverinfo="skip",
            )
        )

    if rotulo:
        # O nome fica perto do canto 3, um pouco para dentro do quadrado.
        posicao = centro - 0.4 * tamanho * u + 0.4 * tamanho * v
        fig = adicionar_rotulo(fig, posicao, rotulo, cor=cor, tamanho=18)
    return fig


def desenhar_plano_por_tres_pontos(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    c: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.35,
    tamanho: float = 6,
    mostrar_borda: bool = True,
) -> go.Figure:
    """Desenha o plano determinado por três pontos não colineares.

    O pedaço desenhado fica centrado no baricentro dos três pontos (o "ponto de
    equilíbrio" do triângulo ABC). Se o `tamanho` pedido for pequeno demais para
    os três pontos aparecerem dentro do quadrado, ele é aumentado automaticamente.

    Parâmetros:
        fig: a figura.
        a, b, c: os três pontos, cada um como (x, y, z).
        rotulo: nome do plano (opcional).
        cor: cor do plano. Se não for informada, usa CORES["plano"].
        opacidade: de 0 (invisível) a 1 (totalmente opaco).
        tamanho: lado do quadrado desenhado (tamanho mínimo).
        mostrar_borda: se True, desenha o contorno do quadrado.

    Retorna:
        A mesma figura, com o plano desenhado.

    Levanta:
        ValueError: se os três pontos forem colineares.

    Exemplo:
        >>> fig = desenhar_plano_por_tres_pontos(fig, (0, 0, 0), (1, 0, 0), (0, 1, 1))
    """
    pa, pb, pc = _como_vetor(a), _como_vetor(b), _como_vetor(c)

    # O construtor já confere se os pontos são colineares (e avisa, se forem).
    baricentro, normal = plano_por_tres_pontos(pa, pb, pc)

    # Medimos o quanto cada ponto se afasta do centro nas direções u e v do plano.
    # O quadrado precisa de lado maior que o dobro do maior afastamento (com folga).
    u, v = _base_do_plano(normal)
    maior_afastamento = max(
        max(abs(np.dot(p - baricentro, u)), abs(np.dot(p - baricentro, v)))
        for p in (pa, pb, pc)
    )
    tamanho = max(tamanho, 2 * maior_afastamento * 1.3)

    return desenhar_plano(
        fig, baricentro, normal, rotulo=rotulo, cor=cor, opacidade=opacidade,
        tamanho=tamanho, mostrar_borda=mostrar_borda,
    )


def desenhar_plano_paralelo_a_reta(
    fig: go.Figure,
    a1: Vetor3D,
    a2: Vetor3D,
    b1: Vetor3D,
    b2: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.35,
    tamanho: float = 6,
    mostrar_borda: bool = True,
) -> go.Figure:
    """Desenha o plano que contém a reta a1a2 e é paralelo à reta b1b2.

    O pedaço desenhado fica centrado no ponto médio de a1a2. Para obter o plano
    como dado, sem desenhar, use `plano_paralelo_a_reta`.

    Parâmetros:
        fig: a figura.
        a1, a2: dois pontos distintos da reta contida no plano.
        b1, b2: dois pontos distintos da reta paralela ao plano.
        rotulo: nome do plano (opcional).
        cor: cor do plano. Se não for informada, usa CORES["plano"].
        opacidade: de 0 (invisível) a 1 (totalmente opaco).
        tamanho: lado do quadrado desenhado.
        mostrar_borda: se True, desenha o contorno do quadrado.

    Retorna:
        A mesma figura, com o plano desenhado.

    Levanta:
        ValueError: se as duas retas forem paralelas (ou coincidentes).

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> fig = desenhar_plano_paralelo_a_reta(fig, v["A"], v["B"], v["C"], v["G"])
    """
    ponto, normal = plano_paralelo_a_reta(a1, a2, b1, b2)
    return desenhar_plano(
        fig, ponto, normal, rotulo=rotulo, cor=cor, opacidade=opacidade,
        tamanho=tamanho, mostrar_borda=mostrar_borda,
    )


def desenhar_plano_perpendicular_a_reta(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    ponto: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.35,
    tamanho: float = 6,
    mostrar_borda: bool = True,
) -> go.Figure:
    """Desenha o plano que passa por `ponto` e é perpendicular à reta ab.

    O pedaço desenhado fica centrado em `ponto`. Para obter o plano como dado, sem
    desenhar, use `plano_perpendicular_a_reta`.

    Parâmetros:
        fig: a figura.
        a, b: dois pontos distintos da reta.
        ponto: o ponto por onde o plano passa, como (x, y, z).
        rotulo: nome do plano (opcional).
        cor: cor do plano. Se não for informada, usa CORES["plano"].
        opacidade: de 0 (invisível) a 1 (totalmente opaco).
        tamanho: lado do quadrado desenhado.
        mostrar_borda: se True, desenha o contorno do quadrado.

    Retorna:
        A mesma figura, com o plano desenhado.

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> fig = desenhar_plano_perpendicular_a_reta(fig, v["A"], v["B"], v["B"])
    """
    ponto_do_plano, normal = plano_perpendicular_a_reta(a, b, ponto)
    return desenhar_plano(
        fig, ponto_do_plano, normal, rotulo=rotulo, cor=cor, opacidade=opacidade,
        tamanho=tamanho, mostrar_borda=mostrar_borda,
    )


def desenhar_plano_perpendicular_por_reta(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    plano: Plano,
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.35,
    tamanho: float = 6,
    mostrar_borda: bool = True,
) -> go.Figure:
    """Desenha o plano que contém a reta ab e é perpendicular a `plano`.

    O pedaço desenhado fica centrado no ponto médio de ab. Para obter o plano como
    dado, sem desenhar, use `plano_perpendicular_por_reta`.

    Parâmetros:
        fig: a figura.
        a, b: dois pontos distintos da reta contida no plano desenhado.
        plano: o plano dado, como (ponto, normal).
        rotulo: nome do plano (opcional).
        cor: cor do plano. Se não for informada, usa CORES["plano"].
        opacidade: de 0 (invisível) a 1 (totalmente opaco).
        tamanho: lado do quadrado desenhado.
        mostrar_borda: se True, desenha o contorno do quadrado.

    Retorna:
        A mesma figura, com o plano desenhado.

    Levanta:
        ValueError: se a e b forem iguais, ou se a reta ab for perpendicular ao
            plano dado.

    Exemplo:
        >>> chao = ((0, 0, 0), (0, 0, 1))
        >>> fig = desenhar_plano_perpendicular_por_reta(fig, (0, 0, 0), (1, 0, 1), chao)
    """
    ponto_do_plano, normal = plano_perpendicular_por_reta(a, b, plano)
    return desenhar_plano(
        fig, ponto_do_plano, normal, rotulo=rotulo, cor=cor, opacidade=opacidade,
        tamanho=tamanho, mostrar_borda=mostrar_borda,
    )


def desenhar_intersecao_de_planos(
    fig: go.Figure,
    plano1: Plano,
    plano2: Plano,
    rotulo: str | None = None,
    cor: str | None = None,
    comprimento: float = 10,
    espessura: float = 6,
) -> go.Figure:
    """Desenha a reta em que dois planos secantes se cortam.

    Cada plano é dado como uma tupla (ponto, normal), a mesma informação usada em
    `desenhar_plano`. Se os planos forem paralelos ou coincidentes, não há uma reta
    de interseção para desenhar: a figura volta sem alterações (use
    `planos_sao_paralelos` para saber qual é o caso).

    Parâmetros:
        fig: a figura.
        plano1: o primeiro plano, como (ponto, normal).
        plano2: o segundo plano, como (ponto, normal).
        rotulo: nome da reta (opcional).
        cor: cor da reta. Se não for informada, usa CORES["destaque"].
        comprimento: tamanho do trecho desenhado.
        espessura: espessura da linha.

    Retorna:
        A mesma figura, com a reta de interseção desenhada (se existir).

    Exemplo:
        >>> alfa = ((0, 0, 0), (0, 0, 1))
        >>> beta = ((0, 0, 0), (1, 0, 0))
        >>> fig = desenhar_intersecao_de_planos(fig, alfa, beta, rotulo="r")
    """
    reta = _intersecao_de_planos(plano1, plano2)
    if reta is None:
        return fig
    ponto, direcao = reta
    return desenhar_reta(
        fig, ponto, direcao, rotulo=rotulo, cor=cor or CORES["destaque"],
        comprimento=comprimento, espessura=espessura,
    )


def desenhar_intersecao_reta_plano(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    plano: Plano,
    rotulo: str | None = None,
    cor: str | None = None,
    tamanho: float = 7,
) -> go.Figure:
    """Marca o ponto onde a reta ab "fura" o plano.

    Se a reta for paralela ao plano ou estiver contida nele, não existe um único
    ponto de interseção: a figura volta sem alterações (use
    `classificar_posicao_reta_plano` para saber qual é o caso).

    Parâmetros:
        fig: a figura.
        a, b: dois pontos distintos da reta.
        plano: o plano, como (ponto, normal).
        rotulo: nome do ponto, por exemplo "P" (opcional).
        cor: cor do ponto. Se não for informada, usa CORES["destaque"].
        tamanho: tamanho do marcador.

    Retorna:
        A mesma figura, com o ponto de interseção marcado (se existir).

    Exemplo:
        >>> alfa = ((0, 0, 0), (0, 0, 1))
        >>> fig = desenhar_intersecao_reta_plano(fig, (0, 0, 1), (1, 1, 2), alfa, "P")
    """
    ponto = intersecao_reta_plano(a, b, plano)
    if ponto is None:
        return fig
    return desenhar_ponto(
        fig, ponto, rotulo=rotulo, cor=cor or CORES["destaque"], tamanho=tamanho
    )


def desenhar_poligono(
    fig: go.Figure,
    vertices: Sequence[Vetor3D],
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.4,
    mostrar_borda: bool = True,
) -> go.Figure:
    """Desenha um polígono plano e convexo, preenchido e com borda.

    A função supõe que todos os vértices estão num mesmo plano e que o polígono é
    convexo (sem "dentes"), como um triângulo, um quadrado ou um hexágono regular.
    Os vértices devem vir na ordem em que aparecem ao percorrer o contorno.

    Parâmetros:
        fig: a figura.
        vertices: lista de vértices, cada um como (x, y, z), na ordem do contorno.
        rotulo: nome do polígono, escrito no centro dele (opcional).
        cor: cor do polígono. Se não for informada, usa CORES["solido"].
        opacidade: de 0 (invisível) a 1 (totalmente opaco).
        mostrar_borda: se True, desenha o contorno do polígono.

    Retorna:
        A mesma figura, com o polígono desenhado.

    Levanta:
        ValueError: se houver menos de 3 vértices.

    Exemplo:
        >>> fig = desenhar_poligono(fig, [(0, 0, 0), (2, 0, 0), (0, 2, 0)], "ABC")
    """
    if len(vertices) < 3:
        raise ValueError("Um polígono precisa de pelo menos 3 vértices.")
    pontos = np.array([_como_vetor(v) for v in vertices])
    cor = cor or CORES["solido"]
    quantidade = len(pontos)

    # Triangulação "em leque": ligamos o primeiro vértice a cada par de vértices
    # vizinhos seguintes. Num polígono convexo, isso cobre tudo sem sobrar buraco:
    # (0, 1, 2), (0, 2, 3), (0, 3, 4), ...
    fig.add_trace(
        go.Mesh3d(
            x=pontos[:, 0], y=pontos[:, 1], z=pontos[:, 2],
            i=[0] * (quantidade - 2),
            j=list(range(1, quantidade - 1)),
            k=list(range(2, quantidade)),
            color=cor,
            opacity=opacidade,
            lighting=dict(ambient=1, diffuse=0, specular=0, fresnel=0),
            hoverinfo="skip",
        )
    )

    if mostrar_borda:
        # Repetimos o primeiro vértice no fim para fechar o contorno.
        contorno = np.vstack([pontos, pontos[:1]])
        fig.add_trace(
            go.Scatter3d(
                x=contorno[:, 0], y=contorno[:, 1], z=contorno[:, 2],
                mode="lines",
                line=dict(color=cor, width=3),
                hoverinfo="skip",
            )
        )

    if rotulo:
        # O centroide (a média dos vértices) fica dentro de um polígono convexo.
        fig = adicionar_rotulo(fig, pontos.mean(axis=0), rotulo, cor=cor)
    return fig


def desenhar_circunferencia(
    fig: go.Figure,
    centro: Vetor3D,
    raio: float,
    normal: Vetor3D = (0, 0, 1),
    rotulo: str | None = None,
    cor: str | None = None,
    espessura: float = 3,
    tracejado: bool = False,
    resolucao: int = 80,
) -> go.Figure:
    """Desenha uma circunferência no espaço.

    A circunferência fica no plano que passa por `centro` e tem o vetor `normal`
    como perpendicular. Com o normal padrão (0, 0, 1), ela fica "deitada", num
    plano horizontal.

    Parâmetros:
        fig: a figura.
        centro: o centro da circunferência, como (x, y, z).
        raio: o raio.
        normal: vetor perpendicular ao plano da circunferência.
        rotulo: nome da circunferência, escrito junto dela (opcional).
        cor: cor da linha. Se não for informada, usa CORES["reta"].
        espessura: espessura da linha.
        tracejado: se True, desenha a linha tracejada.
        resolucao: quantidade de pontos usados para desenhar a curva (mais pontos,
            curva mais lisa).

    Retorna:
        A mesma figura, com a circunferência desenhada.

    Exemplo:
        >>> fig = desenhar_circunferencia(fig, (0, 0, 0), 2, rotulo="λ")
    """
    c = _como_vetor(centro)
    u, v = _base_do_plano(normal)
    cor = cor or CORES["reta"]

    # Seno e cosseno dão os pontos da circunferência: andamos raio·cos(t) na
    # direção u e raio·sen(t) na direção v, com t dando uma volta completa.
    angulos = np.linspace(0, 2 * np.pi, resolucao + 1)
    curva = np.array([c + raio * (np.cos(t) * u + np.sin(t) * v) for t in angulos])
    fig.add_trace(
        go.Scatter3d(
            x=curva[:, 0], y=curva[:, 1], z=curva[:, 2],
            mode="lines",
            line=dict(
                color=cor, width=espessura, dash="dash" if tracejado else "solid"
            ),
            hoverinfo="skip",
        )
    )
    if rotulo:
        # O nome fica um pouco para fora da curva, na direção u.
        fig = adicionar_rotulo(fig, c + 1.12 * raio * u, rotulo, cor=cor)
    return fig


def desenhar_nuvem_de_pontos(
    fig: go.Figure,
    pontos: Sequence[Vetor3D] | np.ndarray,
    cor: str | None = None,
    tamanho: float = 2,
    opacidade: float = 0.6,
) -> go.Figure:
    """Desenha muitos pontos de uma vez, sem rótulos.

    Diferente de `desenhar_pontos`, que cria um desenho para cada ponto, esta
    função junta todos os pontos num único desenho. Assim ela continua rápida mesmo
    com milhares de pontos (por exemplo, pontos sorteados ao acaso).

    Parâmetros:
        fig: a figura.
        pontos: lista de pontos (x, y, z) ou array do NumPy de formato (N, 3).
        cor: cor dos pontos. Se não for informada, usa CORES["ponto"].
        tamanho: tamanho dos marcadores.
        opacidade: de 0 (invisível) a 1 (totalmente opaco).

    Retorna:
        A mesma figura, com os pontos desenhados. Se a lista estiver vazia, a
        figura volta sem alterações.

    Exemplo:
        >>> sorteados = np.random.default_rng(1).uniform(-1, 1, size=(500, 3))
        >>> fig = desenhar_nuvem_de_pontos(fig, sorteados)
    """
    if len(pontos) == 0:
        return fig
    nuvem = np.asarray(pontos, dtype=float)
    if nuvem.ndim != 2 or nuvem.shape[1] != 3:
        raise ValueError(
            "Esperava uma lista de pontos (x, y, z) ou um array de formato (N, 3)."
        )
    fig.add_trace(
        go.Scatter3d(
            x=nuvem[:, 0], y=nuvem[:, 1], z=nuvem[:, 2],
            mode="markers",
            marker=dict(
                size=tamanho, color=cor or CORES["ponto"], opacity=opacidade
            ),
            hoverinfo="skip",
        )
    )
    return fig


def desenhar_projecao_ortogonal(
    fig: go.Figure,
    pontos: Sequence[Vetor3D],
    plano: Plano,
    rotulos: Sequence[str] | None = None,
    cor: str | None = None,
    ligar_pontos: bool = True,
    fechar: bool = False,
    preencher: bool = False,
    mostrar_projetantes: bool = True,
    tamanho: float = 5,
) -> go.Figure:
    """Desenha a projeção ortogonal de uma figura sobre um plano.

    Cada ponto é projetado no pé da sua perpendicular ao plano (veja
    `pe_da_perpendicular_ao_plano`). Ligando as projeções na ordem dada, obtemos a
    projeção de um segmento, de uma linha poligonal ou de um polígono.

    Parâmetros:
        fig: a figura.
        pontos: os pontos da figura, cada um como (x, y, z), na ordem do contorno.
        plano: o plano de projeção, como (ponto, normal).
        rotulos: nomes dos pontos projetados, na mesma ordem (opcional), por
            exemplo ["A'", "B'"].
        cor: cor da projeção. Se não for informada, usa CORES["destaque"].
        ligar_pontos: se True, liga os pontos projetados em sequência.
        fechar: se True (com ligar_pontos=True), liga também o último ao primeiro.
        preencher: se True, pinta o polígono projetado (exige fechar=True).
        mostrar_projetantes: se True, desenha tracejados os segmentos que vão de
            cada ponto até a sua projeção.
        tamanho: tamanho dos marcadores dos pontos projetados.

    Retorna:
        A mesma figura, com a projeção desenhada.

    Levanta:
        ValueError: se preencher=True sem fechar=True, ou se a lista de rótulos não
            tiver um nome para cada ponto.

    Exemplo:
        >>> chao = ((0, 0, 0), (0, 0, 1))
        >>> fig = desenhar_projecao_ortogonal(
        ...     fig, [(0, 0, 1), (2, 1, 2)], chao, rotulos=["A'", "B'"]
        ... )
    """
    if preencher and not fechar:
        raise ValueError(
            "Para preencher a projeção, ela precisa ser um polígono: use fechar=True."
        )
    cor = cor or CORES["destaque"]
    originais = [_como_vetor(p) for p in pontos]
    projetados = [np.array(pe_da_perpendicular_ao_plano(p, plano)) for p in originais]

    if mostrar_projetantes:
        for original, projetado in zip(originais, projetados, strict=True):
            # Um ponto que já está no plano é a sua própria projeção: não há
            # projetante para desenhar.
            if np.linalg.norm(original - projetado) > TOLERANCIA:
                fig = desenhar_segmento(
                    fig, original, projetado, cor=CORES["apoio"], espessura=2,
                    tracejado=True,
                )

    if preencher:
        fig = desenhar_poligono(
            fig, projetados, cor=cor, opacidade=0.3, mostrar_borda=False
        )

    if ligar_pontos and len(projetados) >= 2:
        # Liga cada ponto ao seguinte; com fechar=True, o último volta ao primeiro.
        pares = [
            (projetados[i], projetados[i + 1]) for i in range(len(projetados) - 1)
        ]
        if fechar and len(projetados) >= 3:
            pares.append((projetados[-1], projetados[0]))
        for inicio, fim in pares:
            fig = desenhar_segmento(fig, inicio, fim, cor=cor, espessura=4)

    return desenhar_pontos(fig, projetados, rotulos=rotulos, cor=cor, tamanho=tamanho)


def desenhar_perpendicular_comum(
    fig: go.Figure,
    a1: Vetor3D,
    a2: Vetor3D,
    b1: Vetor3D,
    b2: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    espessura: float = 6,
    marcar_angulos_retos: bool = True,
    mostrar_pes: bool = True,
) -> go.Figure:
    """Desenha o segmento da perpendicular comum às retas reversas a1a2 e b1b2.

    O segmento liga o pé p, na reta r = a1a2, ao pé q, na reta s = b1b2, e o seu
    comprimento é a distância entre as retas (veja `perpendicular_comum`).

    Parâmetros:
        fig: a figura.
        a1, a2: dois pontos distintos da reta r.
        b1, b2: dois pontos distintos da reta s.
        rotulo: nome do segmento, escrito no ponto médio (opcional).
        cor: cor do segmento. Se não for informada, usa CORES["destaque"].
        espessura: espessura da linha.
        marcar_angulos_retos: se True, desenha os quadradinhos de ângulo reto nas
            duas pontas.
        mostrar_pes: se True, marca os pés da perpendicular comum.

    Retorna:
        A mesma figura, com a perpendicular comum desenhada.

    Levanta:
        ValueError: se as retas não forem reversas.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> fig = desenhar_perpendicular_comum(fig, v["A"], v["G"], v["B"], v["F"])
    """
    p, q = perpendicular_comum(a1, a2, b1, b2)
    p, q = np.array(p), np.array(q)
    cor = cor or CORES["destaque"]
    fig = desenhar_segmento(fig, p, q, rotulo=rotulo, cor=cor, espessura=espessura)

    if marcar_angulos_retos:
        # Em cada ponta, o quadradinho fica entre o segmento pq e a reta daquela ponta.
        # O tamanho acompanha o comprimento do segmento, para caber no desenho.
        u = _normalizar(_como_vetor(a2) - _como_vetor(a1))
        v = _normalizar(_como_vetor(b2) - _como_vetor(b1))
        lado = min(0.3, 0.25 * np.linalg.norm(q - p))
        fig = desenhar_angulo_reto(fig, p, p + u, q, cor=cor, tamanho=lado)
        fig = desenhar_angulo_reto(fig, q, q + v, p, cor=cor, tamanho=lado)
    if mostrar_pes:
        fig = desenhar_pontos(fig, [p, q], cor=cor, tamanho=5)
    return fig


def desenhar_angulo(
    fig: go.Figure,
    vertice: Vetor3D,
    ponto1: Vetor3D,
    ponto2: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    raio: float = 0.6,
    espessura: float = 3,
) -> go.Figure:
    """Desenha o arco que marca o ângulo entre as semirretas VP1 e VP2.

    O arco é um pedaço de circunferência de centro no vértice V, desenhado no plano
    das duas semirretas (como o arco que fazemos com o compasso na Geometria Plana).

    Parâmetros:
        fig: a figura.
        vertice: o vértice V do ângulo, como (x, y, z).
        ponto1: um ponto da primeira semirreta (diferente de V).
        ponto2: um ponto da segunda semirreta (diferente de V).
        rotulo: texto escrito junto ao arco, por exemplo "θ" ou "45°" (opcional).
        cor: cor do arco. Se não for informada, usa CORES["destaque"].
        raio: raio do arco.
        espessura: espessura da linha.

    Retorna:
        A mesma figura, com o arco desenhado.

    Exemplo:
        >>> fig = desenhar_angulo(fig, (0, 0, 0), (1, 0, 0), (1, 1, 0), rotulo="45°")
    """
    v = _como_vetor(vertice)
    u1 = _normalizar(_como_vetor(ponto1) - v)
    u2 = _normalizar(_como_vetor(ponto2) - v)
    cor = cor or CORES["destaque"]

    # Abertura do ângulo, em radianos. O clip evita erros de arredondamento como
    # cos = 1.0000000002, que o arco-cosseno não aceita.
    abertura = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
    if abertura < TOLERANCIA:
        # Semirretas sobrepostas: o ângulo é nulo e não há arco para desenhar.
        return fig

    # Montamos duas direções perpendiculares no plano do ângulo: e1 sobre a
    # primeira semirreta e e2 "girada de 90°" em direção à segunda.
    e1 = u1
    componente = u2 - np.dot(u2, u1) * u1
    if np.linalg.norm(componente) < TOLERANCIA:
        # Ângulo raso (180°): qualquer direção perpendicular serve para o arco.
        e2, _ = _base_do_plano(u1)
    else:
        e2 = _normalizar(componente)

    # Cerca de 30 pontos do arco, indo de e1 (ângulo 0) até u2 (ângulo `abertura`).
    angulos = np.linspace(0, abertura, 30)
    arco = np.array([v + raio * (np.cos(t) * e1 + np.sin(t) * e2) for t in angulos])
    fig.add_trace(
        go.Scatter3d(
            x=arco[:, 0], y=arco[:, 1], z=arco[:, 2],
            mode="lines",
            line=dict(color=cor, width=espessura),
            hoverinfo="skip",
        )
    )

    if rotulo:
        # O texto fica sobre a bissetriz (metade da abertura), um pouco além do arco.
        meio = abertura / 2
        posicao = v + 1.4 * raio * (np.cos(meio) * e1 + np.sin(meio) * e2)
        fig = adicionar_rotulo(fig, posicao, rotulo, cor=cor)
    return fig


def desenhar_angulo_reto(
    fig: go.Figure,
    vertice: Vetor3D,
    ponto1: Vetor3D,
    ponto2: Vetor3D,
    cor: str | None = None,
    tamanho: float = 0.3,
) -> go.Figure:
    """Desenha o "quadradinho" que marca um ângulo reto entre VP1 e VP2.

    Parâmetros:
        fig: a figura.
        vertice: o vértice V do ângulo, como (x, y, z).
        ponto1: um ponto da primeira semirreta (diferente de V).
        ponto2: um ponto da segunda semirreta (diferente de V).
        cor: cor do quadradinho. Se não for informada, usa CORES["destaque"].
        tamanho: lado do quadradinho.

    Retorna:
        A mesma figura, com a marca de ângulo reto.

    Exemplo:
        >>> fig = desenhar_angulo_reto(fig, (0, 0, 0), (1, 0, 0), (0, 0, 1))
    """
    v = _como_vetor(vertice)
    u1 = _normalizar(_como_vetor(ponto1) - v)
    u2 = _como_vetor(ponto2) - v

    # Se o ângulo não for exatamente reto (erro de arredondamento), "endireitamos"
    # a segunda direção para o quadradinho ficar com cantos de 90°.
    u2 = _normalizar(u2 - np.dot(u2, u1) * u1)

    # O quadradinho é uma linha com três trechos: sai de perto da primeira
    # semirreta, faz a quina e volta para a segunda semirreta.
    cantos = np.array([
        v + tamanho * u1,
        v + tamanho * u1 + tamanho * u2,
        v + tamanho * u2,
    ])
    fig.add_trace(
        go.Scatter3d(
            x=cantos[:, 0], y=cantos[:, 1], z=cantos[:, 2],
            mode="lines",
            line=dict(color=cor or CORES["destaque"], width=3),
            hoverinfo="skip",
        )
    )
    return fig


def desenhar_angulo_entre_retas(
    fig: go.Figure,
    a1: Vetor3D,
    a2: Vetor3D,
    b1: Vetor3D,
    b2: Vetor3D,
    vertice: Vetor3D | None = None,
    rotulo: str | None = None,
    cor: str | None = None,
    raio: float = 0.6,
) -> go.Figure:
    """Desenha o ângulo entre a reta r = a1a2 e a reta s = b1b2.

    Segue a definição do notebook 02: o ângulo entre duas retas é sempre agudo ou
    reto.
    - Concorrentes: o arco (ou o quadradinho, se for reto) fica no ponto de encontro.
    - Reversas: traçamos, tracejada, a paralela a s pelo vértice (o ponto `vertice`
      ou, se ele não for dado, o ponto a1) e marcamos o ângulo entre r e essa
      paralela.
    - Paralelas ou coincidentes: o ângulo é 0° e nada é desenhado.

    Parâmetros:
        fig: a figura.
        a1, a2: dois pontos distintos da reta r.
        b1, b2: dois pontos distintos da reta s.
        vertice: para retas reversas, o ponto de r onde a paralela a s é traçada
            (opcional; padrão: a1). É ignorado nos outros casos.
        rotulo: texto escrito junto ao arco, por exemplo "θ" (opcional).
        cor: cor do arco. Se não for informada, usa CORES["destaque"].
        raio: raio do arco (e lado do quadradinho, proporcionalmente).

    Retorna:
        A mesma figura, com o ângulo desenhado.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> fig = desenhar_angulo_entre_retas(fig, v["A"], v["B"], v["C"], v["H"])
    """
    pa1, pa2 = _como_vetor(a1), _como_vetor(a2)
    pb1, pb2 = _como_vetor(b1), _como_vetor(b2)
    posicao = classificar_posicao_retas(pa1, pa2, pb1, pb2)
    if posicao in ("paralelas", "coincidentes"):
        return fig

    # Cada reta forma dois ângulos com a outra (um agudo e seu suplemento). Para
    # ficar com o agudo, escolhemos o sentido de s que "concorda" com o de r:
    # se os dois vetores apontam para lados opostos, invertemos o de s.
    u = _normalizar(pa2 - pa1)
    w = _normalizar(pb2 - pb1)
    if np.dot(u, w) < 0:
        w = -w

    if posicao == "concorrentes":
        v = _encontro_de_retas(pa1, pa2, pb1, pb2)
    else:
        # Reversas: o vértice fica em r, e por ele passa a paralela a s.
        v = pa1 if vertice is None else _como_vetor(vertice)
        fig = desenhar_reta(
            fig, v, w, cor=CORES["apoio"], comprimento=6, espessura=3, tracejado=True
        )

    if abs(np.dot(u, w)) < TOLERANCIA:
        # Ângulo reto: no lugar do arco, o quadradinho.
        return desenhar_angulo_reto(
            fig, v, v + u, v + w, cor=cor, tamanho=raio / 2
        )
    return desenhar_angulo(fig, v, v + u, v + w, rotulo=rotulo, cor=cor, raio=raio)


def desenhar_semiplano(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    ponto: Vetor3D,
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.35,
    comprimento: float = 6,
    largura: float = 4,
    mostrar_borda: bool = True,
) -> go.Figure:
    """Desenha (um pedaço de) um semiplano.

    O semiplano tem origem na reta ab e fica do lado do ponto `ponto`. Como ele é
    infinito, desenhamos um retângulo: um lado fica sobre a reta de origem (centrado
    no ponto médio de ab, com tamanho `comprimento`) e o retângulo avança `largura`
    para o lado de `ponto`. Lembre-se: o semiplano continua além da borda, mas
    nunca passa para o outro lado da reta de origem.

    Parâmetros:
        fig: a figura.
        a, b: dois pontos distintos da reta de origem.
        ponto: um ponto fora da reta de origem, que indica o lado do semiplano.
        rotulo: nome do semiplano, por exemplo "α" (opcional).
        cor: cor do semiplano. Se não for informada, usa CORES["plano"].
        opacidade: de 0 (invisível) a 1 (totalmente opaco).
        comprimento: tamanho do lado que fica sobre a reta de origem.
        largura: quanto o retângulo avança para o lado de `ponto`.
        mostrar_borda: se True, desenha o contorno do retângulo.

    Retorna:
        A mesma figura, com o semiplano desenhado.

    Levanta:
        ValueError: se `ponto` estiver na reta ab.

    Exemplo:
        >>> fig = desenhar_semiplano(fig, (0, 0, 0), (1, 0, 0), (0, 1, 1), rotulo="α")
    """
    d, w = _direcao_do_semiplano(a, b, ponto)
    meio = (_como_vetor(a) + _como_vetor(b)) / 2
    cor = cor or CORES["plano"]
    metade = comprimento / 2

    # Dois cantos sobre a reta de origem e dois cantos afastados `largura` dela.
    cantos = [
        meio - metade * d,
        meio + metade * d,
        meio + metade * d + largura * w,
        meio - metade * d + largura * w,
    ]
    fig = desenhar_poligono(
        fig, cantos, cor=cor, opacidade=opacidade, mostrar_borda=mostrar_borda
    )
    if rotulo:
        # O nome fica perto da borda de fora, longe da reta de origem (onde as faces
        # de um diedro se encontram e os rótulos se misturariam).
        posicao = meio + 0.35 * comprimento * d + 0.8 * largura * w
        fig = adicionar_rotulo(fig, posicao, rotulo, cor=cor, tamanho=18)
    return fig


def desenhar_diedro(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    p1: Vetor3D,
    p2: Vetor3D,
    rotulos: tuple[str, str] | None = None,
    rotulo_aresta: str | None = None,
    cores: tuple[str, str] | None = None,
    opacidade: float = 0.35,
    comprimento: float = 6,
    largura: float = 4,
    mostrar_aresta: bool = True,
) -> go.Figure:
    """Desenha um diedro: duas faces (semiplanos) com a mesma aresta.

    O diedro é dado pela aresta (os pontos a e b) e por um ponto de cada face: a
    primeira face é o semiplano de origem ab do lado de p1, e a segunda, o do lado
    de p2. Cada face é desenhada com `desenhar_semiplano`.

    Parâmetros:
        fig: a figura.
        a, b: dois pontos distintos da aresta.
        p1: um ponto da primeira face (fora da aresta).
        p2: um ponto da segunda face (fora da aresta).
        rotulos: os nomes das duas faces, por exemplo ("α", "β") (opcional).
        rotulo_aresta: o nome da aresta, por exemplo "r" (opcional).
        cores: as cores das duas faces. Se não forem informadas, usa
            CORES["plano"] e CORES["plano_secundario"].
        opacidade: opacidade das faces, de 0 a 1.
        comprimento: tamanho do trecho da aresta (e do lado das faces sobre ela).
        largura: quanto cada face avança a partir da aresta.
        mostrar_aresta: se True, desenha a aresta em destaque.

    Retorna:
        A mesma figura, com o diedro desenhado.

    Levanta:
        ValueError: se p1 ou p2 estiver na reta da aresta.

    Exemplo:
        >>> d = vertices_do_diedro(70)
        >>> fig = desenhar_diedro(
        ...     fig, d["aresta_inicio"], d["aresta_fim"], d["ponto_face1"],
        ...     d["ponto_face2"], rotulos=("α", "β"), rotulo_aresta="r",
        ... )
    """
    cor1, cor2 = cores or (CORES["plano"], CORES["plano_secundario"])
    rotulo1, rotulo2 = rotulos or (None, None)
    for ponto_da_face, rotulo, cor in ((p1, rotulo1, cor1), (p2, rotulo2, cor2)):
        fig = desenhar_semiplano(
            fig, a, b, ponto_da_face, rotulo=rotulo, cor=cor, opacidade=opacidade,
            comprimento=comprimento, largura=largura,
        )
    if mostrar_aresta:
        pa, pb = _como_vetor(a), _como_vetor(b)
        fig = desenhar_reta(
            fig, (pa + pb) / 2, pb - pa, rotulo=rotulo_aresta,
            cor=CORES["destaque"], comprimento=comprimento, espessura=6,
        )
    return fig


def desenhar_secao_do_diedro(
    fig: go.Figure,
    a: Vetor3D,
    b: Vetor3D,
    p1: Vetor3D,
    p2: Vetor3D,
    plano: Plano,
    rotulo: str | None = None,
    cor: str | None = None,
    comprimento: float = 3,
    mostrar_plano: bool = True,
    opacidade_plano: float = 0.15,
    mostrar_arco: bool = True,
) -> go.Figure:
    """Desenha a seção do diedro pelo plano dado: as duas semirretas e o ângulo.

    Usa `secao_do_diedro` para encontrar o vértice e as semirretas da seção. Se o
    plano for paralelo à aresta ou a contiver, não há seção em forma de ângulo: a
    figura volta sem alterações.

    Parâmetros:
        fig: a figura.
        a, b: dois pontos distintos da aresta.
        p1: um ponto da primeira face (fora da aresta).
        p2: um ponto da segunda face (fora da aresta).
        plano: o plano de corte, como (ponto, normal).
        rotulo: texto escrito junto ao arco, por exemplo "60°" (opcional).
        cor: cor das semirretas. Se não for informada, usa CORES["reta"].
        comprimento: tamanho dos trechos desenhados das semirretas.
        mostrar_plano: se True, desenha o plano de corte, quase transparente
            (cor CORES["plano_terciario"]).
        opacidade_plano: opacidade do plano de corte.
        mostrar_arco: se True, marca o ângulo da seção com um arco.

    Retorna:
        A mesma figura, com a seção desenhada (se existir).

    Exemplo:
        >>> d = vertices_do_diedro(60)
        >>> fig = desenhar_secao_do_diedro(
        ...     fig, d["aresta_inicio"], d["aresta_fim"], d["ponto_face1"],
        ...     d["ponto_face2"], ((0, 0, 0), (1, 0, 0)), rotulo="60°",
        ... )
    """
    secao = secao_do_diedro(a, b, p1, p2, plano)
    if secao is None:
        return fig
    v, lado1, lado2 = (np.array(p) for p in secao)
    cor = cor or CORES["reta"]

    if mostrar_plano:
        # O plano de corte fica centrado no vértice da seção.
        fig = desenhar_plano(
            fig, v, plano[1], cor=CORES["plano_terciario"], opacidade=opacidade_plano,
            tamanho=2 * comprimento,
        )
    # lado1 e lado2 estão a distância 1 do vértice: esticamos até `comprimento`.
    for lado in (lado1, lado2):
        fig = desenhar_segmento(
            fig, v, v + comprimento * (lado - v), cor=cor, espessura=5
        )
    if mostrar_arco:
        fig = desenhar_angulo(
            fig, v, lado1, lado2, rotulo=rotulo, raio=0.3 * comprimento
        )
    return desenhar_ponto(fig, v, cor=cor, tamanho=4)


def adicionar_rotulo(
    fig: go.Figure,
    posicao: Vetor3D,
    texto: str,
    cor: str | None = None,
    tamanho: float = 14,
) -> go.Figure:
    """Escreve um texto solto no espaço.

    Aceita qualquer caractere, inclusive letras gregas (α, β, γ, π).

    Parâmetros:
        fig: a figura.
        posicao: onde o texto aparece, como (x, y, z).
        texto: o texto a escrever.
        cor: cor do texto. Se não for informada, usa CORES["texto"].
        tamanho: tamanho da fonte.

    Retorna:
        A mesma figura, com o texto.

    Exemplo:
        >>> fig = adicionar_rotulo(fig, (0, 0, 2), "figura espacial")
    """
    x, y, z = _como_vetor(posicao)
    fig.add_trace(
        go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="text",
            text=[texto],
            textfont=dict(color=cor or CORES["texto"], size=tamanho),
            hoverinfo="skip",
        )
    )
    return fig


# ==================================================================================
# Sólidos
# ==================================================================================
def vertices_do_cubo(
    aresta: float = 2, origem: Vetor3D = (0, 0, 0)
) -> dict[str, tuple[float, float, float]]:
    """Devolve as posições dos 8 vértices de um cubo, com os nomes do módulo.

    Convenção de nomes: a base de baixo é ABCD (no sentido anti-horário, olhando de
    cima) e a base de cima é EFGH, com E acima de A, F acima de B, G acima de C e
    H acima de D.

    Parâmetros:
        aresta: medida da aresta do cubo.
        origem: posição do vértice A.

    Retorna:
        Um dicionário {"A": (x, y, z), ..., "H": (x, y, z)}.

    Exemplo:
        >>> vertices = vertices_do_cubo()
        >>> vertices["G"]
        (2.0, 2.0, 2.0)
    """
    ox, oy, oz = _como_vetor(origem)
    a = float(aresta)
    base = {
        "A": (0.0, 0.0),
        "B": (a, 0.0),
        "C": (a, a),
        "D": (0.0, a),
    }
    vertices = {}
    # Base de baixo (altura 0) e base de cima (altura igual à aresta).
    for nome, (x, y) in base.items():
        vertices[nome] = (ox + x, oy + y, oz)
    for nome_baixo, nome_cima in zip("ABCD", "EFGH", strict=True):
        x, y, z = vertices[nome_baixo]
        vertices[nome_cima] = (x, y, z + a)
    # Devolvemos em ordem alfabética, de A a H.
    return {nome: vertices[nome] for nome in "ABCDEFGH"}


def desenhar_cubo(
    fig: go.Figure,
    aresta: float = 2,
    origem: Vetor3D = (0, 0, 0),
    rotulos: bool = True,
    cor: str | None = None,
    opacidade: float = 0.08,
    cor_arestas: str | None = None,
) -> go.Figure:
    """Desenha um cubo com faces quase transparentes e as 12 arestas.

    Parâmetros:
        fig: a figura.
        aresta: medida da aresta do cubo.
        origem: posição do vértice A.
        rotulos: se True, escreve as letras A a H nos vértices.
        cor: cor das faces. Se não for informada, usa CORES["solido"].
        opacidade: opacidade das faces (baixa, para enxergar através delas).
        cor_arestas: cor das arestas. Se não for informada, usa CORES["apoio"].

    Retorna:
        A mesma figura, com o cubo desenhado.

    Exemplo:
        >>> fig = desenhar_cubo(fig, aresta=2)
    """
    vertices = vertices_do_cubo(aresta, origem)
    pontos = np.array(list(vertices.values()))  # linhas na ordem A, B, ..., H

    # As 6 faces, cada uma dividida em 2 triângulos. Os números são as posições dos
    # vértices na ordem A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7.
    triangulos = [
        (0, 1, 2), (0, 2, 3),  # base de baixo ABCD
        (4, 5, 6), (4, 6, 7),  # base de cima EFGH
        (0, 1, 5), (0, 5, 4),  # face ABFE
        (1, 2, 6), (1, 6, 5),  # face BCGF
        (2, 3, 7), (2, 7, 6),  # face CDHG
        (3, 0, 4), (3, 4, 7),  # face DAEH
    ]
    fig.add_trace(
        go.Mesh3d(
            x=pontos[:, 0], y=pontos[:, 1], z=pontos[:, 2],
            i=[t[0] for t in triangulos],
            j=[t[1] for t in triangulos],
            k=[t[2] for t in triangulos],
            color=cor or CORES["solido"],
            opacity=opacidade,
            lighting=dict(ambient=1, diffuse=0, specular=0, fresnel=0),
            hoverinfo="skip",
        )
    )

    arestas = ["AB", "BC", "CD", "DA", "EF", "FG", "GH", "HE", "AE", "BF", "CG", "DH"]
    for nome in arestas:
        fig = desenhar_segmento(
            fig, vertices[nome[0]], vertices[nome[1]],
            cor=cor_arestas or CORES["apoio"], espessura=3,
        )

    if rotulos:
        # Cada letra fica um pouco para fora do cubo, na direção que vai do centro
        # do cubo até o vértice.
        centro = pontos.mean(axis=0)
        for nome, vertice in vertices.items():
            vertice = np.array(vertice)
            para_fora = _normalizar(vertice - centro)
            fig = adicionar_rotulo(fig, vertice + 0.15 * aresta * para_fora, nome)
    return fig


def desenhar_esfera(
    fig: go.Figure,
    centro: Vetor3D,
    raio: float,
    rotulo: str | None = None,
    cor: str | None = None,
    opacidade: float = 0.25,
    resolucao: int = 40,
) -> go.Figure:
    """Desenha uma superfície esférica semitransparente.

    A superfície esférica é o conjunto dos pontos do espaço que estão a uma mesma
    distância (o raio) do centro (notebook 04). O estudo completo da esfera fica
    para o notebook 13.

    Parâmetros:
        fig: a figura.
        centro: o centro da esfera, como (x, y, z).
        raio: o raio.
        rotulo: nome da esfera, escrito acima dela (opcional).
        cor: cor da superfície. Se não for informada, usa CORES["solido"].
        opacidade: de 0 (invisível) a 1 (totalmente opaca).
        resolucao: quantidade de divisões usadas no desenho (mais divisões,
            superfície mais lisa).

    Retorna:
        A mesma figura, com a esfera desenhada.

    Exemplo:
        >>> fig = desenhar_esfera(fig, (0, 0, 0), 2, rotulo="esfera")
    """
    c = _como_vetor(centro)
    cor = cor or CORES["solido"]

    # Como num globo terrestre, cada ponto da esfera é dado por uma "longitude"
    # (giro em volta do eixo vertical) e uma "latitude" (do polo sul ao polo norte).
    # Seno e cosseno transformam esses dois ângulos em posições no espaço.
    longitude, latitude = np.meshgrid(
        np.linspace(0, 2 * np.pi, resolucao + 1),
        np.linspace(-np.pi / 2, np.pi / 2, resolucao // 2 + 1),
    )
    x = c[0] + raio * np.cos(latitude) * np.cos(longitude)
    y = c[1] + raio * np.cos(latitude) * np.sin(longitude)
    z = c[2] + raio * np.sin(latitude)
    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            # Uma escala de cores com uma cor só: a esfera fica toda da mesma cor.
            colorscale=[[0, cor], [1, cor]],
            showscale=False,
            opacity=opacidade,
            lighting=dict(ambient=0.8, diffuse=0.3, specular=0.05, fresnel=0),
            hoverinfo="skip",
        )
    )
    if rotulo:
        fig = adicionar_rotulo(fig, c + np.array([0, 0, 1.15 * raio]), rotulo, cor=cor)
    return fig


# ==================================================================================
# Funções de conferência
# ==================================================================================
def classificar_posicao_retas(
    a1: Vetor3D, a2: Vetor3D, b1: Vetor3D, b2: Vetor3D
) -> str:
    """Classifica a posição relativa de duas retas no espaço.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        a1, a2: dois pontos distintos da reta r.
        b1, b2: dois pontos distintos da reta s.

    Retorna:
        Exatamente uma das palavras: "coincidentes", "concorrentes", "paralelas"
        ou "reversas".

    Levanta:
        ValueError: se os dois pontos de alguma das retas forem iguais.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> classificar_posicao_retas(v["A"], v["B"], v["C"], v["G"])
        'reversas'
    """
    pa1, pa2 = _como_vetor(a1), _como_vetor(a2)
    pb1, pb2 = _como_vetor(b1), _como_vetor(b2)
    u = pa2 - pa1  # direção de r
    v = pb2 - pb1  # direção de s
    if np.linalg.norm(u) < TOLERANCIA or np.linalg.norm(v) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    w = pb1 - pa1  # vai de um ponto de r até um ponto de s

    # Comprimentos usados para que a tolerância acompanhe o tamanho do desenho.
    tam_u = np.linalg.norm(u)
    tam_v = np.linalg.norm(v)
    tam_w = max(np.linalg.norm(w), 1.0)

    # Produto vetorial (quase) zero quer dizer: r e s têm a mesma direção.
    if np.linalg.norm(np.cross(u, v)) <= TOLERANCIA * tam_u * tam_v:
        # Mesma direção: ou são a mesma reta, ou são paralelas distintas.
        # Basta ver se o ponto b1 (de s) está sobre r.
        if np.linalg.norm(np.cross(u, w)) <= TOLERANCIA * tam_u * tam_w:
            return "coincidentes"
        return "paralelas"

    # Direções diferentes: o produto misto diz se as retas estão num mesmo plano.
    # Se for zero, são coplanares (e então se cortam); se não, são reversas.
    produto_misto = np.dot(w, np.cross(u, v))
    if abs(produto_misto) <= TOLERANCIA * tam_u * tam_v * tam_w:
        return "concorrentes"
    return "reversas"


def planos_sao_paralelos(plano1: Plano, plano2: Plano) -> bool:
    """Diz se dois planos são paralelos ou coincidentes.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        plano1: o primeiro plano, como (ponto, normal).
        plano2: o segundo plano, como (ponto, normal).

    Retorna:
        True se os planos forem paralelos ou coincidentes (normais com a mesma
        direção); False se forem secantes.

    Exemplo:
        >>> planos_sao_paralelos(((0, 0, 0), (0, 0, 1)), ((0, 0, 3), (0, 0, 2)))
        True
    """
    n1 = _como_vetor(plano1[1])
    n2 = _como_vetor(plano2[1])
    produto = np.linalg.norm(np.cross(n1, n2))
    return bool(produto <= TOLERANCIA * np.linalg.norm(n1) * np.linalg.norm(n2))


def classificar_posicao_reta_plano(a: Vetor3D, b: Vetor3D, plano: Plano) -> str:
    """Classifica a posição relativa da reta ab e de um plano.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        a, b: dois pontos distintos da reta.
        plano: o plano, como (ponto, normal).

    Retorna:
        Exatamente uma das palavras: "contida", "concorrente" ou "paralela".

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> base = plano_por_tres_pontos(v["A"], v["B"], v["C"])
        >>> classificar_posicao_reta_plano(v["E"], v["F"], base)
        'paralela'
    """
    pa, pb = _como_vetor(a), _como_vetor(b)
    ponto, normal = _como_vetor(plano[0]), _como_vetor(plano[1])
    direcao = pb - pa
    if np.linalg.norm(direcao) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    tam_n = np.linalg.norm(normal)

    # Se a direção da reta não é perpendicular ao normal, a reta "atravessa" o
    # plano: tem exatamente um ponto em comum com ele.
    if abs(np.dot(normal, direcao)) > TOLERANCIA * tam_n * np.linalg.norm(direcao):
        return "concorrente"

    # Direção perpendicular ao normal: a reta corre "deitada" em relação ao plano.
    # Ou ela está dentro dele (o ponto a pertence ao plano), ou fica fora, paralela.
    distancia = abs(np.dot(normal, pa - ponto)) / tam_n
    if distancia <= TOLERANCIA * max(np.linalg.norm(pa - ponto), 1.0):
        return "contida"
    return "paralela"


def classificar_posicao_planos(plano1: Plano, plano2: Plano) -> str:
    """Classifica a posição relativa de dois planos.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        plano1: o primeiro plano, como (ponto, normal).
        plano2: o segundo plano, como (ponto, normal).

    Retorna:
        Exatamente uma das palavras: "coincidentes", "secantes" ou "paralelos".

    Exemplo:
        >>> classificar_posicao_planos(((0, 0, 0), (0, 0, 1)), ((0, 0, 3), (0, 0, 2)))
        'paralelos'
    """
    if not planos_sao_paralelos(plano1, plano2):
        return "secantes"

    # Normais com a mesma direção: os planos são paralelos ou o mesmo plano.
    # Basta ver se o ponto do segundo plano pertence ao primeiro.
    p1, n1 = _como_vetor(plano1[0]), _como_vetor(plano1[1])
    p2 = _como_vetor(plano2[0])
    distancia = abs(np.dot(n1, p2 - p1)) / np.linalg.norm(n1)
    if distancia <= TOLERANCIA * max(np.linalg.norm(p2 - p1), 1.0):
        return "coincidentes"
    return "paralelos"


def medir_distancia(p: Vetor3D | np.ndarray, q: Vetor3D) -> float | np.ndarray:
    """Mede a distância entre dois pontos: é a nossa trena.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta. A fórmula da distância entre dois
    pontos com coordenadas (uma "Pitágoras em 3D") aparece no módulo 07.

    Também mede de uma vez as distâncias de muitos pontos até um mesmo ponto q:
    basta passar em p um array do NumPy de formato (N, 3).

    Parâmetros:
        p: um ponto, como (x, y, z), ou um array de formato (N, 3) com N pontos.
        q: o outro ponto, como (x, y, z).

    Retorna:
        A distância entre p e q (um número) ou, se p tiver N pontos, um array com
        as N distâncias.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> round(medir_distancia(v["A"], v["G"]), 4)
        3.4641
        >>> medir_distancia(np.array([v["A"], v["G"]]), v["A"]).round(4)
        array([0.    , 3.4641])
    """
    varios = np.asarray(p, dtype=float)
    if varios.ndim == 2:
        if varios.shape[1] != 3:
            raise ValueError("Esperava um array de pontos de formato (N, 3).")
        # Uma distância para cada linha do array (cada ponto).
        return np.linalg.norm(varios - _como_vetor(q), axis=1)
    return float(np.linalg.norm(_como_vetor(q) - _como_vetor(p)))


def medir_angulo_entre_retas(
    a1: Vetor3D, a2: Vetor3D, b1: Vetor3D, b2: Vetor3D
) -> float:
    """Mede o ângulo entre as retas a1a2 e b1b2, em graus.

    Segue a definição do notebook 02: o resultado está sempre entre 0° e 90°
    (inclusive). Funciona para retas concorrentes, paralelas e reversas; para as
    reversas, é o ângulo entre uma delas e uma paralela à outra que a corta.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        a1, a2: dois pontos distintos da reta r.
        b1, b2: dois pontos distintos da reta s.

    Retorna:
        O ângulo entre r e s, em graus.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> round(medir_angulo_entre_retas(v["A"], v["B"], v["C"], v["H"]), 1)
        45.0
    """
    u = _normalizar(_como_vetor(a2) - _como_vetor(a1))
    w = _normalizar(_como_vetor(b2) - _como_vetor(b1))

    # O valor absoluto escolhe sempre o ângulo agudo (ou reto) entre as duas
    # direções; o clip protege o arco-cosseno de erros de arredondamento.
    cosseno = min(abs(float(np.dot(u, w))), 1.0)
    return float(np.degrees(np.arccos(cosseno)))


def medir_distancia_ponto_plano(
    ponto: Vetor3D | np.ndarray, plano: Plano
) -> float | np.ndarray:
    """Mede a distância de um ponto a um plano.

    É o comprimento do segmento perpendicular do ponto ao plano (notebook 04).
    Também mede de uma vez as distâncias de muitos pontos: basta passar um array do
    NumPy de formato (N, 3).

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        ponto: um ponto, como (x, y, z), ou um array de formato (N, 3) com N pontos.
        plano: o plano, como (ponto, normal).

    Retorna:
        A distância (um número) ou, se forem N pontos, um array com as N distâncias.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> base = plano_por_tres_pontos(v["A"], v["B"], v["C"])
        >>> medir_distancia_ponto_plano(v["G"], base)
        2.0
    """
    ponto_do_plano = _como_vetor(plano[0])
    n = _normalizar(plano[1])
    varios = np.asarray(ponto, dtype=float)
    if varios.ndim == 2:
        if varios.shape[1] != 3:
            raise ValueError("Esperava um array de pontos de formato (N, 3).")
        # A "altura" de cada ponto em relação ao plano, medida na direção do normal.
        return np.abs((varios - ponto_do_plano) @ n)
    return float(abs(np.dot(_como_vetor(ponto) - ponto_do_plano, n)))


def medir_distancia_entre_retas(
    a1: Vetor3D, a2: Vetor3D, b1: Vetor3D, b2: Vetor3D
) -> float:
    """Mede a distância entre as retas a1a2 e b1b2.

    - Concorrentes ou coincidentes: a distância é 0.
    - Paralelas: é a distância de qualquer ponto de uma até a outra.
    - Reversas: é o comprimento do segmento da perpendicular comum (notebook 04).

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        a1, a2: dois pontos distintos da reta r.
        b1, b2: dois pontos distintos da reta s.

    Retorna:
        A distância entre r e s.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> medir_distancia_entre_retas(v["A"], v["B"], v["C"], v["G"])
        2.0
    """
    posicao = classificar_posicao_retas(a1, a2, b1, b2)
    if posicao in ("concorrentes", "coincidentes"):
        return 0.0
    if posicao == "paralelas":
        # Retas paralelas: basta medir de um ponto de s até o pé da perpendicular
        # traçada dele até r.
        pe = pe_da_perpendicular_a_reta(b1, a1, a2)
        return medir_distancia(b1, pe)
    p, q = perpendicular_comum(a1, a2, b1, b2)
    return medir_distancia(p, q)


def medir_angulo_reta_plano(a: Vetor3D, b: Vetor3D, plano: Plano) -> float:
    """Mede o ângulo entre a reta ab e um plano, em graus.

    Segue a definição do notebook 04: 90° se a reta for perpendicular ao plano, 0°
    se for paralela a ele ou estiver contida nele e, nos outros casos, o ângulo
    entre a reta e a sua projeção ortogonal sobre o plano. O resultado está sempre
    entre 0° e 90°.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        a, b: dois pontos distintos da reta.
        plano: o plano, como (ponto, normal).

    Retorna:
        O ângulo entre a reta e o plano, em graus.

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> base = plano_por_tres_pontos(v["A"], v["B"], v["C"])
        >>> round(medir_angulo_reta_plano(v["A"], v["F"], base), 1)
        45.0
    """
    direcao = _como_vetor(b) - _como_vetor(a)
    if np.linalg.norm(direcao) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    u = _normalizar(direcao)
    n = _normalizar(plano[1])

    # O ângulo com o normal e o ângulo com o plano são complementares: por isso o
    # seno do ângulo com o plano é o cosseno do ângulo com o normal. O valor
    # absoluto e o min protegem de sentidos invertidos e de arredondamentos.
    seno = min(abs(float(np.dot(u, n))), 1.0)
    return float(np.degrees(np.arcsin(seno)))


def medir_angulo(vertice: Vetor3D, ponto1: Vetor3D, ponto2: Vetor3D) -> float:
    """Mede o ângulo entre as semirretas VP1 e VP2, em graus: é o nosso transferidor.

    Diferente de `medir_angulo_entre_retas`, que mede sempre o ângulo agudo entre
    duas retas, aqui as semirretas têm sentido: o resultado vai de 0° a 180°.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        vertice: o vértice V do ângulo, como (x, y, z).
        ponto1: um ponto da primeira semirreta (diferente de V).
        ponto2: um ponto da segunda semirreta (diferente de V).

    Retorna:
        A medida do ângulo, em graus, entre 0 e 180.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> round(medir_angulo(v["A"], v["B"], v["C"]), 1)
        45.0
    """
    v = _como_vetor(vertice)
    u1 = _normalizar(_como_vetor(ponto1) - v)
    u2 = _normalizar(_como_vetor(ponto2) - v)
    # O clip protege o arco-cosseno de erros de arredondamento (como 1.0000000002).
    cosseno = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosseno)))


def medir_angulo_do_diedro(a: Vetor3D, b: Vetor3D, p1: Vetor3D, p2: Vetor3D) -> float:
    """Mede um diedro, em graus: é a medida da sua seção reta.

    O diedro é dado pela aresta (os pontos a e b) e por um ponto de cada face (p1 e
    p2). Os pontos das faces não precisam estar na mesma seção reta: a função
    encontra a seção reta sozinha.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        a, b: dois pontos distintos da aresta.
        p1: um ponto da primeira face (fora da aresta).
        p2: um ponto da segunda face (fora da aresta).

    Retorna:
        A medida do diedro, em graus, entre 0 e 180.

    Levanta:
        ValueError: se p1 ou p2 estiver na reta da aresta.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> round(medir_angulo_do_diedro(v["A"], v["B"], v["D"], v["G"]), 1)
        45.0
    """
    # w1 e w2 são as semirretas da seção reta que passa por a: cada uma está numa
    # face e forma 90° com a aresta. O diedro mede o ângulo entre elas.
    _, w1 = _direcao_do_semiplano(a, b, p1)
    _, w2 = _direcao_do_semiplano(a, b, p2)
    cosseno = float(np.clip(np.dot(w1, w2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosseno)))


def medir_angulo_entre_planos(plano1: Plano, plano2: Plano) -> float:
    """Mede o ângulo entre dois planos, em graus.

    Segue a definição do notebook 05: 0° se os planos forem paralelos ou
    coincidentes; se forem secantes, a medida do menor dos diedros que eles formam.
    O resultado está sempre entre 0° e 90°.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        plano1: o primeiro plano, como (ponto, normal).
        plano2: o segundo plano, como (ponto, normal).

    Retorna:
        O ângulo entre os planos, em graus.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> base = plano_por_tres_pontos(v["A"], v["B"], v["C"])
        >>> round(medir_angulo_entre_planos(base, plano_por_tres_pontos(
        ...     v["A"], v["B"], v["G"])), 1)
        45.0
    """
    if planos_sao_paralelos(plano1, plano2):
        return 0.0
    n1 = _normalizar(plano1[1])
    n2 = _normalizar(plano2[1])
    # O ângulo entre os planos é igual ao ângulo entre retas perpendiculares a eles
    # (os normais). O valor absoluto escolhe o menor dos dois ângulos possíveis.
    cosseno = min(abs(float(np.dot(n1, n2))), 1.0)
    return float(np.degrees(np.arccos(cosseno)))


def reta_perpendicular_ao_plano(a: Vetor3D, b: Vetor3D, plano: Plano) -> bool:
    """Diz se a reta ab é perpendicular ao plano.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        a, b: dois pontos distintos da reta.
        plano: o plano, como (ponto, normal).

    Retorna:
        True se a reta for perpendicular ao plano; False caso contrário (reta
        oblíqua, paralela ou contida no plano).

    Levanta:
        ValueError: se a e b forem iguais.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> base = plano_por_tres_pontos(v["A"], v["B"], v["C"])
        >>> reta_perpendicular_ao_plano(v["A"], v["E"], base)
        True
    """
    direcao = _como_vetor(b) - _como_vetor(a)
    normal = _como_vetor(plano[1])
    if np.linalg.norm(direcao) < TOLERANCIA:
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    # A reta é perpendicular ao plano quando tem a mesma direção do normal:
    # nesse caso, o produto vetorial da direção com o normal dá (quase) zero.
    produto = np.linalg.norm(np.cross(direcao, normal))
    escala = np.linalg.norm(direcao) * np.linalg.norm(normal)
    return bool(produto <= TOLERANCIA * escala)


def planos_sao_perpendiculares(plano1: Plano, plano2: Plano) -> bool:
    """Diz se dois planos são perpendiculares.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta.

    Parâmetros:
        plano1: o primeiro plano, como (ponto, normal).
        plano2: o segundo plano, como (ponto, normal).

    Retorna:
        True se os planos forem perpendiculares; False caso contrário.

    Exemplo:
        >>> planos_sao_perpendiculares(((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (1, 0, 0)))
        True
    """
    n1 = _como_vetor(plano1[1])
    n2 = _como_vetor(plano2[1])
    # Dois planos são perpendiculares quando os seus normais formam 90°: o normal
    # de um é uma reta "deitada" no outro. O produto escalar mede isso: é zero
    # exatamente quando os dois vetores formam ângulo reto.
    produto = abs(np.dot(n1, n2))
    return bool(produto <= TOLERANCIA * np.linalg.norm(n1) * np.linalg.norm(n2))
