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


def medir_distancia(p: Vetor3D, q: Vetor3D) -> float:
    """Mede a distância entre dois pontos: é a nossa trena.

    Esta função usa vetores, assunto do módulo 07 (Geometria Analítica). Aqui ela
    serve apenas para conferir a sua resposta. A fórmula da distância entre dois
    pontos com coordenadas (uma "Pitágoras em 3D") aparece no módulo 07.

    Parâmetros:
        p, q: os dois pontos, cada um como (x, y, z).

    Retorna:
        A distância entre p e q.

    Exemplo:
        >>> v = vertices_do_cubo()
        >>> round(medir_distancia(v["A"], v["G"]), 4)
        3.4641
    """
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
