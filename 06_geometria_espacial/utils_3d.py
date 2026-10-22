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

    # Se AB e AC apontam na mesma direção, o produto vetorial dá (quase) zero:
    # os três pontos estão numa mesma reta.
    normal = np.cross(pb - pa, pc - pa)
    escala = np.linalg.norm(pb - pa) * np.linalg.norm(pc - pa)
    if np.linalg.norm(normal) <= TOLERANCIA * max(escala, 1.0):
        raise ValueError(
            "Os três pontos são colineares: eles não determinam um único plano."
        )

    baricentro = (pa + pb + pc) / 3

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
