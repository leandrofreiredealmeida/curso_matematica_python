"""Funções de desenho do Módulo 07 — Geometria Analítica Plana e Espacial.

Propósito
---------
Este arquivo reúne as funções que desenham o plano cartesiano, pontos, segmentos e
polígonos nos notebooks do módulo 07. Assim os notebooks ficam curtos e todas as
figuras ficam com a mesma cara (a mesma do `utils_3d.py` do módulo 06).

Aos poucos, ele também guarda as **ferramentas que você já deduziu** (distância, ponto
médio etc.), para que os notebooks seguintes possam usá-las sem reescrevê-las.

Você não precisa entender este arquivo para estudar a matemática. Mas, se gosta de
programar, fique à vontade para ler, testar e modificar: o código foi escrito para
ser lido por quem está começando.

Como usar
---------
    import utils_ga as uga

    fig = uga.criar_plano_cartesiano(titulo="Meu primeiro desenho")
    fig = uga.desenhar_ponto(fig, (3, 2), rotulo="A", mostrar_coordenadas=True)
    fig = uga.desenhar_segmento(fig, (3, 2), (-1, -2))
    uga.mostrar(fig)

Toda função `desenhar_...` recebe a figura, acrescenta algo a ela e devolve a mesma
figura. Por isso escrevemos sempre `fig = uga.desenhar_...(fig, ...)`.

A regra das caixas abertas
--------------------------
Neste módulo, quem faz as contas é você. O arquivo tem duas partes:

1. **Desenho**: funções que só desenham. Podem ser usadas em qualquer notebook.
2. **Ferramentas já estudadas**: funções que calculam conteúdo do módulo
   (`distancia`, `ponto_medio` etc.). Uma função só entra aqui DEPOIS de ter sido
   deduzida e programada por você num notebook, e só é usada pelo `uga` a partir do
   notebook SEGUINTE. A docstring de cada uma diz onde ela foi deduzida.

Se um desenho precisar de uma conta que você ainda não estudou (por exemplo, os
pontos de uma circunferência), a conta fica escondida numa função interna (com
sublinhado no começo do nome) e o notebook avisa quando ela será estudada.

Histórico de versões
--------------------
v1 — notebook 01: núcleo de desenho 2D (plano cartesiano, ponto, pontos, segmento,
     polígono, projeções, triângulo das diferenças, rótulo, círculo de alcance);
     ferramentas distancia, ponto_medio, ponto_do_segmento, baricentro
v2 — notebook 02: desenhar_seta, desenhar_sentido_de_percurso,
     desenhar_reta_por_dois_pontos, criar_grafico; parâmetros rotulo_x, rotulo_y e
     passo em criar_plano_cartesiano; ferramentas area_orientada, area_triangulo,
     orientacao, estao_alinhados, area_poligono
"""

# ==================================================================================
# Importações
# ==================================================================================
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import plotly.graph_objects as go

# Um ponto pode ser escrito como tupla, lista ou array do NumPy: (x, y) no plano e,
# a partir da parte espacial do módulo, (x, y, z).
Ponto: TypeAlias = Sequence[float] | np.ndarray

# Tolerância numérica: o computador faz contas com pequenos erros de arredondamento,
# então "zero" na prática quer dizer "menor que este número".
TOLERANCIA = 1e-9


# ==================================================================================
# Cores (paleta Nord, tema escuro)
# ==================================================================================
# NORD e as chaves antigas de CORES são idênticas às do utils_3d.py (módulo 06):
# assim as figuras dos dois módulos têm a mesma identidade visual.
NORD = {
    "nord0": "#2E3440", "nord1": "#3B4252", "nord2": "#434C5E", "nord3": "#4C566A",
    "nord4": "#D8DEE9", "nord5": "#E5E9F0", "nord6": "#ECEFF4",
    "nord7": "#8FBCBB", "nord8": "#88C0D0", "nord9": "#81A1C1", "nord10": "#5E81AC",
    "nord11": "#BF616A", "nord12": "#D08770", "nord13": "#EBCB8B",
    "nord14": "#A3BE8C", "nord15": "#B48EAD",
}

# Cores por papel: assim o "elemento em destaque" tem a mesma cor em todos os
# notebooks. O fundo é escuro (padrão do curso), por isso textos, pontos e linhas de
# apoio usam os tons claros da paleta.
CORES = {
    # --- Chaves do utils_3d.py (mesmos valores) ---
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
    # --- Chaves novas do módulo 07 ---
    # Eixos e grade usam tons claros com transparência (veja criar_plano_cartesiano):
    # o eixo um pouco mais forte, a grade bem suave, para não competir com os dados.
    "eixo": NORD["nord4"],
    "grade": NORD["nord4"],
    "projecao": NORD["nord4"],   # linhas tracejadas de um ponto até os eixos
    "cateto_x": NORD["nord12"],  # o "quanto andou na horizontal", Δx
    "cateto_y": NORD["nord14"],  # o "quanto andou na vertical", Δy
    "alcance": NORD["nord13"],   # regiões de alcance (o círculo de visão do gancho)
}

# Transparências usadas no plano cartesiano (0 = invisível, 1 = cor cheia).
_OPACIDADE_EIXO = 0.6
_OPACIDADE_GRADE = 0.15
_OPACIDADE_PROJECAO = 0.7


# ==================================================================================
# Funções auxiliares internas
# ==================================================================================
def _como_coordenadas(ponto: Ponto) -> np.ndarray:
    """Transforma um ponto (tupla, lista ou array) num array de números decimais."""
    return np.asarray(ponto, dtype=float)


def _como_ponto_2d(ponto: Ponto) -> np.ndarray:
    """Como _como_coordenadas, mas confere que o ponto tem exatamente 2 coordenadas."""
    coordenadas = _como_coordenadas(ponto)
    if coordenadas.shape != (2,):
        raise ValueError(
            f"Um ponto do plano precisa ter 2 coordenadas, (x, y), mas recebi {ponto!r}."
        )
    return coordenadas


def _conferir_mesma_dimensao(a: np.ndarray, b: np.ndarray) -> None:
    """Confere que dois pontos têm a mesma quantidade de coordenadas (2 ou 3)."""
    if a.shape != b.shape or a.ndim != 1 or len(a) not in (2, 3):
        raise ValueError(
            "Os pontos precisam ter a mesma quantidade de coordenadas: "
            "2 no plano, (x, y), ou 3 no espaço, (x, y, z)."
        )


def _cor_transparente(cor: str, opacidade: float) -> str:
    """Converte uma cor como "#D8DEE9" no formato "rgba(216, 222, 233, 0.2)".

    Os dois primeiros dígitos depois do # são o vermelho, os dois seguintes o verde e
    os dois últimos o azul, escritos na base 16.
    """
    vermelho, verde, azul = (int(cor[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({vermelho}, {verde}, {azul}, {opacidade})"


def _formatar_numero(valor: float) -> str:
    """Escreve um número do jeito do texto do curso: 2, -1, 1,5 ou 3,33.

    Inteiros aparecem sem casas decimais (2, e não 2.0); os demais, com no máximo 2
    casas e vírgula decimal.
    """
    arredondado = round(float(valor), 2)
    if abs(arredondado - round(arredondado)) < TOLERANCIA:
        # round() sem casas decimais devolve um inteiro; isso também transforma um
        # "-0.0" em 0.
        return str(round(arredondado))
    # "1.50" -> "1.5" -> "1,5": tiramos os zeros do fim e trocamos o ponto pela vírgula.
    texto = f"{arredondado:.2f}".rstrip("0")
    return texto.replace(".", ",")


def _formatar_ponto(rotulo: str | None, ponto: Ponto) -> str:
    """Monta o texto de um ponto: "A(2, 3)" ou, com decimais, "M(1,5; 2)".

    Quando alguma coordenada tem vírgula decimal, as coordenadas são separadas por
    ponto e vírgula, para não confundir "1,5" com duas coordenadas.
    """
    textos = [_formatar_numero(c) for c in _como_coordenadas(ponto)]
    separador = "; " if any("," in texto for texto in textos) else ", "
    return f"{rotulo or ''}({separador.join(textos)})"


def _posicao_do_texto(direcao_x: float, direcao_y: float) -> str:
    """Escolhe o "textposition" do Plotly que empurra um texto na direção dada.

    Exemplo: direção (1, 1), para cima e para a direita, dá "top right".
    """
    vertical = "top" if direcao_y >= 0 else "bottom"
    if abs(direcao_x) < 0.3 * abs(direcao_y):
        horizontal = "center"   # direção quase vertical: texto centralizado
    elif direcao_x > 0:
        horizontal = "right"
    else:
        horizontal = "left"
    return f"{vertical} {horizontal}"


def _escrever(
    fig: go.Figure,
    posicao: Ponto,
    texto: str,
    cor: str,
    tamanho: float,
    posicao_do_texto: str = "middle center",
) -> go.Figure:
    """Escreve um texto no plano, sem marcador e sem informação ao passar o mouse."""
    x, y = _como_ponto_2d(posicao)
    fig.add_trace(
        go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=[texto],
            textposition=posicao_do_texto,
            textfont=dict(color=cor, size=tamanho),
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def _pontos_da_circunferencia(
    centro: Ponto, raio: float, quantidade: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve os x e os y de `quantidade` pontos igualmente espaçados numa
    circunferência (o último repete o primeiro, para a linha se fechar).

    A conta usa seno e cosseno, como no ciclo trigonométrico (módulo 05): o ponto do
    ângulo t é (cx + r·cos t, cy + r·sen t). A equação da circunferência será
    estudada no notebook 08.
    """
    cx, cy = _como_ponto_2d(centro)
    t = np.linspace(0, 2 * np.pi, quantidade + 1)
    return cx + raio * np.cos(t), cy + raio * np.sin(t)


def _recortar_reta_na_janela(
    a: Ponto, b: Ponto, x_min: float, x_max: float, y_min: float, y_max: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Encontra o pedaço da reta AB que aparece dentro da janela retangular.

    A reta é percorrida na forma P = A + t·(B − A): cada número real t dá um ponto
    dela (é o `ponto_do_segmento` do notebook 01, agora com qualquer t). Para cada
    eixo, descobrimos o intervalo de valores de t em que o ponto fica entre o mínimo
    e o máximo daquele eixo. O pedaço visível corresponde aos t que servem nos DOIS
    eixos ao mesmo tempo (a interseção dos dois intervalos).

    Retorna as duas pontas do pedaço visível, ou None se a reta passa fora da janela.
    """
    pa, pb = _como_ponto_2d(a), _como_ponto_2d(b)
    direcao = pb - pa
    t_inicio, t_fim = -np.inf, np.inf
    for eixo, (minimo, maximo) in enumerate([(x_min, x_max), (y_min, y_max)]):
        if abs(direcao[eixo]) < TOLERANCIA:
            # A reta não anda nesta direção (é vertical, para o eixo x, ou horizontal,
            # para o eixo y): ou ela está inteira dentro da faixa, ou inteira fora.
            if not minimo <= pa[eixo] <= maximo:
                return None
            continue
        # Os valores de t em que a reta cruza as duas bordas deste eixo.
        t1 = (minimo - pa[eixo]) / direcao[eixo]
        t2 = (maximo - pa[eixo]) / direcao[eixo]
        t_inicio = max(t_inicio, min(t1, t2))
        t_fim = min(t_fim, max(t1, t2))
    if t_inicio > t_fim:
        return None
    return pa + t_inicio * direcao, pa + t_fim * direcao


# ==================================================================================
# Figura e exibição
# ==================================================================================
def criar_plano_cartesiano(
    titulo: str | None = None,
    x_min: float = -6,
    x_max: float = 6,
    y_min: float = -6,
    y_max: float = 6,
    grade: bool = True,
    altura: int = 500,
    rotulo_x: str = "x",
    rotulo_y: str = "y",
    passo: float | None = None,
) -> go.Figure:
    """Cria um plano cartesiano vazio, com eixos, grade e o visual padrão do módulo.

    A escala é a mesma nos dois eixos (uma unidade em x tem o mesmo tamanho que uma
    unidade em y): assim um quadrado parece quadrado e um ângulo reto parece reto.
    Os intervalos são fixos, para a figura não "pular" quando um ponto se move.

    Parâmetros:
        titulo: texto que aparece no alto da figura (opcional).
        x_min, x_max: o intervalo visível no eixo x.
        y_min, y_max: o intervalo visível no eixo y.
        grade: se False, esconde as linhas da grade (os eixos continuam).
        altura: altura da figura, em pixels.
        rotulo_x: nome escrito na ponta positiva do eixo x, por exemplo "x (m)".
        rotulo_y: nome escrito na ponta positiva do eixo y.
        passo: distância entre as marcações dos eixos (e as linhas da grade). Com
            None, as marcações vão de 1 em 1 quando o intervalo tem até 20
            unidades; acima disso, o Plotly escolhe sozinho.

    Retorna:
        Uma figura do Plotly (`go.Figure`), pronta para receber desenhos.

    Levanta:
        ValueError: se algum intervalo estiver invertido (mínimo maior ou igual ao
            máximo), ou se o passo não for positivo.

    Exemplo:
        >>> fig = criar_plano_cartesiano(titulo="Pontos no plano", x_min=-2, x_max=10)
        >>> terreno = criar_plano_cartesiano(
        ...     x_min=-40, x_max=200, y_min=-30, y_max=170, rotulo_x="x (m)",
        ...     rotulo_y="y (m)", passo=20,
        ... )
    """
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            "Cada intervalo precisa ter o mínimo menor que o máximo, "
            "por exemplo x_min=-6 e x_max=6."
        )
    if passo is not None and passo <= 0:
        raise ValueError("O passo entre as marcações dos eixos precisa ser positivo.")

    cor_da_grade = _cor_transparente(CORES["grade"], _OPACIDADE_GRADE)
    cor_do_eixo = _cor_transparente(CORES["eixo"], _OPACIDADE_EIXO)

    def configurar_eixo(minimo: float, maximo: float) -> dict:
        # Com o passo pedido, usamos ele. Sem passo: marcações de 1 em 1 quando
        # cabem bem; senão, o Plotly escolhe sozinho.
        if passo is not None:
            marcacao = passo
        else:
            marcacao = 1 if maximo - minimo <= 20 else None
        return dict(
            range=[minimo, maximo],
            dtick=marcacao,
            showgrid=grade,
            gridcolor=cor_da_grade,
            zeroline=True,               # a "linha do zero" é o próprio eixo
            zerolinecolor=cor_do_eixo,
            zerolinewidth=2,
            color=NORD["nord4"],         # cor dos números dos eixos
            # "domain": para manter a escala igual, o Plotly encolhe a área do
            # gráfico em vez de mudar os intervalos pedidos.
            constrain="domain",
        )

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, x=0.5) if titulo else None,
        height=altura,
        paper_bgcolor=CORES["fundo"],
        plot_bgcolor=CORES["fundo"],
        font=dict(color=CORES["texto"], size=13),
        margin=dict(l=60, r=20, b=50, t=50 if titulo else 20),
        showlegend=False,
        hovermode="closest",
        hoverlabel=dict(bgcolor=NORD["nord1"], font=dict(color=CORES["texto"])),
        xaxis=configurar_eixo(x_min, x_max),
        yaxis=configurar_eixo(y_min, y_max),
    )
    # Escala igual: o eixo y fica "preso" ao eixo x, com razão 1.
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Os nomes dos eixos ("x" e "y", por padrão) nas pontas positivas. Se o zero
    # estiver fora do intervalo visível, o nome fica na borda mais próxima.
    y_do_eixo_x = min(max(0, y_min), y_max)
    x_do_eixo_y = min(max(0, x_min), x_max)
    fig.add_annotation(
        x=x_max, y=y_do_eixo_x, text=f"<i>{rotulo_x}</i>", showarrow=False,
        xanchor="right", yanchor="bottom", font=dict(size=16, color=CORES["texto"]),
    )
    fig.add_annotation(
        x=x_do_eixo_y, y=y_max, text=f"<i>{rotulo_y}</i>", showarrow=False,
        xanchor="left", yanchor="top", xshift=4,
        font=dict(size=16, color=CORES["texto"]),
    )
    return fig


def criar_grafico(
    titulo: str | None = None,
    rotulo_x: str | None = None,
    rotulo_y: str | None = None,
    altura: int = 400,
    proporcao_igual: bool = False,
) -> go.Figure:
    """Cria um gráfico 2D vazio, para curvas que não são figuras geométricas.

    É a mesma função do `utils_3d.py` (módulo 06), com o mesmo nome e os mesmos
    parâmetros. Use-a, por exemplo, para o gráfico de uma conta em função de uma
    coordenada. Ao contrário de `criar_plano_cartesiano`, aqui cada eixo se ajusta
    sozinho ao espaço disponível (a escala NÃO é igual nos dois eixos, a menos que
    você peça) e os intervalos não são fixos.

    Parâmetros:
        titulo: texto que aparece no alto do gráfico (opcional).
        rotulo_x: nome do eixo horizontal (opcional).
        rotulo_y: nome do eixo vertical (opcional).
        altura: altura do gráfico, em pixels.
        proporcao_igual: se True, uma unidade no eixo vertical tem o mesmo tamanho
            que uma unidade no eixo horizontal.

    Retorna:
        Uma figura do Plotly (`go.Figure`), pronta para receber curvas e pontos.

    Exemplo:
        >>> fig = criar_grafico(rotulo_x="altura de C", rotulo_y="D")
    """
    # A grade usa a cor de apoio bem transparente: ela ajuda a ler os valores sem
    # chamar mais atenção do que a curva.
    grade = _cor_transparente(CORES["grade"], _OPACIDADE_GRADE)
    eixo = _cor_transparente(CORES["eixo"], _OPACIDADE_EIXO)
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, x=0.5) if titulo else None,
        height=altura,
        paper_bgcolor=CORES["fundo"],
        plot_bgcolor=CORES["fundo"],
        font=dict(color=CORES["texto"], size=13),
        margin=dict(l=60, r=20, b=50, t=50 if titulo else 20),
        showlegend=True,
        hovermode="closest",
        hoverlabel=dict(bgcolor=NORD["nord1"], font=dict(color=CORES["texto"])),
        xaxis=dict(title=rotulo_x, gridcolor=grade, zerolinecolor=eixo,
                   color=NORD["nord4"]),
        yaxis=dict(title=rotulo_y, gridcolor=grade, zerolinecolor=eixo,
                   color=NORD["nord4"]),
    )
    if proporcao_igual:
        # O eixo y fica "preso" ao eixo x, com a mesma escala (razão 1).
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
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
# Desenho no plano
# ==================================================================================
def desenhar_ponto(
    fig: go.Figure,
    ponto: Ponto,
    rotulo: str | None = None,
    cor: str | None = None,
    tamanho: float = 9,
    mostrar_coordenadas: bool = False,
    posicao_rotulo: str = "top right",
) -> go.Figure:
    """Desenha um ponto, com um rótulo opcional ao lado.

    Ao passar o mouse sobre o ponto, a figura mostra o nome e as coordenadas dele,
    por exemplo A(2, 3).

    Parâmetros:
        fig: a figura (criada com `criar_plano_cartesiano`).
        ponto: a posição do ponto, como (x, y).
        rotulo: nome do ponto, por exemplo "A" (opcional).
        cor: cor do ponto. Se não for informada, usa CORES["ponto"].
        tamanho: tamanho do marcador.
        mostrar_coordenadas: se True, escreve as coordenadas junto do nome, como
            "A(2, 3)" ou, com decimais, "M(1,5; 2)".
        posicao_rotulo: onde o texto fica em relação ao ponto, no formato do Plotly
            ("top right", "bottom center", "middle left" etc.).

    Retorna:
        A mesma figura, com o ponto desenhado.

    Exemplo:
        >>> fig = desenhar_ponto(fig, (2, 3), rotulo="A", mostrar_coordenadas=True)
    """
    x, y = _como_ponto_2d(ponto)
    cor = cor or CORES["ponto"]
    texto = _formatar_ponto(rotulo, (x, y)) if mostrar_coordenadas else rotulo
    fig.add_trace(
        go.Scatter(
            x=[x], y=[y],
            mode="markers+text" if texto else "markers",
            marker=dict(size=tamanho, color=cor),
            text=[texto] if texto else None,
            textposition=posicao_rotulo,
            textfont=dict(color=cor, size=14),
            # O texto que aparece ao passar o mouse: "A(2, 3)". O "<extra></extra>"
            # esconde a caixinha extra que o Plotly mostraria ao lado.
            hovertemplate=_formatar_ponto(rotulo, (x, y)) + "<extra></extra>",
            # Deixa o texto de um ponto perto da borda aparecer inteiro, sem ser cortado.
            cliponaxis=False,
            showlegend=False,
        )
    )
    return fig


def desenhar_pontos(
    fig: go.Figure,
    pontos: Sequence[Ponto],
    rotulos: Sequence[str] | None = None,
    cor: str | None = None,
    tamanho: float = 9,
    mostrar_coordenadas: bool = False,
) -> go.Figure:
    """Desenha vários pontos de uma vez.

    Parâmetros:
        fig: a figura.
        pontos: lista de pontos, cada um como (x, y).
        rotulos: lista de nomes, na mesma ordem dos pontos (opcional).
        cor: cor dos pontos. Se não for informada, usa CORES["ponto"].
        tamanho: tamanho dos marcadores.
        mostrar_coordenadas: se True, escreve as coordenadas junto de cada nome.

    Retorna:
        A mesma figura, com os pontos desenhados.

    Levanta:
        ValueError: se a lista de rótulos não tiver um nome para cada ponto.

    Exemplo:
        >>> fig = desenhar_pontos(fig, [(0, 0), (3, 4)], rotulos=["O", "P"])
    """
    if rotulos is not None and len(rotulos) != len(pontos):
        raise ValueError("A lista de rótulos precisa ter um nome para cada ponto.")
    for indice, ponto in enumerate(pontos):
        rotulo = rotulos[indice] if rotulos is not None else None
        fig = desenhar_ponto(
            fig, ponto, rotulo=rotulo, cor=cor, tamanho=tamanho,
            mostrar_coordenadas=mostrar_coordenadas,
        )
    return fig


def desenhar_segmento(
    fig: go.Figure,
    inicio: Ponto,
    fim: Ponto,
    rotulo: str | None = None,
    cor: str | None = None,
    espessura: float = 3,
    tracejado: bool = False,
) -> go.Figure:
    """Desenha o segmento de reta que liga dois pontos.

    Parâmetros:
        fig: a figura.
        inicio: uma das pontas do segmento, como (x, y).
        fim: a outra ponta.
        rotulo: texto escrito no meio do segmento, um pouco afastado dele (opcional).
        cor: cor do segmento. Se não for informada, usa CORES["reta"].
        espessura: espessura da linha.
        tracejado: se True, desenha a linha tracejada.

    Retorna:
        A mesma figura, com o segmento desenhado.

    Exemplo:
        >>> fig = desenhar_segmento(fig, (0, 0), (4, 3), rotulo="5")
    """
    a = _como_ponto_2d(inicio)
    b = _como_ponto_2d(fim)
    cor = cor or CORES["reta"]
    fig.add_trace(
        go.Scatter(
            x=[a[0], b[0]], y=[a[1], b[1]],
            mode="lines",
            line=dict(color=cor, width=espessura, dash="dash" if tracejado else "solid"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if rotulo:
        # O texto vai para o meio do segmento, empurrado para o lado de cima (ou para
        # a direita, se o segmento for vertical), para não ficar em cima da linha.
        meio = (a + b) / 2
        direcao = b - a
        lado = np.array([-direcao[1], direcao[0]])   # perpendicular ao segmento
        if lado[1] < 0 or (lado[1] == 0 and lado[0] < 0):
            lado = -lado
        fig = _escrever(
            fig, meio, rotulo, cor, 14,
            posicao_do_texto=_posicao_do_texto(lado[0], lado[1]),
        )
    return fig


def desenhar_poligono(
    fig: go.Figure,
    vertices: Sequence[Ponto],
    rotulos: Sequence[str] | None = None,
    cor: str | None = None,
    preenchido: bool = True,
    opacidade: float = 0.2,
    espessura: float = 2,
) -> go.Figure:
    """Desenha um polígono fechado, com o interior pintado (opcional).

    Parâmetros:
        fig: a figura.
        vertices: os vértices, cada um como (x, y), na ordem do contorno. Não é
            preciso repetir o primeiro no fim: o polígono é fechado automaticamente.
        rotulos: nomes dos vértices, na mesma ordem (opcional). Com eles, os vértices
            são marcados e nomeados, com o nome do lado de fora do polígono.
        cor: cor do contorno e do interior. Se não for informada, usa CORES["solido"].
        preenchido: se True, pinta o interior.
        opacidade: opacidade do interior, de 0 a 1 (o contorno fica opaco).
        espessura: espessura do contorno.

    Retorna:
        A mesma figura, com o polígono desenhado.

    Levanta:
        ValueError: se houver menos de 3 vértices, ou se a lista de rótulos não
            tiver um nome para cada vértice.

    Exemplo:
        >>> fig = desenhar_poligono(fig, [(0, 0), (4, 0), (1, 3)], rotulos=["A", "B", "C"])
    """
    if len(vertices) < 3:
        raise ValueError("Um polígono precisa de pelo menos 3 vértices.")
    if rotulos is not None and len(rotulos) != len(vertices):
        raise ValueError("A lista de rótulos precisa ter um nome para cada vértice.")
    cor = cor or CORES["solido"]
    pontos = [_como_ponto_2d(v) for v in vertices]
    # Repetimos o primeiro vértice no fim da lista para a linha "voltar" ao início.
    xs = [p[0] for p in pontos] + [pontos[0][0]]
    ys = [p[1] for p in pontos] + [pontos[0][1]]
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="lines",
            # "toself" pinta a região cercada pela própria linha.
            fill="toself" if preenchido else None,
            fillcolor=_cor_transparente(cor, opacidade) if preenchido else None,
            line=dict(color=cor, width=espessura),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if rotulos is not None:
        # Para escrever cada nome do lado de fora, olhamos para onde o vértice fica
        # em relação ao "meio" do polígono (a média dos vértices, usada aqui só para
        # posicionar os textos).
        meio = np.mean(pontos, axis=0)
        for ponto, rotulo in zip(pontos, rotulos):
            direcao = ponto - meio
            fig = desenhar_ponto(
                fig, ponto, rotulo=rotulo, cor=cor,
                posicao_rotulo=_posicao_do_texto(direcao[0], direcao[1]),
            )
    return fig


def desenhar_sentido_de_percurso(
    fig: go.Figure,
    vertices: Sequence[Ponto],
    cor: str | None = None,
    fechado: bool = True,
) -> go.Figure:
    """Desenha uma setinha no meio de cada lado, mostrando a ordem dos vértices.

    Cada seta vai de 40% a 60% do caminho entre um vértice e o seguinte (com o
    `ponto_do_segmento` do notebook 01), apontando para o vértice seguinte na lista.

    Atenção: esta função NÃO decide se o percurso é horário ou anti-horário. Ela só
    mostra a ordem em que os vértices foram dados. Descobrir o sentido é com você.

    Parâmetros:
        fig: a figura.
        vertices: os vértices, cada um como (x, y), na ordem do percurso.
        cor: cor das setas. Se não for informada, usa CORES["texto"].
        fechado: se True, desenha também a seta do último vértice de volta ao
            primeiro (o percurso de um polígono). Se False, o percurso termina no
            último vértice.

    Retorna:
        A mesma figura, com as setas desenhadas.

    Levanta:
        ValueError: se houver menos de 2 vértices.

    Exemplo:
        >>> fig = desenhar_sentido_de_percurso(fig, [(0, 0), (4, 0), (1, 3)])
    """
    if len(vertices) < 2:
        raise ValueError("Um percurso precisa de pelo menos 2 vértices.")
    cor = cor or CORES["texto"]
    pontos = [_como_ponto_2d(v) for v in vertices]
    quantidade_de_lados = len(pontos) if fechado else len(pontos) - 1
    for i in range(quantidade_de_lados):
        # O "% len(pontos)" faz o vértice depois do último ser o primeiro.
        atual, seguinte = pontos[i], pontos[(i + 1) % len(pontos)]
        fig = desenhar_seta(
            fig,
            ponto_do_segmento(atual, seguinte, 0.4),
            ponto_do_segmento(atual, seguinte, 0.6),
            cor=cor, espessura=2.5,
        )
    return fig


def desenhar_seta(
    fig: go.Figure,
    inicio: Ponto,
    fim: Ponto,
    rotulo: str | None = None,
    cor: str | None = None,
    espessura: float = 2,
) -> go.Figure:
    """Desenha uma seta que sai de `inicio` e tem a ponta em `fim`.

    No notebook 03 ela será usada para representar vetores.

    Parâmetros:
        fig: a figura.
        inicio: o ponto de onde a seta sai, como (x, y).
        fim: o ponto onde fica a ponta da seta.
        rotulo: texto escrito perto do meio da seta, um pouco afastado dela
            (opcional).
        cor: cor da seta. Se não for informada, usa CORES["reta"].
        espessura: espessura da linha.

    Retorna:
        A mesma figura, com a seta desenhada. Se `inicio` e `fim` forem o mesmo
        ponto, não há seta para desenhar e a figura volta sem mudanças.

    Exemplo:
        >>> fig = desenhar_seta(fig, (1, 1), (4, 3), rotulo="AB")
    """
    a = _como_ponto_2d(inicio)
    b = _como_ponto_2d(fim)
    if np.all(np.abs(b - a) < TOLERANCIA):
        return fig
    cor = cor or CORES["reta"]
    # No Plotly, uma seta é uma "anotação": (ax, ay) é a cauda e (x, y) é a ponta.
    # O "axref"/"ayref" igual a "x"/"y" diz que a cauda também é dada em coordenadas
    # do gráfico (e não em pixels).
    fig.add_annotation(
        x=b[0], y=b[1], ax=a[0], ay=a[1],
        xref="x", yref="y", axref="x", ayref="y",
        text="", showarrow=True,
        arrowhead=2, arrowsize=1.2, arrowwidth=espessura, arrowcolor=cor,
    )
    if rotulo:
        # Como em desenhar_segmento: o texto vai para o meio, empurrado para o lado.
        meio = (a + b) / 2
        direcao = b - a
        lado = np.array([-direcao[1], direcao[0]])
        if lado[1] < 0 or (lado[1] == 0 and lado[0] < 0):
            lado = -lado
        fig = _escrever(
            fig, meio, rotulo, cor, 14,
            posicao_do_texto=_posicao_do_texto(lado[0], lado[1]),
        )
    return fig


def desenhar_reta_por_dois_pontos(
    fig: go.Figure,
    a: Ponto,
    b: Ponto,
    rotulo: str | None = None,
    cor: str | None = None,
    espessura: float = 2,
    tracejado: bool = True,
) -> go.Figure:
    """Desenha a reta que passa por A e B, atravessando toda a janela da figura.

    As equações da reta serão estudadas no notebook 04. Aqui a reta é só desenhada.

    A janela é lida dos intervalos da figura (os x_min, x_max, y_min e y_max usados
    em `criar_plano_cartesiano`). Se a reta passar fora da janela, nada é desenhado.

    Parâmetros:
        fig: a figura (criada com `criar_plano_cartesiano`).
        a: um ponto da reta, como (x, y).
        b: outro ponto da reta, diferente de A.
        rotulo: nome da reta, escrito perto de uma das pontas visíveis (opcional).
        cor: cor da reta. Se não for informada, usa CORES["apoio"].
        espessura: espessura da linha.
        tracejado: se True (o padrão), desenha a linha tracejada.

    Retorna:
        A mesma figura, com a reta desenhada.

    Levanta:
        ValueError: se A e B forem o mesmo ponto, ou se a figura não tiver os
            intervalos dos eixos definidos.

    Exemplo:
        >>> fig = desenhar_reta_por_dois_pontos(fig, (-3, -2), (4, 1), rotulo="r")
    """
    pa, pb = _como_ponto_2d(a), _como_ponto_2d(b)
    if np.all(np.abs(pb - pa) < TOLERANCIA):
        raise ValueError(
            "Os dois pontos são iguais: é preciso dois pontos distintos para "
            "determinar uma reta."
        )
    if fig.layout.xaxis.range is None or fig.layout.yaxis.range is None:
        raise ValueError(
            "A figura precisa ter os intervalos dos eixos definidos: crie-a com "
            "criar_plano_cartesiano."
        )
    x_min, x_max = fig.layout.xaxis.range
    y_min, y_max = fig.layout.yaxis.range
    pedaco = _recortar_reta_na_janela(pa, pb, x_min, x_max, y_min, y_max)
    if pedaco is None:
        return fig
    inicio, fim = pedaco
    cor = cor or CORES["apoio"]
    fig.add_trace(
        go.Scatter(
            x=[inicio[0], fim[0]], y=[inicio[1], fim[1]],
            mode="lines",
            line=dict(color=cor, width=espessura, dash="dash" if tracejado else "solid"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if rotulo:
        # O nome fica perto de uma das pontas (a 8% do comprimento visível), do lado
        # de cima da reta (ou à direita, se ela for vertical).
        posicao = inicio + 0.92 * (fim - inicio)
        direcao = fim - inicio
        lado = np.array([-direcao[1], direcao[0]])
        if lado[1] < 0 or (lado[1] == 0 and lado[0] < 0):
            lado = -lado
        fig = _escrever(
            fig, posicao, rotulo, cor, 14,
            posicao_do_texto=_posicao_do_texto(lado[0], lado[1]),
        )
    return fig


def desenhar_projecoes(
    fig: go.Figure, ponto: Ponto, cor: str | None = None
) -> go.Figure:
    """Desenha as linhas tracejadas que ligam um ponto aos dois eixos.

    A linha vertical desce (ou sobe) até o eixo x, na abscissa do ponto; a linha
    horizontal vai até o eixo y, na ordenada do ponto. Elas mostram de onde vêm as
    coordenadas.

    Parâmetros:
        fig: a figura.
        ponto: o ponto, como (x, y).
        cor: cor das linhas. Se não for informada, usa CORES["projecao"].

    Retorna:
        A mesma figura, com as duas linhas desenhadas.

    Exemplo:
        >>> fig = desenhar_projecoes(fig, (3, 2))
    """
    x, y = _como_ponto_2d(ponto)
    cor = cor or CORES["projecao"]
    for fim in [(x, 0), (0, y)]:   # até o eixo x e até o eixo y
        fig.add_trace(
            go.Scatter(
                x=[x, fim[0]], y=[y, fim[1]],
                mode="lines",
                line=dict(color=cor, width=1.5, dash="dot"),
                opacity=_OPACIDADE_PROJECAO,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return fig


def desenhar_triangulo_das_diferencas(
    fig: go.Figure, a: Ponto, b: Ponto, rotulos: bool = True
) -> go.Figure:
    """Desenha o triângulo retângulo "escondido" entre dois pontos A e B.

    São os dois caminhos paralelos aos eixos: de A, anda-se na horizontal até o canto
    (x_B, y_A) e, de lá, na vertical até B. Um quadradinho marca o ângulo reto no
    canto. Esta função APENAS DESENHA os catetos: a distância de A até B é você quem
    calcula.

    Se os pontos estiverem na mesma horizontal (Δy = 0) ou na mesma vertical
    (Δx = 0), só o cateto que existe é desenhado, sem o quadradinho.

    Parâmetros:
        fig: a figura.
        a: o ponto A, como (x, y).
        b: o ponto B, como (x, y).
        rotulos: se True, escreve |Δx| = ... e |Δy| = ... no meio de cada cateto.

    Retorna:
        A mesma figura, com os catetos desenhados.

    Exemplo:
        >>> fig = desenhar_triangulo_das_diferencas(fig, (-3, -1), (3, 2))
    """
    xa, ya = _como_ponto_2d(a)
    xb, yb = _como_ponto_2d(b)
    delta_x, delta_y = xb - xa, yb - ya
    canto = (xb, ya)
    tem_cateto_x = abs(delta_x) > TOLERANCIA
    tem_cateto_y = abs(delta_y) > TOLERANCIA

    if tem_cateto_x:
        fig = desenhar_segmento(
            fig, (xa, ya), canto, cor=CORES["cateto_x"], espessura=3, tracejado=True
        )
        if rotulos:
            # O texto fica do lado de FORA do triângulo: abaixo do cateto se B está
            # acima de A, e acima se B está abaixo.
            posicao = "bottom center" if delta_y >= 0 else "top center"
            fig = _escrever(
                fig, ((xa + xb) / 2, ya), f"|Δx| = {_formatar_numero(abs(delta_x))}",
                CORES["cateto_x"], 13, posicao_do_texto=posicao,
            )
    if tem_cateto_y:
        fig = desenhar_segmento(
            fig, canto, (xb, yb), cor=CORES["cateto_y"], espessura=3, tracejado=True
        )
        if rotulos:
            # Do lado de fora: à direita do cateto se A está à esquerda de B.
            posicao = "middle right" if delta_x >= 0 else "middle left"
            fig = _escrever(
                fig, (xb, (ya + yb) / 2), f" |Δy| = {_formatar_numero(abs(delta_y))} ",
                CORES["cateto_y"], 13, posicao_do_texto=posicao,
            )
    if tem_cateto_x and tem_cateto_y:
        # O quadradinho do ângulo reto: sai do canto em direção a A (na horizontal) e
        # em direção a B (na vertical). O lado é pequeno, mas cabe nos catetos.
        lado = min(0.35, abs(delta_x) / 3, abs(delta_y) / 3)
        passo_x = -lado if delta_x > 0 else lado
        passo_y = lado if delta_y > 0 else -lado
        fig.add_trace(
            go.Scatter(
                x=[xb + passo_x, xb + passo_x, xb],
                y=[ya, ya + passo_y, ya + passo_y],
                mode="lines",
                line=dict(color=CORES["apoio"], width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return fig


def adicionar_rotulo(
    fig: go.Figure,
    posicao: Ponto,
    texto: str,
    cor: str | None = None,
    tamanho: float = 14,
) -> go.Figure:
    """Escreve um texto solto no plano, centralizado na posição dada.

    Parâmetros:
        fig: a figura.
        posicao: onde o texto aparece, como (x, y).
        texto: o texto a escrever.
        cor: cor do texto. Se não for informada, usa CORES["texto"].
        tamanho: tamanho da fonte.

    Retorna:
        A mesma figura, com o texto.

    Exemplo:
        >>> fig = adicionar_rotulo(fig, (2, 5), "1º quadrante")
    """
    return _escrever(fig, posicao, texto, cor or CORES["texto"], tamanho)


def desenhar_circulo_de_alcance(
    fig: go.Figure,
    centro: Ponto,
    raio: float,
    cor: str | None = None,
    opacidade: float = 0.15,
) -> go.Figure:
    """Desenha um disco semitransparente com borda: a região de alcance de alguém.

    A equação da circunferência será estudada no notebook 08. Aqui ela é usada apenas
    para desenhar.

    Parâmetros:
        fig: a figura.
        centro: o centro do disco, como (x, y).
        raio: o raio do disco (positivo).
        cor: cor da borda e do interior. Se não for informada, usa CORES["alcance"].
        opacidade: opacidade do interior, de 0 a 1 (a borda fica opaca).

    Retorna:
        A mesma figura, com o disco desenhado.

    Levanta:
        ValueError: se o raio não for positivo.

    Exemplo:
        >>> fig = desenhar_circulo_de_alcance(fig, (6, 4), 3)
    """
    if raio <= 0:
        raise ValueError("O raio do alcance precisa ser positivo.")
    cor = cor or CORES["alcance"]
    xs, ys = _pontos_da_circunferencia(centro, raio, quantidade=120)
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="lines",
            fill="toself",
            fillcolor=_cor_transparente(cor, opacidade),
            line=dict(color=cor, width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


# ==================================================================================
# Desenho no espaço
# ==================================================================================
# (Esta seção começa no notebook 16, "Sistema de coordenadas no espaço".)


# ==================================================================================
# Ferramentas já estudadas
# ==================================================================================
# Regra das caixas abertas: cada função abaixo foi deduzida e programada por você num
# notebook, e só é usada pelo `uga` a partir do notebook seguinte.
# Todas funcionam no plano, (x, y), e no espaço, (x, y, z).
def distancia(a: Ponto, b: Ponto) -> float:
    """Calcula a distância entre os pontos A e B.

    Deduzida no notebook 01 (Plano cartesiano, ponto e distância).

    É o Teorema de Pitágoras aplicado às diferenças de coordenadas:
    d = √((x_B − x_A)² + (y_B − y_A)²). No espaço, entra também (z_B − z_A)².

    Parâmetros:
        a: o ponto A, como (x, y) ou (x, y, z).
        b: o ponto B, com a mesma quantidade de coordenadas.

    Retorna:
        A distância, um número maior ou igual a zero.

    Levanta:
        ValueError: se os pontos tiverem quantidades diferentes de coordenadas.

    Exemplo:
        >>> distancia((1, 2), (4, 6))
        5.0
    """
    pa, pb = _como_coordenadas(a), _como_coordenadas(b)
    _conferir_mesma_dimensao(pa, pb)
    # Soma dos quadrados das diferenças, coordenada a coordenada, e depois a raiz.
    soma_dos_quadrados = sum((cb - ca) ** 2 for ca, cb in zip(pa, pb))
    return float(math.sqrt(soma_dos_quadrados))


def ponto_medio(a: Ponto, b: Ponto) -> tuple[float, ...]:
    """Calcula o ponto médio do segmento AB.

    Deduzido no notebook 01 (Plano cartesiano, ponto e distância).

    É a média dos dois pontos, coordenada a coordenada: ((x_A + x_B)/2, ...).

    Parâmetros:
        a: o ponto A, como (x, y) ou (x, y, z).
        b: o ponto B, com a mesma quantidade de coordenadas.

    Retorna:
        O ponto médio, como uma tupla.

    Levanta:
        ValueError: se os pontos tiverem quantidades diferentes de coordenadas.

    Exemplo:
        >>> ponto_medio((-2, 5), (6, -1))
        (2.0, 2.0)
    """
    pa, pb = _como_coordenadas(a), _como_coordenadas(b)
    _conferir_mesma_dimensao(pa, pb)
    return tuple(float((ca + cb) / 2) for ca, cb in zip(pa, pb))


def ponto_do_segmento(a: Ponto, b: Ponto, t: float) -> tuple[float, ...]:
    """Calcula o ponto que está na fração t do caminho de A até B.

    Deduzido no notebook 01 (Plano cartesiano, ponto e distância).

    P = (x_A + t·Δx, y_A + t·Δy). Com t = 0, P = A; com t = 1, P = B; com t = 1/2,
    P é o ponto médio. Qualquer t real é aceito: com t < 0 ou t > 1, o ponto sai do
    segmento e anda sobre a reta AB (a equação paramétrica da reta, notebook 04).

    Parâmetros:
        a: o ponto A, como (x, y) ou (x, y, z).
        b: o ponto B, com a mesma quantidade de coordenadas.
        t: a fração do caminho (um número real qualquer).

    Retorna:
        O ponto P, como uma tupla.

    Levanta:
        ValueError: se os pontos tiverem quantidades diferentes de coordenadas.

    Exemplo:
        >>> ponto_do_segmento((-1, 2), (8, -4), 1 / 3)
        (2.0, 0.0)
    """
    pa, pb = _como_coordenadas(a), _como_coordenadas(b)
    _conferir_mesma_dimensao(pa, pb)
    return tuple(float(ca + t * (cb - ca)) for ca, cb in zip(pa, pb))


def baricentro(a: Ponto, b: Ponto, c: Ponto) -> tuple[float, ...]:
    """Calcula o baricentro G do triângulo ABC.

    Deduzido no notebook 01 (Plano cartesiano, ponto e distância).

    É a média dos três vértices, coordenada a coordenada: ((x_A + x_B + x_C)/3, ...).

    Parâmetros:
        a: o vértice A, como (x, y) ou (x, y, z).
        b: o vértice B, com a mesma quantidade de coordenadas.
        c: o vértice C, com a mesma quantidade de coordenadas.

    Retorna:
        O baricentro, como uma tupla.

    Levanta:
        ValueError: se os pontos tiverem quantidades diferentes de coordenadas.

    Exemplo:
        >>> baricentro((2, 1), (7, 3), (3, 8))
        (4.0, 4.0)
    """
    pa, pb, pc = _como_coordenadas(a), _como_coordenadas(b), _como_coordenadas(c)
    _conferir_mesma_dimensao(pa, pb)
    _conferir_mesma_dimensao(pa, pc)
    return tuple(float((ca + cb + cc) / 3) for ca, cb, cc in zip(pa, pb, pc))


def area_orientada(a: Ponto, b: Ponto, c: Ponto) -> float:
    """Calcula a área orientada (área com sinal) do triângulo ABC.

    Deduzida no notebook 02 (Alinhamento e áreas por determinantes).

    É D/2, em que D é o determinante dos três pontos:
    D = x_A(y_B − y_C) + x_B(y_C − y_A) + x_C(y_A − y_B).
    O resultado é positivo quando o percurso A → B → C é anti-horário, negativo
    quando é horário e zero quando os três pontos estão alinhados.

    Parâmetros:
        a: o ponto A, como (x, y).
        b: o ponto B, como (x, y).
        c: o ponto C, como (x, y).

    Retorna:
        A área orientada, um número que pode ser positivo, negativo ou zero.

    Exemplo:
        >>> area_orientada((0, 0), (4, 0), (1, 3))
        6.0
        >>> area_orientada((0, 0), (1, 3), (4, 0))
        -6.0
    """
    xa, ya = _como_ponto_2d(a)
    xb, yb = _como_ponto_2d(b)
    xc, yc = _como_ponto_2d(c)
    determinante = xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb)
    return float(determinante / 2)


def area_triangulo(a: Ponto, b: Ponto, c: Ponto) -> float:
    """Calcula a área do triângulo ABC a partir das coordenadas dos vértices.

    Deduzida no notebook 02 (Alinhamento e áreas por determinantes).

    É o valor absoluto da área orientada: S = |D| / 2.

    Parâmetros:
        a: o vértice A, como (x, y).
        b: o vértice B, como (x, y).
        c: o vértice C, como (x, y).

    Retorna:
        A área, um número maior ou igual a zero (zero se os pontos estão alinhados).

    Exemplo:
        >>> area_triangulo((1, 2), (5, 3), (2, 7))
        9.5
    """
    return abs(area_orientada(a, b, c))


def orientacao(
    a: Ponto, b: Ponto, c: Ponto, tolerancia: float = TOLERANCIA
) -> str:
    """Diz em que sentido é o percurso A → B → C.

    Deduzida no notebook 02 (Alinhamento e áreas por determinantes).

    Usa o sinal do determinante dos três pontos, D: positivo é anti-horário,
    negativo é horário e zero (a menos da tolerância) quer dizer pontos alinhados.
    Os textos devolvidos não têm acento, porque são valores usados no código.

    Parâmetros:
        a: o ponto A, como (x, y).
        b: o ponto B, como (x, y).
        c: o ponto C, como (x, y).
        tolerancia: se |D| for menor ou igual a este número, D conta como zero.

    Retorna:
        Exatamente um destes textos: "anti-horario", "horario" ou "alinhados".

    Exemplo:
        >>> orientacao((0, 0), (4, 0), (1, 3))
        'anti-horario'
    """
    determinante = 2 * area_orientada(a, b, c)
    if abs(determinante) <= tolerancia:
        return "alinhados"
    return "anti-horario" if determinante > 0 else "horario"


def estao_alinhados(
    a: Ponto, b: Ponto, c: Ponto, tolerancia: float = TOLERANCIA
) -> bool:
    """Diz se os pontos A, B e C estão sobre uma mesma reta.

    Deduzida no notebook 02 (Alinhamento e áreas por determinantes).

    Condição de alinhamento: o determinante dos três pontos vale zero.

    Parâmetros:
        a: o ponto A, como (x, y).
        b: o ponto B, como (x, y).
        c: o ponto C, como (x, y).
        tolerancia: se |D| for menor ou igual a este número, D conta como zero.

    Retorna:
        True se os pontos estão alinhados; False se formam um triângulo.

    Exemplo:
        >>> estao_alinhados((0, 5), (2, 5), (7, 5))
        True
    """
    return orientacao(a, b, c, tolerancia) == "alinhados"


def area_poligono(vertices: Sequence[Ponto]) -> float:
    """Calcula a área de um polígono pela fórmula do cadarço.

    Deduzida no notebook 02 (Alinhamento e áreas por determinantes).

    S = ½·|Σ (x_i·y_(i+1) − x_(i+1)·y_i)|, com o vértice depois do último voltando
    a ser o primeiro. Cada parcela é o dobro da área orientada do triângulo formado
    pela origem e por um lado do polígono.

    Os vértices devem estar em ordem (seguindo o contorno), sem que os lados se
    cruzem. O sentido (horário ou anti-horário) não importa.

    Parâmetros:
        vertices: os vértices, cada um como (x, y), na ordem do contorno.

    Retorna:
        A área do polígono.

    Levanta:
        ValueError: se houver menos de 3 vértices.

    Exemplo:
        >>> area_poligono([(0, 0), (6, 0), (5, 4), (1, 5)])
        22.5
    """
    if len(vertices) < 3:
        raise ValueError("Um polígono precisa de pelo menos 3 vértices.")
    pontos = [_como_ponto_2d(v) for v in vertices]
    n = len(pontos)
    soma = 0.0
    for i in range(n):
        xi, yi = pontos[i]
        xj, yj = pontos[(i + 1) % n]   # o "% n" faz o último vértice voltar ao primeiro
        soma += xi * yj - xj * yi
    return float(abs(soma) / 2)
