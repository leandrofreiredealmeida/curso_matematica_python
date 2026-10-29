# Glossário — Módulo 06: Geometria Espacial

Termos do módulo em ordem alfabética, com uma definição curta e o notebook em que aparecem pela primeira vez. Use para revisar ou para tirar uma dúvida rápida; a explicação completa (com figuras e widgets) está no notebook indicado.

| Termo | Definição curta | Notebook |
|---|---|---|
| Ângulo entre dois planos | 0° se os planos são paralelos ou coincidentes; se são secantes, a medida do **menor** dos diedros que eles formam (entre 0° e 90°). É igual ao ângulo entre retas perpendiculares aos planos. | [0605a](0605a_diedros.ipynb) |
| Ângulo entre duas retas | O menor ângulo formado por elas: entre 0° e 90°. Para retas reversas, é o ângulo entre uma delas e uma paralela à outra que a corta. | [0602a](0602a_paralelismo.ipynb) |
| Ângulo entre reta e plano | 90° se a reta é perpendicular ao plano; 0° se é paralela a ele ou está contida nele; nos outros casos, o ângulo entre a reta e a sua projeção ortogonal sobre o plano (o menor ângulo que ela forma com as retas do plano). | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Ângulo poliédrico | Figura formada por $n \ge 3$ semirretas de mesma origem $V$ (três consecutivas quaisquer não coplanares) e pelos ângulos entre semirretas consecutivas, as **faces**. O triedro é o caso $n = 3$. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Ângulo poliédrico convexo | Ângulo poliédrico em que o plano de cada face deixa todas as outras arestas do mesmo lado. A soma das faces é menor que 360°, e a soma dos diedros fica entre $(n-2) \cdot 180°$ e $n \cdot 180°$. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Ângulo retilíneo | O ângulo obtido numa seção reta de um diedro. A sua medida é a medida do diedro. | [0605a](0605a_diedros.ipynb) |
| Área da projeção | Uma figura plana de área $S$ num plano que forma ângulo $\theta$ com $\beta$ tem projeção ortogonal sobre $\beta$ de área $S' = S \cos\theta$. Telhado: $S_{\text{telhado}} = S_{\text{planta}} / \cos\theta$. | [0605a](0605a_diedros.ipynb) |
| Aresta do diedro | A reta $r$, origem comum das duas faces de um diedro. | [0605a](0605a_diedros.ipynb) |
| Arestas do triedro | As três semirretas $a$, $b$ e $c$ de um triedro $V(a, b, c)$. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Coincidentes (retas ou planos) | São o mesmo objeto: todos os pontos em comum. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Colineares (pontos) | Pontos que pertencem a uma mesma reta. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Conceitos primitivos | Ponto, reta e plano: aceitos sem definição, a partir da intuição. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Concorrentes (retas) | Retas com um único ponto em comum. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Condição de paralelismo entre planos | Se um plano contém duas retas **concorrentes**, ambas paralelas a outro plano, os dois planos são paralelos. | [0602a](0602a_paralelismo.ipynb) |
| Condição de paralelismo entre reta e plano | Se uma reta não está contida num plano e é paralela a uma reta dele, ela é paralela ao plano. | [0602a](0602a_paralelismo.ipynb) |
| Congruência de triedros (FDF, DFD, FFF, DDD) | Dois triedros são congruentes quando há uma correspondência entre as arestas em que as faces e os diedros correspondentes são congruentes. Bastam três elementos: FDF, DFD, FFF ou DDD (este último sem análogo nos triângulos, porque triedros não têm tamanho). | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Contida (reta num plano) | Todos os pontos da reta pertencem ao plano: $r \subset \alpha$. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Coplanares | Pontos ou retas que pertencem a (ou estão contidos em) um mesmo plano. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Defeito angular | A diferença $360° - $ (soma das faces) num vértice: o quanto "falta" para o papel ficar plano. Descartes mostrou que, num poliedro convexo, a soma dos defeitos de todos os vértices é sempre 720°. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Desigualdade das faces | Em todo triedro, cada face é menor que a soma das outras duas ($f_a < f_b + f_c$ etc.). Junto com a soma das faces menor que 360°, é a condição para três ângulos serem as faces de um triedro. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Determinação de um plano | Um plano fica determinado por: três pontos não colineares; uma reta e um ponto fora dela; duas retas concorrentes; ou duas retas paralelas distintas. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Diedro | Figura formada por dois semiplanos de mesma origem que não estão contidos num mesmo plano: $\mathrm{di}(\alpha\, r\, \beta)$. É o "ângulo" do espaço. | [0605a](0605a_diedros.ipynb) |
| Diedro agudo, reto e obtuso | Diedro de medida menor que 90°, igual a 90° (faces em planos perpendiculares) ou maior que 90°, respectivamente. | [0605a](0605a_diedros.ipynb) |
| Diedros adjacentes | Diedros com a mesma aresta e uma face em comum, sem sobreposição. | [0605a](0605a_diedros.ipynb) |
| Diedros complementares e suplementares | Diedros cujas medidas somam 90° (complementares) ou 180° (suplementares). | [0605a](0605a_diedros.ipynb) |
| Diedros congruentes | Diedros com a mesma medida. | [0605a](0605a_diedros.ipynb) |
| Diedros do triedro | Os três diedros $d_a$, $d_b$ e $d_c$ de um triedro, um em cada aresta. A soma deles fica entre 180° e 540°. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Diedros opostos pela aresta | Dois dos quatro diedros formados por dois planos secantes que não são adjacentes. São congruentes, como os ângulos opostos pelo vértice. | [0605a](0605a_diedros.ipynb) |
| Distância de ponto a plano | O comprimento do segmento perpendicular do ponto ao plano: é o menor segmento entre eles. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Distância de um ponto a uma reta | O comprimento do segmento perpendicular que liga o ponto à reta: é o menor segmento entre eles. Com as três perpendiculares, $PM = \sqrt{PO^2 + OM^2}$. | [0603a](0603a_perpendicularidade.ipynb) |
| Distância entre figuras | O comprimento do **menor** segmento que liga um ponto de uma figura a um ponto da outra. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Distância entre planos paralelos | A distância de **qualquer** ponto de um dos planos ao outro (todos dão o mesmo valor). É a altura de um prisma. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Distância entre retas reversas | O comprimento do segmento da perpendicular comum. É igual à distância entre os planos paralelos que contêm as duas retas. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Faces do diedro | Os dois semiplanos $\alpha$ e $\beta$ que formam um diedro. | [0605a](0605a_diedros.ipynb) |
| Faces do triedro | Os três ângulos de um triedro. Cada face recebe o nome da aresta oposta: $f_a = b\hat{V}c$, $f_b = a\hat{V}c$, $f_c = a\hat{V}b$. A soma das faces é menor que 360°. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Feixe de planos | O conjunto de todos os planos que passam por uma mesma reta. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Figura espacial | Figura cujos pontos não estão todos num mesmo plano. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Figura plana | Figura cujos pontos estão todos num mesmo plano. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Fio de prumo | Peso pendurado num fio: pela gravidade, o fio fica na vertical, perpendicular ao plano horizontal. Ferramenta usada desde egípcios e romanos. | [0603a](0603a_perpendicularidade.ipynb) |
| Geometria de Posição | Parte da Geometria Espacial que estuda as posições relativas de pontos, retas e planos, sem medir. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Geometria Descritiva | Método criado por Gaspard Monge no fim do século XVIII para representar objetos 3D por projeções ortogonais em dois ou mais planos. É a base do desenho técnico. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Geometria não euclidiana | Geometria coerente em que o postulado das paralelas é falso (Gauss, Bolyai, Lobachevsky). | [0602a](0602a_paralelismo.ipynb) |
| Inclinação percentual | A tangente do ângulo de inclinação vezes 100: $\tan\theta \cdot 100\%$. Uma inclinação de 100% corresponde a 45°, e não a uma parede vertical. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Lugar geométrico | O conjunto de **todos** os pontos que têm uma certa propriedade, e **somente** eles. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Medida de um diedro | A medida do seu ângulo retilíneo (de qualquer seção reta). | [0605a](0605a_diedros.ipynb) |
| Paralelas (retas) | Retas coplanares sem ponto em comum (ou coincidentes). | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Pé da perpendicular | O ponto $O$ onde uma reta perpendicular a um plano (ou a uma reta) encontra esse plano (ou essa reta). | [0603a](0603a_perpendicularidade.ipynb) |
| Perpendicular comum | A única reta concorrente com duas retas reversas e perpendicular a ambas. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Planificação | As faces de uma figura "abertas" lado a lado num plano, como o papel antes de ser dobrado. Na planificação de um triedro, as três faces ficam em volta do vértice, e sobra uma folga de $360° - $ (soma das faces). | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Plano mediador | Plano perpendicular a um segmento $AB$ pelo seu ponto médio: o lugar geométrico dos pontos equidistantes de $A$ e $B$. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Planos bissetores | Os dois planos formados pelos pontos equidistantes de dois planos secantes. Contêm a reta de interseção e são perpendiculares entre si. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Planos paralelos | Planos sem ponto em comum. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Planos perpendiculares | Dois planos tais que um deles contém uma reta perpendicular ao outro: $\alpha \perp \beta$. | [0603a](0603a_perpendicularidade.ipynb) |
| Planos secantes | Planos distintos que se cortam numa reta. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Postulado | Afirmação aceita sem demonstração; as "regras do jogo" da Geometria. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Postulado da inclusão | Se uma reta tem dois pontos distintos num plano, ela está contida nele. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Postulado da interseção | Se dois planos distintos têm um ponto em comum, eles têm uma reta em comum passando por esse ponto. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Postulado das paralelas (de Euclides) | Por um ponto fora de uma reta passa uma única reta paralela a ela. | [0602a](0602a_paralelismo.ipynb) |
| Projeção ortogonal | A projeção de um ponto $P$ sobre um plano $\alpha$ é o pé $P'$ da perpendicular a $\alpha$ por $P$; a de uma figura é o conjunto das projeções de todos os seus pontos. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Projetante | A reta $PP'$ que liga um ponto $P$ à sua projeção ortogonal $P'$. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Reta concorrente (secante) a um plano | Reta com exatamente um ponto em comum com o plano: ela "fura" o plano. | [0602a](0602a_paralelismo.ipynb) |
| Reta de maior declive | Reta de um plano $\alpha$ perpendicular à reta de interseção $i$ de $\alpha$ com outro plano $\beta$. Entre as retas de $\alpha$, é a que forma o **maior** ângulo com $\beta$. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Reta oblíqua a um plano | Reta concorrente com o plano que não é perpendicular a ele. | [0603a](0603a_perpendicularidade.ipynb) |
| Reta paralela a um plano | Reta sem ponto em comum com o plano. | [0602a](0602a_paralelismo.ipynb) |
| Reta perpendicular a um plano | Reta concorrente com o plano que forma 90° com **todas** as retas dele (perpendicular às que passam pelo pé, ortogonal às outras): $r \perp \alpha$. | [0603a](0603a_perpendicularidade.ipynb) |
| Reta que se apoia em outra | Reta concorrente com a outra (encosta nela num ponto). | [0602a](0602a_paralelismo.ipynb) |
| Retas ortogonais | Retas **reversas** que formam ângulo de 90°. | [0602a](0602a_paralelismo.ipynb) |
| Retas perpendiculares | Retas **concorrentes** que formam ângulo de 90°. | [0602a](0602a_paralelismo.ipynb) |
| Retas reversas | Retas que não estão num mesmo plano. Nunca se encontram e não são paralelas. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Seção de um diedro | O ângulo obtido cortando o diedro por um plano que corta a aresta num único ponto. Seções oblíquas podem ser maiores ou menores que a abertura verdadeira. | [0605a](0605a_diedros.ipynb) |
| Seção igualmente inclinada | Ângulo $x\hat{O}y$, com $O$ na aresta, $Ox$ numa face e $Oy$ na outra, cujos lados formam com a aresta, do mesmo lado, um mesmo ângulo $\varphi$ (a inclinação). Diedros congruentes ⟺ seções igualmente inclinadas congruentes de mesma inclinação. | [0605a](0605a_diedros.ipynb) |
| Seção reta (seção normal) | Seção de um diedro feita por um plano **perpendicular à aresta**. Todas as seções retas de um diedro são congruentes. | [0605a](0605a_diedros.ipynb) |
| Segmento oblíquo | Segmento $PA$ que liga um ponto $P$ fora de um plano a um ponto $A$ do plano, diferente do pé $O$ da perpendicular. A sua projeção no plano é $OA$, e $PA^2 = PO^2 + OA^2$. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Segmento perpendicular | Segmento $PO$ que liga um ponto $P$ ao pé $O$ da perpendicular a um plano. É menor que qualquer segmento oblíquo traçado de $P$. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Semiespaço | Cada uma das duas partes em que um plano divide o espaço. | [0605a](0605a_diedros.ipynb) |
| Semiplano | Cada uma das duas partes em que uma reta $r$ divide um plano, junto com a própria reta $r$ (a **origem** do semiplano). | [0605a](0605a_diedros.ipynb) |
| Semiplano bissetor | Semiplano com origem na aresta que divide um diedro em dois diedros adjacentes congruentes. Os seus pontos são equidistantes das duas faces. | [0605a](0605a_diedros.ipynb) |
| Superfície esférica | O conjunto dos pontos do espaço que estão a uma distância $R$ (o raio) de um ponto $C$ (o centro). | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Teorema | Afirmação demonstrada a partir dos postulados e de outros teoremas. | [0601a](0601a_introducao_geometria_de_posicao.ipynb) |
| Teorema das três perpendiculares | Se $r \perp \alpha$ em $O$, $s \subset \alpha$ e $OM \perp s$ (com $M \in s$), então $PM \perp s$ para todo $P \in r$. | [0603a](0603a_perpendicularidade.ipynb) |
| Teorema de Tales no espaço | Três planos paralelos cortados por duas transversais determinam segmentos proporcionais: $\frac{AB}{BC} = \frac{A'B'}{B'C'}$. | [0602a](0602a_paralelismo.ipynb) |
| Teorema fundamental da perpendicularidade | Se uma reta forma 90° com **duas retas concorrentes** de um plano, ela é perpendicular ao plano. | [0603a](0603a_perpendicularidade.ipynb) |
| Transitividade do paralelismo | Se $a \parallel b$ e $b \parallel c$, então $a \parallel c$ (vale também no espaço). | [0602a](0602a_paralelismo.ipynb) |
| Triângulo esférico | Figura que as faces de um triedro com vértice no centro de uma esfera marcam na superfície. Os lados correspondem às faces e os ângulos aos diedros; por isso a soma dos ângulos é maior que 180°. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Triedro | Figura formada por três semirretas $a$, $b$ e $c$ de mesma origem $V$, não coplanares, e pelos ângulos $a\hat{V}b$, $b\hat{V}c$ e $a\hat{V}c$. Notação: $V(a, b, c)$. Tem seis elementos: três faces e três diedros. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Triedro polar (suplementar) | Triedro $V(a', b', c')$ em que $a'$ é perpendicular ao plano da face $f_a$, do mesmo lado que $a$ (e o mesmo para $b'$ e $c'$). Suas faces e diedros são suplementos dos diedros e faces do original: $f'_a = 180° - d_a$ e $d'_a = 180° - f_a$. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Triedro trirretângulo | Triedro com as três faces retas, como o canto de uma sala ou o vértice de um cubo. É polar de si mesmo. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Triedros simétricos (espelhados) | Triedros com os mesmos seis elementos, mas "espelhados", como a mão direita e a esquerda: não dá para encaixar um no outro só girando. São congruentes. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
| Verdadeira grandeza | Diz-se de uma figura cuja projeção ortogonal tem o mesmo tamanho e a mesma forma que ela; acontece quando a figura é paralela ao plano de projeção. | [0604a](0604a_projecoes_distancias_e_angulos.ipynb) |
| Vértice do triedro | O ponto $V$, origem comum das três arestas de um triedro. | [0606a](0606a_triedros_e_angulos_poliedricos.ipynb) |
