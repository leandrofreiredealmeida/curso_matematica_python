"""Executa um notebook do curso, contornando os travamentos do nbconvert.

Uso (a partir da raiz do projeto):
    venv/bin/python3 ferramentas/executar_notebook.py <notebook.ipynb> [...]

O que faz:
- executa o notebook com o `nbconvert --execute` padrão (que embute o estado dos
  widgets em `metadata.widgets`);
- NÃO executa as células de exercício e de verificação (as que começam com
  "# Exercício" ou "# Verificação"), porque o placeholder `...` quebraria a execução;
  elas voltam ao notebook com `execution_count: null` e `outputs: []`;
- com o ipykernel 7.3 do `venv`, a execução às vezes trava sem erro (o kernel termina a
  célula, mas a resposta se perde). Por isso usamos um timeout curto por célula e,
  se aparecer `CellTimeoutError`, repetimos a execução inteira. Qualquer outro erro é
  real e interrompe na hora.

Este arquivo é uma ferramenta de manutenção: não faz parte do material do aluno.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parent.parent
PYTHON = RAIZ / "venv" / "bin" / "python3"

TIMEOUT_POR_CELULA = 30
MAXIMO_DE_TENTATIVAS = 15
PREFIXOS_NAO_EXECUTADOS = ("# Exercício", "# Verificação")


def deve_executar(celula: nbformat.NotebookNode) -> bool:
    """Diz se a célula entra na execução (exercícios e verificações ficam de fora)."""
    if celula.cell_type != "code":
        return False
    return not celula.source.lstrip().startswith(PREFIXOS_NAO_EXECUTADOS)


def rodar_nbconvert(caminho: Path) -> None:
    """Executa o notebook com nbconvert, repetindo a execução se ela travar.

    Raises:
        RuntimeError: se a execução tiver um erro de verdade ou travar em todas as
            tentativas.
    """
    comando = [
        str(PYTHON), "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--inplace", "--ExecutePreprocessor.kernel_name=python3",
        f"--ExecutePreprocessor.timeout={TIMEOUT_POR_CELULA}", str(caminho),
    ]
    original = caminho.read_text(encoding="utf-8")
    for tentativa in range(1, MAXIMO_DE_TENTATIVAS + 1):
        # Cada tentativa parte do notebook original, sem saídas parciais.
        caminho.write_text(original, encoding="utf-8")
        resultado = subprocess.run(comando, capture_output=True, text=True)
        if resultado.returncode == 0:
            print(f"  executado na tentativa {tentativa}")
            return
        if "CellTimeoutError" not in resultado.stderr:
            print(resultado.stderr[-3000:])
            raise RuntimeError("Erro de verdade na execução (não é travamento).")
        print(f"  tentativa {tentativa} travou; repetindo...")
    raise RuntimeError(f"Travou em todas as {MAXIMO_DE_TENTATIVAS} tentativas.")


def executar_notebook(caminho: Path) -> None:
    """Executa as células executáveis e mescla de volta as de exercício, sem saída."""
    notebook = nbformat.read(caminho, as_version=4)
    executaveis = [celula for celula in notebook.cells if deve_executar(celula)]

    # O temporário fica na mesma pasta, para que imports locais (utils_3d) funcionem.
    temporario = caminho.with_name(f"_tmp_{caminho.name}")
    copia = nbformat.v4.new_notebook(cells=executaveis, metadata=notebook.metadata)
    nbformat.write(copia, temporario)
    try:
        rodar_nbconvert(temporario)
        executado = nbformat.read(temporario, as_version=4)
    finally:
        temporario.unlink(missing_ok=True)

    celulas_executadas = iter(executado.cells)
    celulas = []
    for celula in notebook.cells:
        if deve_executar(celula):
            celulas.append(next(celulas_executadas))
        else:
            if celula.cell_type == "code":
                celula.execution_count = None
                celula.outputs = []
            celulas.append(celula)

    notebook.cells = celulas
    notebook.metadata = executado.metadata
    nbformat.validate(notebook)
    nbformat.write(notebook, caminho)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for argumento in sys.argv[1:]:
        caminho = Path(argumento).resolve()
        print(caminho.name)
        executar_notebook(caminho)
