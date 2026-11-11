# Classificação de Dígitos no SVHN — Classificador Linear, MLP e CNN

**Grupo:** João Pedro, Samyr e Valber

## Entregáveis

| Entregável | Arquivo |
|---|---|
| Código e experimentos | `trabalho_svhn.ipynb` (executado, com as saídas; grava `svhn_lib.py`) |
| Relatório | `relatorio.pdf` |
| Pesos do melhor modelo | `melhor_modelo.pt` (CNN final, com a média e o desvio da normalização) |
| Predições do conjunto adicional | `JOAOPEDRO_SAMYR_VALBER_RESULTS.txt` (203.856 linhas) |

## Resultados

| Exp. | Modificação | Validação | Teste |
|---|---|---|---|
| E1 | Classificador linear | 28,83% | 24,02% |
| E2 | MLP baseline da aula | 58,35% | 55,01% |
| E3 | E2 + normalização | 83,03% | 79,48% |
| E4 | E3 com lr 1e-2 | 19,70% | 19,59% |
| E5 | E3 com lr 1e-4 | 86,19% | 83,45% |
| E6 | E5 com rede maior | 86,23% | 83,57% |
| E7 | E6 + dropout | 86,91% | 84,55% |
| E8 | E7 + BatchNorm | 87,02% | 84,40% |
| E9 | E8 com lr 1e-3 | 87,01% | 84,39% |
| E10 | E8 com batch 256 | 86,99% | 83,98% |
| E11 | E8 + lr cosseno (melhor MLP) | 87,78% | 85,46% |
| E12 | CNN | 95,45% | 95,17% |
| **E13** | **E12 + augmentation (escolhido)** | **95,80%** | **96,27%** |
| **Final** | **E13 retreinado com todo o treino** | — | **96,39%** |

## Como reproduzir

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pandas matplotlib pillow scipy pyarrow pytest jupyter nbconvert reportlab pypdfium2
.venv/bin/pytest -q                        # 23 testes
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 trabalho_svhn.ipynb   # ~4 h em CPU
.venv/bin/python tools/gerar_relatorio.py  # relatorio.pdf
.venv/bin/python tools/verificar.py        # re-treina 2 experimentos e confere os entregáveis
```

- **Dados:** `torchvision.datasets.SVHN` baixa os arquivos em `data/`.
- **Conjunto adicional:** descompacte `images.zip` de modo que as imagens fiquem em `extra/`.
- **Semente:** 42 em tudo. Na mesma máquina, a reexecução reproduz os mesmos números bit a bit.
- **Colab:** basta o notebook; a primeira célula grava `svhn_lib.py`.
- `tools/build_notebook.py` gera o notebook a partir de `svhn_lib.py`; `tools/relatorio_texto.py` tem o texto do relatório.
- `reserva_mlp/` contém as predições e os pesos da melhor MLP (E11). **Não faz parte da entrega**; só serve se o
  modelo final precisar ser uma MLP.
