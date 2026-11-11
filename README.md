# Trabalho SVHN — Classificador Linear, MLP e CNN

| Entregável | Arquivo |
|---|---|
| Código + experimentos | `trabalho_svhn.ipynb` (executado, com saídas) + `svhn_lib.py` |
| Relatório | `relatorio.pdf` |
| Pesos do melhor modelo | `melhor_modelo.pt` (CNN E8 + média/desvio da normalização) |
| Predições do conjunto adicional | `NOME1_NOME2_RESULTS.txt` (renomear com os nomes do grupo) |

## Resultados

| Exp. | Modelo | Val. | Teste |
|---|---|---|---|
| E1 | Classificador linear | 28,41% | 23,41% |
| E2 | MLP baseline (aula) | 58,35% | 55,01% |
| E3 | + normalização | 83,06% | 79,62% |
| E4 | + MLP maior | 83,48% | 79,77% |
| E5 | + dropout | 81,93% | 78,70% |
| E6 | + BatchNorm | 87,21% | 84,09% |
| E7 | batch 256 + lr cosseno | 88,03% | 85,89% |
| **E8** | **CNN + augmentation (final)** | **96,10%** | **96,18%** |

## Como reproduzir

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pandas matplotlib pillow pyarrow pytest jupyter nbconvert reportlab
.venv/bin/pytest -q                                   # 19 testes
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 trabalho_svhn.ipynb   # ~1h50 em CPU
.venv/bin/python tools/gerar_relatorio.py             # relatorio.pdf
```

- Dados: `torchvision.datasets.SVHN` em `data/`. Se o servidor de Stanford falhar, coloque `train.parquet` e
  `test.parquet` do espelho `ufldl-stanford/svhn` (cropped_digits) em `data_hf/`. É o mesmo conteúdo, na mesma ordem.
- Conjunto adicional: descompacte `images.zip` e coloque as imagens em `extra/`.
- O notebook é gerado por `tools/build_notebook.py`, e o texto do relatório fica em `tools/relatorio_texto.py`.
- Colab: envie `svhn_lib.py` junto com o notebook.
