# SVHN com redes totalmente conectadas (PyTorch)

Até onde dá para levar uma **MLP** (rede só com camadas lineares, sem convolução) na classificação de dígitos do
[SVHN](http://ufldl.stanford.edu/housenumbers/)? Este repositório registra o caminho, um experimento de cada vez:
de um classificador linear com 24% de acurácia até uma MLP com **93,9% no teste**, com uma CNN pequena (96,3%) como
referência.

Tudo roda em CPU, com semente fixa, e os resultados são reproduzíveis bit a bit na mesma máquina.

## Resultados

| Exp. | O que mudou | Validação | Teste |
|---|---|---|---|
| E1 | Classificador linear | 28,83% | 24,02% |
| E2 | MLP 2×500, entrada em [0, 1] | 58,35% | 55,01% |
| E3 | E2 + normalização por canal | 83,03% | 79,48% |
| E4 | E3 com lr 1e-2 | 19,70% | 19,59% |
| E5 | E3 com lr 1e-4 | 86,19% | 83,45% |
| E6 | E5 com rede maior (1024-512-256) | 86,23% | 83,57% |
| E7 | E6 + dropout 0,3 | 86,91% | 84,55% |
| E8 | E7 + BatchNorm | 87,02% | 84,40% |
| E9 | E8 com lr 1e-3 | 87,01% | 84,39% |
| E10 | E8 com batch 256 | 86,99% | 83,98% |
| E11 | E8 + lr com decaimento cosseno | 87,78% | 85,46% |
| E12 | CNN pequena (referência) | 95,45% | 95,17% |
| E13 | E12 + translação aleatória (referência) | 95,80% | 96,27% |
| E14 | E11 + translação aleatória de ±2 px | 87,84% | 86,21% |
| E15 | E14 com rede 4×2048, dropout 0,1, 60 épocas, lr one-cycle | 92,22% | 90,50% |
| E16 | E15 + normalização de contraste por imagem | 93,23% | 91,69% |
| **E17** | **E16 + rotação e escala (melhor MLP)** | **93,54%** | **92,39%** |
| | E17 + média sobre 25 translações na predição | 94,32% | 93,48% |
| **Final** | **E17 retreinado com todo o treino, com a média** | — | **93,92%** |

As curvas de cada experimento estão em [`resultados/`](resultados/) e a análise completa em
[`relatorio.pdf`](relatorio.pdf).

![Acurácia e loss de validação de todos os experimentos](resultados/comparacao_validacao.png)

## O que aprendi

- **Normalizar a entrada vale mais do que qualquer mudança de arquitetura.** Subtrair a média e dividir pelo desvio
  de cada canal levou a mesma MLP de 58% para 83%.
- **A taxa de aprendizado "padrão" pode estar errada para o seu problema.** Com 1e-3 a MLP sofria overfitting cedo;
  com 1e-4 ganhou 3 pontos; com 1e-2 colapsou e passou a responder sempre o dígito mais frequente.
- **Ler a diferença entre treino e validação diz o que fazer em seguida.** Treino muito acima da validação pede
  regularização (dropout resolveu). Treino *abaixo* da validação, como aconteceu ao ligar a augmentation, pede o
  contrário: mais capacidade e mais épocas. Foi esse diagnóstico que rendeu o maior salto da MLP (+4,4 pontos).
- **Uma MLP não sabe que um dígito deslocado é o mesmo dígito.** Augmentation no treino e média das probabilidades
  sobre pequenas translações na predição compensam parte disso (+0,8 ponto só com a média, sem mudar os pesos).
- **Normalização por imagem** (cada foto pela própria média e desvio) remove variações de brilho e contraste: +1 ponto.
- **A arquitetura certa ainda ganha.** Uma CNN com 23 vezes menos parâmetros supera a melhor MLP em 2,4 pontos no
  teste, sem nenhum dos truques acima.
- **Semente fixa não basta para reproduzir.** A média de 67 milhões de valores em float32 mudava na 7ª casa conforme
  o layout de memória, e isso bastava para alterar o treino. Somas inteiras exatas resolveram.
- **Um fator por experimento, e a validação decide.** Cada experimento parte do melhor anterior na validação; o
  conjunto de teste só é usado para relatar.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `trabalho_svhn.ipynb` | Notebook executado, com todos os experimentos e saídas |
| `svhn_lib.py` | Modelos, laço de treino, augmentation, avaliação e predição |
| `tests/` | 29 testes automatizados da biblioteca |
| `tools/build_notebook.py` | Gera o notebook a partir da biblioteca |
| `tools/gerar_relatorio.py`, `tools/relatorio_texto.py` | Geram o `relatorio.pdf` a partir de `resultados/` |
| `tools/verificar.py` | Re-treina dois experimentos e confere os pesos e as predições |
| `resultados/` | Histórico (JSON), curvas e tabelas de cada experimento |
| `melhor_modelo.pt` | Pesos da MLP final, com a normalização e a configuração |

## Como reproduzir

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pandas matplotlib pillow scipy pyarrow pytest jupyter nbconvert reportlab pypdfium2
.venv/bin/pytest -q                        # 29 testes
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 trabalho_svhn.ipynb   # ~10 h em CPU
.venv/bin/python tools/gerar_relatorio.py  # relatorio.pdf
.venv/bin/python tools/verificar.py        # confere a reprodutibilidade e os arquivos gerados
```

- **Dados:** `torchvision.datasets.SVHN` baixa os splits `train` e `test` em `data/`.
- **Predição de imagens avulsas:** a última seção do notebook classifica as imagens de uma pasta `extra/` (PNG 32×32)
  e grava uma predição por linha, em ordem do nome do arquivo.
- **Semente:** 42 em tudo.
- **Colab:** basta o notebook; a primeira célula grava `svhn_lib.py`.
